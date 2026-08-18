import os
import logging
import signal
import dotenv
import time
import volcenginesdkcore
import volcenginesdkecs.models as ecs_models
from volcenginesdkecs.api import ECSApi

from desktop_env.providers.base import VMManager

import random
import fcntl
from contextlib import contextmanager

# Load environment variables from .env file
dotenv.load_dotenv()

for env_name in [
    "VOLCENGINE_ACCESS_KEY_ID",
    "VOLCENGINE_SECRET_ACCESS_KEY",
    "VOLCENGINE_REGION",
    "VOLCENGINE_SUBNET_ID",
    "VOLCENGINE_SECURITY_GROUP_ID",
    "VOLCENGINE_INSTANCE_TYPE",
    "VOLCENGINE_IMAGE_ID",
    "VOLCENGINE_ZONE_ID",
    "VOLCENGINE_DEFAULT_PASSWORD",
]:
    if not os.getenv(env_name):
        raise EnvironmentError(f"{env_name} must be set in the environment variables.")

logger = logging.getLogger("desktopenv.providers.volcengine.VolcengineVMManager")
logger.setLevel(logging.INFO)

# ---- Global throttling for RunInstances (same-host / shared-fs) ----
RUNINST_LOCK_FILE = os.getenv("VOLCENGINE_RUNINST_LOCK_FILE", "/tmp/volcengine_runinstances.lock")
RUNINST_MIN_INTERVAL = float(os.getenv("VOLCENGINE_RUNINST_MIN_INTERVAL", "1.0"))  # seconds
RUNINST_MAX_RETRY = int(os.getenv("VOLCENGINE_RUNINST_MAX_RETRY", "80000"))

# ---- Global throttling for DeleteInstance (same-host / shared-fs) ----
DELINST_LOCK_FILE = os.getenv("VOLCENGINE_DELINST_LOCK_FILE", "/tmp/volcengine_deleteinstance.lock")
DELINST_MIN_INTERVAL = float(os.getenv("VOLCENGINE_DELINST_MIN_INTERVAL", "1.0"))  # seconds
DELINST_MAX_RETRY = int(os.getenv("VOLCENGINE_DELINST_MAX_RETRY", "80000"))

VOLCENGINE_ACCESS_KEY_ID = os.getenv("VOLCENGINE_ACCESS_KEY_ID")
VOLCENGINE_SECRET_ACCESS_KEY = os.getenv("VOLCENGINE_SECRET_ACCESS_KEY")
VOLCENGINE_REGION = os.getenv("VOLCENGINE_REGION")
VOLCENGINE_SUBNET_ID = os.getenv("VOLCENGINE_SUBNET_ID")
VOLCENGINE_SECURITY_GROUP_ID = os.getenv("VOLCENGINE_SECURITY_GROUP_ID")
VOLCENGINE_IMAGE_ID = os.getenv("VOLCENGINE_IMAGE_ID")
VOLCENGINE_ZONE_ID = os.getenv("VOLCENGINE_ZONE_ID")
VOLCENGINE_DEFAULT_PASSWORD = os.getenv("VOLCENGINE_DEFAULT_PASSWORD")

raw_instance_type = os.getenv("VOLCENGINE_INSTANCE_TYPE", "").strip()
if not raw_instance_type:
    raise EnvironmentError("VOLCENGINE_INSTANCE_TYPE must be set in the environment variables.")

# Supports comma-separated values: "a,b,c" or a single "a"
VOLCENGINE_INSTANCE_TYPES = [t.strip() for t in raw_instance_type.split(",") if t.strip()]
if not VOLCENGINE_INSTANCE_TYPES:
    raise EnvironmentError("VOLCENGINE_INSTANCE_TYPE must contain at least one valid instance type.")


@contextmanager
def _file_lock(path: str):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _throttle_by_lockfile(lock_file: str, min_interval: float):
    """
    Ensure any two calls guarded by the same lock_file are spaced by min_interval
    across all local processes (and also across nodes if lock_file is on shared FS).
    """
    with _file_lock(lock_file) as fd:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 64).decode("utf-8", "ignore").strip()
        try:
            last_ts = float(raw) if raw else 0.0
        except Exception:
            last_ts = 0.0

        now = time.time()
        wait = float(min_interval) - (now - last_ts)
        if wait > 0:
            time.sleep(wait + random.uniform(0.0, 0.3))

        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(time.time()).encode("utf-8"))


def _is_flow_limited(e: Exception) -> bool:
    s = str(e)
    return ("AccountFlowLimitExceeded" in s) or ("Too Many Requests" in s) or ("429" in s)


def _is_not_found(e: Exception) -> bool:
    s = str(e)
    # Error-code strings vary across SDKs, so match loosely here
    return ("NotFound" in s) or ("InvalidInstance" in s) or ("InstanceNotFound" in s) or ("does not exist" in s)


def run_instances_with_retry(api_instance: ECSApi, req: ecs_models.RunInstancesRequest):
    """
    Cross-process throttle + exponential backoff on 429 for RunInstances.
    """
    base = 1.0
    for attempt in range(RUNINST_MAX_RETRY):
        try:
            _throttle_by_lockfile(RUNINST_LOCK_FILE, RUNINST_MIN_INTERVAL)
            return api_instance.run_instances(req)
        except Exception as e:
            if _is_flow_limited(e) and attempt < RUNINST_MAX_RETRY - 1:
                sleep_s = min(4.0, base * (2 ** attempt)) + random.uniform(0.0, 1.0)
                logger.warning(
                    f"[RunInstances] rate-limited, backoff {sleep_s:.1f}s "
                    f"(attempt {attempt+1}/{RUNINST_MAX_RETRY}). err={e}"
                )
                time.sleep(sleep_s)
                continue
            raise

import sys

def delete_instance_with_retry(api_instance: ECSApi, instance_id: str):
    """
    Cross-process throttle + exponential backoff on 429 for DeleteInstance.
    Also treat NotFound as success (idempotent cleanup).
    """
    base = 1.0
    req = ecs_models.DeleteInstanceRequest(instance_id=instance_id)

    for attempt in range(DELINST_MAX_RETRY):
        try:
            _throttle_by_lockfile(DELINST_LOCK_FILE, DELINST_MIN_INTERVAL)
            return api_instance.delete_instance(req)
        except Exception as e:
            if _is_not_found(e):
                logger.warning(f"[DeleteInstance] instance {instance_id} not found; treat as success. err={e}")
                return None
            if _is_flow_limited(e) and attempt < DELINST_MAX_RETRY - 1:
                sleep_s = min(10.0, base * (2 ** attempt)) + random.uniform(0.0, 1.0)
                logger.warning(
                    f"[DeleteInstance] rate-limited, backoff {sleep_s:.1f}s "
                    f"(attempt {attempt+1}/{DELINST_MAX_RETRY}). instance={instance_id}. err={e}"
                )
                time.sleep(sleep_s)
                continue
            raise


def _allocate_vm(screen_size=(1920, 1080)):
    """Allocate a Volcengine virtual machine"""

    configuration = volcenginesdkcore.Configuration()
    configuration.region = VOLCENGINE_REGION
    configuration.ak = VOLCENGINE_ACCESS_KEY_ID
    configuration.sk = VOLCENGINE_SECRET_ACCESS_KEY
    configuration.client_side_validation = True
    volcenginesdkcore.Configuration.set_default(configuration)

    api_instance = ECSApi()

    instance_id = None
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def signal_handler(sig, frame):
        nonlocal instance_id
        if instance_id:
            signal_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
            logger.warning(f"Received {signal_name} signal, terminating instance {instance_id}...")
            try:
                delete_instance_with_retry(api_instance, instance_id)
                logger.info(f"Successfully terminated instance {instance_id} after {signal_name}.")
            except Exception as cleanup_error:
                logger.error(f"Failed to terminate instance {instance_id} after {signal_name}: {str(cleanup_error)}")

        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)

        if sig == signal.SIGINT:
            raise KeyboardInterrupt
        else:
            sys.exit(0)

    try:
        # Install signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        last_error = None
        chosen_type = None

        # Try multiple instance types in order
        for itype in VOLCENGINE_INSTANCE_TYPES:
            logger.info(f"Trying to create instance with type: {itype}")
            create_instance_params = ecs_models.RunInstancesRequest(
                image_id=VOLCENGINE_IMAGE_ID,
                instance_type=itype,
                network_interfaces=[ecs_models.NetworkInterfaceForRunInstancesInput(
                    subnet_id=VOLCENGINE_SUBNET_ID,
                    security_group_ids=[VOLCENGINE_SECURITY_GROUP_ID],
                )],
                instance_name=f"{os.getenv('OSWORLD_PROJECT')}-{os.getpid()}-{int(time.time())}",
                volumes=[ecs_models.VolumeForRunInstancesInput(
                    volume_type="ESSD_PL0",
                    size=30,
                )],
                zone_id=VOLCENGINE_ZONE_ID,
                password=VOLCENGINE_DEFAULT_PASSWORD,
                description="OSWorld evaluation instance",
            )

            try:
                response = run_instances_with_retry(api_instance, create_instance_params)
                instance_id = response.instance_ids[0]
                chosen_type = itype
                logger.info(f"Successfully created instance {instance_id} with type {itype}")
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to create instance with type {itype}, "
                    f"trying next candidate if available. Error: {e}"
                )

        if instance_id is None:
            logger.error(
                f"Failed to allocate VM with all candidate instance types: {VOLCENGINE_INSTANCE_TYPES}. "
                f"Last error: {last_error}"
            )
            raise last_error or Exception("Failed to allocate VM with all candidate instance types.")

        logger.info(f"Waiting for instance {instance_id} (type={chosen_type}) to be running...")

        # Wait for the instance to become RUNNING
        while True:
            instance_info = api_instance.describe_instances(ecs_models.DescribeInstancesRequest(
                instance_ids=[instance_id]
            ))
            status = instance_info.instances[0].status
            if status == "RUNNING":
                break
            elif status in ["STOPPED", "ERROR"]:
                raise Exception(f"Instance {instance_id} failed to start, status: {status}")
            time.sleep(5)

        logger.info(f"Instance {instance_id} is ready.")

    except KeyboardInterrupt:
        logger.warning("VM allocation interrupted by user (SIGINT).")
        if instance_id:
            logger.info(f"Terminating instance {instance_id} due to interruption.")
            try:
                delete_instance_with_retry(api_instance, instance_id)
            except Exception:
                logger.exception("delete_instance cleanup failed")
        raise
    except Exception as e:
        logger.error(f"Failed to allocate VM: {e}", exc_info=True)
        if instance_id:
            logger.info(f"Terminating instance {instance_id} due to an error.")
            try:
                delete_instance_with_retry(api_instance, instance_id)
            except Exception:
                logger.exception("delete_instance cleanup failed")
        raise
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)

    return instance_id


class VolcengineVMManager(VMManager):
    """
    Volcengine VM Manager for managing virtual machines on Volcengine.
    """
    def __init__(self, **kwargs):
        self.initialize_registry()

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, lock_needed=True, **kwargs):
        pass

    def _add_vm(self, vm_path):
        pass

    def delete_vm(self, vm_path, lock_needed=True, **kwargs):
        pass

    def _delete_vm(self, vm_path):
        pass

    def occupy_vm(self, vm_path, pid, lock_needed=True, **kwargs):
        pass

    def _occupy_vm(self, vm_path, pid):
        pass

    def check_and_clean(self, lock_needed=True, **kwargs):
        pass

    def _check_and_clean(self):
        pass

    def list_free_vms(self, lock_needed=True, **kwargs):
        pass

    def _list_free_vms(self):
        pass

    def get_vm_path(self, screen_size=(1920, 1080), **kwargs):
        logger.info("Allocating a new VM in region: {region}".format(region=VOLCENGINE_REGION))
        new_vm_path = _allocate_vm(screen_size=screen_size)
        return new_vm_path

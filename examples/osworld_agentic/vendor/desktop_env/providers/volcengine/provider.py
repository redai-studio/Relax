import os
import time
import logging
import volcenginesdkcore
import volcenginesdkecs.models as ecs_models
from volcenginesdkcore.rest import ApiException
from volcenginesdkecs.api import ECSApi

from desktop_env.providers.base import Provider
from desktop_env.providers.volcengine.manager import _allocate_vm, delete_instance_with_retry

logger = logging.getLogger("desktopenv.providers.volcengine.VolcengineProvider")
logger.setLevel(logging.INFO)

WAIT_DELAY = 15
MAX_ATTEMPTS = 10


class VolcengineProvider(Provider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region = os.getenv("VOLCENGINE_REGION", "eu-central-1")
        self.client = self._create_client()
        self._terminated = set()  # best-effort de-dupe delete calls in this process

    def _create_client(self) -> ECSApi:
        configuration = volcenginesdkcore.Configuration()
        configuration.ak = os.getenv('VOLCENGINE_ACCESS_KEY_ID')
        configuration.sk = os.getenv('VOLCENGINE_SECRET_ACCESS_KEY')
        configuration.region = os.getenv('VOLCENGINE_REGION')
        configuration.client_side_validation = True
        volcenginesdkcore.Configuration.set_default(configuration)
        return ECSApi()

    def start_emulator(self, path_to_vm: str, headless: bool, *args, **kwargs):
        logger.info("Starting Volcengine VM...")

        try:
            instance_info = self.client.describe_instances(ecs_models.DescribeInstancesRequest(
                instance_ids=[path_to_vm]
            ))
            status = instance_info.instances[0].status
            logger.info(f"Instance {path_to_vm} current status: {status}")

            if status == 'RUNNING':
                logger.info(f"Instance {path_to_vm} is already running. Skipping start.")
                return

            if status == 'STOPPED':
                self.client.start_instance(ecs_models.StartInstancesRequest(instance_ids=[path_to_vm]))
                logger.info(f"Instance {path_to_vm} is starting...")

                for attempt in range(MAX_ATTEMPTS):
                    time.sleep(WAIT_DELAY)
                    instance_info = self.client.describe_instances(ecs_models.DescribeInstancesRequest(
                        instance_ids=[path_to_vm]
                    ))
                    status = instance_info.instances[0].status

                    if status == 'RUNNING':
                        logger.info(f"Instance {path_to_vm} is now running.")
                        break
                    elif status == 'ERROR':
                        raise Exception(f"Instance {path_to_vm} failed to start")
                    elif attempt == MAX_ATTEMPTS - 1:
                        raise Exception(f"Instance {path_to_vm} failed to start within timeout")
            else:
                logger.warning(f"Instance {path_to_vm} is in status '{status}' and cannot be started.")

        except ApiException as e:
            logger.error(f"Failed to start the Volcengine VM {path_to_vm}: {str(e)}")
            raise

    def get_ip_address(self, path_to_vm: str) -> str:
        logger.info("Getting Volcengine VM IP address...")

        try:
            instance_info = self.client.describe_instances(ecs_models.DescribeInstancesRequest(
                instance_ids=[path_to_vm]
            ))
            private_ip = instance_info.instances[0].network_interfaces[0].primary_ip_address
            return private_ip

        except ApiException as e:
            logger.error(f"Failed to retrieve IP address for the instance {path_to_vm}: {str(e)}")
            raise

    def save_state(self, path_to_vm: str, snapshot_name: str):
        logger.info("Saving Volcengine VM state...")
        try:
            response = self.client.create_image(ecs_models.CreateImageRequest(
                snapshot_id=snapshot_name,
                instance_id=path_to_vm,
                description=f"OSWorld snapshot: {snapshot_name}"
            ))
            image_id = response['image_id']
            logger.info(f"Image {image_id} created successfully from instance {path_to_vm}.")
            return image_id
        except ApiException as e:
            logger.error(f"Failed to create image from the instance {path_to_vm}: {str(e)}")
            raise

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        logger.info(f"Reverting Volcengine VM to snapshot: {snapshot_name}...")

        try:
            # Delete the old instance (throttling + 429 backoff is mandatory here, otherwise this is exactly where you die)
            if path_to_vm not in self._terminated:
                delete_instance_with_retry(self.client, path_to_vm)
                self._terminated.add(path_to_vm)
            logger.info(f"Old instance {path_to_vm} has been deleted.")

            # Create a new instance (RunInstances inside _allocate_vm is already throttled)
            new_instance_id = _allocate_vm()
            logger.info(f"New instance {new_instance_id} launched from image {snapshot_name}.")
            logger.info(f"Waiting for instance {new_instance_id} to be running...")

            while True:
                instance_info = self.client.describe_instances(ecs_models.DescribeInstancesRequest(
                    instance_ids=[new_instance_id]
                ))
                status = instance_info.instances[0].status
                if status == 'RUNNING':
                    break
                elif status in ['STOPPED', 'ERROR']:
                    raise Exception(f"New instance {new_instance_id} failed to start, status: {status}")
                time.sleep(5)

            logger.info(f"Instance {new_instance_id} is ready.")
            return new_instance_id

        except ApiException as e:
            logger.error(f"Failed to revert to snapshot {snapshot_name} for the instance {path_to_vm}: {str(e)}")
            raise

    def stop_emulator(self, path_to_vm, region=None):
        logger.info(f"Stopping Volcengine VM {path_to_vm}...")

        # close/cleanup must never raise on a 429 and kill the worker, or it triggers more restarts / a bigger delete storm
        try:
            if path_to_vm in self._terminated:
                logger.info(f"Instance {path_to_vm} already terminated in this process. Skip delete.")
                return

            # Random jitter: avoid a herd of workers all closing at the same time
            time.sleep(random.uniform(0.0, 1.0))

            delete_instance_with_retry(self.client, path_to_vm)
            self._terminated.add(path_to_vm)
            logger.info(f"Instance {path_to_vm} has been terminated.")

        except Exception as e:
            logger.error(f"Failed to stop the Volcengine VM {path_to_vm}: {str(e)}")
            # Do not raise during the cleanup phase
            return


import random  # placing this at the end of the file is fine too

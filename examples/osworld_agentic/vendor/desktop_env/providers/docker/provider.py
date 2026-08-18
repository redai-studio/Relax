import logging
import os
import platform
import time
import docker
import psutil
import requests
from filelock import FileLock
from pathlib import Path

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 600


class PortAllocationError(Exception):
    pass


class DockerProvider(Provider):
    def __init__(self, region: str):
        self.client = docker.from_env()
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.container = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}  # Modify if needed

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        # Use a per-port-range lock so different model groups (with different VNC port
        # start offsets) don't compete for the same lock, reducing contention.
        vnc_start = int(os.environ.get("DOCKER_VNC_PORT_START", 10006))
        self.lock_file = temp_dir / f"docker_port_alloc_{vnc_start}.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_used_ports(self):
        """Get all currently used ports (both system and Docker)."""
        # psutil.net_connections() scans all sockets and is O(n_connections) — with 1600+
        # vLLM connections it takes ~7 seconds, making each lock hold 30s instead of 3s.
        # Use 'ss -tlnH' (instant) to get only LISTEN ports instead.
        system_ports = set()
        try:
            import subprocess
            out = subprocess.check_output(["ss", "-tlnH"], timeout=3, text=True)
            for line in out.splitlines():
                parts = line.split()
                # format: State Recv-Q Send-Q Local-Address:Port ...
                if len(parts) >= 4:
                    addr_port = parts[3]
                    p = addr_port.rsplit(":", 1)[-1]
                    if p.isdigit():
                        system_ports.add(int(p))
        except Exception:
            pass

        # Get Docker container ports (all=True catches created/starting containers too)
        docker_ports = set()
        try:
            for container in self.client.containers.list(all=True):
                # NetworkSettings['Ports'] only populated for running containers
                ports = container.attrs.get('NetworkSettings', {}).get('Ports') or {}
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
                # HostConfig['PortBindings'] is set even for "Created" (not-yet-started) containers
                host_cfg = container.attrs.get('HostConfig', {}).get('PortBindings') or {}
                for port_mappings in host_cfg.values():
                    if port_mappings:
                        for p in port_mappings:
                            hp = p.get('HostPort')
                            if hp:
                                docker_ports.add(int(hp))
        except Exception:
            # Race condition: a container was removed between list() and inspect().
            # Fall back to the raw API which avoids per-container inspect calls.
            try:
                for c in self.client.api.containers(all=True):
                    for port_info in c.get('Ports', []):
                        pub = port_info.get('PublicPort')
                        if pub:
                            docker_ports.add(pub)
            except Exception:
                pass

        return system_ports | docker_ports

    def _get_available_port(self, start_port: int, used_ports: set = None) -> int:
        """Find next available port starting from start_port."""
        if used_ports is None:
            used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10),
                    proxies={"http": None, "https": None},  # bypass any system proxy
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        # Use a single lock for all port allocation and container startup
        lock = FileLock(str(self.lock_file), timeout=LOCK_TIMEOUT)

        try:
            # Retry loop: if Docker reports a port conflict at start time, re-scan and retry.
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    with lock:
                        # Scan ports once, reuse the set for all 4 allocations
                        # (psutil.net_connections is slow ~7s with many vLLM connections)
                        _used = self._get_used_ports()
                        self.vnc_port = self._get_available_port(int(os.environ.get("DOCKER_VNC_PORT_START", 10006)), _used)
                        _used.add(self.vnc_port)
                        self.server_port = self._get_available_port(int(os.environ.get("DOCKER_SERVER_PORT_START", 5000)), _used)
                        _used.add(self.server_port)
                        self.chromium_port = self._get_available_port(int(os.environ.get("DOCKER_CHROME_PORT_START", 9222)), _used)
                        _used.add(self.chromium_port)
                        self.vlc_port = self._get_available_port(int(os.environ.get("DOCKER_VLC_PORT_START", 8080)), _used)

                        # Start container while still holding the lock
                        # Check if KVM is available
                        devices = []
                        if os.path.exists("/dev/kvm"):
                            devices.append("/dev/kvm")
                            logger.info("KVM device found, using hardware acceleration")
                        else:
                            self.environment["KVM"] = "N"
                            logger.warning("KVM device not found, running without hardware acceleration (will be slower)")

                        self.container = self.client.containers.run(
                            "happysixd/osworld-docker",
                            environment=self.environment,
                            cap_add=["NET_ADMIN"],
                            devices=devices,
                            labels={"osworld-rl": "true"},
                            volumes={
                                os.path.abspath(path_to_vm): {
                                    "bind": "/System.qcow2",
                                    "mode": "ro"
                                }
                            },
                            ports={
                                8006: self.vnc_port,
                                5000: self.server_port,
                                9222: self.chromium_port,
                                8080: self.vlc_port
                            },
                            detach=True
                        )
                        # Brief pause so the Docker daemon registers the port binding
                        # before we release the lock and the next process scans ports.
                        time.sleep(1)
                    break  # success — exit retry loop
                except Exception as run_err:
                    err_str = str(run_err)
                    if ("port is already allocated" in err_str or
                            "address already in use" in err_str.lower()):
                        logger.warning(
                            f"Port conflict on attempt {attempt+1}/{max_retries}: {run_err}. Retrying...")
                        # Clean up any partially-created container
                        if self.container:
                            try:
                                self.container.remove(force=True)
                            except Exception:
                                pass
                            self.container = None
                        time.sleep(2)  # brief pause before retry
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise

            logger.info(f"Started container with ports - VNC: {self.vnc_port}, "
                        f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

            # Wait for VM to be ready
            self._wait_for_vm_ready()

        except Exception as e:
            # Clean up if anything goes wrong
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove()
                except:
                    pass
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        # Note: region parameter is ignored for Docker provider
        # but kept for interface consistency with other providers
        if self.container:
            logger.info("Stopping VM...")
            try:
                self.container.stop()
                self.container.remove()
                time.sleep(WAIT_TIME)
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None

from __future__ import annotations

import logging
import os
import time
import re
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.providers import create_vm_manager_and_provider

logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]

MAX_RETRIES = 5 # Maximum retries for environment setup
            


def _fix_pyautogui_less_than_bug(command: str) -> str:
    """
    Fix PyAutoGUI '<' character bug by converting it to hotkey("shift", ',') calls.
    
    This fixes the known PyAutoGUI issue where typing '<' produces '>' instead.
    References:
    - https://github.com/asweigart/pyautogui/issues/198
    - https://github.com/xlang-ai/OSWorld/issues/257
    
    Args:
        command (str): The original pyautogui command
        
    Returns:
        str: The fixed command with '<' characters handled properly
    """
    # Pattern to match press('<') or press('\u003c') calls  
    press_pattern = r'pyautogui\.press\(["\'](?:<|\\u003c)["\']\)'

    # Handle press('<') calls
    def replace_press_less_than(match):
        return 'pyautogui.hotkey("shift", ",")'
    
    # First handle press('<') calls
    command = re.sub(press_pattern, replace_press_less_than, command)

    # Pattern to match typewrite calls with quoted strings
    typewrite_pattern = r'pyautogui\.typewrite\((["\'])(.*?)\1\)'
    
    # Then handle typewrite calls
    def process_typewrite_match(match):
        quote_char = match.group(1)
        content = match.group(2)
        
        # Preprocess: Try to decode Unicode escapes like \u003c to actual '<'
        # This handles cases where '<' is represented as escaped Unicode
        try:
            # Attempt to decode unicode escapes
            decoded_content = content.encode('utf-8').decode('unicode_escape')
            content = decoded_content
        except UnicodeDecodeError:
            # If decoding fails, proceed with original content to avoid breaking existing logic
            pass  # English comment: Graceful degradation - fall back to original content if decoding fails
        
        # Check if content contains '<'
        if '<' not in content:
            return match.group(0)
        
        # Split by '<' and rebuild
        parts = content.split('<')
        result_parts = []
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part
                if part:
                    result_parts.append(f"pyautogui.typewrite({quote_char}{part}{quote_char})")
            else:
                # Add hotkey for '<' and then typewrite for the rest
                result_parts.append('pyautogui.hotkey("shift", ",")')
                if part:
                    result_parts.append(f"pyautogui.typewrite({quote_char}{part}{quote_char})")
        
        return '; '.join(result_parts)
    
    command = re.sub(typewrite_pattern, process_typewrite_match, command)
    
    return command


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """
    def __init__(
            self,
            provider_name: str = "vmware",
            region: str = None,
            path_to_vm: str = None,
            snapshot_name: str = "init_state",
            action_space: str = "pyautogui",
            cache_dir: str = "cache",
            screen_size: Tuple[int] = (int(os.environ.get("SCREEN_WIDTH", 1920)), int(os.environ.get("SCREEN_HEIGHT", 1080))),
            headless: bool = False,
            require_a11y_tree: bool = True,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
            enable_proxy: bool = False,
            client_password: str = "",
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
            os_type (str): operating system type, default to "Ubuntu"
            enable_proxy (bool): whether to enable proxy support, default to False
        """
        # Initialize VM manager and vitualization provider
        self.region = region
        self.provider_name = provider_name
        self.enable_proxy = enable_proxy  # Store proxy enablement setting
        if client_password == "":
            if self.provider_name == "aws":
                self.client_password = "osworld-public-evaluation"
            else:
                self.client_password = "password"
        else:
            self.client_password = client_password

        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]

        # Default 
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080
        
        # Initialize with default (no proxy) provider
        self.current_use_proxy = False
        self.manager, self.provider = create_vm_manager_and_provider(provider_name, region, use_proxy=False)

        self.os_type = os_type

        # Track whether environment has been used (step/setup) to optimize snapshot revert
        # docker, aws, gcp, azure are always unused as the emulator starts from a clean state
        # vmware, virtualbox are always used as the emulator starts from a dirty state
        if self.provider_name in {"docker", "aws", "gcp", "azure", "aliyun", "volcengine"}:
            self.is_environment_used = False
        elif self.provider_name in {"vmware", "virtualbox"}:
            self.is_environment_used = True
        else:
            raise ValueError(f"Invalid provider name: {self.provider_name}")

        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm))) \
                if provider_name in {"vmware", "virtualbox"} else path_to_vm
        else:
            self.path_to_vm = self.manager.get_vm_path(os_type=self.os_type, region=region, screen_size=(self.screen_width, self.screen_height))
        
        self.snapshot_name = snapshot_name
        self.cache_dir_base: str = cache_dir
        # todo: add the logic to get the screen size from the VM
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal

        # Initialize emulator and controller
        logger.info("Initializing...")
        self._start_emulator()

        # mode: human or machine
        self.instruction = None
        assert action_space in ["computer_13", "pyautogui", "claude_computer_use", "autoglm_computer_use"]
        self.action_space = action_space  # todo: refactor it to the ActType

        # episodic stuffs, like counters, will be updated or reset
        # when calling self.reset()
        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: List[Dict[str, any]] = []


    def _start_emulator(self):
        try:
            # Power on the virtual machine
            self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)

            # Get the ip from the virtual machine, and setup the controller
            vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
            self.vm_ip = vm_ip_ports[0]
            # Get the ports from the virtual machine (for Docker provider only)
            if len(vm_ip_ports) > 1:
                self.server_port = int(vm_ip_ports[1])
                self.chromium_port = int(vm_ip_ports[2])
                self.vnc_port = int(vm_ip_ports[3])
                self.vlc_port = int(vm_ip_ports[4])
            self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port)
            self.setup_controller = SetupController(vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port, vlc_port=self.vlc_port, cache_dir=self.cache_dir_base, client_password=self.client_password, screen_width=self.screen_width, screen_height=self.screen_height)

        except Exception as e:
            try:
                self.provider.stop_emulator(self.path_to_vm)
            except Exception as stop_err:
                logger.warning(f"Cleanup after interrupt failed: {stop_err}")
            raise

    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path 
            
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def close(self):
        # Close (release) the virtual machine
        self.provider.stop_emulator(self.path_to_vm)

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        for attempt in range(MAX_RETRIES):
            # Only revert to snapshot if environment has been used (step/setup)
            # This optimization is especially important for cloud providers like AWS
            # where unnecessary snapshot operations are costly and time-consuming
            
            if task_config is not None:
                # Only consider task proxy requirement if proxy is enabled at system level
                task_use_proxy = task_config.get("proxy", False) and self.enable_proxy
                if not self.enable_proxy and task_config.get("proxy", False):
                    logger.info("Task requires proxy but proxy is disabled at system level, ignoring proxy requirement.")
                
                if task_use_proxy != self.current_use_proxy:
                    # keep because get_info_from_website depend on this
                    self.current_use_proxy = task_use_proxy
            
            if self.is_environment_used:
                logger.info("Environment has been used, reverting to snapshot {}...".format(self.snapshot_name))
                self._revert_to_snapshot()
                logger.info("Starting emulator...")
                self._start_emulator()
                logger.info("Emulator started.")
                # Reset the usage flag after reverting
                self.is_environment_used = False
            else:
                logger.info("Environment is clean, skipping snapshot revert (provider: {}).".format(self.provider_name))

            if task_config is not None:
                if task_config.get("proxy", False) and self.enable_proxy:
                    # If using proxy and proxy is enabled, set up the proxy configuration
                    self.setup_controller._proxy_setup(self.client_password)
                self._set_task_info(task_config)
                self.setup_controller.reset_cache_dir(self.cache_dir)
                logger.info("Setting up environment...")
                success = self.setup_controller.setup(self.config, task_config.get("proxy", False) and self.enable_proxy)
                if success:
                    # Mark environment as used when setup is successfully executed
                    if self.config:  # Only mark as used if there were actual setup operations
                        self.is_environment_used = True
                    break
                else:
                    logger.error(
                        "Environment setup failed, retrying (%d/%d)...",
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(5)
            else:
                break
            
        logger.info("Environment setup complete.")

        # Enable UNO socket on :2002 for MCP libreoffice tools.
        # Mirrors OSWorld-MCP/osworld/desktop_env/desktop_env.py:536 — relies on
        # soffice single-instance behavior: launching with --accept while a
        # soffice is already running adds the listener to the existing
        # process WITHOUT closing the documents opened by _open_setup.
        try:
            self.setup_controller._launch_setup(
                'soffice --accept="socket,host=localhost,port=2002;urp;" --norestore --nologo --nodefault',
                shell=True,
            )
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Failed to enable soffice UNO accept socket: {e}")

        # Ensure MCP server + file stubs are ready BEFORE the final _get_obs,
        # because _get_obs calls get_mcp_tool_list which imports OsworldMcpClient
        # in the VM — and that import fails with NameError if the QCOW's
        # osworld_mcp_client.py is still a 0-byte stub.
        try:
            self._ensure_mcp_server()
        except Exception as e:
            logger.warning(f"Failed to ensure MCP server: {e}")

        observation = self._get_obs()
        return observation

    def maximize_window(self):
        window_state = r"""import subprocess;
command = "xprop -id $(xprop -root _NET_ACTIVE_WINDOW | awk -F' ' '{print $5}') _NET_WM_STATE"
output = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip();
print(output);"""
        for _ in range(5):
            try:
                self.setup_controller._launch_setup('wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz', shell=True)
                time.sleep(2)
                output = self.controller.execute_python_command(window_state)['output'].strip()
                if '_NET_WM_STATE_FOCUSED' not in output or '_NET_WM_STATE_SKIP_TASKBAR' in output or '_NET_WM_STATE_MODAL' in output or '_NET_WM_STATE_MAXIMIZED' in output:
                    return
            except Exception as e:
                logger.error(f"Failed to maximize window: {e}")
                time.sleep(1)

    def _normalize_wm_class(self, raw):
        """Map WM_CLASS string to canonical app name or None."""
        if not raw:
            return None
        s = raw.lower().replace('-', '_')
        if 'libreoffice' in s or 'soffice' in s:
            if 'calc' in s: return 'libreoffice_calc'
            if 'writer' in s: return 'libreoffice_writer'
            if 'impress' in s: return 'libreoffice_impress'
            return None
        if 'code' in s and 'qtcreator' not in s:
            return 'code'
        if 'chrome' in s or 'chromium' in s or 'firefox' in s:
            return 'google_chrome'
        if 'vlc' in s:
            return 'vlc'
        if 'thunderbird' in s:
            return 'thunderbird'
        if 'terminal' in s or 'gnome_terminal' in s or 'xterm' in s:
            return 'os'
        return None

    def get_app_state(self):
        """One VM RPC: returns (cur_app, app_info).

        - cur_app: canonical name from WM_CLASS, e.g. 'libreoffice_calc' (used
          for ToolRetriever filter)
        - app_info: active window title (e.g. "Sample.docx — LibreOffice Writer"
          or "Untitled 1 — LibreOffice Calc - Sheet1") — gives model the current
          file + sheet/slide hint to avoid index errors.

        Always returns; if probe fails, both None.
        """
        code = r"""
import subprocess, re
try:
    aw = subprocess.run("xprop -root _NET_ACTIVE_WINDOW", shell=True, capture_output=True, text=True).stdout
    win_id = aw.strip().split()[-1] if aw.strip() else ''
    if not win_id or win_id == '0x0':
        print('||')
    else:
        cls = subprocess.run(f"xprop -id {win_id} WM_CLASS", shell=True, capture_output=True, text=True).stdout
        m = re.search(r'"([^"]+)",\s*"([^"]+)"', cls)
        wm_class = m.group(2) if m else ''
        title_out = subprocess.run(f"xprop -id {win_id} _NET_WM_NAME", shell=True, capture_output=True, text=True).stdout
        tm = re.search(r'_NET_WM_NAME.*?=\s*"(.*)"\s*$', title_out, re.MULTILINE)
        title = tm.group(1) if tm else ''
        print(f"{wm_class}||{title}")
except Exception:
    print('||')
"""
        try:
            out = self.controller.execute_python_command(code)['output'].strip()
        except Exception as e:
            logger.warning(f"Failed to probe app state: {e}")
            return None, None
        if not out or '||' not in out:
            return None, None
        wm_class, _, title = out.partition('||')
        cur_app = self._normalize_wm_class(wm_class)
        app_info = title.strip() or None
        return cur_app, app_info

    def get_active_app(self):
        """Back-compat: detect active app name only. Calls get_app_state internally."""
        cur_app, _ = self.get_app_state()
        return cur_app

    # Mapping from canonical app name to the MCP Tools class name used by
    # osworld_mcp_client / FastMCP server (matches OSWorld-MCP desktop_env.py:640).
    _APP_TO_MCP_TOOL_NAME = {
        "libreoffice_calc": "libreoffice_calc",
        "libreoffice_impress": "libreoffice_impress",
        "libreoffice_writer": "libreoffice_writer",
        "code": "code",
        "vlc": "vlc",
        "google_chrome": "google_chrome",
        "thunderbird": "thunderbird",
        "os": "os",
    }

    # Keyword -> canonical app inference, used when xprop can't identify the
    # active window (e.g. Start Center, splash screens). Keeps tool_list pointed
    # at the right app from step 0 instead of falling through to the os.* tools.
    _INSTRUCTION_APP_KEYWORDS = [
        ("libreoffice_calc", ("spreadsheet", "workbook", "csv", "pivot", "cell ", "cells",
                              " sheet", "sheets", " column", "columns", " row ", " rows",
                              "libreoffice calc", "excel")),
        ("libreoffice_impress", ("slide", "slides", "presentation", "impress", "powerpoint", "pptx")),
        ("libreoffice_writer", ("paragraph", "subscript", "superscript", "strikethrough", "underline",
                                "document", "writer", "page break", " font", "word doc", ".docx", ".doc ")),
        ("vlc", ("vlc", "playlist", "video player", "media player", "audio track")),
        ("code", ("vs code", "vscode", "visual studio code", "extension", "editor settings")),
        ("google_chrome", ("chrome", "firefox", "browser", "website", "url ", "bookmark",
                           "browser tab", "web page")),
        ("thunderbird", ("thunderbird", "email", "inbox")),
        ("os", ("terminal", "shell", "bash")),
    ]

    @classmethod
    def _infer_app_from_instruction(cls, instruction):
        if not instruction:
            return None
        s = " " + instruction.lower() + " "
        for app, kws in cls._INSTRUCTION_APP_KEYWORDS:
            if any(k in s for k in kws):
                return app
        return None

    # ────────────────────────────────────────────────────────────────────
    # MCP server / tool methods — ported verbatim from
    # OSWorld-MCP/osworld/desktop_env/desktop_env.py:326-590 so we use the
    # same VM-side path (no rebuilds in lib_run_single.py).
    # ────────────────────────────────────────────────────────────────────

    # Source-of-truth files on host for VM injection (the QCOW2 in this repo
    # ships with empty stubs — the same situation OSWorld-MCP handles by
    # injecting at runtime in desktop_env._inject_mcp_files).
    # Override via env var MCP_SRC_ROOT (e.g. to swap in ToolCUA's patched
    # mcp_server for ablation). Both layouts must expose `{root}/mcp/` with
    # `mcp_server/` and `osworld_mcp_client.py` underneath.
    _MCP_SRC_ROOT = os.environ.get(
        "MCP_SRC_ROOT",
        "OSWorld-MCP",
    )

    def _inject_mcp_files_if_empty(self):
        """Idempotent wrapper around _inject_mcp_files: skip if already populated."""
        check_cmd = (
            "import os; "
            "paths = ('/home/user/osworld_mcp_client.py', "
            "'/home/user/mcp_server/server.py', "
            "'/home/user/mcp_server/tools/package/libreoffice_calc.py'); "
            "print('|'.join(str(os.path.getsize(p)) if os.path.exists(p) else 'X' for p in paths))"
        )
        result = self.controller.execute_python_command(check_cmd)
        sizes = ((result or {}).get("output", "") or "").strip()
        logger.info(f"MCP file sizes (client|server|calc_pkg): {sizes}")
        if sizes and all(s.isdigit() and int(s) > 0 for s in sizes.split('|')):
            return
        self._inject_mcp_files()

    def _inject_mcp_files(self):
        """Inject MCP server source files from host into the VM via base64-encoded tar.gz.

        Ported verbatim from
        OSWorld-MCP/osworld/desktop_env/desktop_env.py:266-324.
        """
        import tarfile, io, base64
        from pathlib import Path

        mcp_dir = Path(self._MCP_SRC_ROOT) / "mcp"
        if not mcp_dir.exists():
            logger.warning(f"MCP source dir not found: {mcp_dir}; skipping file injection")
            return

        logger.info(f"Injecting MCP server files from {mcp_dir} into VM...")

        # Build in-memory tar.gz of mcp/ contents
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            for fpath in sorted(mcp_dir.rglob('*')):
                if fpath.is_file() and '__pycache__' not in str(fpath):
                    arcname = str(fpath.relative_to(mcp_dir))
                    tar.add(str(fpath), arcname=arcname)
        tar_bytes = buf.getvalue()
        tar_b64 = base64.b64encode(tar_bytes).decode('ascii')

        # Write base64 data in 40KB chunks to /tmp/mcp_inject.b64 in VM
        chunk_size = 40 * 1024
        chunks = [tar_b64[i:i+chunk_size] for i in range(0, len(tar_b64), chunk_size)]
        logger.info(f"MCP tar.gz: {len(tar_bytes)//1024}KB, {len(chunks)} chunks")

        # Write first chunk (overwrite)
        write_cmd = (
            f"f=open('/tmp/mcp_inject.b64','w'); "
            f"f.write({repr(chunks[0])}); f.close()"
        )
        self.controller.execute_python_command(write_cmd)

        # Append remaining chunks
        for chunk in chunks[1:]:
            append_cmd = (
                f"f=open('/tmp/mcp_inject.b64','a'); "
                f"f.write({repr(chunk)}); f.close()"
            )
            self.controller.execute_python_command(append_cmd)

        # Decode and extract in VM
        extract_cmd = (
            "import base64, tarfile, io; "
            "data = base64.b64decode(open('/tmp/mcp_inject.b64').read()); "
            "tarfile.open(fileobj=io.BytesIO(data), mode='r:gz').extractall('/home/user'); "
            "print('MCP_INJECT_OK')"
        )
        result = self.controller.execute_python_command(extract_cmd)
        if result and 'MCP_INJECT_OK' in result.get('output', ''):
            logger.info("MCP files injected successfully.")
        else:
            logger.error(f"MCP file injection failed: {result}")

    def _start_mcp_server(self):
        """Start the MCP HTTP server in the VM using double-fork daemonization."""
        logger.info("Starting MCP server in VM...")

        # Step 0: make sure the QCOW's MCP file stubs are populated.
        # (setup_mcp_vm.py is the canonical one-time builder; we do the same
        # injection at runtime via the same vm_write_file pattern.)
        self._inject_mcp_files_if_empty()

        # Step 1: verify the injected files exist
        verify_cmd = (
            "import os; "
            "srv = '/home/user/mcp_server/server.py'; "
            "print('SERVER_EXISTS:' + str(os.path.exists(srv))); "
            "print('MSERVER_DIR:' + str(os.listdir('/home/user/mcp_server') if os.path.exists('/home/user/mcp_server') else 'MISSING'))"
        )
        vr = self.controller.execute_python_command(verify_cmd)
        logger.info(f"MCP server dir check: {vr.get('output','')[:200] if vr else 'N/A'}")

        # Step 2: kill any existing server process
        kill_cmd = (
            "import subprocess; "
            "subprocess.run(['pkill', '-f', 'python3.*server.py'], check=False); "
            "print('KILLED_OLD')"
        )
        self.controller.execute_python_command(kill_cmd)
        time.sleep(2)

        # Step 3: double-fork daemonize so the server survives the Flask
        # parent's SIGTERM after the HTTP response is sent. setsid + grandchild
        # adopted by init.
        fork_script = (
            "import os, subprocess, sys\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            "    pid2 = os.fork()\n"
            "    if pid2 == 0:\n"
            "        log = open('/tmp/mcp_server.log', 'w')\n"
            "        subprocess.Popen(['python3', 'server.py'],\n"
            "            cwd='/home/user/mcp_server',\n"
            "            stdout=log, stderr=subprocess.STDOUT,\n"
            "            stdin=open('/dev/null', 'r'),\n"
            "            close_fds=True)\n"
            "        os._exit(0)\n"
            "    else:\n"
            "        os._exit(0)\n"
            "else:\n"
            "    os.waitpid(pid, 0)\n"
            "    print('MCP_FORK_DONE')\n"
        )
        escaped = fork_script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        fork_cmd = f'exec("{escaped}")'
        result = self.controller.execute_python_command(fork_cmd)
        logger.info(f"MCP double-fork result: {result.get('output','')[:100] if result else 'N/A'}")

        time.sleep(10)
        logger.info("MCP server start command issued.")
        self._verify_mcp_server()

    def _verify_mcp_server(self, max_wait=60):
        """Wait until MCP server is fully ready.

        Two-stage check:
          1) TCP-level: port 9292 LISTEN
          2) FastMCP-level: list_tools() returns non-empty via fastmcp Client

        FastMCP binds the socket BEFORE finishing tool registration, so a
        TCP-only check can return OK while list_tools still returns empty.
        Under heavy concurrency (96 envs starting at once), this race caused
        ~80% of list_tools calls to silently return [] and fall back to the
        host-side BM25 registry (lost the entire point of injecting an MCP
        server). Polling list_tools end-to-end fixes it.
        """
        logger.info("Verifying MCP server readiness (TCP + FastMCP list_tools)...")
        tcp_check = (
            "import socket; s = socket.socket(); s.settimeout(2); "
            "rc = s.connect_ex(('localhost', 9292)); s.close(); "
            "print('TCP_OK' if rc == 0 else 'TCP_DOWN:' + str(rc))"
        )
        list_tools_check = (
            "import asyncio, sys; "
            "sys.path.insert(0, '/home/user'); "
            "import os; "
            "os.environ['no_proxy'] = '127.0.0.1,localhost'; "
            "from fastmcp import Client\n"
            "_cfg = {'mcpServers':{'osworld_mcp':{'url':'http://localhost:9292/mcp','transport':'streamable-http'}}}\n"
            "async def _check():\n"
            "    c = Client(_cfg)\n"
            "    async with c:\n"
            "        return len(await c.list_tools())\n"
            "try:\n"
            "    n = asyncio.run(_check())\n"
            "    print(f'FASTMCP_OK:n={n}')\n"
            "except Exception as e:\n"
            "    print(f'FASTMCP_FAIL:{type(e).__name__}:{str(e)[:120]}')"
        )

        deadline = time.time() + max_wait
        while time.time() < deadline:
            tcp_out = ((self.controller.execute_python_command(tcp_check) or {})
                       .get('output', '') or '')
            if 'TCP_OK' in tcp_out:
                tools_out = ((self.controller.execute_python_command(list_tools_check) or {})
                             .get('output', '') or '').strip()
                if 'FASTMCP_OK:n=' in tools_out:
                    n = int(tools_out.split('FASTMCP_OK:n=')[1].split()[0])
                    if n > 0:
                        logger.info(f"MCP server FULLY ready ({tools_out})")
                        return True
                    else:
                        logger.warning(f"MCP server has 0 tools registered ({tools_out}); re-polling")
                else:
                    logger.info(f"MCP TCP up but list_tools not ready: {tools_out[:200]}")
            else:
                logger.debug(f"MCP TCP not up: {tcp_out[:120]}")
            time.sleep(2)

        # Final diagnostic: dump server log so caller can debug
        log_cmd = (
            "import os; "
            "path='/tmp/mcp_server.log'; "
            "print(open(path).read()[-800:] if os.path.exists(path) else 'NO_LOG')"
        )
        log_output = ((self.controller.execute_python_command(log_cmd) or {})
                      .get('output', '') or '')
        logger.error(
            "MCP server did NOT become fully ready within %ds. /tmp/mcp_server.log tail:\n%s",
            max_wait, log_output[:800],
        )
        return False

    def _ensure_mcp_server(self):
        """Check if MCP server is listening on port 9292; restart if not."""
        check_cmd = (
            "import socket; s = socket.socket(); s.settimeout(3); "
            "rc = s.connect_ex(('localhost', 9292)); s.close(); "
            "print('MCP_OK' if rc == 0 else 'MCP_DOWN:errno=' + str(rc))"
        )
        result = self.controller.execute_python_command(check_cmd)
        output = result.get('output', '') if result else ''
        if 'MCP_OK' in output:
            logger.info("MCP server healthy.")
            return
        logger.warning(f"MCP server not listening ({output[:80]}), restarting via double-fork...")
        self._start_mcp_server()

    def call_mcp_tool(self, name, params):
        """Invoke an MCP tool inside the VM via the FastMCP server on :9292.

        Ported verbatim from OSWorld-MCP desktop_env.py:575-587. Returns the
        raw str of CallToolResult (truncated by VM-side print to ~stdout limit).
        """
        ENV_SETTING = (
            "import os, sys; "
            "os.environ['PATH'] = '/home/user/.nvm/versions/node/v22.18.0/bin:/home/user/.local/bin:' + os.environ['PATH']; "
            "sys.path.insert(0, '/home/user'); "
        )
        command = ENV_SETTING
        command += "from osworld_mcp_client import *; "
        command += f"OsworldMcpClient.call_tool(name={name!r}, params={str(params)}); "
        response = self.controller.execute_python_command(command).get('output', '').strip()
        return response

    def get_mcp_tool_list(self, tool_name, instruction=None, shuffle=False, rag=True):
        """Fetch MCP tool list from the live FastMCP server inside the VM.

        Mirrors OSWorld-MCP/osworld/desktop_env/desktop_env.py:548-573 so that
        the model sees the same JSON Schemas and the same RAG-filtered top-K
        set that OSWorld-MCP eval pipeline produces.

        Retries up to 3 times on empty / exception. Under heavy concurrency
        the FastMCP server can be briefly unresponsive even after
        _verify_mcp_server returned; without retry every transient stutter
        causes the agent to silently fall back to the host BM25 registry,
        defeating the point of injecting a live MCP server.
        """
        env_setting = (
            "import os, sys; "
            "os.environ['PATH'] = '/home/user/.nvm/versions/node/v22.18.0/bin:/home/user/.local/bin:' + os.environ['PATH']; "
            "sys.path.insert(0, '/home/user'); "
        )
        escaped_instruction = (instruction or '').replace('\\', '\\\\').replace("'", "\\'")
        command = env_setting
        command += "from osworld_mcp_client import *; "
        command += (
            f"OsworldMcpClient.list_tools(tool_name={tool_name!r}, "
            f"instruction='{escaped_instruction}', shuffle={shuffle}, rag={rag}); "
        )
        last_err = None
        last_raw = None
        for attempt in range(3):
            try:
                raw = self.controller.execute_python_command(command)['output'].strip()
            except Exception as e:
                last_err = f"RPC: {e}"
                time.sleep(1)
                continue
            last_raw = raw
            if not raw:
                last_err = "empty stdout from list_tools"
                time.sleep(1)
                continue
            try:
                parsed = eval(raw)
            except Exception as e:
                last_err = f"eval: {e}"
                time.sleep(1)
                continue
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
            # Empty list is treated as transient — server may still be
            # registering tools. Retry rather than fall back immediately.
            last_err = f"empty tool list (n={len(parsed) if isinstance(parsed, list) else 'N/A'})"
            time.sleep(1)
        logger.warning(
            f"get_mcp_tool_list failed after 3 attempts: {last_err}; "
            f"raw_last={(last_raw or '')[:150]!r}"
        )
        return []

    def _get_obs(self):
        self.maximize_window()
        cur_app, app_info = self.get_app_state()
        tool_name = self._APP_TO_MCP_TOOL_NAME.get(cur_app)
        # When xprop can't recognise the active window (Start Center, splash,
        # blank screen at task start), fall back to instruction-keyword
        # inference so tool_list points at the right app from step 0 instead
        # of the os.* fallback set (which the model can't use to e.g. fill
        # cells in a not-yet-focused Calc workbook).
        if not tool_name:
            tool_name = self._infer_app_from_instruction(getattr(self, "instruction", None))
        tool_list = []
        if tool_name is not None:
            try:
                tool_list = self.get_mcp_tool_list(tool_name, instruction=getattr(self, "instruction", None))
            except Exception as e:
                logger.warning(f"get_mcp_tool_list failed in _get_obs: {e}")
        return {
            "screenshot": self.controller.get_screenshot(),
            "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
            "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
            "instruction": self.instruction,
            "cur_app": cur_app,
            "app_info": app_info,
            "tool_name": tool_name,
            "tool_list": tool_list,
        }

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    def _set_task_info(self, task_config: Dict[str, Any]):
        """Set task info (proxy logic is handled in reset method)"""
        self.task_id: str = task_config["id"]
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config["config"] if "config" in task_config else []
        
        self._set_evaluator_info(task_config)

    def _set_evaluator_info(self, task_config: Dict[str, Any]):
        """Set evaluator information from task config"""
        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric: Metric = [getattr(metrics, func) for func in self.evaluator["func"]] \
            if isinstance(self.evaluator["func"], list) \
            else getattr(metrics, self.evaluator["func"])
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter: Getter = [getattr(getters, "get_{:}".format(res["type"])) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["result"]["type"]))
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter: Getter = [getattr(getters, "get_{:}".format(exp["type"])) if exp else None for exp in
                                            self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else getattr(getters, "get_{:}".format(self.evaluator["expected"]["type"]))
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        assert (not isinstance(self.evaluator["func"], list)
                or (len(self.metric) == len(self.result_getter) == len(self.expected_getter) == len(
                    self.metric_options)))

    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)
        
        # Mark environment as used when step is called
        self.is_environment_used = True

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}
        logger.info(f"Step {self._step_no} in trajectory {self._traj_no} with action: {action}")
        # handle the special actions
        if action in ['WAIT', 'FAIL', 'DONE'] or (type(action) == dict and action['action_type'] in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}

        if self.action_space == "computer_13":
            # the set of all possible actions defined in the action representation
            self.controller.execute_action(action)
        elif self.action_space == "pyautogui" or self.action_space == "claude_computer_use":
            if action in ['WAIT', 'FAIL', 'DONE']:
                self.controller.execute_action(action)
            else:
                # the set of all possible python commands insides `pyautogui`
                if type(action) == str:
                    # Fix PyAutoGUI '<' character bug before execution
                    fixed_command = _fix_pyautogui_less_than_bug(action)
                    self.controller.execute_python_command(fixed_command)
                elif type(action) == dict:
                    # Fix PyAutoGUI '<' character bug before execution
                    fixed_command = _fix_pyautogui_less_than_bug(action['command'])
                    self.controller.execute_python_command(fixed_command)

        time.sleep(pause)
        observation = self._get_obs()

        return observation, reward, done, info

    def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """

        postconfig = self.evaluator.get("postconfig", [])
        self.setup_controller.setup(postconfig, self.enable_proxy)
        # Mark environment as used if there were postconfig setup operations
        if postconfig:
            self.is_environment_used = True

        if self.evaluator['func'] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            # Multiple metrics to evaluate whether the task is successfully completed
            results = []
            assert len(self.metric) == len(self.result_getter), "The number of metrics and result getters must be the same"
            if "expected" in self.evaluator:
                assert len(self.metric) == len(self.expected_getter), "The number of metrics and expected getters must be the same"
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = self.result_getter[idx](self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == 'and':
                        return 0

                if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                    expected_state = self.expected_getter[idx](self, self.evaluator["expected"][idx])
                    metric: int = metric(result_state, expected_state, **self.metric_options[idx])
                else:
                    metric: int = metric(result_state, **self.metric_options[idx])

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)

            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            # Single metric to evaluate whether the task is successfully completed
            try:
                result_state = self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                expected_state = self.expected_getter(self, self.evaluator["expected"])
                metric: float = self.metric(result_state, expected_state, **self.metric_options)
            else:
                metric: float = self.metric(result_state, **self.metric_options)

        return metric

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.controller.get_screenshot()
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))

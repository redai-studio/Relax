"""
unified_tools.py – 将 TerminalSizeTools / SystemVolumeTools / … 等
全部整合到 **一个** 顶层类 UnifiedTools 。

最末尾提供兼容别名：
    TerminalSizeTools   = UnifiedTools
    SystemVolumeTools   = UnifiedTools
    …（共 10 个）
这样 JSON descriptor 中的 name 字段无需改变。
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyautogui


class UnifiedTools:
    ret: str = ""

    @classmethod
    def open_shell(cls):
        try:
            pyautogui.hotkey('ctrl', 'alt', 't')
            cls.ret = "Opened a terminal window."
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to open a terminal window: {exc}"
            return False
    # ──────────────────────────────────────────────────────────────────────
    # 0) 通用 / 内部小工具
    # ──────────────────────────────────────────────────────────────────────
    class _Shell:
        """SystemVolumeTools 原来的安全调用器"""
        @staticmethod
        def run(cmd: str) -> Tuple[int, str, str]:
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = proc.communicate()
            return proc.returncode, out.strip(), err.strip()

    # ──────────────────────────────────────────────────────────────────────
    # TerminalSizeTools  ---------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ts_run(cmd: str) -> Tuple[int, str, str]:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = proc.communicate()
        return proc.returncode, out.strip(), err.strip()

    @classmethod
    def _ts_get_default_profile_uuid(cls) -> str:
        rc, out, _ = cls._ts_run(
            "gsettings get org.gnome.Terminal.ProfilesList default"
        )
        if rc == 0 and out.startswith("'") and out.endswith("'"):
            return out.strip("'")
        return ""

    # ---- public ----------------------------------------------------------
    @classmethod
    def set_default_terminal_size(cls, columns: int, rows: int) -> str:
        if columns <= 0 or rows <= 0:
            return "❌  Columns / rows must be positive integers."
        if cls._ts_run("command -v gsettings")[0] != 0:
            return "❌  gsettings not found in PATH."

        profile_id = cls._ts_get_default_profile_uuid()
        if not profile_id:
            return "❌  Could not detect default GNOME-Terminal profile UUID."

        schema = (
            f"org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/"
            f"legacy/profiles:/:{profile_id}/"
        )
        for cmd in (
            f"gsettings set '{schema}' default-size-columns {columns}",
            f"gsettings set '{schema}' default-size-rows    {rows}",
        ):
            rc, _, err = cls._ts_run(cmd)
            if rc != 0:
                return f"❌  Failed while running: {cmd}\n{err}"

        return (
            f"✅  Persisted default GNOME-Terminal size: {columns}×{rows} "
            f"(profile {profile_id})."
        )

    # ──────────────────────────────────────────────────────────────────────
    # SystemVolumeTools  ---------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sv_detect_default_sink() -> Optional[str]:
        rc, out, _ = UnifiedTools._Shell.run("pactl info")
        if rc != 0:
            return None
        for line in out.splitlines():
            if line.lower().startswith("default sink:"):
                sink = line.split(":", 1)[1].strip()
                return (
                    sink
                    if sink and sink != "auto_null" and sink.lower() != "n/a"
                    else None
                )
        return None

    @classmethod
    def get_volume(cls) -> Dict[str, Any]:
       
        shell_script = '''
DEFAULT_SINK=$(pactl info | awk -F': ' '/Default Sink:/ {print $2}')
CURRENT_VOL=$(pactl get-sink-volume "$DEFAULT_SINK" | head -n1 | awk -F'/' '{print $2}' | xargs)
echo "$CURRENT_VOL"'''.strip()
        try:
            result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)
            success = True
            err = None
            stdout = result.stdout
        except:
            err = "subprocess Err"
            stdout = None
            success = False

        percent = None
        if success:
            try:
                percent = int(result.stdout.strip().rstrip("%"))
            except Exception:
                percent = None
                success = False
                err = f"Unable to parse pactl output: {stdout}"
        if success:
            return {
                "success": True,
                "result": {
                    "volume_percent": percent,
                    "stdout": stdout,
                },
                "error_message": None,
            }
        return {
            "success": False,
            "result": None,
            "error_message": err or f"Unable to parse pactl output: {stdout}",
        }

    @classmethod
    def set_volume(cls, percent: int) -> Dict[str, Any]:

        percent = max(0, min(percent, 100))
        shell_script = '''DEFAULT_SINK=$(pactl info | awk -F': ' '/Default Sink:/ {print $2}')''' + f'''
pactl set-sink-volume "$DEFAULT_SINK" {percent}%
if [ $? -ne 0 ]; then
  echo "Failed to set volume."
  exit 2
else
  echo "Command sent successfully."
fi'''
        try:
            result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "error_message": f"Failed to set volume: {exc}",
            }

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            return {
                "success": False,
                "result": None,
                "error_message": stderr or stdout or "Failed to set volume.",
            }

        if stdout != "Command sent successfully.":
            return {
                "success": False,
                "result": None,
                "error_message": stdout or "Volume command did not confirm success.",
            }

        info = cls.get_volume()
        if not info.get("success"):
            return {
                "success": False,
                "result": None,
                "error_message": info.get("error_message") or "Volume was changed, but reading back the new volume failed.",
            }

        return {
            "success": True,
            "result": {
                "volume_percent": info["result"]["volume_percent"],
                "stdout": stdout,
            },
            "error_message": None,
        }

    # ──────────────────────────────────────────────────────────────────────
    # GnomeAccessibilityTools  --------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    _GA_SCHEMA = "org.gnome.desktop.interface"
    _GA_KEY = "text-scaling-factor"

    @classmethod
    def _ga_run(cls, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["gsettings", *args], capture_output=True, text=True, check=True
        )

    @staticmethod
    def _ga_parse_float(value: str) -> float:
        return float(value.strip())

    @classmethod
    def get_text_scale(cls) -> float:
        result = cls._ga_run("get", cls._GA_SCHEMA, cls._GA_KEY)
        return cls._ga_parse_float(result.stdout)

    @classmethod
    def set_text_scale(cls, scale: float) -> None:
        if not (0.5 <= scale <= 3.0):
            raise ValueError("scale should be between 0.5 and 3.0")
        subprocess.run(["which", "gsettings"], check=True, stdout=subprocess.PIPE)
        cls._ga_run("set", cls._GA_SCHEMA, cls._GA_KEY, str(scale))
        new_val = cls.get_text_scale()
        if abs(new_val - scale) > 1e-3:
            raise RuntimeError(
                f"Tried to set text scale to {scale}, but system reports {new_val}"
            )

    @classmethod
    def change_text_scale(cls, new_scale: float) -> float:
        old = cls.get_text_scale()
        cls.set_text_scale(new_scale)
        return old

    # ──────────────────────────────────────────────────────────────────────
    # FileTools  -----------------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def copy_matching_files_with_hierarchy(
        cls,
        root_dir: str = ".",
        dest_dir: str = "fails",
        pattern: str = "*failed.ipynb",
        dry_run: bool = False,
    ) -> List[Tuple[str, str]]:
        copied: list[tuple[str, str]] = []
        root_dir = os.path.abspath(os.path.expanduser(root_dir))
        dest_dir_abs = os.path.abspath(os.path.join(root_dir, dest_dir))

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [
                d
                for d in dirnames
                if os.path.abspath(os.path.join(dirpath, d)) != dest_dir_abs
            ]
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    src_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(src_path, root_dir)
                    dst_path = os.path.join(dest_dir_abs, rel_path)
                    copied.append((src_path, dst_path))
                    if dry_run:
                        continue
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
        return copied

    # -- copy_file_to_directories -----------------------------------------
    file_ret: str = ""

    @classmethod
    def print_result(cls) -> None:
        print(cls.file_ret)

    # ──────────────────────────────────────────────────────────────────────
    # TrashTools  ----------------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    _TRASH_FILES = Path.home() / ".local/share/Trash/files"
    _TRASH_INFO = Path.home() / ".local/share/Trash/info"

    @classmethod
    def _trash_ensure(cls) -> None:
        if not cls._TRASH_FILES.is_dir() or not cls._TRASH_INFO.is_dir():
            raise FileNotFoundError("Trash directory not found – expected "
                                    f"{cls._TRASH_FILES} and {cls._TRASH_INFO}")

    @classmethod
    def _trash_info_path(cls, filename: str) -> Path:
        return cls._TRASH_INFO / f"{filename}.trashinfo"

    @classmethod
    def get_trash_directory(cls) -> str:
        """
        Return the absolute path to the user Trash ‘files’ directory, e.g.
        /home/USER/.local/share/Trash/files
        """
        return str(cls._TRASH_FILES.resolve())

    # ---------- 通用文件搜索（替代原 search_files） --------------- #
    @classmethod
    def search_files(cls,
                     keyword: str,
                     root_dir: str = ".") -> List[str]:
        """
        Recursively search *root_dir* for files whose **basename**
        contains *keyword* (case-insensitive).

        Returns a list of absolute paths.
        """
        matches: List[str] = []
        root = Path(root_dir).expanduser().resolve()
        keyword = keyword.lower()

        for p in root.rglob("*"):
            if p.is_file() and keyword in p.name.lower():
                matches.append(str(p))

        return matches

    @classmethod
    def restore_file(cls, file_name: str) -> str:
        """
        Restore *file_name* from the user Trash back to its original path.

        Parameters
        ----------
        file_name : str
            Exact basename as shown in ~/.local/share/Trash/files .
            Example:  "holiday_poster.png"

        Returns
        -------
        str
            Absolute path where the file was restored.

        Raises
        ------
        FileNotFoundError – if the given file does not exist in Trash
        RuntimeError      – if .trashinfo is missing or move fails
        """
        trash_dir = Path(cls.get_trash_directory())
        info_dir  = cls._TRASH_INFO
        if str(trash_dir) in file_name:
            file_name = file_name.split(str(trash_dir) + '/')[-1]
        elif str(info_dir) in file_name:
            file_name = file_name.split(str(info_dir) + '/')[-1]
        chosen_path = trash_dir / file_name
        info_path   = info_dir  / f"{file_name}.trashinfo"

        # 1) 目标是否存在
        if not chosen_path.is_file():
            raise FileNotFoundError(f"{file_name!r} not found in Trash")

        # 2) 解析原始路径
        orig_path: Optional[str] = None
        if info_path.is_file():
            with info_path.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("Path="):
                        orig_path = line[len("Path="):].strip()
                        break
        if not orig_path:
            raise RuntimeError(f"Could not read original path from {info_path}")

        orig_path = os.path.expanduser(orig_path)
        Path(os.path.dirname(orig_path)).mkdir(parents=True, exist_ok=True)

        # 3) 执行恢复
        shutil.move(str(chosen_path), orig_path)
        info_path.unlink(missing_ok=True)
        return orig_path

    # ──────────────────────────────────────────────────────────────────────
    # UbuntuPackageTools  --------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _upt_run(cmd: List[str]) -> Tuple[bool, List[str]]:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = completed.returncode == 0
        out = (completed.stdout or "").splitlines()
        err = (completed.stderr or "").splitlines()
        return ok, [f"$ {' '.join(cmd)}"] + out + err

    # ──────────────────────────────────────────────────────────────────────
    # ScreenLockTools  -----------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    _SL_SCHEMA_SESSION = "org.gnome.desktop.session"
    _SL_SCHEMA_SAVER = "org.gnome.desktop.screensaver"

    @staticmethod
    def _sl_run(cmd: str) -> str:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    @staticmethod
    def _sl_ensure_gsettings() -> None:
        if shutil.which("gsettings") is None:
            raise RuntimeError(
                "`gsettings` command not found. Install it via "
                "`sudo apt install dconf-cli`."
            )


    @classmethod
    def configure_auto_lock(
        cls,
        idle_delay_seconds: int,
        lock_delay_seconds: int = 0,
        enable_lock: bool = True,
    ) -> Dict[str, Any]:
        if idle_delay_seconds < 0 or lock_delay_seconds < 0:
            raise ValueError("Delays must be non-negative integers")
        cls._sl_ensure_gsettings()
        cls._sl_run(
            f"gsettings set {cls._SL_SCHEMA_SESSION} idle-delay {idle_delay_seconds}"
        )
        cls._sl_run(
            f"gsettings set {cls._SL_SCHEMA_SAVER} lock-enabled "
            f"{'true' if enable_lock else 'false'}"
        )
        cls._sl_run(
            f"gsettings set {cls._SL_SCHEMA_SAVER} lock-delay {lock_delay_seconds}"
        )
        return cls.get_current_settings()

    # ──────────────────────────────────────────────────────────────────────
    # TimeZoneTools  -------------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _tz_run(cmd: str, sudo_password: Optional[str] = None) -> Tuple[int, str, str]:
        if sudo_password:
            cmd = f"echo {shlex.quote(sudo_password)} | sudo -S {cmd}"
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = proc.communicate()
        return proc.returncode, out.strip(), err.strip()

    # ──────────────────────────────────────────────────────────────────────
    # FsTools  -------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    fs_ret: str = ""

    @classmethod
    def rename_directory(
        cls, old_path: str, new_path: str, create_intermediate: bool = False
    ) -> bool:
        try:
            old_dir = Path(os.path.expanduser(old_path)).resolve()
            new_dir = Path(os.path.expanduser(new_path)).resolve()
            if not old_dir.exists():
                cls.fs_ret = f"Error: source directory not found: {old_dir}"
                return False
            if not old_dir.is_dir():
                cls.fs_ret = f"Error: source path is not a directory: {old_dir}"
                return False
            if new_dir.exists():
                cls.fs_ret = f"Error: target already exists: {new_dir}"
                return False
            if create_intermediate:
                new_dir.parent.mkdir(parents=True, exist_ok=True)
            elif not new_dir.parent.exists():
                cls.fs_ret = (
                    f"Error: target parent directory does not exist: {new_dir.parent}"
                )
                return False
            shutil.move(str(old_dir), str(new_dir))
            cls.fs_ret = (
                f"Success: '{old_dir.name}' renamed to '{new_dir.name}'.\n"
                f"Full path: {new_dir}"
            )
            return True
        except Exception as exc:
            cls.fs_ret = f"Unhandled exception: {exc}"
            return False

    # ──────────────────────────────────────────────────────────────────────
    # NotificationTools  ---------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _nt_run_cmd(cmd: List[str]) -> Tuple[bool, str]:
        try:
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True
            )
            return True, res.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, e.output.strip()

    @staticmethod
    def _nt_gsettings_available() -> bool:
        return shutil.which("gsettings") is not None

    @classmethod
    def set_do_not_disturb(cls, enable: bool) -> bool:
        if not cls._nt_gsettings_available():
            cls.ret = "Error: `gsettings` not found."
            return False
        value = "false" if enable else "true"   # GNOME 逻辑
        ok, out = cls._nt_run_cmd(
            ["gsettings", "set", "org.gnome.desktop.notifications", "show-banners", value]
        )
        if not ok:
            cls.ret = f"Error: Failed to change notification setting: {out}"
            return False
        cls.ret = f"Do Not Disturb {'enabled' if enable else 'disabled'}."
        return ok

    @classmethod
    def get_do_not_disturb_status(cls) -> Dict[str, Any]:
        if not cls._nt_gsettings_available():
            return {
                "success": False,
                "result": None,
                "error_message": "Error: `gsettings` not found.",
            }
        ok, out = cls._nt_run_cmd(
            ["gsettings", "get", "org.gnome.desktop.notifications", "show-banners"]
        )
        if not ok:
            return {
                "success": False,
                "result": None,
                "error_message": f"Error: Failed to read notification setting: {out}",
            }
        out = out.strip().lower()
        if out == "false":
            return {
                "success": True,
                "result": {"enabled": True},
                "error_message": None,
            }
        if out == "true":
            return {
                "success": True,
                "result": {"enabled": False},
                "error_message": None,
            }
        return {
            "success": False,
            "result": None,
            "error_message": f"Error: Unexpected gsettings output: {out}",
        }


# ────────────────────────────────────────────────────────────────────────────
# 兼容别名：保持旧的 Fully-Qualified Name 不变
# ────────────────────────────────────────────────────────────────────────────
TerminalSizeTools   = UnifiedTools
SystemVolumeTools   = UnifiedTools
GnomeAccessibilityTools = UnifiedTools
FileTools           = UnifiedTools
TrashTools          = UnifiedTools
UbuntuPackageTools  = UnifiedTools
ScreenLockTools     = UnifiedTools
TimeZoneTools       = UnifiedTools
FsTools             = UnifiedTools
NotificationTools   = UnifiedTools

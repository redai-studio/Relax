import time
import platform
import pyautogui


class VSCodeTools:
    settings_path = None
    ret = ""

    @staticmethod
    def _update_settings(new_settings: dict):
        """
        更新 settings.json 内容。
        """
        import json
        import os
        if not VSCodeTools.settings_path:
            from pathlib import Path
            # Windows/macOS/Linux 路径不同，可按需调整
            VSCodeTools.settings_path = os.path.expanduser(
                "~/.config/Code/User/settings.json")
        try:
            if os.path.exists(VSCodeTools.settings_path):
                with open(VSCodeTools.settings_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
            else:
                content = {}
            content.update(new_settings)
            with open(VSCodeTools.settings_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4)
            VSCodeTools.ret = "Success"
            return True
        except Exception as e:
            VSCodeTools.ret = f"Error: {e}"
            return False

    @classmethod
    def set_word_wrap_column(cls, column: int):
        """设置代码自动换行行长"""
        return cls._update_settings({"editor.wordWrapColumn": column})

    @classmethod
    def set_auto_save_delay(cls, delay_ms: int):
        """设置自动保存延迟时间（毫秒）"""
        return cls._update_settings({
            "files.autoSave": "afterDelay",
            "files.autoSaveDelay": delay_ms
        })

    @classmethod
    def set_focus_editor_on_break(cls, value: bool):
        """设置 Debug: Focus Editor On Break"""
        return cls._update_settings({"debug.focusEditorOnBreak": value})

    @classmethod
    def set_color_theme(cls, theme_name: str):
        """设置编辑器颜色主题"""
        return cls._update_settings({"workbench.colorTheme": theme_name})

    @classmethod
    def set_wrap_tabs(cls, value: bool):
        """设置 Wrap Tabs"""
        return cls._update_settings({"workbench.editor.wrapTabs": value})

    @classmethod
    def add_files_exclude(cls, pattern: str):
        """添加排除的文件模式"""
        return cls._update_settings({"files.exclude": {pattern: True}})

    @classmethod
    def set_python_diagnostics_override(cls, rule: str, severity: str):
        """设置 Python 分析规则的严重性覆盖"""
        return cls._update_settings({
            "python.analysis.diagnosticSeverityOverrides": {
                rule: severity
            }
        })

    @classmethod
    def search_text(cls, text: str, all_files: bool = True):
        """
        在 VSCode 中搜索文本

        Args:
            text (str): 要搜索的内容
            all_files (bool): True 表示在所有文件中搜索，False 表示仅当前文件搜索
        """
        try:
            if all_files:
                if platform.system() == "Darwin":
                    pyautogui.hotkey('command', 'shift', 'f')
                else:
                    pyautogui.hotkey('ctrl', 'shift', 'f')
            else:
                if platform.system() == "Darwin":
                    pyautogui.hotkey('command', 'f')
                else:
                    pyautogui.hotkey('ctrl', 'f')

            time.sleep(0.3)
            pyautogui.write(text, interval=0.1)
            time.sleep(0.2)
            pyautogui.press('enter')
            cls.ret = f"Searched for '{text}' in {'all files' if all_files else 'the current file'}."
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to search text in VSCode: {exc}"
            return False

    @classmethod
    def replace_text(cls, search_text: str, replace_text: str, all_files: bool = True):
        """
        在 VSCode 中替换文本

        Args:
            search_text (str): 查找的文本内容
            replace_text (str): 替换为的文本内容
            all_files (bool): True 表示在所有文件中替换，False 表示仅当前文件替换
        """
        try:
            if all_files:
                if platform.system() == "Darwin":
                    pyautogui.hotkey('command', 'shift', 'h')
                else:
                    pyautogui.hotkey('ctrl', 'shift', 'h')
            else:
                if platform.system() == "Darwin":
                    pyautogui.hotkey('command', 'h')
                else:
                    pyautogui.hotkey('ctrl', 'h')

            time.sleep(0.3)
            pyautogui.write(search_text, interval=0.1)
            time.sleep(0.2)
            pyautogui.press('tab')
            time.sleep(0.2)
            pyautogui.write(replace_text, interval=0.1)
            time.sleep(0.3)

            if all_files:
                if platform.system() == "Darwin":
                    pyautogui.hotkey('shift', 'option', 'enter')
                else:
                    pyautogui.hotkey('shift', 'alt', 'enter')
            else:
                if platform.system() == "Darwin":
                    pyautogui.hotkey('command', 'option', 'enter')
                else:
                    pyautogui.hotkey('ctrl', 'alt', 'enter')

            cls.ret = (
                f"Replaced '{search_text}' with '{replace_text}' in "
                f"{'all files' if all_files else 'the current file'}."
            )
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to replace text in VSCode: {exc}"
            return False

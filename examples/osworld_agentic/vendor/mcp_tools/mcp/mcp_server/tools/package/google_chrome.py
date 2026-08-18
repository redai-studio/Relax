import time
import os
import logging
import pyautogui
import asyncio
from playwright.async_api import async_playwright  # 改成异步 API

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BrowserTools:
    vm_ip = "127.0.0.1"
    chromium_port = 9222  # 默认 Chrome Remote Debugging Port
    ret = ""

    @classmethod
    async def _chrome_open_tabs_setup(cls, urls_to_open):
        """
        连接到已启动的 Chrome 并打开指定的标签页 (async 版本)
        """
        host = cls.vm_ip
        port = cls.chromium_port
        remote_debugging_url = f"http://{host}:{port}"
        logger.info("Connect to Chrome @: %s", remote_debugging_url)
        logger.debug("PLAYWRIGHT ENV: %s", repr(os.environ))

        for attempt in range(15):
            if attempt > 0:
                await asyncio.sleep(5)   # 异步 sleep

            try:
                playwright = await async_playwright().start()
                browser = await playwright.chromium.connect_over_cdp(remote_debugging_url)
            except Exception as e:
                if attempt < 14:
                    logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                    continue
                else:
                    logger.error(f"Failed to connect after multiple attempts: {e}")
                    raise e

            if not browser:
                cls.ret = "Error: Failed to connect to Chrome browser instance."
                return False

            logger.info("Opening %s...", urls_to_open)
            if not browser.contexts:
                cls.ret = "Error: Connected to Chrome but no browser context is available."
                return False
            context = browser.contexts[0]

            for i, url in enumerate(urls_to_open):
                page = await context.new_page()
                try:
                    await page.goto(url, timeout=60000)
                except Exception:
                    logger.warning("Opening %s exceeds time limit", url)

                logger.info(f"Opened tab {i + 1}: {url}")

                if i == 0:
                    # 关闭默认的空白页
                    default_page = context.pages[0] if context.pages else None
                    if default_page is not None and default_page != page:
                        await default_page.close()

            cls.ret = f"Opened {len(urls_to_open)} tab(s): {', '.join(urls_to_open)}"
            return True

    # ====== Chrome Pages (全改成 async) ======
    @classmethod
    async def chrome_open_tabs_setup(cls, url):
        return await cls._chrome_open_tabs_setup([url])
    @classmethod
    async def open_profile_settings(cls):
        return await cls._chrome_open_tabs_setup(["chrome://settings/people"])

    @classmethod
    async def open_password_settings(cls):
        return await cls._chrome_open_tabs_setup(["chrome://settings/autofill"])

    @classmethod
    async def open_privacy_settings(cls):
        return await cls._chrome_open_tabs_setup(["chrome://settings/privacy"])

    @classmethod
    async def open_appearance_settings(cls):
        return await cls._chrome_open_tabs_setup(["chrome://settings/appearance"])

    @classmethod
    async def open_search_engine_settings(cls):
        return await cls._chrome_open_tabs_setup(["chrome://settings/search"])

    @classmethod
    async def open_extensions(cls):
        return await cls._chrome_open_tabs_setup(["chrome://extensions"])

    @classmethod
    async def open_bookmarks(cls):
        return await cls._chrome_open_tabs_setup(["chrome://bookmarks"])

    # ====== Keyboard Shortcut Actions (保持同步) ======
    @classmethod
    def bring_back_last_tab(cls):
        """恢复上次关闭的标签页 (Ctrl+Shift+T)"""
        try:
            pyautogui.hotkey('ctrl', 'shift', 't')
            cls.ret = "Brought back the last closed tab."
            logger.info(cls.ret)
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to restore the last closed tab: {exc}"
            logger.error(cls.ret)
            return False

    @classmethod
    def print(cls):
        """打开打印对话框 (Ctrl+P)"""
        try:
            pyautogui.hotkey('ctrl', 'p')
            cls.ret = "Opened the print dialog."
            logger.info(cls.ret)
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to open the print dialog: {exc}"
            logger.error(cls.ret)
            return False

    @classmethod
    def delete_browsing_data(cls):
        """打开清除浏览数据窗口 (Ctrl+Shift+Del)"""
        try:
            pyautogui.hotkey('ctrl', 'shift', 'del')
            cls.ret = "Opened the clear browsing data dialog."
            logger.info(cls.ret)
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to open the clear browsing data dialog: {exc}"
            logger.error(cls.ret)
            return False

    @classmethod
    def bookmark_page(cls):
        """收藏当前页面 (Ctrl+D)"""
        try:
            pyautogui.hotkey('ctrl', 'd')
            cls.ret = "Opened the bookmark dialog for the current page."
            logger.info(cls.ret)
            return True
        except Exception as exc:
            cls.ret = f"Error: Failed to bookmark the current page: {exc}"
            logger.error(cls.ret)
            return False


# 示例用法
if __name__ == "__main__":
    async def main():
        BrowserTools.vm_ip = "127.0.0.1"
        BrowserTools.chromium_port = 9222

        await BrowserTools.open_privacy_settings()  # 现在用 await 调用
        await asyncio.sleep(1)
        BrowserTools.bring_back_last_tab()  # 依旧同步调用

    asyncio.run(main())

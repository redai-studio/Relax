# file: uno_presentation_tools.py
"""
PresentationTools (UNO 版)

把当前打开的 LibreOffice Impress 演示文稿里的全部文字，按照
“Slide N” → 若干段落的形式，写入一个新的 Writer 文档。

示例用法
--------
soffice --accept="socket,host=localhost,port=2002;urp;" &
python -m uno_presentation_tools                # 仅打开 Writer，不落盘
python -m uno_presentation_tools ~/Desktop/script.docx   # 同时保存为 docx
"""
import os
import re
from pathlib import Path
from typing import Optional

import uno
from com.sun.star.beans import PropertyValue


class PresentationToolsUNO:
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    ret = ""

    # ---------- public API -------------------------------------------------
    @classmethod
    def convert_to_docx(cls, output_path: Optional[str | Path] = None) -> str | None:
        """
        读取 **当前前台 Im​press 文档**，把文字复制到新的 Writer 文档。

        Parameters
        ----------
        output_path : str | Path | None
            若提供则立即保存为 .docx；不提供时只打开 Writer 让用户自行保存。

        Returns
        -------
        str | None
            保存成功时返回绝对路径；若未保存文件则返回 None。
        """
        desktop, ctx = cls._connect_desktop()

        impress = desktop.getCurrentComponent()
        if not impress or not impress.supportsService("com.sun.star.presentation.PresentationDocument"):
            raise RuntimeError("当前组件不是 Im​press 演示文稿")

        # 1) 创建新的 Writer 文档（private:factory/swriter 即可，不需要实际文件）
        writer = desktop.loadComponentFromURL("private:factory/swriter", "_blank", 0, ())
        w_text   = writer.getText()
        w_cursor = w_text.createTextCursor()

        draw_pages = impress.getDrawPages()

        for slide_idx in range(draw_pages.getCount()):
            slide = draw_pages.getByIndex(slide_idx)

            # 题目 “Slide N”
            # w_text.insertString(w_cursor, f"Slide {slide_idx + 1}\n", False)

            # 遍历形状，抽取文本
            for shp_idx in range(slide.getCount()):
                shape = slide.getByIndex(shp_idx)
                if hasattr(shape, "getText"):
                    txt = cls._clean(shape.getText().getString())
                    if txt:
                        w_text.insertString(w_cursor, txt + "\n", False)

            # w_text.insertString(w_cursor, "\n", False)  # slide 之间空一行

        # 2) 如有要求则立刻保存
        if output_path:
            output_path = Path(output_path).expanduser().resolve()
            if output_path.suffix.lower() != ".docx":
                raise ValueError("output_path 必须以 .docx 结尾")

            url  = uno.systemPathToFileUrl(str(output_path))
            prop = PropertyValue(Name="FilterName", Value="MS Word 2007 XML")
            writer.storeToURL(url, (prop,))
            cls.ret = str(output_path)
            return cls.ret

        # 不保存文件——让用户看到 Writer 文档即可
        cls.ret = "Copied presentation text into a new Writer document."
        return cls.ret

    # ---------- private helpers -------------------------------------------
    @staticmethod
    def _connect_desktop():
        """建立到正在运行的 soffice 进程的 UNO 连接并返回 Desktop 对象。"""
        local_ctx = uno.getComponentContext()
        resolver  = local_ctx.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_ctx
        )
        ctx = resolver.resolve(
            "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
        )
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        return desktop, ctx

    @classmethod
    def _clean(cls, txt: str) -> str:
        """去掉非法控制字符与首尾空白。"""
        txt = cls.CONTROL_CHAR_PATTERN.sub("", txt)
        return txt.strip()


# -------------------------------------------------------------------------
# 便于在终端里 “python -m uno_presentation_tools [save_path]” 直接运行
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        result = PresentationToolsUNO.convert_to_docx(save_path)
        if result:
            print("已保存到：", result)
        else:
            print("文字已复制到新的 Writer 文档（未保存）")
    except Exception as exc:
        print("失败：", exc)

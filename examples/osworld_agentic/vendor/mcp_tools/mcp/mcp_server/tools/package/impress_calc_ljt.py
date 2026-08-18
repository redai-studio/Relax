import uno
from functools import wraps
from decimal import Decimal, ROUND_HALF_UP
import subprocess
from PIL import Image
import sys
import os
import shutil
import re

# import pandas as pd
from typing import List, Optional


class CalcToolsPlus:
    localContext = None
    resolver = None
    ctx = None
    desktop = None
    doc = None
    sheet = None
    ret = ""
    COPY_CELL_LIMIT = 10000

    @staticmethod
    def with_context(func):
        @wraps(func)
        def wrapper(cls, **kwargs):
            cls.localContext = uno.getComponentContext()
            cls.resolver = cls.localContext.ServiceManager.createInstanceWithContext(
                "com.sun.star.bridge.UnoUrlResolver", cls.localContext)
            cls.ctx = cls.resolver.resolve(
                "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
            cls.desktop = cls.ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", cls.ctx)
            cls.doc = cls.desktop.getCurrentComponent()
            cls.sheet = cls.doc.CurrentController.ActiveSheet
            return func(cls, **kwargs)
        return wrapper

    @staticmethod
    def _column_name_to_index(column_name: str) -> int:
        value = 0
        for char in column_name.upper():
            if not ("A" <= char <= "Z"):
                raise ValueError(f"Invalid column label: {column_name}")
            value = value * 26 + (ord(char) - ord("A") + 1)
        return value - 1

    @classmethod
    def _parse_a1_range(cls, range_str: str):
        pattern = r"^\s*([A-Za-z]+)(\d+)(?::([A-Za-z]+)(\d+))?\s*$"
        match = re.match(pattern, range_str or "")
        if not match:
            raise ValueError(f"Invalid A1 range: {range_str}")

        start_col_name, start_row_str, end_col_name, end_row_str = match.groups()
        end_col_name = end_col_name or start_col_name
        end_row_str = end_row_str or start_row_str

        start_col = cls._column_name_to_index(start_col_name)
        end_col = cls._column_name_to_index(end_col_name)
        start_row = int(start_row_str)
        end_row = int(end_row_str)

        if start_row <= 0 or end_row <= 0:
            raise ValueError(f"Invalid A1 range: row index must be >= 1 in {range_str}")
        if end_col < start_col or end_row < start_row:
            raise ValueError(f"Invalid A1 range: end must not precede start in {range_str}")

        return start_col, start_row, end_col, end_row

    @classmethod
    @with_context
    def scale_first_sheet_and_export_pdf(cls, output_path, pages_x=1, pages_y=1):
        """
        Scale the first sheet to fit specified pages and export to PDF.
        """
        try:
            # 获取第一个 sheet
            first_sheet = cls.doc.Sheets.getByIndex(0)
            page_styles = cls.doc.getStyleFamilies().getByName("PageStyles")
            default_style_name = first_sheet.PageStyle
            page_style = page_styles.getByName(default_style_name)

            # 缩放到指定页数
            page_style.setPropertyValue("ScaleToPagesX", pages_x)
            page_style.setPropertyValue("ScaleToPagesY", pages_y)

            # 导出 PDF
            props = [uno.createUnoStruct(
                "com.sun.star.beans.PropertyValue") for _ in range(1)]
            props[0].Name = "FilterName"
            props[0].Value = "calc_pdf_Export"

            cls.doc.storeToURL(uno.systemPathToFileUrl(
                output_path), tuple(props))
            cls.ret = f"PDF exported successfully to {output_path}"
            return True
        except Exception as e:
            cls.ret = f"Error exporting PDF: {e}"
            return False

    # @classmethod
    # @with_context
    # def fill_blank_down(cls, columns: Optional[List[str]] = None) -> bool:
    #     """
    #     Forward-fill all blank cells in the specified columns of the active sheet.
    #     """
    #     try:
    #         cursor = cls.sheet.createCursor()
    #         cursor.gotoEndOfUsedArea(False)
    #         end_col = cursor.RangeAddress.EndColumn
    #         end_row = cursor.RangeAddress.EndRow

    #         # 读数据
    #         data = []
    #         for r in range(end_row + 1):
    #             row_data = [cls.sheet.getCellByPosition(
    #                 c, r).getString() for c in range(end_col + 1)]
    #             data.append(row_data)

    #         df = pd.DataFrame(data)

    #         # 第一行作为表头
    #         header = df.iloc[0].tolist()
    #         df.columns = header
    #         df = df.drop(index=0).reset_index(drop=True)

    #         target_cols = columns if columns else df.columns.tolist()

    #         df[target_cols] = df[target_cols].replace("", pd.NA).ffill()

    #         # 回写 (保留表头)
    #         for r in range(df.shape[0]):
    #             for c in range(df.shape[1]):
    #                 text = "" if pd.isna(df.iat[r, c]) else str(df.iat[r, c])
    #                 cls.sheet.getCellByPosition(c, r + 1).setString(text)

    #         cls.ret = "Blanks forward-filled successfully."
    #         return True
    #     except Exception as e:
    #         cls.ret = f"Error while filling blanks: {e}"
    #         return False

    @classmethod
    @with_context
    def copy_cells_between_sheets(cls, source_range, target_sheet_name, target_start_cell):
        """
        Copy a range from active sheet to another sheet's target location.
        """
        try:
            start_col, start_row, end_col, end_row = cls._parse_a1_range(source_range)
            row_count = end_row - start_row + 1
            col_count = end_col - start_col + 1
            cell_count = row_count * col_count
            if cell_count > cls.COPY_CELL_LIMIT:
                cls.ret = (
                    f"Error copying cells: source range {source_range} contains "
                    f"{cell_count} cells, which exceeds the limit of {cls.COPY_CELL_LIMIT}."
                )
                return False

            src_range = cls.sheet.getCellRangeByName(source_range)
            target_sheet = cls.doc.Sheets.getByName(target_sheet_name)
            dest_cell = target_sheet.getCellRangeByName(target_start_cell)
            # 使用公式复制内容（UNO 中没有直接 copyRangeTo 这种 API）
            for r in range(src_range.Rows.getCount()):
                for c in range(src_range.Columns.getCount()):
                    val = src_range.getCellByPosition(c, r).getFormula()
                    target_cell = target_sheet.getCellByPosition(
                        dest_cell.RangeAddress.StartColumn + c,
                        dest_cell.RangeAddress.StartRow + r
                    )
                    target_cell.setFormula(val)
            cls.ret = "Copied successfully"
            return True
        except Exception as e:
            cls.ret = f"Error copying cells: {e}"
            return False

    @classmethod
    @with_context
    def format_numbers_to_human_readable(cls, cell_range):
        """
        Format numbers to readable form with M or B suffix.
        """
        try:
            rng = cls.sheet.getCellRangeByName(cell_range)
            rows, cols = rng.Rows.getCount(), rng.Columns.getCount()
            for r in range(rows):
                for c in range(cols):
                    cell = rng.getCellByPosition(c, r)
                    if cell.Type.value == 2:  # 数值
                        num = cell.Value
                        if abs(num) >= 1e9:
                            val = Decimal(
                                num / 1e9).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                            cell.String = f"{val}B"
                        elif abs(num) >= 1e6:
                            val = Decimal(
                                num / 1e6).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                            cell.String = f"{val}M"
            cls.ret = "Formatted successfully"
            return True
        except Exception as e:
            cls.ret = f"Error formatting: {e}"
            return False

    @classmethod
    @with_context
    def fill_blank_down(cls, columns: Optional[List[str]] = None) -> bool:
        """
        Forward-fill all blank cells in the specified columns of the active sheet
        without using pandas.

        Args:
            columns: List of column labels (e.g., ['A', 'C']) to process.
                        If None, process all columns.

        Returns:
            True on success, False on error.
        """
        try:
            # 1. 获取已使用区域范围
            cursor = cls.sheet.createCursor()
            cursor.gotoEndOfUsedArea(False)
            end_col = cursor.RangeAddress.EndColumn
            end_row = cursor.RangeAddress.EndRow

            # 2. 全部列标
            all_columns = [chr(ord('A') + i) for i in range(end_col + 1)]
            target_columns = columns if columns else all_columns

            # 3. 转换列字母到列索引
            col_indices = [ord(col.upper()) - ord('A')
                           for col in target_columns]

            # 4. 遍历每个目标列，向前填充
            for col_idx in col_indices:
                prev_value = None
                for row in range(1, end_row + 1):  # 从第2行开始（索引1），跳过标题行
                    cell = cls.sheet.getCellByPosition(col_idx, row)
                    cell_value_str = cell.getString().strip()

                    if cell_value_str == "":
                        if prev_value is not None:
                            # 根据 prev_value 类型填充数据
                            if isinstance(prev_value, (int, float)):
                                cell.Value = prev_value
                            else:
                                cell.String = str(prev_value)
                    else:
                        # 判断是数字还是字符串
                        try:
                            num_val = float(cell_value_str)
                            # 如果是整数，保持整数显示
                            if num_val.is_integer():
                                num_val = int(num_val)
                            prev_value = num_val
                        except ValueError:
                            prev_value = cell_value_str

            cls.ret = "Blanks forward-filled successfully."
            return True

        except Exception as exc:
            cls.ret = f"Error while filling blanks: {exc}"
            return False

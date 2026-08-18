import json
import os
import subprocess
import sys

import uno
from com.sun.star.beans import PropertyValue
from functools import wraps

class CalcTools:
    localContext = None
    resolver = None
    ctx = None
    desktop = None
    doc = None
    sheet = None
    ret = ""

    @staticmethod
    def _normalize_color(color):
        if isinstance(color, bool):
            raise ValueError(f"Unsupported color value: {color}")
        if isinstance(color, int):
            return color
        if isinstance(color, str):
            value = color.strip()
            if not value:
                raise ValueError("Color value cannot be empty.")
            if value.startswith("#"):
                return int(value[1:], 16)
            if value.lower().startswith("0x"):
                return int(value, 16)
            return int(value)
        raise ValueError(f"Unsupported color value: {color}")

    @staticmethod
    def with_context(func):
        """装饰器：在调用目标方法前初始化 UNO 上下文"""
        @wraps(func)
        def wrapper(cls, **kwargs):
            cls.localContext = uno.getComponentContext()
            cls.resolver = cls.localContext.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", cls.localContext)
            cls.ctx = cls.resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
            cls.desktop = cls.ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", cls.ctx)
            cls.doc = cls.desktop.getCurrentComponent()
            cls.sheet = cls.doc.CurrentController.ActiveSheet
            return func(cls, **kwargs)
        return wrapper

    # @classmethod
    # @with_context
    # def _initialize_context(cls):
    #     cls.localContext = uno.getComponentContext()
    #     cls.resolver = cls.localContext.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", cls.localContext)
    #     cls.ctx = cls.resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
    #     cls.desktop = cls.ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", cls.ctx)
    #     cls.doc = cls.desktop.getCurrentComponent()
    #     cls.sheet = cls.doc.CurrentController.ActiveSheet

    @classmethod
    @with_context
    def close_other_window(cls):
        """关闭除当前文档外的所有文档"""
        # 获取所有打开的文档
        components = cls.desktop.getComponents().createEnumeration()
        current_url = cls.doc.getURL()

        while components.hasMoreElements():
            doc = components.nextElement()
            if doc.getURL() != current_url:  # 如果不是当前文档
                doc.close(True)  # True 表示保存更改

    @classmethod
    @with_context
    def maximize_window(cls):
        """
        将窗口设置为工作区最大尺寸
        使用工作区域大小（考虑任务栏等）
        """
        window = cls.doc.getCurrentController().getFrame().getContainerWindow()
        toolkit = window.getToolkit()
        device = toolkit.createScreenCompatibleDevice(0, 0)

        # 获取工作区域（排除任务栏等）
        workarea = toolkit.getWorkArea()

        # 设置窗口位置和大小为工作区域
        window.setPosSize(workarea.X, workarea.Y, workarea.Width, workarea.Height, 15)

    @classmethod
    @with_context
    def print_result(cls):
        print(cls.ret)

    @classmethod
    @with_context
    def save(cls):
        """
        Save the current workbook to its current location

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Just save the document
            cls.doc.store()
            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    def _get_column_index(cls, column_name, sheet=None):
        """
        Get the index of a column by its name (A, B, C, ...)

        Args:
            column_name (str): Name of the column

        Returns:
            int: Index of the column
        """
        try:
            if not isinstance(column_name, str):
                return None
            normalized = column_name.strip().upper()
            if not normalized or not normalized.isalpha():
                return None
            return cls._column_name_to_index(normalized)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _get_last_used_column(cls):
        """
        Get the last used column index

        Args:
            None

        Returns:
            int: Index of the last used column
        """
        cursor = cls.sheet.createCursor()
        cursor.gotoEndOfUsedArea(False)
        return cursor.RangeAddress.EndColumn

    @classmethod
    def _get_last_used_row(cls):
        """
        Get the last used row index

        Args:
            None

        Returns:
            int: Index of the last used row
        """
        cursor = cls.sheet.createCursor()
        cursor.gotoEndOfUsedArea(False)
        return cursor.RangeAddress.EndRow

    @classmethod
    def _column_name_to_index(cls, column_name):
        """
        将列名转换为列索引

        Args:
            column_name (str): 列名，如 'A', 'AB'

        Returns:
            int: 列索引（从0开始）
        """
        column_name = column_name.upper()
        result = 0
        for char in column_name:
            result = result * 26 + (ord(char) - ord("A") + 1)
        return result - 1

    @classmethod
    @with_context
    def get_workbook_info(cls):
        """
        Get workbook information

        Args:
            None

        Returns:
            dict: Workbook information, including file path, file name, sheets and active sheet
        """
        try:
            info = {
                "file_path": cls.doc.getLocation(),
                "file_title": cls.doc.getTitle(),
                "sheets": [],
                "active_sheet": cls.sheet.Name,
            }

            # Get sheets information
            sheets = cls.doc.getSheets()
            info["sheet_count"] = sheets.getCount()

            # Get all sheet names and info
            for i in range(sheets.getCount()):
                sheet = sheets.getByIndex(i)
                cursor = sheet.createCursor()
                cursor.gotoEndOfUsedArea(False)
                end_col = cursor.getRangeAddress().EndColumn
                end_row = cursor.getRangeAddress().EndRow

                sheet_info = {
                    "name": sheet.getName(),
                    "index": i,
                    "visible": sheet.IsVisible,
                    "row_count": end_row + 1,
                    "column_count": end_col + 1,
                }
                info["sheets"].append(sheet_info)

                # Check if this is the active sheet
                if sheet == cls.sheet:
                    info["active_sheet"] = sheet_info

            cls.ret = json.dumps(info, ensure_ascii=False)
            return info

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def env_info(cls, sheet_name=None):
        """
        Get content of the specified or active sheet

        Args:
            sheet_name (str, optional): Name of the sheet to read. If None, uses active sheet

        Returns:
            dict: Sheet information including name, headers and data
        """
        try:
            # Get the target sheet
            if sheet_name is not None:
                sheet = cls.doc.getSheets().getByName(sheet_name)
            else:
                sheet = cls.sheet

            # Create cursor to find used range
            cursor = sheet.createCursor()
            cursor.gotoEndOfUsedArea(False)
            end_col = cursor.getRangeAddress().EndColumn
            end_row = cursor.getRangeAddress().EndRow

            # Generate column headers (A, B, C, ...)
            col_headers = [chr(65 + i) for i in range(end_col + 1)]

            # Get displayed values from cells
            data_array = []
            for row in range(end_row + 1):
                row_data = []
                for col in range(end_col + 1):
                    cell = sheet.getCellByPosition(col, row)
                    row_data.append(cell.getString())
                data_array.append(row_data)

            # Calculate maximum width for each column
            col_widths = [len(header) for header in col_headers]  # Initialize with header lengths
            for row in data_array:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

            # Format the header row
            header_row = "  | " + " | ".join(f"{h:<{w}}" for h, w in zip(col_headers, col_widths)) + " |"
            separator = "--|-" + "-|-".join("-" * w for w in col_widths) + "-|"

            # Format data rows with row numbers
            formatted_rows = []
            for row_idx, row in enumerate(data_array, 1):
                row_str = f"{row_idx:<2}| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |"
                formatted_rows.append(row_str)

            # Combine all parts
            formated_data = header_row + "\n" + separator + "\n" + "\n".join(formatted_rows)

            # Get sheet properties
            sheet_info = {
                "name": sheet.getName(),
                "data": formated_data,
                "row_count": end_row + 1,
                "column_count": end_col + 1,
            }

            cls.ret = json.dumps(sheet_info, ensure_ascii=False)
            cls.print_result()
            return sheet_info

        except Exception as e:
            cls.ret = f"Error: {e}"
            cls.print_result()

    @classmethod
    @with_context
    def get_column_data(cls, column_name):
        """
        Get data from the specified column

        Args:
            column_name (str): Name of the column to read

        Returns:
            list: List of values in the specified column
        """
        column_index = cls._get_column_index(column_name)
        if column_index is None:
            cls.ret = f"Error: Column not found: {column_name}"
            return False
        last_row = cls._get_last_used_row()
        _range = cls.sheet.getCellRangeByPosition(column_index, 0, column_index, last_row)
        # 获取数据数组并展平
        cls.ret = json.dumps([row[0] for row in _range.getDataArray()], ensure_ascii=False)
        return [row[0] for row in _range.getDataArray()]

    @classmethod
    @with_context
    def switch_active_sheet(cls, sheet_name):
        """
        Switch to the specified sheet and make it active, create if not exist

        Args:
            sheet_name (str): Name of the sheet to switch to or create

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 获取所有工作表
            sheets = cls.doc.getSheets()

            # 检查工作表是否存在
            if not sheets.hasByName(sheet_name):
                # 创建新工作表
                new_sheet = cls.doc.createInstance("com.sun.star.sheet.Spreadsheet")
                sheets.insertByName(sheet_name, new_sheet)

            # 获取目标工作表
            sheet = sheets.getByName(sheet_name)

            # 切换到目标工作表
            cls.doc.getCurrentController().setActiveSheet(sheet)
            
            # 验证切换是否生效
            active = cls.doc.getCurrentController().ActiveSheet
            if active.getName() != sheet_name:
                cls.ret = f"Error: Failed to switch to sheet '{sheet_name}', current active sheet is still '{active.getName()}'"
                return False


            # 更新当前工作表引用
            cls.sheet = sheet
            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def set_column_values(cls, column_name, data, start_index=2):
        """
        Set data to the specified column

        Args:
            column_name (str): Name of the column to write
            data (list): List of values to write to the column
            start_index (int): The index of the first row to write to, default is 2 (skip the first row)

        Returns:
            bool: True if successful, False otherwise
        """
        # 获取列的索引
        column_index = cls._get_column_index(column_name)
        if column_index is None:
            cls.ret = f"Error: Column not found: {column_name}"
            return False
        for i, value in enumerate(data):
            cell = cls.sheet.getCellByPosition(column_index, i + start_index - 1)
            if type(value) == float and value.is_integer():
                cell.setNumber(int(value))
            else:
                cell.setString(str(value))
        cls.ret = "Success"
        return True

    @classmethod
    @with_context
    def highlight_range(cls, range_str, color="#FF0000"):
        """
        highlight the specified range with the specified color

        Args:
            range_str (str): Range to highlight, in the format of "A1:B10"
            color (str): Color to highlight with, default is '0xFF0000' (red)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            _range = cls.sheet.getCellRangeByName(range_str)
            _range.CellBackColor = cls._normalize_color(color)
            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: Failed to highlight range '{range_str}': {e}"
            return False

    @classmethod
    @with_context
    def transpose_range(cls, source_range, target_cell):
        """
        Transpose the specified range and paste it to the target cell

        Args:
            source_range (str): Range to transpose, in the format of "A1:B10"
            target_cell (str): Target cell to paste the transposed data, in the format of "A1"

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            source = cls.sheet.getCellRangeByName(source_range)
            target = cls.sheet.getCellRangeByName(target_cell)

            data = source.getDataArray()
            # 转置数据
            transposed_data = list(map(list, zip(*data)))

            # 设置转置后的数据
            target_range = cls.sheet.getCellRangeByPosition(
                target.CellAddress.Column,
                target.CellAddress.Row,
                target.CellAddress.Column + len(transposed_data[0]) - 1,
                target.CellAddress.Row + len(transposed_data) - 1,
            )
            target_range.setDataArray(transposed_data)
            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def export_to_csv(cls):
        """
        Export the current document to a CSV file

        Args:
            None

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 获取当前文档的URL
            doc_url = cls.doc.getURL()
            if not doc_url:
                raise ValueError("Document must be saved first")

            # 构造CSV文件路径
            if doc_url.startswith("file://"):
                base_path = doc_url[7:]  # 移除 'file://' 前缀
            else:
                base_path = doc_url

            # 获取基本路径和文件名
            csv_path = os.path.splitext(base_path)[0] + ".csv"

            # 确保路径是绝对路径
            csv_path = os.path.abspath(csv_path)

            # 转换为 LibreOffice URL 格式
            csv_url = uno.systemPathToFileUrl(csv_path)

            # 设置CSV导出选项
            props = (
                PropertyValue(Name="FilterName", Value="Text - txt - csv (StarCalc)"),
                PropertyValue(
                    Name="FilterOptions", Value="44,0,76,0"
                ),  # 44=comma, 34=quote, 76=UTF-8, 1=first row as header
            )

            # 导出文件
            cls.doc.storeToURL(csv_url, props)
            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def sort_column(cls, column_name, ascending=True, start_index=1):
        """
        Sorts the data in the specified column in ascending or descending order

        Args:
            column_name (str): The name of the column to sort (e.g. 'A') or the title
            ascending (bool): Whether to sort in ascending order (default True)
            start_index (int): The index of the first row to sort, default is 1

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            column_data = cls.get_column_data(column_name=column_name)[start_index - 1 :]
            column_data = sorted(column_data, key=lambda x: float(x), reverse=not ascending)
        except:
            cls.ret = "Error: Invalid column name or data type"
            return False

        return cls.set_column_values(column_name=column_name, data=column_data, start_index=start_index)

    @classmethod
    @with_context
    def set_validation_list(cls, column_name, values):
        """
        Set a validation list for the specified column

        Args:
            column_name (str): The name of the column to set the validation list for
            values (list): The list of values to use for the validation list

        Returns:
            None
        """
        try:
            column_index = cls._get_column_index(column_name)
            last_row = cls._get_last_used_row()
            cell_range = cls.sheet.getCellRangeByPosition(column_index, 1, column_index, last_row)

            # 获取现有的验证对象
            validation = cell_range.getPropertyValue("Validation")

            # 设置基本验证类型
            validation.Type = uno.Enum("com.sun.star.sheet.ValidationType", "LIST")
            validation.Operator = uno.Enum("com.sun.star.sheet.ConditionOperator", "EQUAL")

            # 设置下拉列表
            validation.ShowList = True
            values_str = ";".join(str(val) for val in values)
            validation.Formula1 = values_str

            # 应用验证设置回单元格范围
            cell_range.setPropertyValue("Validation", validation)

            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def hide_row_data(cls, value="N/A"):
        """
        Hide rows that contain the specified value

        Args:
            value (str): The value to hide rows for, default is 'N/A'

        Returns:
            None
        """
        last_row = cls._get_last_used_row()
        last_col = cls._get_last_used_column()

        for row in range(1, last_row + 1):
            has_value = False
            for col in range(last_col + 1):
                cell = cls.sheet.getCellByPosition(col, row)
                if cell.getString() == value:
                    has_value = True
                    break
            row_range = cls.sheet.getRows().getByIndex(row)
            row_range.IsVisible = not has_value

        cls.ret = "Success"
        return True

    @classmethod
    @with_context
    def reorder_columns(cls, column_order):
        """
        Reorder the columns in the sheet according to the specified order

        Args:
            column_order (list): A list of column names in the desired order

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 获取原始列索引列表
            old_indices = [cls._get_column_index(col) for col in column_order]

            # 一次性读取所有需要重排列的数据
            last_row = cls._get_last_used_row()
            columns_data = []
            for old_idx in old_indices:
                col_range = cls.sheet.getCellRangeByPosition(
                    old_idx, 0, old_idx, last_row)
                columns_data.append(col_range.getDataArray())

            # 确定实际要操作的列范围（取 old_indices 的最小到最大）
            min_col = min(old_indices)
            max_col = max(len(old_indices) - 1, max(old_indices))

            # 按目标顺序将数据回写到对应位置
            for new_idx, col_data in enumerate(columns_data):
                target_col_idx = min_col + new_idx
                col_range = cls.sheet.getCellRangeByPosition(
                    target_col_idx, 0, target_col_idx, last_row)
                col_range.setDataArray(col_data)
            
            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def create_pivot_table(
        cls,
        source_sheet,
        table_name,
        row_fields=None,
        col_fields=None,
        value_fields=None,
        aggregation_function="sum",
        target_cell="A1",
    ):
        """
        Create a pivot table in the active worksheet based on data from the active sheet.
        """
        try:
            source = cls.doc.getSheets().getByName(source_sheet)

            # 获取数据范围
            cursor = source.createCursor()
            cursor.gotoEndOfUsedArea(False)
            end_col = cursor.getRangeAddress().EndColumn
            end_row = cursor.getRangeAddress().EndRow

            # 获取完整的数据范围
            source_range = source.getCellRangeByPosition(0, 0, end_col, end_row)

            # 获取数据透视表集合
            dp_tables = cls.sheet.getDataPilotTables()

            # 创建数据透视表描述符
            dp_descriptor = dp_tables.createDataPilotDescriptor()

            # 设置数据源
            dp_descriptor.setSourceRange(source_range.getRangeAddress())

            # 设置行字段
            if row_fields:
                for field in row_fields:
                    field_index = cls._get_column_index(field)
                    dimension = dp_descriptor.getDataPilotFields().getByIndex(field_index)
                    dimension.Orientation = uno.Enum("com.sun.star.sheet.DataPilotFieldOrientation", "ROW")

            # 设置列字段
            if col_fields:
                for field in col_fields:
                    field_index = cls._get_column_index(field)
                    dimension = dp_descriptor.getDataPilotFields().getByIndex(field_index)
                    dimension.Orientation = uno.Enum("com.sun.star.sheet.DataPilotFieldOrientation", "COLUMN")

            # 设置数据字段
            for field in value_fields:
                field_index = cls._get_column_index(field)
                dimension = dp_descriptor.getDataPilotFields().getByIndex(field_index)
                dimension.Orientation = uno.Enum("com.sun.star.sheet.DataPilotFieldOrientation", "DATA")

                # 设置聚合函数
                function_map = {
                    "count": "COUNT",
                    "sum": "SUM",
                    "average": "AVERAGE",
                    "min": "MIN",
                    "max": "MAX"
                }
                if aggregation_function.lower() in function_map:
                    dimension.Function = uno.Enum(
                        "com.sun.star.sheet.GeneralFunction",
                        function_map[aggregation_function.lower()]
                    )

            # 在当前工作表中创建数据透视表
            dp_tables.insertNewByName(
                table_name,  # 透视表名称
                cls.sheet.getCellRangeByName(target_cell).CellAddress,  # 目标位置
                dp_descriptor,  # 描述符
            )

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def merge_cells(cls, range_str):
        """
        合并活动工作表中指定范围的单元格

        Args:
            range_str (str): 要合并的单元格范围，格式为'A1:B10'

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取当前活动工作表
            sheet = cls.sheet

            # 获取单元格范围
            cell_range = sheet.getCellRangeByName(range_str)

            # 获取单元格范围的属性
            range_props = cell_range.getIsMerged()

            # 如果单元格范围尚未合并，则进行合并
            if not range_props:
                cell_range.merge(True)

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def set_cell_value(cls, cell, value):
        """
        Set a value to a specific cell in the active worksheet.

        Args:
            cell (str): Cell reference (e.g., 'A1')
            value (str): Value to set in the cell

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            range_obj = cls.sheet.getCellRangeByName(cell)
            addr = range_obj.getRangeAddress()
            cell_obj = cls.sheet.getCellByPosition(
                addr.StartColumn, addr.StartRow
            )
            print(f"[DEBUG] cell_obj type: {type(cell_obj)}")  # 临时调试

            if isinstance(value, str) and value.startswith("="):
                # LibreOffice setFormula 要求分号作为参数分隔符
                # 但引号内的逗号不能替换（如 "Alice,Bob" 里的逗号）
                # 用状态机处理：只替换引号外的逗号
                formula = value
                result = []
                in_quotes = False
                for ch in formula:
                    if ch == '"':
                        in_quotes = not in_quotes
                    if ch == ',' and not in_quotes:
                        result.append(';')
                    else:
                        result.append(ch)
                formula_converted = ''.join(result)
                print(f"[DEBUG] converted formula: {formula_converted}")
                cell_obj.setFormula(formula_converted)
                cls.ret = "Success"
                return True

            try:
                int_value = int(value)
                cell_obj.setValue(int_value)
            except (ValueError, TypeError):
                try:
                    float_value = float(value)
                    cell_obj.setValue(float_value)
                except (ValueError, TypeError):
                    cell_obj.setString(str(value))

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def format_range(cls, range_str, background_color=None, font_color=None, bold=None, alignment=None):
        """
        Apply formatting to the specified range in the active worksheet

        Args:
            range_str (str): Range to format, in the format of 'A1:B10'
            background_color (str, optional): Background color in hex format (e.g., '#0000ff')
            font_color (str, optional): Font color in hex format (e.g., '#ffffff')
            bold (bool, optional): Whether to make the text bold
            italic (bool, optional): Whether to make the text italic
            alignment (str, optional): Text alignment (left, center, right)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 获取指定范围
            cell_range = cls.sheet.getCellRangeByName(range_str)

            # 设置背景颜色
            if background_color:
                cell_range.CellBackColor = cls._normalize_color(background_color)

            # 设置字体颜色
            if font_color:
                cell_range.CharColor = cls._normalize_color(font_color)

            # 设置粗体
            if bold is not None:
                cell_range.CharWeight = 150.0 if bold else 100.0  # 150.0 是粗体，100.0 是正常

            # 设置对齐方式
            if alignment:
                # 设置水平对齐方式
                struct = cell_range.getPropertyValue("HoriJustify")
                if alignment == "left":
                    struct.value = "LEFT"
                elif alignment == "center":
                    struct.value = "CENTER"
                elif alignment == "right":
                    struct.value = "RIGHT"
                cell_range.setPropertyValue("HoriJustify", struct)

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def create_chart(cls, chart_type, data_range, title=None, x_axis_title=None, y_axis_title=None):
        """
        Create a chart in the active worksheet based on the specified data range.

        Args:
            chart_type (str): Type of chart to create (bar, column, line, pie, scatter, area)
            data_range (str): Range containing the data for the chart, in the format of 'A1:B10'
            title (str, optional): Title for the chart
            x_axis_title (str, optional): Title for the X axis
            y_axis_title (str, optional): Title for the Y axis

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            chart_type_map = {
                "bar": "com.sun.star.chart.BarDiagram",
                "column": "com.sun.star.chart.BarDiagram",
                "line": "com.sun.star.chart.LineDiagram",
                "pie": "com.sun.star.chart.PieDiagram",
                "scatter": "com.sun.star.chart.ScatterDiagram",
                "area": "com.sun.star.chart.AreaDiagram",
            }

            range_addresses = []
            for segment in data_range.split(","):
                segment = segment.strip()
                if "." in segment:
                    dot_idx = segment.index(".")
                    seg_sheet_name = segment[:dot_idx]
                    seg_range = segment[dot_idx + 1:]
                    # 去掉可能的 $ 符号
                    seg_range = seg_range.replace("$", "")
                    seg_sheet = cls.doc.getSheets().getByName(seg_sheet_name)
                    addr = seg_sheet.getCellRangeByName(seg_range).getRangeAddress()
                    range_addresses.append(addr)
                else:
                    addr = cls.sheet.getCellRangeByName(segment).getRangeAddress()
                    range_addresses.append(addr)

            charts = cls.sheet.getCharts()
            rect = uno.createUnoStruct("com.sun.star.awt.Rectangle")
            rect.X = 0
            rect.Y = 0
            rect.Width = 10000
            rect.Height = 7000

            import time
            chart_name = f"Chart_{int(time.time())}"
            charts.addNewByName(
                chart_name, rect, tuple(range_addresses), True, True
            )
            time.sleep(0.3)

            chart = charts.getByName(chart_name)
            chart_doc = chart.getEmbeddedObject()

            if chart_type in chart_type_map:
                diagram = chart_doc.createInstance(chart_type_map[chart_type])
                # column 需要设置 Vertical=False
                if chart_type == "column":
                    diagram.Vertical = False
                elif chart_type == "bar":
                    diagram.Vertical = True
                chart_doc.setDiagram(diagram)

            if title:
                chart_doc.HasMainTitle = True
                chart_doc.Title.String = title

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def freeze_panes(cls, rows=0, columns=0):
        """
        冻结活动工作表中的行和/或列

        Args:
            rows (int): 从顶部开始冻结的行数
            columns (int): 从左侧开始冻结的列数

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取当前视图
            view = cls.doc.getCurrentController()

            # 设置冻结窗格
            view.freezeAtPosition(columns, rows)

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def rename_sheet(cls, old_name, new_name):
        """
        重命名工作表

        Args:
            old_name (str): 要重命名的工作表的当前名称
            new_name (str): 工作表的新名称

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取所有工作表
            sheets = cls.doc.getSheets()

            # 检查原工作表是否存在
            if not sheets.hasByName(old_name):
                cls.ret = f"Error: Sheet not found: {old_name}"
                return False

            # 检查新名称是否已存在
            if sheets.hasByName(new_name):
                cls.ret = f"Error: Target sheet name already exists: {new_name}"
                return False

            # 获取要重命名的工作表
            sheet = sheets.getByName(old_name)

            # 重命名工作表
            sheet.setName(new_name)

            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def copy_sheet(cls, source_sheet, new_sheet_name=None):
        """
        创建工作簿中现有工作表的副本

        Args:
            source_sheet (str): 要复制的工作表名称
            new_sheet_name (str, optional): 新工作表副本的名称，如果不提供则自动生成

        Returns:
            str: 新创建的工作表名称，如果失败则返回None
        """
        try:
            # 获取所有工作表
            sheets = cls.doc.getSheets()

            # 检查源工作表是否存在
            if not sheets.hasByName(source_sheet):
                cls.ret = f"Error: Source sheet not found: {source_sheet}"
                return False

            # 如果没有提供新名称，则生成一个
            if not new_sheet_name:
                # 生成类似 "Sheet1 (2)" 的名称
                base_name = source_sheet
                counter = 1
                new_sheet_name = f"{base_name} ({counter})"

                # 确保名称不重复
                while sheets.hasByName(new_sheet_name):
                    counter += 1
                    new_sheet_name = f"{base_name} ({counter})"

            # 检查新名称是否已存在
            if sheets.hasByName(new_sheet_name):
                cls.ret = f"Error: Target sheet name already exists: {new_sheet_name}"
                return False

            # 获取源工作表的索引
            source_index = -1
            for i in range(sheets.getCount()):
                if sheets.getByIndex(i).getName() == source_sheet:
                    source_index = i
                    break

            if source_index == -1:
                cls.ret = f"Error: Unable to locate source sheet index: {source_sheet}"
                return False

            # 复制工作表
            sheets.copyByName(source_sheet, new_sheet_name, source_index + 1)

            cls.ret = f"New sheet created: {new_sheet_name}"
            return new_sheet_name

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def reorder_sheets(cls, sheet_name, position):
        """
        重新排序工作表在工作簿中的位置

        Args:
            sheet_name (str): 要移动的工作表名称
            position (int): 要移动到的位置(基于0的索引)

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取所有工作表
            sheets = cls.doc.getSheets()

            # 检查工作表是否存在
            if not sheets.hasByName(sheet_name):
                return False

            # 获取工作表总数
            sheet_count = sheets.getCount()

            # 检查位置是否有效
            if position < 0 or position >= sheet_count:
                return False

            # 获取要移动的工作表
            sheet = sheets.getByName(sheet_name)

            # 获取工作表当前索引
            current_index = -1
            for i in range(sheet_count):
                if sheets.getByIndex(i).Name == sheet_name:
                    current_index = i
                    break

            if current_index == -1:
                return False

            # 移动工作表到指定位置
            sheets.moveByName(sheet_name, position)

            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def set_chart_legend_position(cls, position):
        """
        Set the position of the legend in a chart in the active worksheet.

        Args:
            position (str): Position of the legend ('top', 'bottom', 'left', 'right', 'none')

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 获取当前工作表中的所有图表
            charts = cls.sheet.getCharts()
            if charts.getCount() == 0:
                return False

            # 获取第一个图表（假设我们要修改的是第一个图表）
            chart = charts.getByIndex(0)
            chart_obj = chart.getEmbeddedObject()

            # 获取图表的图例
            diagram = chart_obj.getDiagram()
            legend = chart_obj.getLegend()

            # 根据指定的位置设置图例位置
            if position == "none":
                # 如果选择"none"，则隐藏图例
                chart_obj.HasLegend = False
            else:
                # 确保图例可见
                chart_obj.HasLegend = True

                import inspect

                print(inspect.getmembers(legend))

                # 设置图例位置
                if position == "top":
                    pos = uno.Enum("com.sun.star.chart.ChartLegendPosition", "TOP")
                elif position == "bottom":
                    pos = uno.Enum("com.sun.star.chart.ChartLegendPosition", "BOTTOM")
                elif position == "left":
                    pos = uno.Enum("com.sun.star.chart.ChartLegendPosition", "LEFT")
                elif position == "right":
                    pos = uno.Enum("com.sun.star.chart.ChartLegendPosition", "RIGHT")

                legend.Alignment = pos

            cls.ret = "Success"
            return True
        except Exception:
            cls.ret = "Error"
            return False

    @classmethod
    @with_context
    def set_number_format(cls, range_str, format_type, decimal_places=None):
        """
        Apply a specific number format to a range of cells in the active worksheet.

        Args:
            range_str (str): Range to format, in the format of 'A1:B10'
            format_type (str): Type of number format to apply
            decimal_places (int, optional): Number of decimal places to display

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 获取单元格范围
            cell_range = cls.sheet.getCellRangeByName(range_str)

            # 获取数字格式化服务
            number_formats = cls.doc.NumberFormats
            locale = cls.doc.CharLocale

            # 根据格式类型设置格式字符串
            format_string = ""

            if format_type == "general":
                format_string = "General"
            elif format_type == "number":
                if decimal_places is not None:
                    format_string = f"0{('.' + '0' * decimal_places) if decimal_places > 0 else ''}"
                else:
                    format_string = "0"
            elif format_type == "currency":
                if decimal_places is not None:
                    format_string = f"[$¥-804]#,##0{('.' + '0' * decimal_places) if decimal_places > 0 else ''}"
                else:
                    format_string = "[$¥-804]#,##0.00"
            elif format_type == "accounting":
                if decimal_places is not None:
                    format_string = f"_-[$¥-804]* #,##0{('.' + '0' * decimal_places) if decimal_places > 0 else ''}_-;-[$¥-804]* #,##0{('.' + '0' * decimal_places) if decimal_places > 0 else ''}_-;_-[$¥-804]* \"-\"_-;_-@_-"
                else:
                    format_string = '_-[$¥-804]* #,##0.00_-;-[$¥-804]* #,##0.00_-;_-[$¥-804]* "-"??_-;_-@_-'
            elif format_type == "date":
                format_string = "YYYY/MM/DD"
            elif format_type == "time":
                format_string = "HH:MM:SS"
            elif format_type == "percentage":
                if decimal_places is not None:
                    format_string = f"0{('.' + '0' * decimal_places) if decimal_places > 0 else ''}%"
                else:
                    format_string = "0.00%"
            elif format_type == "fraction":
                format_string = "# ?/?"
            elif format_type == "scientific":
                if decimal_places is not None:
                    format_string = f"0{('.' + '0' * decimal_places) if decimal_places > 0 else ''}E+00"
                else:
                    format_string = "0.00E+00"
            elif format_type == "text":
                format_string = "@"

            # 获取格式键
            format_key = number_formats.queryKey(format_string, locale, True)

            # 如果格式不存在，则添加
            if format_key == -1:
                format_key = number_formats.addNew(format_string, locale)

            # 应用格式
            cell_range.NumberFormat = format_key

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def adjust_column_width(cls, columns, width=None, autofit=False):
        """
        调整活动工作表中指定列的宽度

        Args:
            columns (str): 要调整的列范围，例如 'A:C' 表示从A列到C列
            width (float, optional): 要设置的宽度（以字符为单位）
            autofit (bool, optional): 是否自动调整列宽以适应内容

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 解析列范围
            col_range = columns.split(":")
            start_col = cls._column_name_to_index(col_range[0])

            if len(col_range) > 1:
                end_col = cls._column_name_to_index(col_range[1])
            else:
                end_col = start_col

            # 获取列对象
            columns_obj = cls.sheet.getColumns()

            # 遍历指定的列范围
            for col_idx in range(start_col, end_col + 1):
                column = columns_obj.getByIndex(col_idx)

                if autofit:
                    # 自动调整列宽
                    column.OptimalWidth = True
                elif width is not None:
                    # 设置指定宽度（转换为1/100毫米）
                    # 大约一个字符宽度为256 (1/100 mm)
                    column.Width = int(width * 256)

            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def adjust_row_height(cls, rows, height=None, autofit=False):
        """
        调整活动工作表中指定行的高度

        Args:
            rows (str): 要调整的行范围，例如 '1:10' 表示第1行到第10行
            height (float, optional): 要设置的高度（以点为单位）
            autofit (bool, optional): 是否自动调整行高以适应内容

        Returns:
            bool: 操作成功返回True，否则返回False
        """
        try:
            # 解析行范围
            row_range = rows.split(":")
            start_row = int(row_range[0])
            end_row = int(row_range[1]) if len(row_range) > 1 else start_row

            # 获取行对象
            for row_index in range(start_row, end_row + 1):
                row = cls.sheet.getRows().getByIndex(row_index - 1)  # 索引从0开始

                if autofit:
                    # 自动调整行高以适应内容
                    row.OptimalHeight = True
                elif height is not None:
                    # 设置指定高度（将点转换为1/100毫米，LibreOffice使用的单位）
                    # 1点 ≈ 35.28 1/100毫米
                    row.Height = int(height * 35.28)
                    row.OptimalHeight = False

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def export_to_pdf(cls, file_path=None, sheets=None, open_after_export=False):
        """
        将当前文档或指定工作表导出为PDF文件

        Args:
            file_path (str, optional): PDF文件保存路径，如果不指定则使用当前文档路径
            sheets (list, optional): 要包含在PDF中的工作表名称列表，如果不指定则包含所有工作表
            open_after_export (bool, optional): 导出后是否打开PDF文件

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 如果未指定文件路径，则使用当前文档路径并更改扩展名为.pdf
            if not file_path:
                if cls.doc.hasLocation():
                    url = cls.doc.getLocation()
                    file_path = uno.fileUrlToSystemPath(url)
                    file_path = os.path.splitext(file_path)[0] + ".pdf"
                else:
                    # 如果文档尚未保存，则在用户桌面创建临时文件
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                    file_path = os.path.join(desktop_path, "LibreOffice_Export.pdf")

            # 确保文件路径是系统路径，然后转换为URL
            pdf_url = uno.systemPathToFileUrl(os.path.abspath(file_path))

            # 创建导出属性
            export_props = []

            # 设置过滤器名称
            export_props.append(PropertyValue(Name="FilterName", Value="calc_pdf_Export"))

            # 如果指定了特定工作表，则只导出这些工作表
            if sheets and isinstance(sheets, list) and len(sheets) > 0:
                # 获取所有工作表
                all_sheets = cls.doc.getSheets()
                selection = []

                # 查找指定的工作表
                for sheet_name in sheets:
                    if all_sheets.hasByName(sheet_name):
                        sheet = all_sheets.getByName(sheet_name)
                        selection.append(sheet)

                # 如果找到了指定的工作表，则设置导出选择
                if selection:
                    export_props.append(PropertyValue(Name="Selection", Value=tuple(selection)))

            # 导出PDF
            cls.doc.storeToURL(pdf_url, tuple(export_props))

            # 如果需要，导出后打开PDF
            if open_after_export:
                if sys.platform.startswith("darwin"):  # macOS
                    subprocess.call(("open", file_path))
                elif os.name == "nt":  # Windows
                    os.startfile(file_path)
                elif os.name == "posix":  # Linux
                    subprocess.call(("xdg-open", file_path))

            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

    @classmethod
    @with_context
    def set_zoom_level(cls, zoom_percentage):
        """
        调整当前工作表的缩放级别，使单元格看起来更大或更小

        Args:
            zoom_percentage (int): 缩放级别的百分比（例如，75表示75%，100表示正常大小，150表示放大）。
                                有效范围通常为10-400。

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取当前控制器
            controller = cls.doc.getCurrentController()

            # 设置缩放值
            # 确保缩放值在合理范围内
            if zoom_percentage < 10:
                zoom_percentage = 10
            elif zoom_percentage > 400:
                zoom_percentage = 400

            # 应用缩放值
            controller.ZoomValue = zoom_percentage
            cls.ret = "Success"
            return True

        except Exception as e:
            cls.ret = f"Error: {e}"
            return False


if __name__ == "__main__":
    # print(CalcTools._get_column_index("A"))
    # print(CalcTools.get_workbook_info())
    # print(CalcTools.get_content())
    # CalcTools.switch_active_sheet("Sheet2")
    # # helper.set_column_values('A', [1, 2, 3, 4, 5])
    # # helper.highlight_range('A1:A3', 'Red')
    # # helper.transpose_range('A1:D5', 'B8')
    # print(CalcTools.get_column_data("A"))
    # CalcTools.sort_column("A", True)
    # CalcTools.hide_row_data("N/A")
    # CalcTools.reorder_columns(["B", "A", "C"])
    # CalcTools.freeze_panes(1, 1)
    # # helper.set_validation_list('C', ['Pass', 'Fail', 'Held'])
    # CalcTools.export_to_csv()

    CalcTools.rename_sheet('Sheet1', 'Sheet2')

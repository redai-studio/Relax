import os
import re

import uno
from com.sun.star.awt.FontSlant import ITALIC, NONE, OBLIQUE
from com.sun.star.awt.FontWeight import BOLD, NORMAL
from com.sun.star.beans import PropertyValue
from com.sun.star.style.ParagraphAdjust import CENTER, LEFT, RIGHT, BLOCK as JUSTIFY
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
from com.sun.star.text.TextContentAnchorType import AS_CHARACTER
from functools import wraps


class WriterTools:
    localContext = None
    resolver = None
    ctx = None
    desktop = None
    doc = None
    text = None
    cursor = None
    ret = ""

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
            cls.text = cls.doc.Text
            cls.cursor = cls.text.createTextCursor()
            return func(cls, **kwargs)
        return wrapper

    @staticmethod
    def _normalize_color(color):
        if isinstance(color, bool):
            raise ValueError(f"Unsupported color value: {color}")
        if isinstance(color, int):
            return color
        if isinstance(color, str):
            value = color.strip()
            if value.startswith("#"):
                return int(value[1:], 16)
            if value.lower().startswith("0x"):
                return int(value, 16)
            return int(value)
        raise ValueError(f"Unsupported color value: {color}")

    @classmethod
    @with_context
    def close_other_window(cls):
        """关闭除当前文档外的所有文档"""
        components = cls.desktop.getComponents().createEnumeration()
        current_url = cls.doc.getURL()
        while components.hasMoreElements():
            doc = components.nextElement()
            if doc.getURL() != current_url:
                doc.close(True)

    @classmethod
    @with_context
    def save(cls):
        """保存文档到当前位置"""
        try:
            if cls.doc.hasLocation():
                cls.doc.store()
                cls.ret = "Success"
            else:
                raise Exception("文档没有保存位置，请使用另存为功能")
            return True
        except Exception as e:
            cls.ret = f"Error: {e}"
            return False

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
        workarea = toolkit.getWorkArea()
        window.setPosSize(workarea.X, workarea.Y, workarea.Width, workarea.Height, 15)

    @classmethod
    @with_context
    def print_result(cls):
        print(cls.ret)

    @classmethod
    @with_context
    def write_text(cls, text, bold=False, italic=False, size=None):
        """写入文本"""
        cls.cursor.CharWeight = BOLD if bold else NORMAL
        cls.cursor.CharPosture = ITALIC if italic else NONE
        if size:
            cls.cursor.CharHeight = size
        cls.text.insertString(cls.cursor, text, False)
        cls.ret = "Success"

    @classmethod
    @with_context
    def get_paragraphs(cls, start_index=0, count=None):
        """Retrieves paragraphs from the document as a list."""
        text = cls.doc.getText()
        paragraphs = text.createEnumeration()
        paragraph_list = []
        while paragraphs.hasMoreElements():
            paragraph = paragraphs.nextElement()
            if paragraph.supportsService("com.sun.star.text.Paragraph"):
                paragraph_list.append(paragraph.getString())
        if start_index < 0:
            start_index = 0
        elif start_index >= len(paragraph_list):
            cls.ret = []
        if count is not None:
            end_index = min(start_index + count, len(paragraph_list))
            cls.ret = paragraph_list[start_index:end_index]
        else:
            cls.ret = paragraph_list[start_index:]
        return cls.ret

    @classmethod
    @with_context
    def env_info(cls):
        paras = cls.get_paragraphs()
        para_str = ""
        for i, para in enumerate(paras):
            para = para[:500] + "..." if len(para) > 500 else para
            para_str += "Paragraph " + str(i) + ": " + para.strip() + "\n"
        cls.ret = para_str
        cls.print_result()
        return cls.ret

    @classmethod
    @with_context
    def set_color(cls, pattern, color, paragraph_indices=None):
        """
        Changes the color of matched text in the document for specified paragraphs.

        Args:
            pattern (str): Regular expression pattern to match text
            color (str | int): Color value like '#000000', '0x000000', or 0
            paragraph_indices (list, optional): List of paragraph indices to modify (0-based).
                If None, applies to all paragraphs.
        """
        try:
            color = cls._normalize_color(color)
            enum = cls.doc.Text.createEnumeration()
            paragraphs = []
            while enum.hasMoreElements():
                paragraphs.append(enum.nextElement())
            if not paragraph_indices:
                paragraphs_to_process = range(len(paragraphs))
            else:
                paragraphs_to_process = paragraph_indices
            regex = re.compile(pattern)
            for idx in paragraphs_to_process:
                if idx < 0 or idx >= len(paragraphs):
                    continue
                paragraph = paragraphs[idx]
                if not paragraph.supportsService("com.sun.star.text.Paragraph"):
                    continue
                para_text = paragraph.getString()
                matches = regex.finditer(para_text)
                for match in matches:
                    para_cursor = cls.text.createTextCursorByRange(paragraph.getStart())
                    para_cursor.goRight(match.start(), False)
                    para_cursor.goRight(match.end() - match.start(), True)
                    para_cursor.CharColor = color
            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def find_and_replace(cls, pattern, replacement, paragraph_indices=None):
        """
        Finds all occurrences of a specified text pattern and replaces them with another text in the document.

        Args:
            pattern (str): The pattern to match in the document, should be a regular expression
            replacement (str): The text to replace the found text with
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing)

        Returns:
            str: Success message with number of replacements made
        """
        try:
            text = cls.doc.getText()
            enum = text.createEnumeration()

            # 分离正文段落和表格
            body_paragraphs = []
            tables = []
            while enum.hasMoreElements():
                element = enum.nextElement()
                if element.supportsService("com.sun.star.text.Paragraph"):
                    body_paragraphs.append(element)
                elif element.supportsService("com.sun.star.text.TextTable"):
                    tables.append(element)

            total_replacements = 0
            regex = re.compile(pattern)

            def replace_in_paragraph(paragraph):
                """对单个段落执行替换，返回替换次数"""
                if not paragraph.supportsService("com.sun.star.text.Paragraph"):
                    return 0
                text_content = paragraph.getString()
                new_text, count = regex.subn(replacement, text_content)
                if count > 0:
                    paragraph.setString(new_text)
                return count

            # 处理正文段落
            if not paragraph_indices:
                paragraphs_to_process = list(range(len(body_paragraphs)))
            else:
                paragraphs_to_process = [
                    i for i in paragraph_indices if 0 <= i < len(body_paragraphs)
                ]

            for idx in paragraphs_to_process:
                total_replacements += replace_in_paragraph(body_paragraphs[idx])

            # 处理表格内所有段落
            # 注意：表格内不受 paragraph_indices 控制，始终全量处理
            for table in tables:
                cell_names = table.getCellNames()
                for cell_name in cell_names:
                    cell = table.getCellByName(cell_name)
                    cell_text = cell.getText()
                    cell_enum = cell_text.createEnumeration()
                    while cell_enum.hasMoreElements():
                        cell_para = cell_enum.nextElement()
                        total_replacements += replace_in_paragraph(cell_para)

            cls.ret = f"Successfully made {total_replacements} replacements"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error during find and replace: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def set_font(cls, font_name, paragraph_indices=None):
        """
        Changes the font of text in the document or specified paragraphs.

        Args:
            font_name (str): The name of the font to apply (e.g., 'Times New Roman', 'Arial', 'Calibri')
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                                             If not provided, applies to all paragraphs.
        """
        try:
            text = cls.doc.getText()
            enum = text.createEnumeration()
            paragraphs = []
            while enum.hasMoreElements():
                paragraphs.append(enum.nextElement())

            if not paragraph_indices:
                paragraph_indices = range(len(paragraphs))

            for idx in paragraph_indices:
                if 0 <= idx < len(paragraphs):
                    paragraph = paragraphs[idx]

                    # 处理正文段落
                    if paragraph.supportsService("com.sun.star.text.Paragraph"):
                        # 1. 先设段落级默认
                        para_cursor = text.createTextCursorByRange(paragraph)
                        para_cursor.setPropertyValue("CharFontName", font_name)
                        para_cursor.setPropertyValue("CharFontNameAsian", font_name)
                        para_cursor.setPropertyValue("CharFontNameComplex", font_name)

                        # 2. 再逐 portion 覆盖字符级 override
                        portion_enum = paragraph.createEnumeration()
                        while portion_enum.hasMoreElements():
                            portion = portion_enum.nextElement()
                            portion_cursor = text.createTextCursorByRange(portion)
                            portion_cursor.setPropertyValue("CharFontName", font_name)
                            portion_cursor.setPropertyValue("CharFontNameAsian", font_name)
                            portion_cursor.setPropertyValue("CharFontNameComplex", font_name)

                    # 处理表格内段落
                    elif paragraph.supportsService("com.sun.star.text.TextTable"):
                        cell_names = paragraph.getCellNames()
                        for cell_name in cell_names:
                            cell = paragraph.getCellByName(cell_name)
                            cell_text = cell.getText()
                            cell_enum = cell_text.createEnumeration()
                            while cell_enum.hasMoreElements():
                                cell_para = cell_enum.nextElement()
                                if cell_para.supportsService("com.sun.star.text.Paragraph"):
                                    # 段落级
                                    cell_para_cursor = cell_text.createTextCursorByRange(cell_para)
                                    cell_para_cursor.setPropertyValue("CharFontName", font_name)
                                    cell_para_cursor.setPropertyValue("CharFontNameAsian", font_name)
                                    cell_para_cursor.setPropertyValue("CharFontNameComplex", font_name)
                                    # portion 级
                                    portion_enum = cell_para.createEnumeration()
                                    while portion_enum.hasMoreElements():
                                        portion = portion_enum.nextElement()
                                        portion_cursor = cell_text.createTextCursorByRange(portion)
                                        portion_cursor.setPropertyValue("CharFontName", font_name)
                                        portion_cursor.setPropertyValue("CharFontNameAsian", font_name)
                                        portion_cursor.setPropertyValue("CharFontNameComplex", font_name)

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_line_spacing(cls, spacing_value, paragraph_indices=None):
        """
        Sets the line spacing for specified paragraphs in the document.

        Args:
            spacing_value (float): The line spacing value to apply (1.0 for single spacing, 2.0 for double spacing, etc.)
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                                            If not provided, applies to all paragraphs.
        """
        try:
            text = cls.doc.getText()
            paragraph_enum = text.createEnumeration()
            line_spacing_value = int(spacing_value * 100)
            current_index = 0

            while paragraph_enum.hasMoreElements():
                paragraph = paragraph_enum.nextElement()
                
                # 只处理真正的段落（跳过 TextTable 等）
                if not paragraph.supportsService("com.sun.star.text.Paragraph"):
                    current_index += 1
                    continue

                if not paragraph_indices or current_index in paragraph_indices:
                    line_spacing = uno.createUnoStruct("com.sun.star.style.LineSpacing")
                    
                    line_spacing.Mode = 0
                    line_spacing.Height = line_spacing_value
                    
                     # 通过 cursor 的 setPropertyValue 才能真正写入
                    para_cursor = text.createTextCursorByRange(paragraph)
                    para_cursor.setPropertyValue("ParaLineSpacing", line_spacing)

                current_index += 1  # 对所有段落统一计数（已在之前修复）
                

            cls.ret = "Success"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def remove_highlighting(cls, paragraph_indices=None):
        """
        Removes ALL highlighting from text in the document for specified paragraphs.

        Args:
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                If not provided, applies to all paragraphs.

        Returns:
            str: Success message or error message
        """
        try:
            text = cls.doc.getText()
            paragraphs = text.createEnumeration()
            target_indices = set(paragraph_indices) if paragraph_indices else None
            current_index = 0

            while paragraphs.hasMoreElements():
                paragraph = paragraphs.nextElement()
                if target_indices is None or current_index in target_indices:
                    if paragraph.supportsService("com.sun.star.text.Paragraph"):
                        para_cursor = text.createTextCursorByRange(paragraph)
                        # Remove all highlighting by setting back color to -1
                        para_cursor.CharBackColor = -1

                        # Additional cleanup for individual text portions (optional)
                        text_portions = paragraph.createEnumeration()
                        while text_portions.hasMoreElements():
                            text_portion = text_portions.nextElement()
                            if hasattr(text_portion, "CharBackColor"):
                                portion_cursor = text.createTextCursorByRange(text_portion)
                                portion_cursor.CharBackColor = -1
                current_index += 1

            cls.ret = "Successfully removed all highlighting"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error removing highlighting: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def find_highlighted_text(cls, highlight_color):
        """
        Finds all text in the document that has a specific highlight color applied to it.

        Args:
            highlight_color (str): The highlight color to search for. Can be a color name (e.g., 'yellow', 'green') or hex code.

        Returns:
            list: A list of strings containing all text segments with the specified highlight color.
        """
        color_map = {
            "yellow": 16776960,
            "green": 65280,
            "blue": 255,
            "red": 16711680,
            "cyan": 65535,
            "magenta": 16711935,
            "black": 0,
            "white": 16777215,
            "gray": 8421504,
            "lightgray": 12632256,
        }
        target_color = None
        if highlight_color.lower() in color_map:
            target_color = color_map[highlight_color.lower()]
        elif highlight_color.startswith("#") and len(highlight_color) == 7:
            try:
                hex_color = highlight_color[1:]
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                target_color = (r << 16) + (g << 8) + b
            except ValueError:
                cls.ret = f"Invalid hex color format: {highlight_color}"
                return []
        else:
            cls.ret = f"Unsupported color format: {highlight_color}"
            return []
        highlighted_text = []
        text = cls.doc.getText()
        enum_paragraphs = text.createEnumeration()
        while enum_paragraphs.hasMoreElements():
            paragraph = enum_paragraphs.nextElement()
            if paragraph.supportsService("com.sun.star.text.Paragraph"):
                enum_portions = paragraph.createEnumeration()
                while enum_portions.hasMoreElements():
                    text_portion = enum_portions.nextElement()
                    if hasattr(text_portion, "CharBackColor") and text_portion.CharBackColor == target_color:
                        if text_portion.getString().strip():
                            highlighted_text.append(text_portion.getString())
        cls.ret = f"Found {len(highlighted_text)} text segments with highlight color {highlight_color}"
        return highlighted_text

    @classmethod
    @with_context
    def insert_formula_at_cursor(cls, formula):
        """
        Inserts a formula at the current cursor position in the document.

        Args:
            formula (str): The formula to insert at the current cursor position.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            embedded_obj = cls.doc.createInstance("com.sun.star.text.TextEmbeddedObject")
            embedded_obj.setPropertyValue("CLSID", "078B7ABA-54FC-457F-8551-6147e776a997")
            embedded_obj.setPropertyValue("AnchorType", AS_CHARACTER)
            cls.text.insertTextContent(cls.cursor, embedded_obj, False)
            math_obj = embedded_obj.getEmbeddedObject()
            math_obj.Formula = formula
            cls.ret = "Formula inserted successfully"
            return True
        except Exception as e:
            cls.ret = f"Error inserting formula: {str(e)}"
            return False

    @classmethod
    @with_context
    def insert_image_at_cursor(cls, image_path, width=None, height=None):
        """
        Inserts an image at the current cursor position in the document.

        Args:
            image_path (str): Full path to the image file to insert
            width (int, optional): Width to display the image in pixels
            height (int, optional): Height to display the image in pixels

        Returns:
            str: Success message or error message
        """
        try:
            if image_path.startswith("~"):
                image_path = os.path.expanduser(image_path)
            if not os.path.exists(image_path):
                cls.ret = f"Error: Image file not found at {image_path}"
                return cls.ret
            image_path = os.path.abspath(image_path)
            if os.name == "nt":
                file_url = "file:///" + image_path.replace("\\", "/")
            else:
                file_url = "file://" + image_path
            graphic = cls.doc.createInstance("com.sun.star.text.GraphicObject")
            graphic.GraphicURL = file_url
            graphic.AnchorType = AS_CHARACTER
            if width is not None:
                graphic.Width = width * 100
            if height is not None:
                graphic.Height = height * 100
            cls.text.insertTextContent(cls.cursor, graphic, False)
            cls.ret = "Success: Image inserted"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def set_strikethrough(cls, pattern, paragraph_indices=None):
        """
        Sets the strikethrough formatting for text matching the specified pattern in the document.

        Args:
            pattern (str): The regular expression pattern to match in the document
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                                               If not provided, applies to all paragraphs.

        Returns:
            str: Success message or error information
        """
        try:
            paragraphs = cls.doc.getText().createEnumeration()
            para_index = 0
            found_matches = 0
            while paragraphs.hasMoreElements():
                paragraph = paragraphs.nextElement()
                if paragraph.supportsService("com.sun.star.text.Paragraph"):
                    if paragraph_indices and para_index not in paragraph_indices:
                        para_index += 1
                        continue
                    para_text = paragraph.getString()
                    matches = list(re.finditer(pattern, para_text))
                    for match in matches:
                        text_range = paragraph.getStart()
                        cursor = cls.doc.getText().createTextCursorByRange(text_range)
                        cursor.goRight(match.start(), False)
                        cursor.goRight(match.end() - match.start(), True)
                        cursor.CharStrikeout = 1
                        found_matches += 1
                    para_index += 1
            cls.ret = f"Successfully applied strikethrough to {found_matches} matches of pattern: {pattern}"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error applying strikethrough: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def set_font_size(cls, font_size, pattern, paragraph_indices=None):
        """
        Changes the font size of specified text in the document.

        Args:
            font_size (float): The font size to apply (in points).
            pattern (str): The pattern to match in the document, should be a regular expression.
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                                               If not provided, applies to all paragraphs.

        Returns:
            str: Result message indicating success or failure.
        """
        try:
            regex = re.compile(pattern)
            paragraphs = cls.doc.getText().createEnumeration()
            current_index = 0
            while paragraphs.hasMoreElements():
                paragraph = paragraphs.nextElement()
                if paragraph_indices and current_index not in paragraph_indices:
                    current_index += 1
                    continue
                if paragraph.supportsService("com.sun.star.text.Paragraph"):
                    para_text = paragraph.getString()
                    matches = list(regex.finditer(para_text))
                    for match in matches:
                        start_pos = match.start()
                        end_pos = match.end()
                        # 每次从段落起点重新创建 cursor，避免 gotoStart 跳到文档开头
                        match_cursor = cls.text.createTextCursorByRange(paragraph.getStart())
                        match_cursor.goRight(start_pos, False)
                        match_cursor.goRight(end_pos - start_pos, True)
                        match_cursor.CharHeight = font_size
                current_index += 1
            cls.ret = f"Successfully changed font size to {font_size} for text matching '{pattern}'"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error changing font size: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def export_to_pdf(cls, output_path=None, output_filename=None, include_comments=False, quality="standard"):
        """
        Exports the current document to PDF format.

        Args:
            output_path (str, optional): The full path where the PDF should be saved.
                If not provided, uses the same location as the original document.
            output_filename (str, optional): The filename to use for the PDF.
                If not provided, uses the original document's filename with .pdf extension.
            include_comments (bool, optional): Whether to include comments in the exported PDF.
                Defaults to False.
            quality (str, optional): The quality of the PDF export ('standard', 'high', 'print').
                Defaults to 'standard'.

        Returns:
            str: Path to the exported PDF file or error message
        """
        try:
            doc_url = cls.doc.getURL()
            if not doc_url and not output_path:
                return "Error: Document has not been saved and no output path provided"
            if doc_url:
                doc_path = uno.fileUrlToSystemPath(os.path.dirname(doc_url))
                doc_filename = os.path.basename(doc_url)
                doc_name = os.path.splitext(doc_filename)[0]
            else:
                doc_path = ""
                doc_name = "export"
            final_path = output_path if output_path else doc_path
            final_filename = output_filename if output_filename else f"{doc_name}.pdf"
            if not final_filename.lower().endswith(".pdf"):
                final_filename += ".pdf"
            full_output_path = os.path.join(final_path, final_filename)
            output_url = uno.systemPathToFileUrl(full_output_path)
            export_props = []
            if quality == "high":
                export_props.append(PropertyValue(Name="SelectPdfVersion", Value=1))
            elif quality == "print":
                export_props.append(PropertyValue(Name="SelectPdfVersion", Value=2))
            else:
                export_props.append(PropertyValue(Name="SelectPdfVersion", Value=0))
            export_props.append(PropertyValue(Name="ExportNotes", Value=include_comments))
            export_props.extend(
                [
                    PropertyValue(Name="FilterName", Value="writer_pdf_Export"),
                    PropertyValue(Name="Overwrite", Value=True),
                ]
            )
            cls.doc.storeToURL(output_url, tuple(export_props))
            cls.ret = f"PDF exported to: {full_output_path}"
            return full_output_path
        except Exception as e:
            cls.ret = f"Error exporting to PDF: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def set_paragraph_alignment(cls, alignment, paragraph_indices=None):
        """
        Sets the text alignment for specified paragraphs in the document.

        Args:
            alignment (str): The alignment to apply ('left', 'center', 'right', 'justify').
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                                               If not provided, applies to all paragraphs.

        Returns:
            str: Success message or error message
        """
        try:
            alignment_map = {"left": LEFT, "center": CENTER, "right": RIGHT, "justify": JUSTIFY}
            if alignment.lower() not in alignment_map:
                cls.ret = f"Error: Invalid alignment '{alignment}'. Use 'left', 'center', 'right', or 'justify'."
                return cls.ret
            alignment_value = alignment_map[alignment.lower()]
            text = cls.doc.getText()
            paragraph_enum = text.createEnumeration()
            paragraphs = []
            while paragraph_enum.hasMoreElements():
                paragraph = paragraph_enum.nextElement()
                if paragraph.supportsService("com.sun.star.text.Paragraph"):
                    paragraphs.append(paragraph)
            if paragraph_indices is not None:
                valid_indices = [i for i in paragraph_indices if 0 <= i < len(paragraphs)]
                if len(valid_indices) != len(paragraph_indices):
                    cls.ret = f"Error: Some paragraph indices were out of range (0-{len(paragraphs) - 1})"
                    return cls.ret
                for idx in valid_indices:
                    paragraphs[idx].ParaAdjust = alignment_value
            else:
                for paragraph in paragraphs:
                    paragraph.ParaAdjust = alignment_value
            cls.ret = f"Successfully applied '{alignment}' alignment to paragraphs"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error setting paragraph alignment: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def capitalize_words(cls, paragraph_indices=None):
        """
        Capitalizes the first letter of each word for specified paragraphs in the document.

        Args:
            paragraph_indices (list, optional): Indices of paragraphs to modify (0-based indexing).
                                               If not provided, applies to all paragraphs.

        Returns:
            str: Success message or error message
        """
        try:
            text = cls.doc.getText()
            enum = text.createEnumeration()
            paragraphs = []
            while enum.hasMoreElements():
                paragraph = enum.nextElement()
                if paragraph.supportsService("com.sun.star.text.Paragraph"):
                    paragraphs.append(paragraph)
            if not paragraph_indices:
                target_paragraphs = list(range(len(paragraphs)))
            else:
                target_paragraphs = paragraph_indices
            valid_indices = [idx for idx in target_paragraphs if 0 <= idx < len(paragraphs)]
            for idx in valid_indices:
                paragraph = paragraphs[idx]
                text_content = paragraph.getString()
                if not text_content.strip():
                    continue
                capitalized_text = " ".join(word.capitalize() if word else "" for word in text_content.split(" "))
                para_cursor = text.createTextCursorByRange(paragraph.getStart())
                para_cursor.gotoRange(paragraph.getEnd(), True)
                para_cursor.setString(capitalized_text)
            cls.ret = f"Successfully capitalized words in {len(valid_indices)} paragraphs"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error capitalizing words: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def set_default_font(cls, font_name, font_size=None):
        """
        Sets the default font for new text in the document without changing existing text.

        Args:
            font_name (str): The name of the font to set as default (e.g., 'Times New Roman', 'Arial', 'Calibri')
            font_size (float, optional): The default font size in points.

        Returns:
            str: Success message or error message
        """
        try:
            # ── 1. 写入系统级用户配置（registrymodifications.xcu）──────────────
            config_provider = cls.ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.configuration.ConfigurationProvider", cls.ctx
            )

            # Writer 默认字体节点
            node_prop = PropertyValue()
            node_prop.Name = "nodepath"
            node_prop.Value = "/org.openoffice.Office.Writer/DefaultFont"

            try:
                config_access = config_provider.createInstanceWithArguments(
                    "com.sun.star.configuration.ConfigurationUpdateAccess",
                    (node_prop,)
                )
                # Western（拉丁）字体
                config_access.setPropertyValue("Standard", font_name)
                config_access.setPropertyValue("Heading", font_name)
                config_access.setPropertyValue("List", font_name)
                config_access.setPropertyValue("Caption", font_name)
                config_access.setPropertyValue("Index", font_name)
                if font_size is not None:
                    config_access.setPropertyValue("StandardHeight", int(font_size * 100))

                # CJK 字体节点
                node_prop_cjk = PropertyValue()
                node_prop_cjk.Name = "nodepath"
                node_prop_cjk.Value = "/org.openoffice.Office.Writer/DefaultFont"
                # Asian / Complex 通常在同一节点下有 Asian* 属性
                try:
                    config_access.setPropertyValue("StandardAsian", font_name)
                    config_access.setPropertyValue("HeadingAsian", font_name)
                    config_access.setPropertyValue("ListAsian", font_name)
                    config_access.setPropertyValue("CaptionAsian", font_name)
                    config_access.setPropertyValue("IndexAsian", font_name)
                    config_access.setPropertyValue("StandardComplex", font_name)
                    config_access.setPropertyValue("HeadingComplex", font_name)
                    config_access.setPropertyValue("ListComplex", font_name)
                    config_access.setPropertyValue("CaptionComplex", font_name)
                    config_access.setPropertyValue("IndexComplex", font_name)
                    if font_size is not None:
                        config_access.setPropertyValue("StandardHeightAsian", int(font_size * 100))
                        config_access.setPropertyValue("StandardHeightComplex", int(font_size * 100))
                except Exception:
                    pass  # 部分属性名可能不存在，忽略

                config_access.commitChanges()  # 写入 registrymodifications.xcu
            except Exception as e:
                # 系统级配置写入失败时记录但继续执行文档级修改
                cls.ret = f"Warning: Could not write system config: {str(e)}"

            # ── 2. 同时修改当前文档的段落样式（文档级）────────────────────────
            try:
                style_families = cls.doc.getStyleFamilies()
                paragraph_styles = style_families.getByName("ParagraphStyles")
                default_style_names = ["Default", "Standard", "Normal"]
                standard_style = None
                for style_name in default_style_names:
                    if paragraph_styles.hasByName(style_name):
                        standard_style = paragraph_styles.getByName(style_name)
                        break
                if standard_style is None:
                    style_names = paragraph_styles.getElementNames()
                    if style_names:
                        standard_style = paragraph_styles.getByName(style_names[0])

                if standard_style is not None:
                    standard_style.setPropertyValue("CharFontName", font_name)
                    standard_style.setPropertyValue("CharFontNameAsian", font_name)
                    standard_style.setPropertyValue("CharFontNameComplex", font_name)
                    if font_size is not None:
                        standard_style.setPropertyValue("CharHeight", float(font_size))
                        standard_style.setPropertyValue("CharHeightAsian", float(font_size))
                        standard_style.setPropertyValue("CharHeightComplex", float(font_size))
            except Exception as e:
                cls.ret = f"Warning: Could not update document style: {str(e)}"

            # ── 3. 更新当前 cursor 默认属性────────────────────────────────────
            try:
                cls.cursor.setPropertyValue("CharFontName", font_name)
                cls.cursor.setPropertyValue("CharFontNameAsian", font_name)
                cls.cursor.setPropertyValue("CharFontNameComplex", font_name)
                if font_size is not None:
                    cls.cursor.setPropertyValue("CharHeight", float(font_size))
                    cls.cursor.setPropertyValue("CharHeightAsian", float(font_size))
                    cls.cursor.setPropertyValue("CharHeightComplex", float(font_size))
            except Exception as e:
                cls.ret = f"Warning: Could not update cursor properties: {str(e)}"

            cls.ret = (
                f"Default font set to '{font_name}'"
                + (f" with size {font_size}pt" if font_size else "")
            )
            return cls.ret
        except Exception as e:
            cls.ret = f"Error setting default font: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def add_page_numbers(cls, position, start_number=1, format=None):
        """
        Adds page numbers to the document at the specified position.

        Args:
            position (str): Position of the page numbers ('bottom_left', 'bottom_center', 'bottom_right',
                            'top_left', 'top_center', 'top_right')
            start_number (int, optional): The starting page number. Defaults to 1.
            format (str, optional): Format of the page numbers (e.g., '1', 'Page 1', '1 of N').
                                   Defaults to simple number format.

        Returns:
            str: Success message or error message
        """
        try:
            page_styles = cls.doc.StyleFamilies.getByName("PageStyles")
            default_style = page_styles.getByName("Standard")
            try:
                default_style.setPropertyValue("PageNumberOffset", start_number)
            except:
                pass
            if position.startswith("top"):
                default_style.HeaderIsOn = True
                target = default_style.HeaderText
            else:
                default_style.FooterIsOn = True
                target = default_style.FooterText
            cursor = target.createTextCursor()
            cursor.gotoStart(False)
            cursor.gotoEnd(True)
            cursor.setString("")
            cursor.gotoStart(False)
            if position.endswith("_left"):
                cursor.ParaAdjust = LEFT
            elif position.endswith("_center"):
                cursor.ParaAdjust = CENTER
            elif position.endswith("_right"):
                cursor.ParaAdjust = RIGHT
            if not format or format == "1":
                page_number = cls.doc.createInstance("com.sun.star.text.TextField.PageNumber")
                page_number.NumberingType = 4
                target.insertTextContent(cursor, page_number, False)
            elif format == "Page 1" or "Page" in format and "of" not in format:
                target.insertString(cursor, "Page ", False)
                page_number = cls.doc.createInstance("com.sun.star.text.TextField.PageNumber")
                page_number.NumberingType = 4
                target.insertTextContent(cursor, page_number, False)
            elif format == "1 of N" or format == "Page {page} of {total}" or "of" in format:
                if "Page" in format:
                    target.insertString(cursor, "Page ", False)
                page_number = cls.doc.createInstance("com.sun.star.text.TextField.PageNumber")
                page_number.NumberingType = 4
                target.insertTextContent(cursor, page_number, False)
                target.insertString(cursor, " of ", False)
                page_count = cls.doc.createInstance("com.sun.star.text.TextField.PageCount")
                page_count.NumberingType = 4
                target.insertTextContent(cursor, page_count, False)
            else:
                page_number = cls.doc.createInstance("com.sun.star.text.TextField.PageNumber")
                page_number.NumberingType = 4
                target.insertTextContent(cursor, page_number, False)
            cls.ret = "Successfully added page numbers"
            return cls.ret
        except Exception as e:
            cls.ret = f"Error adding page numbers: {str(e)}"
            return cls.ret

    @classmethod
    @with_context
    def insert_page_break(cls, position="at_cursor"):
        """
        Inserts a page break at the specified position.

        Args:
            position (str): Where to insert the page break: 'at_cursor' for current cursor position,
                           'end_of_document' for end of document. Defaults to 'at_cursor'.
        """
        try:
            if position == "end_of_document":
                cls.cursor.gotoEnd(False)
            cls.text.insertControlCharacter(cls.cursor, PARAGRAPH_BREAK, False)
            cls.cursor.gotoStartOfParagraph(True)
            cls.cursor.BreakType = uno.Enum("com.sun.star.style.BreakType", "PAGE_BEFORE")
            cls.ret = "Page break inserted successfully"
            return True
        except Exception as e:
            cls.ret = f"Error inserting page break: {str(e)}"
            return False

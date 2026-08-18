import json
import os
import uno
import time

from com.sun.star.awt.FontSlant import ITALIC, NONE
from com.sun.star.awt.FontWeight import BOLD, NORMAL
from com.sun.star.beans import PropertyValue
from com.sun.star.drawing.TextHorizontalAdjust import CENTER, LEFT, RIGHT

from com.sun.star.style.ParagraphAdjust import (
    LEFT as PARA_LEFT,
    CENTER as PARA_CENTER,
    RIGHT as PARA_RIGHT,
    BLOCK as PARA_JUSTIFY,
)
from com.sun.star.drawing.TextHorizontalAdjust import (
    LEFT as TEXT_LEFT,
    CENTER as TEXT_CENTER,
    RIGHT as TEXT_RIGHT,
)
from com.sun.star.awt.FontStrikeout import NONE as STRIKEOUT_NONE, SINGLE as STRIKEOUT_SINGLE
from functools import wraps



class ImpressTools:
    localContext = None
    resolver = None
    ctx = None
    desktop = None
    doc = None
    ret = ""

    @staticmethod
    def _normalize_color(color):
        color_map = {
            "red": 16711680,
            "green": 65280,
            "blue": 255,
            "yellow": 16776960,
            "black": 0,
            "white": 16777215,
            "purple": 8388736,
            "orange": 16753920,
            "pink": 16761035,
            "gray": 8421504,
            "brown": 10824234,
            "cyan": 65535,
            "magenta": 16711935,
        }
        if isinstance(color, bool):
            raise ValueError(f"Invalid color format: {color}")
        if isinstance(color, int):
            return color
        if not isinstance(color, str):
            raise ValueError(f"Invalid color format: {color}")
        value = color.strip()
        if not value:
            raise ValueError("Color value cannot be empty.")
        lowered = value.lower()
        if lowered in color_map:
            return color_map[lowered]
        if value.startswith("#"):
            if len(value) != 7:
                raise ValueError(f"Invalid color format: {color}")
            return int(value[1:], 16)
        if lowered.startswith("0x"):
            return int(value, 16)
        if value.isdigit():
            return int(value)
        raise ValueError(f"Invalid color format: {color}")

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
            return func(cls, **kwargs)
        return wrapper

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
                return True
            else:
                cls.ret = "Error: Document has no save location"
                return False
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
    def env_info(cls, page_indices=None):
        """
        获取指定页面的内容
        :param page_indices: 页码列表，如果为None则获取所有页面
        :return: 包含各页面内容的列表
        """
        try:
            pages = cls.doc.getDrawPages()
            content_str = ""
            if page_indices is None:
                page_indices = range(1, pages.getCount() + 1)
            for page_index in page_indices:
                zero_based = page_index - 1
                if 0 <= zero_based < pages.getCount():
                    page = pages.getByIndex(zero_based)
                    page_content = []
                    for i in range(page.getCount()):
                        shape = page.getByIndex(i)
                        if hasattr(shape, "getText"):
                            text = shape.getText()
                            if text:
                                page_content.append(
                                    "- Box " + str(i) + ": " + text.getString().strip())

                    c = "\n".join(page_content)
                    content_str += f"Slide {page_index}:\n{c}\n\n"

            cur_idx = cls.get_current_slide_index() + 1
            content_str = content_str + f"Current Slide Index: {cur_idx}"
            cls.ret = content_str
            cls.print_result()
            return content_str
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            cls.print_result()
            return []

    @classmethod
    @with_context
    def get_current_slide_index(cls):
        """
        Gets the index of the currently active slide in the presentation.
        :return: The index of the currently active slide (0-based)
        """
        try:
            controller = cls.doc.getCurrentController()
            current_page = controller.getCurrentPage()
            pages = cls.doc.getDrawPages()
            for i in range(pages.getCount()):
                if pages.getByIndex(i) == current_page:
                    cls.ret = i
                    return i
            cls.ret = "Current slide not found"
            return -1
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return -1

    @classmethod
    @with_context
    def go_to_slide(cls, slide_index):
        """
        Navigates to a specific slide in the presentation based on its index.

        Args:
            slide_index (int): The index of the slide to navigate to (1-based indexing)

        Returns:
            bool: True if navigation was successful, False otherwise
        """
        try:
            zero_based_index = slide_index - 1
            controller = cls.doc.getCurrentController()
            if not controller:
                cls.ret = "Error: Could not get document controller"
                return False
            pages = cls.doc.getDrawPages()
            if zero_based_index < 0 or zero_based_index >= pages.getCount():
                cls.ret = f"Error: Slide index {slide_index} is out of range. Valid range is 1-{pages.getCount()}"
                return False
            target_slide = pages.getByIndex(zero_based_index)
            controller.setCurrentPage(target_slide)
            cls.ret = f"Successfully navigated to slide {slide_index}"
            return True
        except Exception as e:
            cls.ret = f"Error navigating to slide: {str(e)}"
            return False

    @classmethod
    @with_context
    def get_slide_count(cls):
        """
        Gets the total number of slides in the current presentation.
        :return: The total number of slides as an integer
        """
        try:
            pages = cls.doc.getDrawPages()
            count = pages.getCount()
            cls.ret = count
            return count
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return 0
        
    @classmethod
    @with_context
    def duplicate_slide(cls, slide_index, target_position):
        """
        Duplicate slide at `slide_index` and insert it right after `target_position`.
        Both parameters are 1-based indexing.
        """
        try:
            draw_pages = cls.doc.getDrawPages()
            count_before = draw_pages.getCount()

            # 参数合法性检查
            if slide_index < 1 or slide_index > count_before:
                cls.ret = f"Error: Invalid slide index {slide_index}. Valid range is 1-{count_before}"
                return False
            if target_position < 1 or target_position > count_before:
                cls.ret = f"Error: Invalid target position {target_position}. Valid range is 1-{count_before}"
                return False

            zero_based_index = slide_index - 1
            controller = cls.doc.getCurrentController()
            controller.setCurrentPage(draw_pages.getByIndex(zero_based_index))

            dispatcher = cls.ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.DispatchHelper", cls.ctx)
            frame = controller.getFrame()

            # 执行复制，新 slide 总是紧跟在源 slide 后面
            dispatcher.executeDispatch(frame, ".uno:DuplicatePage", "", 0, ())
            time.sleep(0.3)

            draw_pages = cls.doc.getDrawPages()
            new_index = zero_based_index + 1
            desired_index = target_position  # 1-based target 转 0-based 正好是 target_position

            print(f"[DEBUG] new_index={new_index}, desired_index={desired_index}")

            # 每步移动前都 setCurrentPage 确保作用在正确的 slide 上
            current = new_index
            while current < desired_index:
                controller.setCurrentPage(draw_pages.getByIndex(current))
                dispatcher.executeDispatch(frame, ".uno:MovePageDown", "", 0, ())
                time.sleep(0.1)
                current += 1

            while current > desired_index:
                controller.setCurrentPage(draw_pages.getByIndex(current))
                dispatcher.executeDispatch(frame, ".uno:MovePageUp", "", 0, ())
                time.sleep(0.1)
                current -= 1

            cls.ret = f"Slide {slide_index} duplicated successfully and inserted after slide {target_position}"
            return True

        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_slide_font(cls, slide_index, font_name):
        """
        Sets the font style for all text elements in a specific slide, including the title.

        Args:
            slide_index (int): The index of the slide to modify (1-based indexing)
            font_name (str): The name of the font to apply (e.g., 'Arial', 'Times New Roman', 'Calibri')

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            zero_based_index = slide_index - 1
            slides = cls.doc.getDrawPages()
            if zero_based_index < 0 or zero_based_index >= slides.getCount():
                cls.ret = f"Error: Slide index {slide_index} is out of range. Valid range is 1 to {slides.getCount()}."
                return False
            slide = slides.getByIndex(zero_based_index)
            for i in range(slide.getCount()):
                shape = slide.getByIndex(i)
                if hasattr(shape, "getText"):
                    text = shape.getText()
                    if text:
                        cursor = text.createTextCursor()
                        cursor.gotoStart(False)
                        cursor.gotoEnd(True)
                        cursor.setPropertyValue("CharFontName", font_name)
            cls.ret = f"Successfully set font to '{font_name}' for all text elements in slide {slide_index}."
            return True
        except Exception as e:
            cls.ret = f"Error setting font: {str(e)}"
            return False

    @classmethod
    @with_context
    def write_text(cls, content, page_index, box_index, bold=False, italic=False, size=None, append=False):
        """
        Writes text to a specific textbox on a slide

        :param content: The text content to add
        :param page_index: The index of the slide (1-based indexing)
        :param box_index: The index of the textbox to modify (0-based indexing)
        :param bold: Whether to make the text bold, default is False
        :param italic: Whether to make the text italic, default is False
        :param size: The size of the text. If None, uses the box's current font size.
        :param append: Whether to append the text, default is False. If you want to observe some formats(like a bullet at the beginning) or keep the original text, you should set up it.
        :return: True if successful, False otherwise
        """
        try:
            zero_based_page_index = page_index - 1
            pages = cls.doc.getDrawPages()
            if zero_based_page_index < 0 or zero_based_page_index >= pages.getCount():
                cls.ret = f"Error: Page index {page_index} is out of range"
                return False
            page = pages.getByIndex(zero_based_page_index)
            if box_index < 0 or box_index >= page.getCount():
                cls.ret = f"Error: Box index {box_index} is out of range"
                return False
            shape = page.getByIndex(box_index)
            if not hasattr(shape, "getText"):
                cls.ret = f"Error: The shape at index {box_index} cannot contain text"
                return False

            text = shape.getText()
            cursor = text.createTextCursor()

            # libreoffice_impress.py write_text
            if not append:
                text = shape.getText()
                cursor = text.createTextCursor()
                cursor.gotoStart(False)
                cursor.gotoEnd(True)
                # 如果 shape 为空，gotoEnd 后 cursor 仍在起点，insertString replace 无害
                # 如果不为空，replace=True 会替换选中内容
                text.insertString(cursor, content, True)
            else:
                cursor = text.createTextCursor()
                cursor.gotoEnd(False)
                text.insertString(cursor, content, False)

            # 只在显式传入时才修改格式
            cursor2 = text.createTextCursor()
            cursor2.gotoStart(False)
            cursor2.gotoEnd(True)
            if bold is not None:
                cursor2.setPropertyValue("CharWeight", BOLD if bold else NORMAL)
            if italic is not None:
                cursor2.setPropertyValue("CharPosture", ITALIC if italic else NONE)
            if size is not None:
                cursor2.setPropertyValue("CharHeight", float(size))

            cls.ret = f"Text successfully written to page {page_index}, box {box_index}"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_style(cls, slide_index, box_index, bold=None, italic=None, underline=None):
        """
        Sets the style properties for the specified textbox on a slide.

        :param slide_index: The index of the slide to modify (1-based indexing)
        :param box_index: The index of the textbox to modify (0-based indexing)
        :param bold: Whether to make the text bold
        :param italic: Whether to make the text italic
        :param underline: Whether to underline the text
        :return: True if successful, False otherwise
        """
        try:
            pages = cls.doc.getDrawPages()
            if slide_index < 1 or slide_index > pages.getCount():
                cls.ret = f"Error: Invalid slide index {slide_index}. Valid range is 1 to {pages.getCount()}"
                return False
            page = pages.getByIndex(slide_index - 1)
            if box_index < 0 or box_index >= page.getCount():
                cls.ret = f"Error: Invalid box index {box_index}. Valid range is 0 to {page.getCount() - 1}"
                return False
            shape = page.getByIndex(box_index)
            if not hasattr(shape, "getText"):
                cls.ret = "Error: The specified shape does not contain text"
                return False
            text = shape.getText()
            cursor = text.createTextCursor()
            cursor.gotoStart(False)
            cursor.gotoEnd(True)
            if bold is not None:
                cursor.setPropertyValue("CharWeight", BOLD if bold else NORMAL)
            if italic is not None:
                cursor.setPropertyValue("CharPosture", ITALIC if italic else NONE)
            if underline is not None:
                cursor.setPropertyValue("CharUnderline", 1 if underline else 0)
            cls.ret = "Style applied successfully"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def configure_auto_save(cls, enabled, interval_minutes):
        """
        Enables or disables auto-save functionality for the current document and sets the auto-save interval.

        :param enabled: Whether to enable (True) or disable (False) auto-save
        :param interval_minutes: The interval in minutes between auto-saves (minimum 1 minute)
        :return: True if successful, False otherwise
        """
        try:
            if interval_minutes < 1:
                interval_minutes = 1
            config_provider = cls.ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.configuration.ConfigurationProvider", cls.ctx
            )
            prop = PropertyValue()
            prop.Name = "nodepath"
            prop.Value = "/org.openoffice.Office.Common/Save/Document"
            config_access = config_provider.createInstanceWithArguments(
                "com.sun.star.configuration.ConfigurationUpdateAccess", (prop,)
            )
            config_access.setPropertyValue("AutoSave", enabled)
            config_access.setPropertyValue("AutoSaveTimeIntervall", interval_minutes)
            config_access.commitChanges()
            cls.ret = f"Auto-save {'enabled' if enabled else 'disabled'} with interval of {interval_minutes} minutes"
            return True
        except Exception as e:
            cls.ret = f"Error configuring auto-save: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_background_color(cls, slide_index, box_index, color):
        """
        Sets the background color for the specified textbox on a slide.

        Args:
            slide_index (int): The index of the slide containing the textbox (1-based indexing)
            box_index (int): The index of the textbox to modify (0-based indexing)
            color (str): The color to apply to the textbox (e.g., 'red', 'green', 'blue', 'yellow', or hex color code)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            zero_based_slide_index = slide_index - 1
            slides = cls.doc.getDrawPages()
            if zero_based_slide_index < 0 or zero_based_slide_index >= slides.getCount():
                cls.ret = f"Error: Slide index {slide_index} is out of range"
                return False
            slide = slides.getByIndex(zero_based_slide_index)
            if box_index < 0 or box_index >= slide.getCount():
                cls.ret = f"Error: Box index {box_index} is out of range"
                return False
            shape = slide.getByIndex(box_index)
            color_int = 0
            color_int = cls._normalize_color(color)
            shape.FillStyle = uno.Enum("com.sun.star.drawing.FillStyle", "SOLID")
            shape.FillColor = color_int
            cls.ret = f"Background color of textbox {box_index} on slide {slide_index} set to {color}"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_text_color(cls, slide_index, box_index, color):
        """
        Sets the text color for the specified textbox on a slide.

        Args:
            slide_index (int): The index of the slide to modify (1-based indexing)
            box_index (int): The index of the textbox to modify (0-based indexing)
            color (str): The color to apply to the text (e.g., 'red', 'green', 'blue', 'black', or hex color code)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            zero_based_slide_index = slide_index - 1
            slides = cls.doc.getDrawPages()
            if zero_based_slide_index < 0 or zero_based_slide_index >= slides.getCount():
                cls.ret = f"Error: Slide index {slide_index} is out of range"
                return False
            slide = slides.getByIndex(zero_based_slide_index)
            if box_index < 0 or box_index >= slide.getCount():
                cls.ret = f"Error: Box index {box_index} is out of range"
                return False
            shape = slide.getByIndex(box_index)
            if not hasattr(shape, "getText"):
                cls.ret = f"Error: Shape at index {box_index} does not contain text"
                return False
            color_int = cls._normalize_color(color)
            text = shape.getText()
            cursor = text.createTextCursor()
            cursor.gotoStart(False)
            cursor.gotoEnd(True)
            cursor.setPropertyValue("CharColor", color_int)
            cls.ret = f"Successfully set text color to {color} for textbox {box_index} on slide {slide_index}"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def delete_content(cls, slide_index, box_index):
        """
        Deletes the specified textbox from a slide.

        :param slide_index: The index of the slide to modify (1-based indexing)
        :param box_index: The index of the textbox to modify (0-based indexing)
        :return: True if successful, False otherwise
        """
        try:
            pages = cls.doc.getDrawPages()
            zero_based_slide_index = slide_index - 1
            if zero_based_slide_index < 0 or zero_based_slide_index >= pages.getCount():
                cls.ret = f"Error: Invalid slide index {slide_index}. Valid range is 1 to {pages.getCount()}"
                return False
            slide = pages.getByIndex(zero_based_slide_index)
            if box_index < 0 or box_index >= slide.getCount():
                cls.ret = f"Error: Invalid box index {box_index}. Valid range is 0 to {slide.getCount() - 1}"
                return False
            shape = slide.getByIndex(box_index)
            slide.remove(shape)
            cls.ret = f"Successfully deleted textbox {box_index} from slide {slide_index}"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_slide_orientation(cls, orientation):
        """
        Changes the orientation of slides in the presentation between portrait (upright) and landscape (sideways).

        :param orientation: The desired orientation for the slides ('portrait' or 'landscape')
        :return: True if successful, False otherwise
        """
        try:
            draw_pages = cls.doc.getDrawPages()
            first_page = draw_pages.getByIndex(0)
            current_width = first_page.Width
            current_height = first_page.Height
            if orientation == "portrait" and current_width > current_height:
                new_width, new_height = current_height, current_width
            elif orientation == "landscape" and current_width < current_height:
                new_width, new_height = current_height, current_width
            else:
                cls.ret = f"Slides are already in {orientation} orientation"
                return True
            for i in range(draw_pages.getCount()):
                page = draw_pages.getByIndex(i)
                page.Width = new_width
                page.Height = new_height
            cls.ret = f"Changed slide orientation to {orientation}"
            return True
        except Exception as e:
            cls.ret = f"Error changing slide orientation: {str(e)}"
            return False

    @classmethod
    @with_context
    def position_box(cls, slide_index, box_index, position):
        """
        Positions a textbox or image on a slide at a specific location or predefined position.

        :param slide_index: The index of the slide containing the box (1-based indexing)
        :param box_index: The index of the box to position (0-based indexing)
        :param position: Predefined position on the slide (left, right, center, top, bottom, etc.)
        :return: True if successful, False otherwise
        """
        try:
            pages = cls.doc.getDrawPages()
            if slide_index < 1 or slide_index > pages.getCount():
                cls.ret = f"Error: Invalid slide index {slide_index}"
                return False
            page = pages.getByIndex(slide_index - 1)
            if box_index < 0 or box_index >= page.getCount():
                cls.ret = f"Error: Invalid box index {box_index}"
                return False
            shape = page.getByIndex(box_index)
            controller = cls.doc.getCurrentController()
            slide_width = 28000
            slide_height = 21000
            shape_width = shape.Size.Width
            shape_height = shape.Size.Height
            margin = 500
            if position == "left":
                new_x = margin
                new_y = (slide_height - shape_height) / 2
            elif position == "right":
                new_x = slide_width - shape_width - margin
                new_y = (slide_height - shape_height) / 2
            elif position == "center":
                new_x = (slide_width - shape_width) / 2
                new_y = (slide_height - shape_height) / 2
            elif position == "top":
                new_x = (slide_width - shape_width) / 2
                new_y = margin
            elif position == "bottom":
                new_x = (slide_width - shape_width) / 2
                new_y = slide_height - shape_height - margin
            elif position == "top-left":
                new_x = margin
                new_y = margin
            elif position == "top-right":
                new_x = slide_width - shape_width - margin
                new_y = margin
            elif position == "bottom-left":
                new_x = margin
                new_y = slide_height - shape_height - margin
            elif position == "bottom-right":
                new_x = slide_width - shape_width - margin
                new_y = slide_height - shape_height - margin
            else:
                cls.ret = f"Error: Invalid position '{position}'"
                return False
            try:
                shape.Position.X = int(new_x)
                shape.Position.Y = int(new_y)
            except:
                try:
                    shape.setPropertyValue("PositionX", int(new_x))
                    shape.setPropertyValue("PositionY", int(new_y))
                except:
                    point = uno.createUnoStruct("com.sun.star.awt.Point", int(new_x), int(new_y))
                    shape.setPosition(point)
            cls.ret = f"Box positioned at {position} (X: {new_x}, Y: {new_y})"
            return True
        except Exception as e:
            cls.ret = f"Error positioning box: {str(e)}"
            return False

    @classmethod
    @with_context
    def insert_file(cls, file_path, slide_index=None, position=None, size=None, autoplay=False):
        """
        Inserts a video file into the current or specified slide in the presentation.

        Args:
            file_path (str): The full path to the video file to be inserted
            slide_index (int, optional): The index of the slide to insert the video into (1-based indexing).
                                        If not provided, inserts into the current slide.
            position (dict, optional): The position coordinates for the video as percentages of slide dimensions
                                      {'x': float, 'y': float}
            size (dict, optional): The size dimensions for the video as percentages of slide dimensions
                                  {'width': float, 'height': float}
            autoplay (bool, optional): Whether the video should automatically play when the slide is shown

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            expanded_file_path = os.path.expanduser(file_path)
            if not os.path.exists(expanded_file_path):
                cls.ret = f"Error: File not found: {expanded_file_path}"
                return False
            file_url = uno.systemPathToFileUrl(os.path.abspath(expanded_file_path))
            pages = cls.doc.getDrawPages()
            if slide_index is not None:
                zero_based_index = slide_index - 1
                if zero_based_index < 0 or zero_based_index >= pages.getCount():
                    cls.ret = f"Error: Invalid slide index: {slide_index}"
                    return False
                slide = pages.getByIndex(zero_based_index)
            else:
                controller = cls.doc.getCurrentController()
                slide = controller.getCurrentPage()
            slide_width = 21000
            slide_height = 12750
            if position is None:
                position = {"x": 10, "y": 10}
            if size is None:
                size = {"width": 80, "height": 60}
            x = int(position["x"] * slide_width / 100)
            y = int(position["y"] * slide_height / 100)
            width = int(size["width"] * slide_width / 100)
            height = int(size["height"] * slide_height / 100)
            media_shape = cls.doc.createInstance("com.sun.star.presentation.MediaShape")
            slide.add(media_shape)
            media_shape.setPosition(uno.createUnoStruct("com.sun.star.awt.Point", x, y))
            media_shape.setSize(uno.createUnoStruct("com.sun.star.awt.Size", width, height))
            media_shape.setPropertyValue("MediaURL", file_url)
            if autoplay:
                try:
                    media_shape.setPropertyValue("MediaIsAutoPlay", True)
                except:
                    pass
            cls.ret = f"Video inserted successfully from {expanded_file_path}"
            return True
        except Exception as e:
            cls.ret = f"Error inserting video: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_slide_background(cls, slide_index=None, color=None, image_path=None):
        """
        Sets the background color or image for a specific slide or all slides.

        Args:
            slide_index (int, optional): The index of the slide to modify (1-based indexing).
                                        If not provided, applies to all slides.
            color (str, optional): The background color to apply (e.g., 'red', 'green', 'blue', or hex color code)
            image_path (str, optional): Path to an image file to use as background. If provided, overrides color.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not color and not image_path:
                cls.ret = "Error: Either color or image_path must be provided"
                return False
            pages = cls.doc.getDrawPages()
            page_count = pages.getCount()
            rgb_color = None
            if color:
                rgb_color = cls._normalize_color(color)
            if slide_index is not None:
                slide_index = slide_index - 1
                if slide_index < 0 or slide_index >= page_count:
                    cls.ret = f"Error: Slide index {slide_index + 1} is out of range (1-{page_count})"
                    return False
                slides_to_modify = [pages.getByIndex(slide_index)]
            else:
                slides_to_modify = [pages.getByIndex(i) for i in range(page_count)]
            for slide in slides_to_modify:
                if image_path and os.path.exists(image_path):
                    abs_path = os.path.abspath(image_path)
                    file_url = uno.systemPathToFileUrl(abs_path)
                    slide.FillStyle = uno.Enum(
                        "com.sun.star.drawing.FillStyle", "BITMAP")
                    slide.FillBitmapURL = file_url
                    slide.FillBitmapMode = uno.Enum(
                        "com.sun.star.drawing.BitmapMode", "STRETCH")
                elif rgb_color is not None:
                    slide.FillStyle = uno.Enum(
                        "com.sun.star.drawing.FillStyle", "SOLID")
                    slide.FillColor = rgb_color
                
            cls.ret = "Background set successfully"
            return True
        except Exception as e:
            cls.ret = f"Error setting background: {str(e)}"
            return False

    @classmethod
    @with_context
    def save_as(cls, file_path, overwrite=False):
        """
        Saves the current document to a specified location with a given filename.

        :param file_path: The full path where the file should be saved, including the filename and extension
        :param overwrite: Whether to overwrite the file if it already exists (default: False)
        :return: True if successful, False otherwise
        """
        try:
            if os.path.exists(file_path) and not overwrite:
                cls.ret = f"File already exists and overwrite is set to False: {file_path}"
                return False
            abs_path = os.path.abspath(file_path)
            if os.name == "nt":
                url = "file:///" + abs_path.replace("\\", "/")
            else:
                url = "file://" + abs_path
            properties = []
            overwrite_prop = PropertyValue()
            overwrite_prop.Name = "Overwrite"
            overwrite_prop.Value = overwrite
            properties.append(overwrite_prop)
            extension = os.path.splitext(file_path)[1].lower()
            if extension == ".odp":
                filter_name = "impress8"
            elif extension == ".ppt":
                filter_name = "MS PowerPoint 97"
            elif extension == ".pptx":
                filter_name = "Impress MS PowerPoint 2007 XML"
            elif extension == ".pdf":
                filter_name = "impress_pdf_Export"
            else:
                filter_name = "impress8"
            filter_prop = PropertyValue()
            filter_prop.Name = "FilterName"
            filter_prop.Value = filter_name
            properties.append(filter_prop)
            cls.doc.storeAsURL(url, tuple(properties))
            cls.ret = f"Document saved successfully to {file_path}"
            return True
        except Exception as e:
            cls.ret = f"Error saving document: {str(e)}"
            return False

    @classmethod
    @with_context
    def insert_image(cls, slide_index, image_path, width=None, height=None, position=None):
        """
        Inserts an image to a specific slide in the presentation.

        Args:
            slide_index (int): The index of the slide to add the image to (1-based indexing)
            image_path (str): The full path to the image file to be added
            width (float, optional): The width of the image in centimeters
            height (float, optional): The height of the image in centimeters
            position (dict, optional): The position coordinates for the image as percentages
                {
                    'x': float, # The x-coordinate as a percentage of slide width
                    'y': float  # The y-coordinate as a percentage of slide height
                }

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                cls.ret = f"Error: Image file not found at {image_path}"
                return False
            zero_based_index = slide_index - 1
            slides = cls.doc.getDrawPages()
            if zero_based_index < 0 or zero_based_index >= slides.getCount():
                cls.ret = f"Error: Slide index {slide_index} is out of range. Valid range is 1 to {slides.getCount()}"
                return False
            slide = slides.getByIndex(zero_based_index)
            bitmap = cls.doc.createInstance("com.sun.star.drawing.BitmapTable")
            image_url = uno.systemPathToFileUrl(os.path.abspath(image_path))
            shape = cls.doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
            shape.setPropertyValue("GraphicURL", image_url)
            slide.add(shape)
            x_pos = 0
            y_pos = 0
            slide_width = slide.Width
            slide_height = slide.Height
            if position:
                if "x" in position:
                    x_pos = int(position["x"] / 100 * slide_width)
                if "y" in position:
                    y_pos = int(position["y"] / 100 * slide_height)
            current_width = shape.Size.Width
            current_height = shape.Size.Height
            new_width = int(width * 1000) if width is not None else current_width
            new_height = int(height * 1000) if height is not None else current_height
            size = uno.createUnoStruct("com.sun.star.awt.Size")
            size.Width = new_width
            size.Height = new_height
            point = uno.createUnoStruct("com.sun.star.awt.Point")
            point.X = x_pos
            point.Y = y_pos
            shape.Size = size
            shape.Position = point
            cls.ret = f"Image inserted successfully on slide {slide_index}"
            return True
        except Exception as e:
            cls.ret = f"Error inserting image: {str(e)}"
            return False

    @classmethod
    @with_context
    def configure_display_settings(
        cls, use_presenter_view=None, primary_monitor_only=None, monitor_for_presentation=None
    ):
        """
        Configures the display settings for LibreOffice Impress presentations.

        Args:
            use_presenter_view (bool, optional): Whether to use presenter view. Set to false to disable presenter view.
            primary_monitor_only (bool, optional): Whether to use only the primary monitor for the presentation.
            monitor_for_presentation (int, optional): Specify which monitor to use (1 for primary, 2 for secondary, etc.)

        Returns:
            bool: True if settings were successfully applied, False otherwise
        """
        try:
            controller = cls.doc.getCurrentController()
            if not hasattr(controller, "getPropertyValue"):
                cls.ret = "Error: Not an Impress presentation or controller not available"
                return False
            if use_presenter_view is not None:
                try:
                    controller.setPropertyValue("IsPresentationViewEnabled", use_presenter_view)
                except Exception as e:
                    cls.ret = f"Warning: Could not set presenter view: {str(e)}"
            if primary_monitor_only is not None:
                try:
                    controller.setPropertyValue("UsePrimaryMonitorOnly", primary_monitor_only)
                except Exception as e:
                    cls.ret = f"Warning: Could not set primary monitor usage: {str(e)}"
            if monitor_for_presentation is not None:
                try:
                    controller.setPropertyValue("MonitorForPresentation", monitor_for_presentation - 1)
                except Exception as e:
                    cls.ret = f"Warning: Could not set presentation monitor: {str(e)}"
            cls.ret = "Display settings configured successfully"
            return True
        except Exception as e:
            cls.ret = f"Error configuring display settings: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_text_strikethrough(cls, slide_index, box_index, line_numbers, apply):
        """
        Applies or removes strike-through formatting to specific text content in a slide.

        Args:
            slide_index (int): The index of the slide containing the text (1-based indexing)
            box_index (int): The index of the textbox containing the text (0-based indexing)
            line_numbers (list): The line numbers to apply strike-through formatting to (1-based indexing)
            apply (bool): Whether to apply (true) or remove (false) strike-through formatting

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            slides = cls.doc.getDrawPages()
            slide = slides.getByIndex(slide_index - 1)
            shape = slide.getByIndex(box_index)
            if not hasattr(shape, "getText"):
                cls.ret = f"Error: Shape at index {box_index} does not contain text"
                return False
            text = shape.getText()
            cursor = text.createTextCursor()
            text_content = text.getString()
            lines = text_content.split("\n")
            for line_number in line_numbers:
                if 1 <= line_number <= len(lines):
                    start_pos = 0
                    for i in range(line_number - 1):
                        start_pos += len(lines[i]) + 1
                    end_pos = start_pos + len(lines[line_number - 1])
                    cursor.gotoStart(False)
                    cursor.goRight(start_pos, False)
                    cursor.setPropertyValue(
                        "CharStrikeout",
                        STRIKEOUT_SINGLE if apply else STRIKEOUT_NONE
                    )
            cls.ret = f"Strike-through {'applied' if apply else 'removed'} successfully"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_textbox_alignment(cls, slide_index, box_index, alignment):
        """
        Sets the text alignment for the specified textbox on a slide.

        :param slide_index: The index of the slide to modify (1-based indexing)
        :param box_index: The index of the textbox to modify (0-based indexing)
        :param alignment: The text alignment to apply ('left', 'center', 'right', or 'justify')
        :return: True if successful, False otherwise
        """
        try:
            zero_based_slide_index = slide_index - 1
            slides = cls.doc.getDrawPages()
            if zero_based_slide_index < 0 or zero_based_slide_index >= slides.getCount():
                cls.ret = f"Error: Slide index {slide_index} out of range"
                return False
            slide = slides.getByIndex(zero_based_slide_index)
            if box_index < 0 or box_index >= slide.getCount():
                cls.ret = f"Error: Box index {box_index} out of range"
                return False
            shape = slide.getByIndex(box_index)
            if not hasattr(shape, "getText"):
                cls.ret = "Error: Selected shape does not support text"
                return False
          
            alignment_map = {
                "left": PARA_LEFT,
                "center": PARA_CENTER,
                "right": PARA_RIGHT,
                "justify": PARA_JUSTIFY,
            }
            if alignment not in alignment_map:
                cls.ret = f"Error: Invalid alignment value: {alignment}"
                return False

            text = shape.getText()
            cursor = text.createTextCursor()
            cursor.gotoStart(False)
            cursor.gotoEnd(True)
            cursor.setPropertyValue("ParaAdjust", alignment_map[alignment])
          
            cls.ret = f"Successfully set text alignment to {alignment} for textbox {box_index} on slide {slide_index}"
            return True
        except Exception as e:
            cls.ret = f"Error: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_slide_number_properties(
        cls, color=None, font_size=None, visible=None, position=None, apply_to="all", slide_indices=None
    ):
        """
        Modifies the properties of slide numbers in the presentation.

        Args:
            color (str, optional): The color to apply to slide numbers (e.g., 'red', 'green', 'blue', 'black', or hex color code)
            font_size (float, optional): The font size for slide numbers (in points)
            visible (bool, optional): Whether slide numbers should be visible or hidden
            position (str, optional): The position of slide numbers ('bottom-left', 'bottom-center', 'bottom-right',
                                     'top-left', 'top-center', 'top-right')
            apply_to (str, optional): Whether to apply changes to 'all', 'current', or 'selected' slides
            slide_indices (list, optional): Indices of specific slides to change (1-based indexing)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            draw_pages = cls.doc.getDrawPages()
            master_pages = cls.doc.getMasterPages()
            pages_to_modify = []
            if apply_to == "all":
                for i in range(draw_pages.getCount()):
                    pages_to_modify.append(draw_pages.getByIndex(i))
            elif apply_to == "current":
                current_page = cls.doc.getCurrentController().getCurrentPage()
                pages_to_modify.append(current_page)
            elif apply_to == "selected" and slide_indices:
                for idx in slide_indices:
                    if 1 <= idx <= draw_pages.getCount():
                        pages_to_modify.append(draw_pages.getByIndex(idx - 1))
            for i in range(master_pages.getCount()):
                master_page = master_pages.getByIndex(i)
                page_number_shape = None
                for j in range(master_page.getCount()):
                    shape = master_page.getByIndex(j)
                    if hasattr(shape, "TextType"):
                        try:
                            if shape.TextType == 5:
                                page_number_shape = shape
                                break
                        except:
                            pass
                    if hasattr(shape, "getText"):
                        try:
                            text = shape.getText()
                            if text and text.getTextFields().getCount() > 0:
                                fields = text.getTextFields().createEnumeration()
                                while fields.hasMoreElements():
                                    field = fields.nextElement()
                                    if "PageNumber" in field.getImplementationName():
                                        page_number_shape = shape
                                        break
                                if page_number_shape:
                                    break
                        except:
                            pass
                if page_number_shape:
                    if color is not None:
                        color_int = cls._normalize_color(color)
                        text = page_number_shape.getText()
                        cursor = text.createTextCursor()
                        cursor.gotoStart(False)
                        cursor.gotoEnd(True)
                        cursor.CharColor = color_int
                    if font_size is not None:
                        text = page_number_shape.getText()
                        cursor = text.createTextCursor()
                        cursor.gotoStart(False)
                        cursor.gotoEnd(True)
                        cursor.CharHeight = font_size
                    if position is not None:
                        page_width = master_page.Width
                        page_height = master_page.Height
                        width = page_number_shape.Size.Width
                        height = page_number_shape.Size.Height
                        new_x = 0
                        new_y = 0
                        if position.startswith("bottom"):
                            new_y = page_height - height - 100
                        elif position.startswith("top"):
                            new_y = 100
                        if position.endswith("left"):
                            new_x = 100
                        elif position.endswith("center"):
                            new_x = (page_width - width) / 2
                        elif position.endswith("right"):
                            new_x = page_width - width - 100
                        page_number_shape.Position = uno.createUnoStruct("com.sun.star.awt.Point", new_x, new_y)
                        if position.endswith("left"):
                            page_number_shape.ParaAdjust = PARA_LEFT
                        elif position.endswith("center"):
                            page_number_shape.ParaAdjust = PARA_CENTER
                        elif position.endswith("right"):
                            page_number_shape.ParaAdjust = PARA_RIGHT
                    if visible is not None:
                        try:
                            page_number_shape.Visible = visible
                        except:
                            if not visible:
                                page_number_shape.Size = uno.createUnoStruct("com.sun.star.awt.Size", 1, 1)
                                page_number_shape.Position = uno.createUnoStruct("com.sun.star.awt.Point", -1000, -1000)
                elif (
                    visible is True
                    or visible is None
                    and (color is not None or font_size is not None or position is not None)
                ):
                    page_number_shape = cls.doc.createInstance("com.sun.star.drawing.TextShape")
                    master_page.add(page_number_shape)
                    default_width = 2000
                    default_height = 400
                    page_number_shape.Size = uno.createUnoStruct("com.sun.star.awt.Size", default_width, default_height)
                    page_width = master_page.Width
                    page_height = master_page.Height
                    pos_x = page_width - default_width - 100
                    pos_y = page_height - default_height - 100
                    if position is not None:
                        if position.startswith("bottom"):
                            pos_y = page_height - default_height - 100
                        elif position.startswith("top"):
                            pos_y = 100
                        if position.endswith("left"):
                            pos_x = 100
                            page_number_shape.ParaAdjust = PARA_LEFT
                        elif position.endswith("center"):
                            pos_x = (page_width - default_width) / 2
                            page_number_shape.ParaAdjust = PARA_CENTER
                        elif position.endswith("right"):
                            pos_x = page_width - default_width - 100
                            page_number_shape.ParaAdjust = PARA_RIGHT
                    page_number_shape.Position = uno.createUnoStruct("com.sun.star.awt.Point", pos_x, pos_y)
                    text = page_number_shape.getText()
                    cursor = text.createTextCursor()
                    try:
                        page_field = cls.doc.createInstance("com.sun.star.text.TextField.PageNumber")
                        text.insertTextContent(cursor, page_field, False)
                    except:
                        text.setString("<#>")
                    if color is not None:
                        color_int = cls._normalize_color(color)
                        cursor.gotoStart(False)
                        cursor.gotoEnd(True)
                        cursor.CharColor = color_int
                    if font_size is not None:
                        cursor.gotoStart(False)
                        cursor.gotoEnd(True)
                        cursor.CharHeight = font_size
                    if visible is not None:
                        try:
                            page_number_shape.Visible = visible
                        except:
                            if not visible:
                                page_number_shape.Position = uno.createUnoStruct("com.sun.star.awt.Point", -1000, -1000)
                    else:
                        try:
                            page_number_shape.Visible = True
                        except:
                            pass
            try:
                controller = cls.doc.getCurrentController()
                view_data = controller.getViewData()
                controller.restoreViewData(view_data)
            except:
                pass
            cls.ret = "Slide number properties updated successfully"
            return True
        except Exception as e:
            cls.ret = f"Error setting slide number properties: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_slide_number(cls, color=None, font_size=None, visible=None, position=None):
        """
        Sets the slide number in the presentation.

        :param color: The color to apply to slide numbers (e.g., 'red', 'green', 'blue', 'black', or hex color code)
        :param font_size: The font size for slide numbers (in points)
        :param visible: Whether slide numbers should be visible or hidden
        :param position: The position of slide numbers on the slides (bottom-left, bottom-center, bottom-right, top-left, top-center, top-right)
        :return: True if successful, False otherwise
        """
        try:
            controller = cls.doc.getCurrentController()
            dispatcher = cls.ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.DispatchHelper", cls.ctx)
            if visible is False:
                pages = cls.doc.getDrawPages()
                for i in range(pages.getCount()):
                    page = pages.getByIndex(i)
                    for j in range(page.getCount()):
                        try:
                            shape = page.getByIndex(j)
                            if hasattr(shape, "Presentation") and shape.Presentation == "Number":
                                page.remove(shape)
                        except:
                            pass
                master_pages = cls.doc.getMasterPages()
                for i in range(master_pages.getCount()):
                    master_page = master_pages.getByIndex(i)
                    for j in range(master_page.getCount()):
                        try:
                            shape = master_page.getByIndex(j)
                            if hasattr(shape, "Presentation") and shape.Presentation == "Number":
                                master_page.remove(shape)
                        except:
                            pass
                cls.ret = "Slide numbers hidden successfully"
                return True
            if visible is True or color is not None or font_size is not None or position is not None:
                current_slide = controller.getCurrentPage()
                master_pages = cls.doc.getMasterPages()
                if master_pages.getCount() == 0:
                    cls.ret = "No master pages found"
                    return False
                master_page = master_pages.getByIndex(0)
                slide_number_shape = cls.doc.createInstance("com.sun.star.drawing.TextShape")
                slide_number_shape.setSize(uno.createUnoStruct("com.sun.star.awt.Size", 2000, 500))
                pos = position or "bottom-right"
                page_width = master_page.Width
                page_height = master_page.Height
                x, y = 0, 0
                if "bottom" in pos:
                    y = page_height - 1000
                elif "top" in pos:
                    y = 500
                if "left" in pos:
                    x = 500
                elif "center" in pos:
                    x = (page_width - 2000) / 2
                elif "right" in pos:
                    x = page_width - 2500
                slide_number_shape.setPosition(uno.createUnoStruct("com.sun.star.awt.Point", x, y))
                master_page.add(slide_number_shape)
                text = slide_number_shape.getText()
                cursor = text.createTextCursor()
                page_number = cls.doc.createInstance("com.sun.star.text.TextField.PageNumber")
                text.insertTextContent(cursor, page_number, False)
                if "center" in pos:
                    slide_number_shape.setPropertyValue("TextHorizontalAdjust", TEXT_CENTER)
                elif "right" in pos:
                    slide_number_shape.setPropertyValue("TextHorizontalAdjust", TEXT_RIGHT)
                elif "left" in pos:
                    slide_number_shape.setPropertyValue("TextHorizontalAdjust", TEXT_LEFT)
                if font_size is not None:
                    cursor.gotoStart(False)
                    cursor.gotoEnd(True)
                    cursor.setPropertyValue("CharHeight", font_size)
                if color is not None:
                    cursor.gotoStart(False)
                    cursor.gotoEnd(True)
                    cursor.setPropertyValue("CharColor", cls._normalize_color(color))
                cls.ret = "Slide numbers added and configured successfully"
                return True
        except Exception as e:
            cls.ret = f"Error setting slide number: {str(e)}"
            return False

    @classmethod
    @with_context
    def set_slide_number_color(cls, color):
        """
        Sets the color of the slide number in the presentation.

        Args:
            color (str): The color to apply to slide numbers (e.g., 'red', 'green', 'blue', 'black', or hex color code)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            rgb_color = cls._normalize_color(color)
            found = False
            master_pages = cls.doc.getMasterPages()
            for i in range(master_pages.getCount()):
                master_page = master_pages.getByIndex(i)
                for j in range(master_page.getCount()):
                    shape = master_page.getByIndex(j)
                    if hasattr(shape, "getText") and shape.getText() is not None:
                        text = shape.getText()
                        try:
                            enum = text.createEnumeration()
                            while enum.hasMoreElements():
                                para = enum.nextElement()
                                if hasattr(para, "createEnumeration"):
                                    para_enum = para.createEnumeration()
                                    while para_enum.hasMoreElements():
                                        portion = para_enum.nextElement()
                                        if (
                                            hasattr(portion, "TextPortionType")
                                            and portion.TextPortionType == "TextField"
                                        ):
                                            if hasattr(portion, "TextField") and portion.TextField is not None:
                                                field = portion.TextField
                                                if hasattr(field, "supportsService") and (
                                                    field.supportsService(
                                                        "com.sun.star.presentation.TextField.PageNumber"
                                                    )
                                                    or field.supportsService("com.sun.star.text.TextField.PageNumber")
                                                ):
                                                    portion.CharColor = rgb_color
                                                    found = True
                        except Exception as e:
                            continue
            draw_pages = cls.doc.getDrawPages()
            for i in range(draw_pages.getCount()):
                page = draw_pages.getByIndex(i)
                for j in range(page.getCount()):
                    shape = page.getByIndex(j)
                    if hasattr(shape, "getText") and shape.getText() is not None:
                        text = shape.getText()
                        try:
                            enum = text.createEnumeration()
                            while enum.hasMoreElements():
                                para = enum.nextElement()
                                if hasattr(para, "createEnumeration"):
                                    para_enum = para.createEnumeration()
                                    while para_enum.hasMoreElements():
                                        portion = para_enum.nextElement()
                                        if (
                                            hasattr(portion, "TextPortionType")
                                            and portion.TextPortionType == "TextField"
                                        ):
                                            if hasattr(portion, "TextField") and portion.TextField is not None:
                                                field = portion.TextField
                                                if hasattr(field, "supportsService") and (
                                                    field.supportsService(
                                                        "com.sun.star.presentation.TextField.PageNumber"
                                                    )
                                                    or field.supportsService("com.sun.star.text.TextField.PageNumber")
                                                ):
                                                    portion.CharColor = rgb_color
                                                    found = True
                        except Exception as e:
                            continue
            for i in range(draw_pages.getCount()):
                page = draw_pages.getByIndex(i)
                for j in range(page.getCount()):
                    shape = page.getByIndex(j)
                    if hasattr(shape, "getText") and shape.getText() is not None:
                        text = shape.getText()
                        text_string = text.getString()
                        if text_string.isdigit() and len(text_string) <= 3:
                            try:
                                cursor = text.createTextCursor()
                                cursor.gotoStart(False)
                                cursor.gotoEnd(True)
                                cursor.CharColor = rgb_color
                                found = True
                            except Exception as e:
                                continue
            if found:
                cls.ret = f"Slide number color set to {color}"
                return True
            else:
                cls.ret = "Could not find slide numbers to change color"
                return False
        except Exception as e:
            cls.ret = f"Error setting slide number color: {str(e)}"
            return False

    @classmethod
    @with_context
    def export_to_image(cls, file_path, format, slide_index=None):
        """
        Exports the current presentation or a specific slide to an image file format.

        Args:
            file_path (str): The full path where the image file should be saved, including the filename and extension
            format (str): The image format to export to (e.g., 'png', 'jpeg', 'gif')
            slide_index (int, optional): The index of the specific slide to export (1-based indexing).
                                        If not provided, exports the entire presentation as a series of images.

        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            format = format.lower()
            valid_formats = ["png", "jpeg", "jpg", "gif", "bmp", "tiff"]
            if format not in valid_formats:
                cls.ret = f"Error: Invalid format '{format}'. Valid formats are: {', '.join(valid_formats)}"
                return False
            if format == "jpg":
                format = "jpeg"
            pages = cls.doc.getDrawPages()
            page_count = pages.getCount()
            if slide_index is not None:
                slide_index = slide_index - 1
                if slide_index < 0 or slide_index >= page_count:
                    cls.ret = f"Error: Invalid slide index {slide_index + 1}. Valid range is 1 to {page_count}"
                    return False
            controller = cls.doc.getCurrentController()
            filter_name = f"draw_{format}_Export"
            filter_data = PropertyValue(Name="FilterData", Value=())
            if slide_index is not None:
                controller.setCurrentPage(pages.getByIndex(slide_index))
                props = PropertyValue(Name="FilterName", Value=filter_name), filter_data
                cls.doc.storeToURL(uno.systemPathToFileUrl(file_path), props)
                cls.ret = f"Successfully exported slide {slide_index + 1} to {file_path}"
                return True
            else:
                base_name, ext = os.path.splitext(file_path)
                for i in range(page_count):
                    controller.setCurrentPage(pages.getByIndex(i))
                    if page_count == 1:
                        current_file = f"{base_name}.{format}"
                    else:
                        current_file = f"{base_name}_{i + 1}.{format}"
                    props = PropertyValue(Name="FilterName", Value=filter_name), filter_data
                    cls.doc.storeToURL(uno.systemPathToFileUrl(current_file), props)

                if page_count == 1:
                    cls.ret = f"Successfully exported {page_count} slides to {base_name}.{format}"
                else:
                    cls.ret = f"Successfully exported {page_count} slides to {base_name}_[1-{page_count}].{format}"
                return True
        except Exception as e:
            cls.ret = f"Error exporting to image: {str(e)}"
            return False

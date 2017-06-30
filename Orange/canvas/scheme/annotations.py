"""
==================
Scheme Annotations
==================

"""
from AnyQt.QtCore import QObject
from AnyQt.QtCore import pyqtSignal as Signal, pyqtProperty as Property

from ..utils import check_type


class BaseSchemeAnnotation(QObject):
    """
    Base class for scheme annotations.
    """
    # Signal emitted when the geometry of the annotation changes
    geometry_changed = Signal()


class SchemeArrowAnnotation(BaseSchemeAnnotation):
    """
    An arrow annotation in the scheme.
    """

    color_changed = Signal(str)

    def __init__(self, start_pos, end_pos, color="red", anchor=None,
                 parent=None):
        BaseSchemeAnnotation.__init__(self, parent)
        self.__start_pos = start_pos
        self.__end_pos = end_pos
        self.__color = color
        self.__anchor = anchor

    def set_line(self, start_pos, end_pos):
        """
        Set arrow lines start and end position (``(x, y)`` tuples).
        """
        if self.__start_pos != start_pos or self.__end_pos != end_pos:
            self.__start_pos = start_pos
            self.__end_pos = end_pos
            self.geometry_changed.emit()

    def start_pos(self):
        """
        Start position of the arrow (base point).
        """
        return self.__start_pos

    start_pos = Property(tuple, fget=start_pos)

    def end_pos(self):
        """
        End position of the arrow (arrow head points toward the end).
        """
        return self.__end_pos

    end_pos = Property(tuple, fget=end_pos)

    def set_geometry(self, geometry):
        """
        Set the geometry of the arrow as a start and end position tuples
        (e.g. ``set_geometry(((0, 0), (100, 0))``).

        """
        (start_pos, end_pos) = geometry
        self.set_line(start_pos, end_pos)

    def geometry(self):
        """
        Return the start and end positions of the arrow.
        """
        return (self.start_pos, self.end_pos)

    geometry = Property(tuple, fget=geometry, fset=set_geometry)

    def set_color(self, color):
        """
        Set the fill color for the arrow as a string (`#RGB`, `#RRGGBB`,
        `#RRRGGGBBB`, `#RRRRGGGGBBBB` format or one of SVG color keyword
        names).

        """
        check_type(color, str)
        color = str(color)
        if self.__color != color:
            self.__color = color
            self.color_changed.emit(color)

    def color(self):
        """
        The arrow's fill color.
        """
        return self.__color

    color = Property(str, fget=color, fset=set_color)


class SchemeTextAnnotation(BaseSchemeAnnotation):
    """
    Text annotation in the scheme.
    """

    # Signal emitted when the annotation content change.
    content_changed = Signal(str, str)

    # Signal emitted when the annotation text changes.
    text_changed = Signal(str)

    # Signal emitted when the annotation text font changes.
    font_changed = Signal(dict)

    def __init__(self, rect, text="", content_type="text/plain", font=None,
                 anchor=None, parent=None):
        BaseSchemeAnnotation.__init__(self, parent)
        self.__rect = rect
        self.__content = text
        self.__content_type = content_type
        self.__font = {} if font is None else font
        self.__anchor = anchor

    def set_rect(self, rect):
        """
        Set the text geometry bounding rectangle (``(x, y, width, height)``
        tuple).

        """
        if self.__rect != rect:
            self.__rect = rect
            self.geometry_changed.emit()

    def rect(self):
        """
        Text bounding rectangle
        """
        return self.__rect

    rect = Property(tuple, fget=rect, fset=set_rect)

    def set_geometry(self, rect):
        """
        Set the text geometry (same as ``set_rect``)
        """
        self.set_rect(rect)

    def geometry(self):
        """
        Text annotation geometry (same as ``rect``
        """
        return self.rect

    geometry = Property(tuple, fget=geometry, fset=set_geometry)

    def set_text(self, text):
        """
        Set the annotation text.

        Same as `set_content(text, "text/plain")`
        """
        check_type(text, str)
        text = str(text)
        self.set_content(text, "text/plain")

    def text(self):
        """
        Annotation text.

        .. deprecated::
            Use `content` instead.
        """
        return self.__content

    text = Property(tuple, fget=text, fset=set_text)

    @property
    def content_type(self):
        """
        Return the annotations' content type.

        Currently this will be 'text/plain', 'text/html' or 'text/rst'.
        """
        return self.__content_type

    @property
    def content(self):
        """
        The annotation content.

        How the content is interpreted/displayed depends on `content_type`.
        """
        return self.__content

    def set_content(self, content, content_type="text/plain"):
        """
        Set the annotation content.

        Parameters
        ----------
        content : str
            The content.
        content_type : str
            Content type. Currently supported are 'text/plain' 'text/html'
            (subset supported by `QTextDocument`) and `text/rst`.
        """
        if self.__content != content or self.__content_type != content_type:
            text_changed = self.__content != content
            self.__content = content
            self.__content_type = content_type
            self.content_changed.emit(content, content_type)
            if text_changed:
                self.text_changed.emit(content)

    def set_font(self, font):
        """
        Set the annotation's default font as a dictionary of font properties
        (at the moment only family and size are used).

            >>> annotation.set_font({"family": "Helvetica", "size": 16})

        """
        check_type(font, dict)
        font = dict(font)
        if self.__font != font:
            self.__font = font
            self.font_changed.emit(font)

    def font(self):
        """
        Annotation's font property dictionary.
        """
        return dict(self.__font)

    font = Property(str, fget=font, fset=set_font)

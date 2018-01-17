from AnyQt.QtGui import QColor, QIcon, QPixmap, QPainter
from AnyQt.QtCore import Qt, QRectF

from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, DiscreteVariable, Variable
from Orange.widgets.utils import vartype

__all__ = ["createAttributePixmap", "attributeIconDict", "attributeItem"]


def createAttributePixmap(char, background=Qt.black, color=Qt.white):
    """
    Create a QIcon with a given character. The icon is 13 pixels high and wide.

    :param char: The character that is printed in the icon
    :type char: str
    :param background: the background color (default: black)
    :type background: QColor
    :param color: the character color (default: white)
    :type color: QColor
    :rtype: QIcon
    """
    icon = QIcon()
    for size in (13, 16, 18, 20, 22, 24, 28, 32, 64):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(pixmap)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing |
                               painter.SmoothPixmapTransform)
        painter.setPen(background)
        painter.setBrush(background)
        margin = 1 + size // 16
        text_margin = size // 20
        rect = QRectF(margin, margin,
                      size - 2 * margin, size - 2 * margin)
        painter.drawRoundedRect(rect, 30.0, 30.0, Qt.RelativeSize)
        painter.setPen(color)
        font = painter.font()  # type: QFont
        font.setPixelSize(size - 2 * margin - 2 * text_margin)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, char)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class __AttributeIconDict(dict):
    def __getitem__(self, key):
        if not self:
            for tpe, char, col in (
                    (vartype(ContinuousVariable()), "N", (202, 0, 32)),
                    (vartype(DiscreteVariable()), "C", (26, 150, 65)),
                    (vartype(StringVariable()), "S", (0, 0, 0)),
                    (vartype(TimeVariable()), "T", (68, 170, 255)),
                    (-1, "?", (128, 128, 128))):
                self[tpe] = createAttributePixmap(char, QColor(*col))
        if key not in self:
            key = vartype(key) if isinstance(key, Variable) else -1
        return super().__getitem__(key)

#: A dict that returns icons for different attribute types. The dict is
#: constructed on first use since icons cannot be created before initializing
#: the application.
#:
#: Accepted keys are variable type codes and instances
#: of :obj:`Orange.data.variable`: `attributeIconDict[var]` will give the
#: appropriate icon for variable `var` or a question mark if the type is not
#: recognized
attributeIconDict = __AttributeIconDict()


def attributeItem(var):
    """
    Construct a pair (icon, name) for inserting a variable into a combo or
    list box

    :param var: variable
    :type var: Orange.data.Variable
    :rtype: tuple with QIcon and str
    """
    return attributeIconDict[var], var.name

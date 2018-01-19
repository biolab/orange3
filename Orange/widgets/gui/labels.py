import os
import re

from AnyQt.QtWidgets import QLabel, QSizePolicy, QWidget
from AnyQt.QtCore import Qt, QSize, QRect, QPoint
from PyQt5.QtGui import QFontMetrics, QPainter, QPixmap, QIcon

from .base import miscellanea, setLayout
from .boxes import hBox, widgetBox

__all__ = ["widgetLabel", "label", "VerticalLabel", "SmallWidgetLabel"]


__re_label = re.compile(r"(^|[^%])%\((?P<value>[a-zA-Z]\w*)\)")


def widgetLabel(widget, label="", labelWidth=None, **misc):
    """
    Construct a simple, constant label.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param label: The text of the label (default: None)
    :type label: str
    :param labelWidth: The width of the label (default: None)
    :type labelWidth: int
    :return: Constructed label
    :rtype: QLabel
    """
    lbl = QLabel(label, widget)
    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    miscellanea(lbl, None, widget, **misc)
    return lbl


def label(widget, master, label, labelWidth=None, box=None,
          orientation=Qt.Vertical, **misc):
    """
    Construct a label that contains references to the master widget's
    attributes; when their values change, the label is updated.

    Argument :obj:`label` is a format string following Python's syntax
    (see the corresponding Python documentation): the label's content is
    rendered as `label % master.__dict__`. For instance, if the
    :obj:`label` is given as "There are %(mm)i monkeys", the value of
    `master.mm` (which must be an integer) will be inserted in place of
    `%(mm)i`.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param label: The text of the label, including attribute names
    :type label: str
    :param labelWidth: The width of the label (default: None)
    :type labelWidth: int
    :param orientation: layout of the inserted box
    :type orientation: `Qt.Vertical` (default), `Qt.Horizontal` or
        instance of `QLayout`
    :return: label
    :rtype: QLabel
    """
    b = widget if not box else hBox(widget, box, addToLayout=False)
    lbl = QLabel("", b)

    def reprint(*_):
        lbl.setText(label % master.__dict__)

    for mo in __re_label.finditer(label):
        master.connect_control(mo.group("value"), reprint)
    reprint()
    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    miscellanea(lbl, b, widget, **misc)
    return lbl


class VerticalLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setMaximumWidth(self.sizeHint().width() + 2)
        self.setMargin(4)

    def sizeHint(self):
        metrics = QFontMetrics(self.font())
        rect = metrics.boundingRect(self.text())
        size = QSize(rect.height() + self.margin(),
                     rect.width() + self.margin())
        return size

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.geometry()
        text_rect = QRect(0, 0, rect.width(), rect.height())

        painter.translate(text_rect.bottomLeft())
        painter.rotate(-90)
        painter.drawText(
            QRect(QPoint(0, 0), QSize(rect.height(), rect.width())),
            Qt.AlignCenter, self.text())
        painter.end()


# TODO: It doesn't seem anybody uses this. Document or remove.
class SmallWidgetLabel(QLabel):
    def __init__(self, widget, text="", pixmap=None, box=None,
                 orientation=Qt.Vertical, **misc):
        super().__init__(widget)
        if text:
            self.setText("<font color=\"#C10004\">" + text + "</font>")
        elif pixmap is not None:
            iconDir = os.path.join(os.path.dirname(__file__), "icons")
            name = ""
            if isinstance(pixmap, str):
                if os.path.exists(pixmap):
                    name = pixmap
                elif os.path.exists(os.path.join(iconDir, pixmap)):
                    name = os.path.join(iconDir, pixmap)
            elif isinstance(pixmap, (QPixmap, QIcon)):
                name = pixmap
            name = name or os.path.join(iconDir, "arrow_down.png")
            self.setPixmap(QPixmap(name))
        self.autohideWidget = self.widget = AutoHideWidget(None, Qt.Popup)
        setLayout(self.widget, orientation)
        if box:
            self.widget = widgetBox(self.widget, box, orientation)
        self.autohideWidget.hide()
        miscellanea(self, self.widget, widget, **misc)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            self.autohideWidget.move(
                self.mapToGlobal(QPoint(0, self.height())))
            self.autohideWidget.show()


class AutoHideWidget(QWidget):
    def leaveEvent(self, _):
        self.hide()

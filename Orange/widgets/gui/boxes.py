from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGroupBox, QWidget

from .base import setLayout, miscellanea, separator

__all__ = ["rubber", "widgetBox", "hBox", "vBox", "indentedBox"]


def rubber(widget):
    """
    Insert a stretch 100 into the widget's layout
    """
    widget.layout().addStretch(100)


def widgetBox(widget, box=None, orientation=Qt.Vertical, margin=None, spacing=4,
              **misc):
    """
    Construct a box with vertical or horizontal layout, and optionally,
    a border with an optional label.

    If the widget has a frame, the space after the widget is added unless
    explicitly disabled.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param orientation: orientation of the box
    :type orientation: `Qt.Horizontal`, `Qt.Vertical` or instance of `QLayout`
    :param sizePolicy: The size policy for the widget (default: None)
    :type sizePolicy: :obj:`~QSizePolicy`
    :param margin: The margin for the layout. Default is 7 if the widget has
        a border, and 0 if not.
    :type margin: int
    :param spacing: Spacing within the layout (default: 4)
    :type spacing: int
    :return: Constructed box
    :rtype: QGroupBox or QWidget
    """
    if box:
        b = QGroupBox(widget)
        if isinstance(box, str):
            b.setTitle(" " + box.strip() + " ")
        if margin is None:
            margin = 7
    else:
        b = QWidget(widget)
        b.setContentsMargins(0, 0, 0, 0)
        if margin is None:
            margin = 0
    setLayout(b, orientation)
    b.layout().setSpacing(spacing)
    b.layout().setContentsMargins(margin, margin, margin, margin)
    misc.setdefault('addSpace', bool(box))
    miscellanea(b, None, widget, **misc)
    return b


def hBox(*args, **kwargs):
    return widgetBox(orientation=Qt.Horizontal, *args, **kwargs)


def vBox(*args, **kwargs):
    return widgetBox(orientation=Qt.Vertical, *args, **kwargs)


def indentedBox(widget, sep=20, orientation=Qt.Vertical, **misc):
    """
    Creates an indented box. The function can also be used "on the fly"::

        gui.checkBox(gui.indentedBox(box), self, "spam", "Enable spam")

    To align the control with a check box, use :obj:`checkButtonOffsetHint`::

        gui.hSlider(gui.indentedBox(self.interBox), self, "intervals")

    :param widget: the widget into which the box is inserted
    :type widget: QWidget
    :param sep: Indent size (default: 20)
    :type sep: int
    :param orientation: orientation of the inserted box
    :type orientation: `Qt.Vertical` (default), `Qt.Horizontal` or
            instance of `QLayout`
    :return: Constructed box
    :rtype: QGroupBox or QWidget
    """
    outer = hBox(widget, spacing=0)
    separator(outer, sep, 0)
    indented = widgetBox(outer, orientation=orientation)
    miscellanea(indented, outer, widget, **misc)
    indented.box = outer
    return indented

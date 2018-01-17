from functools import partial

from AnyQt.QtWidgets import QPushButton, QToolButton, QApplication, QStyle

from Orange.widgets.utils import getdeepattr
from .base import miscellanea
from .callbacks import connect_control

__all__ = ["button", "toolButton", "toolButtonSizeHint"]


def button(widget, master, label, callback=None, width=None, height=None,
           toggleButton=False, value="", default=False, autoDefault=True,
           buttonType=QPushButton, **misc):
    """
    Insert a button (QPushButton, by default)

    :param widget: the widget into which the button is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param label: label
    :type label: str
    :param callback: a function that is called when the button is pressed
    :type callback: function
    :param width: the width of the button
    :type width: int
    :param height: the height of the button
    :type height: int
    :param toggleButton: if set to `True`, the button is checkable, but it is
        not synchronized with any attribute unless the `value` is given
    :type toggleButton: bool
    :param value: the master's attribute with which the value is synchronized
        (the argument is optional; if present, it makes the button "checkable",
        even if `toggleButton` is not set)
    :type value: str
    :param default: if `True` it makes the button the default button; this is
        the button that is activated when the user presses Enter unless some
        auto default button has current focus
    :type default: bool
    :param autoDefault: all buttons are auto default: they are activated if
        they have focus (or are the next in the focus chain) when the user
        presses enter. By setting `autoDefault` to `False`, the button is not
        activated on pressing Return.
    :type autoDefault: bool
    :param buttonType: the button type (default: `QPushButton`)
    :type buttonType: QPushButton
    :rtype: QPushButton
    """
    button = buttonType(widget)
    if label:
        button.setText(label)
    if width:
        button.setFixedWidth(width)
    if height:
        button.setFixedHeight(height)
    if toggleButton or value:
        button.setCheckable(True)
    if buttonType == QPushButton:
        button.setDefault(default)
        button.setAutoDefault(autoDefault)

    if value:
        button.setChecked(getdeepattr(master, value))
        connect_control(
            master, value, button,
            signal=button.toggled[bool],
            update_control=lambda val: val is not None and button.setChecked(bool(val)),
            callback=callback and partial(callback, **dict(widget=button))
        )
    elif callback:
        button.clicked.connect(callback)

    miscellanea(button, None, widget, **misc)
    return button


def toolButton(widget, master, label="", callback=None,
               width=None, height=None, tooltip=None):
    """
    Insert a tool button. Calls :obj:`button`

    :param widget: the widget into which the button is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param label: label
    :type label: str
    :param callback: a function that is called when the button is pressed
    :type callback: function
    :param width: the width of the button
    :type width: int
    :param height: the height of the button
    :type height: int
    :rtype: QToolButton
    """
    return button(widget, master, label, callback, width, height,
                  buttonType=QToolButton, tooltip=tooltip)


def toolButtonSizeHint(button=None, style=None):
    if button is None and style is None:
        style = QApplication.style()
    elif style is None:
        style = button.style()

    button_size = \
        style.pixelMetric(QStyle.PM_SmallIconSize) + \
        style.pixelMetric(QStyle.PM_ButtonMargin)
    return button_size

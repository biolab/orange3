from AnyQt.QtWidgets import QLineEdit
from AnyQt.QtCore import Qt

from Orange.widgets.utils import getdeepattr

from .base import miscellanea
from .boxes import widgetBox
from .labels import widgetLabel
from .callbacks import connect_control

__all__ = ["lineEdit"]

class LineEditWFocusOut(QLineEdit):
    """
    A class derived from QLineEdit, which postpones the synchronization
    of the control's value with the master's attribute until the user leaves
    the line edit or presses Tab when the value is changed.

    The class also allows specifying a callback function for focus-in event.

    .. attribute:: callback

        Callback that is called when the change is confirmed

    .. attribute:: focusInCallback

        Callback that is called on the focus-in event
    """

    def __init__(self, parent, callback, focusInCallback=None):
        super().__init__(parent)
        if parent.layout() is not None:
            parent.layout().addWidget(self)
        self.callback = callback
        self.focusInCallback = focusInCallback
        self.returnPressed.connect(self.returnPressedHandler)
        # did the text change between focus enter and leave
        self.__changed = False
        self.textEdited.connect(self.__textEdited)

    def __textEdited(self):
        self.__changed = True

    def returnPressedHandler(self):
        self.clearFocus()
        if self.__changed:
            self.__changed = False
            if hasattr(self, "cback") and self.cback:
                self.cback(self.text())
            if self.callback:
                self.callback()

    def setText(self, text):
        self.__changed = False
        super().setText(text)

    def focusOutEvent(self, *e):
        super().focusOutEvent(*e)
        self.returnPressedHandler()

    def focusInEvent(self, *e):
        self.__changed = False
        if self.focusInCallback:
            self.focusInCallback()
        return super().focusInEvent(*e)


def lineEdit(widget, master, value, label=None, labelWidth=None,
             orientation=Qt.Vertical, box=None, callback=None,
             valueType=None, validator=None, controlWidth=None,
             callbackOnType=False, focusInCallback=None, **misc):
    """
    Insert a line edit.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param label: label
    :type label: str
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param orientation: tells whether to put the label above or to the left
    :type orientation: `Qt.Vertical` (default) or `Qt.Horizontal`
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param callback: a function that is called when the check box state is
        changed
    :type callback: function
    :param valueType: the type into which the entered string is converted
        when synchronizing to `value`. If omitted, the type of the current
        `value` is used. If `value` is `None`, the text is left as a string.
    :type valueType: type or `None`
    :param validator: the validator for the input
    :type validator: QValidator
    :param controlWidth: the width of the line edit
    :type controlWidth: int
    :param callbackOnType: if set to `True`, the callback is called at each
        key press (default: `False`)
    :type callbackOnType: bool
    :param focusInCallback: a function that is called when the line edit
        receives focus
    :type focusInCallback: function
    :rtype: QLineEdit or a box
    """
    def update_value(val):
        if val is None:
            return  # Seems wrong, but kept for backward compatibility
        if valueType is not None:
            if valueType in (int, float) and (
                    not val or isinstance(val, str) and val in "+-"):
                val = 0
            val = valueType(val)
        setattr(master, value, val)

    if box or label:
        b = widgetBox(widget, box, orientation, addToLayout=False)
        if label is not None:
            widgetLabel(b, label, labelWidth)
    else:
        b = widget

    baseClass = misc.pop("baseClass", None)
    if baseClass:
        ledit = baseClass(b)
        if b is not widget:
            b.layout().addWidget(ledit)
    elif focusInCallback or callback and not callbackOnType:
        ledit = LineEditWFocusOut(b, callback, focusInCallback)
    else:
        ledit = QLineEdit(b)
        if b is not widget:
            b.layout().addWidget(ledit)

    if value:
        ledit.setText(str(getdeepattr(master, value)))
    if controlWidth:
        ledit.setFixedWidth(controlWidth)
    if validator:
        ledit.setValidator(validator)
    if value:
        if valueType is None:
            current_value = getdeepattr(master, value)
            valueType = str if current_value is None else type(current_value)
        _, ledit.cback, _ = connect_control(
            master, value, ledit,
            signal=ledit.textChanged[str],
            update_control=lambda val: ledit.setText(str(val)),
            update_value=update_value,
            callback=callbackOnType and callback)

    miscellanea(ledit, b, widget, **misc)
    return ledit

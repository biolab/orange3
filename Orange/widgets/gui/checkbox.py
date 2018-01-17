from functools import partial

from AnyQt.QtWidgets import (
    QCheckBox, QWidget, QStyleOptionButton, QStyle, QApplication)
from AnyQt.QtCore import Qt

from Orange.widgets.utils import getdeepattr

from .callbacks import connect_control
from .base import miscellanea
from .boxes import hBox

__all__ = ["checkBox", "checkButtonOffsetHint"]


def checkBox(widget, master, value, label, box=None,
             callback=None, id_=None, labelWidth=None,
             disables=None, **misc):
    """
    A simple checkbox.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param label: label
    :type label: str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param callback: a function that is called when the check box state is
        changed
    :type callback: function
    :param getwidget: If set `True`, the callback function will get a keyword
        argument `widget` referencing the check box
    :type getwidget: bool
    :param id_: If present, the callback function will get a keyword argument
        `id` with this value
    :type id_: any
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param disables: a list of widgets that are disabled if the check box is
        unchecked
    :type disables: list or QWidget or None
    :return: constructed check box; if is is placed within a box, the box is
        return in the attribute `box`
    :rtype: QCheckBox
    """
    if box:
        b = hBox(widget, box, addToLayout=False)
    else:
        b = widget
    cbox = QCheckBox(label, b)

    if labelWidth:
        cbox.setFixedSize(labelWidth, cbox.sizeHint().height())
    cbox.setChecked(getdeepattr(master, value))

    values = [Qt.Unchecked, Qt.Checked, Qt.PartiallyChecked]
    if callback and id_ is not None:
        callback = partial(callback, **dict(widget=cbox, id=id_))
    connect_control(
        master, value, cbox,
        signal=cbox.toggled[bool],
        update_control=lambda val: cbox.setCheckState(values[val]),
        callback=callback)
    if isinstance(disables, QWidget):
        disables = [disables]
    cbox.disables = disables or []
    cbox.makeConsistent = Disabler(cbox, master, value)
    cbox.toggled[bool].connect(cbox.makeConsistent)
    cbox.makeConsistent(value)
    miscellanea(cbox, b, widget, **misc)
    return cbox


def checkButtonOffsetHint(button, style=None):
    option = QStyleOptionButton()
    option.initFrom(button)
    if style is None:
        style = button.style()
    if isinstance(button, QCheckBox):
        pm_spacing = QStyle.PM_CheckBoxLabelSpacing
        pm_indicator_width = QStyle.PM_IndicatorWidth
    else:
        pm_spacing = QStyle.PM_RadioButtonLabelSpacing
        pm_indicator_width = QStyle.PM_ExclusiveIndicatorWidth
    space = style.pixelMetric(pm_spacing, option, button)
    width = style.pixelMetric(pm_indicator_width, option, button)
    # TODO: add other styles (Maybe load corrections from .cfg file?)
    style_correction = {"macintosh (aqua)": -2, "macintosh(aqua)": -2,
                        "plastique": 1, "cde": 1, "motif": 1}
    return space + width + \
        style_correction.get(QApplication.style().objectName().lower(), 0)


DISABLER = 1
HIDER = 2


# noinspection PyShadowingBuiltins
class Disabler:
    """
    A call-back class for check box that can disable/enable other widgets
    according to state (checked/unchecked, enabled/disable) of the check box.

    Note: if self.propagateState is True (default), then if check box is
    disabled the related widgets will be disabled (even if the checkbox is
    checked). If self.propagateState is False, the related widgets will be
    disabled/enabled if check box is checked/clear, disregarding whether the
    check box itself is enabled or not.
    """
    def __init__(self, widget, master, valueName, propagateState=True,
                 type=DISABLER):
        self.widget = widget
        self.master = master
        self.valueName = valueName
        self.propagateState = propagateState
        self.type = type

    def __call__(self, *value):
        def set_state(widget):
            if self.type == DISABLER:
                widget.setDisabled(disabled)
            elif self.type == HIDER:
                if disabled:
                    widget.hide()
                else:
                    widget.show()

        currState = self.widget.isEnabled()
        if currState or not self.propagateState:
            if len(value):
                disabled = not value[0]
            else:
                disabled = not getdeepattr(self.master, self.valueName)
        else:
            disabled = True
        for w in self.widget.disables:
            if isinstance(w, tuple):
                if isinstance(w[0], int):
                    i = 1
                    if w[0] == -1:
                        disabled = not disabled
                else:
                    i = 0
                set_state(w[i])
                if hasattr(w[i], "makeConsistent"):
                    w[i].makeConsistent()
            else:
                set_state(w)

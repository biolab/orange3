from AnyQt.QtWidgets import QRadioButton, QButtonGroup, QSizePolicy
from AnyQt.QtGui import QIcon
from AnyQt.QtCore import Qt

from Orange.widgets.utils import getdeepattr

from .base import miscellanea, _addSpace
from .boxes import widgetBox
from .labels import widgetLabel
from .callbacks import connect_control

__all__ = ["radioButtons", "radioButtonsInBox", "appendRadioButton"]


# btnLabels is a list of either char strings or pixmaps
def radioButtons(widget, master, value, btnLabels=(), tooltips=None,
                 box=None, label=None, orientation=Qt.Vertical,
                 callback=None, **misc):
    """
    Construct a button group and add radio buttons, if they are given.
    The value with which the buttons synchronize is the index of selected
    button.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param btnLabels: a list of labels or icons for radio buttons
    :type btnLabels: list of str or pixmaps
    :param tooltips: a list of tool tips of the same length as btnLabels
    :type tooltips: list of str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: a label that is inserted into the box
    :type label: str
    :param callback: a function that is called when the selection is changed
    :type callback: function
    :param orientation: orientation of the box
    :type orientation: `Qt.Vertical` (default), `Qt.Horizontal` or an
        instance of `QLayout`
    :rtype: QButtonGroup
    """
    bg = widgetBox(widget, box, orientation, addToLayout=False)
    if not label is None:
        widgetLabel(bg, label)

    rb = QButtonGroup(bg)
    if bg is not widget:
        bg.group = rb
    bg.buttons = []
    bg.master = master
    bg.value = value

    for i, lab in enumerate(btnLabels):
        appendRadioButton(bg, lab, tooltip=tooltips and tooltips[i], id=i)
    connect_control(
        master, value, bg,
        signal=bg.group.buttonClicked[int],
        update_control=lambda val: bg.buttons[val].setChecked(True),
        callback=callback)
    misc.setdefault('addSpace', bool(box))
    miscellanea(bg.group, bg, widget, **misc)
    return bg


radioButtonsInBox = radioButtons


def appendRadioButton(group, label, insertInto=None,
                      disabled=False, tooltip=None, sizePolicy=None,
                      addToLayout=True, stretch=0, addSpace=False, id=None):
    """
    Construct a radio button and add it to the group. The group must be
    constructed with :obj:`radioButtons` since it adds additional
    attributes need for the call backs.

    The radio button is inserted into `insertInto` or, if omitted, into the
    button group. This is useful for more complex groups, like those that have
    radio buttons in several groups, divided by labels and inside indented
    boxes.

    :param group: the button group
    :type group: QButtonGroup
    :param label: string label or a pixmap for the button
    :type label: str or QPixmap
    :param insertInto: the widget into which the radio button is inserted
    :type insertInto: QWidget
    :rtype: QRadioButton
    """
    i = len(group.buttons)
    if isinstance(label, str):
        w = QRadioButton(label)
    else:
        w = QRadioButton(str(i))
        w.setIcon(QIcon(label))
    if not hasattr(group, "buttons"):
        group.buttons = []
    group.buttons.append(w)
    group.group.addButton(w, id if id is not None else i)
    w.setChecked(getdeepattr(group.master, group.value) == i)

    # miscellanea for this case is weird, so we do it here
    if disabled:
        w.setDisabled(disabled)
    if tooltip is not None:
        w.setToolTip(tooltip)
    if sizePolicy:
        if isinstance(sizePolicy, tuple):
            sizePolicy = QSizePolicy(*sizePolicy)
        w.setSizePolicy(sizePolicy)
    if addToLayout:
        dest = insertInto or group
        dest.layout().addWidget(w, stretch)
        _addSpace(dest, addSpace)
    return w

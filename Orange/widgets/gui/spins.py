import math

from AnyQt import QtWidgets
from AnyQt.QtCore import Qt

from Orange.widgets.utils import getdeepattr

from .base import miscellanea, is_horizontal
from .labels import widgetLabel
from .boxes import widgetBox, hBox
from .checkbox import checkBox
from .callbacks import connect_control

__all__ = ["spin", "doubleSpin"]


class SpinBoxWFocusOut(QtWidgets.QSpinBox):
    """
    A class derived from QSpinBox, which postpones the synchronization
    of the control's value with the master's attribute until the control looses
    focus or user presses Tab when the value has changed.

    The class overloads :obj:`onChange` event handler to show the commit button,
    and :obj:`onEnter` to commit the change when enter is pressed.
    """

    def __init__(self, minv, maxv, step, parent=None):
        """
        Construct the object and set the range (`minv`, `maxv`) and the step.
        :param minv: Minimal value
        :type minv: int
        :param maxv: Maximal value
        :type maxv: int
        :param step: Step
        :type step: int
        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self.setRange(minv, maxv)
        self.setSingleStep(step)
        self.changed = False

    def onValueChanged(self):
        """
        Sets the flag to determine whether the value has been changed.
        """
        self.changed = True

    def onEnter(self):
        """
        Commits the change by calling the appropriate callbacks.
        """
        if self.cback and self.changed:
            self.cback(int(str(self.text())))
        if self.cfunc and self.changed:
            self.cfunc()
        self.changed = False


class DoubleSpinBoxWFocusOut(QtWidgets.QDoubleSpinBox):
    """
    Same as :obj:`SpinBoxWFocusOut`, except that it is derived from
    :obj:`~QDoubleSpinBox`"""
    def __init__(self, minv, maxv, step, parent):
        super().__init__(parent)
        self.setDecimals(math.ceil(-math.log10(step)))
        self.setRange(minv, maxv)
        self.setSingleStep(step)
        self.changed = False

    def onValueChanged(self):
        """
        Sets the flag to determine whether the value has been changed.
        """
        self.changed = True

    def onEnter(self):
        if self.cback and self.changed:
            self.cback(float(str(self.text()).replace(",", ".")))
        if self.cfunc and self.changed:
            self.cfunc()
        self.changed = False


def spin(widget, master, value, minv, maxv, step=1, box=None, label=None,
         labelWidth=None, orientation=Qt.Horizontal, callback=None,
         controlWidth=None, callbackOnReturn=False, checked=None,
         checkCallback=None, posttext=None, disabled=False,
         alignment=Qt.AlignLeft, keyboardTracking=True,
         decimals=None, spinType=int, **misc):
    """
    A spinbox with lots of bells and whistles, such as a checkbox and various
    callbacks. It constructs a control of type :obj:`SpinBoxWFocusOut` or
    :obj:`DoubleSpinBoxWFocusOut`.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param minv: minimal value
    :type minv: int or float
    :param maxv: maximal value
    :type maxv: int or float
    :param step: step (default: 1)
    :type step: int or float
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: label that is put in above or to the left of the spin box
    :type label: str
    :param labelWidth: optional label width (default: None)
    :type labelWidth: int
    :param orientation: tells whether to put the label above or to the left
    :type orientation: `Qt.Horizontal` (default), `Qt.Vertical` or
        instance of `QLayout`
    :param callback: a function that is called when the value is entered; if
        :obj:`callbackOnReturn` is `True`, the function is called when the
        user commits the value by pressing Enter or clicking the icon
    :type callback: function
    :param controlWidth: the width of the spin box
    :type controlWidth: int
    :param callbackOnReturn: if `True`, the spin box has an associated icon
        that must be clicked to confirm the value (default: False)
    :type callbackOnReturn: bool
    :param checked: if not None, a check box is put in front of the spin box;
        when unchecked, the spin box is disabled. Argument `checked` gives the
        name of the master's attribute given whose value is synchronized with
        the check box's state (default: None).
    :type checked: str
    :param checkCallback: a callback function that is called when the check
        box's state is changed
    :type checkCallback: function
    :param posttext: a text that is put to the right of the spin box
    :type posttext: str
    :param alignment: alignment of the spin box (e.g. `Qt.AlignLeft`)
    :type alignment: Qt.Alignment
    :param keyboardTracking: If `True`, the valueChanged signal is emitted
        when the user is typing (default: True)
    :type keyboardTracking: bool
    :param spinType: determines whether to use QSpinBox (int) or
        QDoubleSpinBox (float)
    :type spinType: type
    :param decimals: number of decimals (if `spinType` is `float`)
    :type decimals: int
    :return: Tuple `(spin box, check box) if `checked` is `True`, otherwise
        the spin box
    :rtype: tuple or gui.SpinBoxWFocusOut
    """

    # b is the outermost box or the widget if there are no boxes;
    #    b is the widget that is inserted into the layout
    # bi is the box that contains the control or the checkbox and the control;
    #    bi can be the widget itself, if there are no boxes
    # cbox is the checkbox (or None)
    # sbox is the spinbox itself
    if box or label and not checked:
        b = widgetBox(widget, box, orientation, addToLayout=False)
        hasHBox = is_horizontal(orientation)
    else:
        b = widget
        hasHBox = False
    if not hasHBox and (checked or callback and callbackOnReturn or posttext):
        bi = hBox(b, addToLayout=False)
    else:
        bi = b

    cbox = None
    if checked is not None:
        cbox = checkBox(bi, master, checked, label, labelWidth=labelWidth,
                        callback=checkCallback)
    elif label:
        b.label = widgetLabel(b, label, labelWidth)
    if posttext:
        widgetLabel(bi, posttext)

    isDouble = spinType == float
    sbox = bi.control = b.control = \
        (SpinBoxWFocusOut, DoubleSpinBoxWFocusOut)[isDouble](minv, maxv,
                                                             step, bi)
    if bi is not widget:
        bi.setDisabled(disabled)
    else:
        sbox.setDisabled(disabled)

    if decimals is not None:
        sbox.setDecimals(decimals)
    sbox.setAlignment(alignment)
    sbox.setKeyboardTracking(keyboardTracking)
    if controlWidth:
        sbox.setFixedWidth(controlWidth)
    if value:
        sbox.setValue(getdeepattr(master, value))

    if not (callback and callbackOnReturn):
        signal = sbox.valueChanged[(int, float)[isDouble]]
    else:
        signal = None
    _, sbox.cback, sbox.cfunc = connect_control(
        master, value, sbox,
        signal=signal,
        update_control=lambda val: val is not None and sbox.setValue(val),
        callback=callback)
    if checked:
        sbox.cbox = cbox
        cbox.disables = [sbox]
        cbox.makeConsistent()
    if callback and callbackOnReturn:
        sbox.valueChanged.connect(sbox.onValueChanged)
        sbox.editingFinished.connect(sbox.onEnter)
        if hasattr(sbox, "upButton"):
            sbox.upButton().clicked.connect(
                lambda c=sbox.editor(): c.setFocus())
            sbox.downButton().clicked.connect(
                lambda c=sbox.editor(): c.setFocus())

    miscellanea(sbox, b if b is not widget else bi, widget, **misc)
    if checked:
        if isDouble and b == widget:
            # TODO Backward compatilibity; try to find and eliminate
            sbox.control = b.control
            return sbox
        return cbox, sbox
    else:
        return sbox


# noinspection PyTypeChecker
def doubleSpin(widget, master, value, minv, maxv, step=1, box=None, label=None,
               labelWidth=None, orientation=Qt.Horizontal, callback=None,
               controlWidth=None, callbackOnReturn=False, checked=None,
               checkCallback=None, posttext=None,
               alignment=Qt.AlignLeft, keyboardTracking=True,
               decimals=None, **misc):
    """
    Backward compatilibity function: calls :obj:`spin` with `spinType=float`.
    """
    return spin(widget, master, value, minv, maxv, step, box=box, label=label,
                labelWidth=labelWidth, orientation=orientation,
                callback=callback, controlWidth=controlWidth,
                callbackOnReturn=callbackOnReturn, checked=checked,
                checkCallback=checkCallback, posttext=posttext,
                alignment=alignment, keyboardTracking=keyboardTracking,
                decimals=decimals, spinType=float, **misc)

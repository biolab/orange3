import math
import os
import re
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, pyqtSignal as Signal
from Orange.widgets.utils import getdeepattr
from Orange.data import ContinuousVariable, StringVariable, DiscreteVariable, Variable
from Orange.widgets.utils import vartype
from Orange.widgets.utils.constants import CONTROLLED_ATTRIBUTES, ATTRIBUTE_CONTROLLERS

YesNo = NoYes = ("No", "Yes")
_enter_icon = None
__re_label = re.compile(r"(^|[^%])%\((?P<value>[a-zA-Z]\w*)\)")


def id_generator(id_):
    while True:
        id_ += 1
        yield id_

OrangeUserRole = id_generator(Qt.UserRole)


class ControlledAttributesDict(dict):
    def __init__(self, master):
        super().__init__()
        self.master = master

    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, [value])
        else:
            dict.__getitem__(self, key).append(value)
        set_controllers(self.master, key, self.master, "")

callbacks = lambda obj: getattr(obj, CONTROLLED_ATTRIBUTES, {})
subcontrollers = lambda obj: getattr(obj, ATTRIBUTE_CONTROLLERS, {})


def notify_changed(obj, name, value):
    if name in callbacks(obj):
        for callback in callbacks(obj)[name]:
            callback(value)
        return

    for controller, prefix in list(subcontrollers(obj)):
        if getdeepattr(controller, prefix, None) != obj:
            del subcontrollers(obj)[(controller, prefix)]
            continue

        full_name = prefix + "." + name
        if full_name in callbacks(controller):
            for callback in callbacks(controller)[full_name]:
                callback(value)
            continue

        prefix = full_name + "."
        prefix_length = len(prefix)
        for controlled in callbacks(controller):
            if controlled[:prefix_length] == prefix:
                set_controllers(value, controlled[prefix_length:], controller, full_name)


def set_controllers(obj, controlled_name, controller, prefix):
    while obj:
        if prefix:
            if hasattr(obj, ATTRIBUTE_CONTROLLERS):
                getattr(obj, ATTRIBUTE_CONTROLLERS)[(controller, prefix)] = True
            else:
                setattr(obj, ATTRIBUTE_CONTROLLERS, {(controller, prefix): True})
        parts = controlled_name.split(".", 1)
        if len(parts) < 2:
            break
        new_prefix, controlled_name = parts
        obj = getattr(obj, new_prefix, None)
        if prefix:
            prefix += '.'
        prefix += new_prefix


class OWComponent:
    def __init__(self, widget):
        setattr(self, CONTROLLED_ATTRIBUTES, ControlledAttributesDict(self))

        if widget.settingsHandler:
            widget.settingsHandler.initialize(self)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        notify_changed(self, key, value)


def miscellanea(control, box, parent,
                addToLayout=True, stretch=0, sizePolicy=None, addSpace=False,
                disabled=False, tooltip=None):
    """
    Helper function that sets various properties of the widget using a common
    set of arguments.

    The function
    - sets the `control`'s attribute `box`, if `box` is given and `control.box`
    is not yet set,
    - attaches a tool tip to the `control` if specified,
    - disables the `control`, if `disabled` is set to `True`,
    - adds the `box` to the `parent`'s layout unless `addToLayout` is set to
    `False`; the stretch factor can be specified,
    - adds the control into the box's layout if the box is given (regardless
    of `addToLayout`!)
    - sets the size policy for the box or the control, if the policy is given,
    - adds space in the `parent`'s layout after the `box` if `addSpace` is set
    and `addToLayout` is not `False`.

    If `box` is the same as `parent` it is set to `None`; this is convenient
    because of the way complex controls are inserted.

    :param control: the control, e.g. a `QCheckBox`
    :type control: PyQt4.QtGui.QWidget
    :param box: the box into which the widget was inserted
    :type box: PyQt4.QtGui.QWidget or None
    :param parent: the parent into whose layout the box or the control will be
        inserted
    :type parent: PyQt4.QtGui.QWidget
    :param addSpace: the amount of space to add after the widget
    :type addSpace: bool or int
    :param disabled: If set to `True`, the widget is initially disabled
    :type disabled: bool
    :param addToLayout: If set to `False` the widget is not added to the layout
    :type addToLayout: bool
    :param stretch: the stretch factor for this widget, used when adding to
        the layout (default: 0)
    :type stretch: int
    :param tooltip: tooltip that is attached to the widget
    :type tooltip: str or None
    :param sizePolicy: the size policy for the box or the control
    :type sizePolicy: PyQt4.QtQui.QSizePolicy
    """
    if disabled:
        # if disabled==False, do nothing; it can be already disabled
        control.setDisabled(disabled)
    if tooltip is not None:
        control.setToolTip(tooltip)
    if box is parent:
        box = None
    elif box and box is not control and not hasattr(control, "box"):
        control.box = box
    if box and box.layout() is not None and \
            isinstance(control, QtGui.QWidget) and \
            box.layout().indexOf(control) == -1:
        box.layout().addWidget(control)
    if sizePolicy is not None:
        (box or control).setSizePolicy(sizePolicy)
    if addToLayout and parent and parent.layout() is not None:
        parent.layout().addWidget(box or control, stretch)
        _addSpace(parent, addSpace)


def setLayout(widget, orientation):
    """
    Set the layout of the widget according to orientation. Argument
    `orientation` can be an instance of :obj:`~PyQt4.QtGui.QLayout`, in which
    case is it used as it is. If `orientation` is `'vertical'` or `True`,
    the layout is set to :obj:`~PyQt4.QtGui.QVBoxLayout`. If it is
    `'horizontal'` or `False`, it is set to :obj:`~PyQt4.QtGui.QVBoxLayout`.

    :param widget: the widget for which the layout is being set
    :type widget: PyQt4.QtGui.QWidget
    :param orientation: orientation for the layout
    :type orientation: str or bool or PyQt4.QtGui.QLayout
    """
    if isinstance(orientation, QtGui.QLayout):
        widget.setLayout(orientation)
    elif orientation == 'horizontal' or not orientation:
        widget.setLayout(QtGui.QHBoxLayout())
    else:
        widget.setLayout(QtGui.QVBoxLayout())


def _enterButton(parent, control, placeholder=True):
    """
    Utility function that returns a button with a symbol for "Enter" and
    optionally a placeholder to show when the enter button is hidden. Both
    are inserted into the parent's layout, if it has one. If placeholder is
    constructed it is shown and the button is hidden.

    The height of the button is the same as the height of the widget passed
    as argument `control`.

    :param parent: parent widget into which the button is inserted
    :type parent: PyQt4.QtGui.QWidget
    :param control: a widget for determining the height of the button
    :type control: PyQt4.QtGui.QWidget
    :param placeholder: a flag telling whether to construct a placeholder
        (default: True)
    :type placeholder: bool
    :return: a tuple with a button and a place holder (or `None`)
    :rtype: PyQt4.QtGui.QToolButton or tuple
    """
    global _enter_icon
    if not _enter_icon:
        _enter_icon = QtGui.QIcon(
            os.path.dirname(__file__) + "/icons/Dlg_enter.png")
    button = QtGui.QToolButton(parent)
    height = control.sizeHint().height()
    button.setFixedSize(height, height)
    button.setIcon(_enter_icon)
    if parent.layout() is not None:
        parent.layout().addWidget(button)
    if placeholder:
        button.hide()
        holder = QtGui.QWidget(parent)
        holder.setFixedSize(height, height)
        if parent.layout() is not None:
            parent.layout().addWidget(holder)
    else:
        holder = None
    return button, holder


def _addSpace(widget, space):
    """
    A helper function that adds space into the widget, if requested.
    The function is called by functions that have the `addSpace` argument.

    :param widget: Widget into which to insert the space
    :type widget: PyQt4.QtGui.QWidget
    :param space: Amount of space to insert. If False, the function does
        nothing. If the argument is an `int`, the specified space is inserted.
        Otherwise, the default space is inserted by calling a :obj:`separator`.
    :type space: bool or int
    """
    if space:
        if type(space) == int:  # distinguish between int and bool!
            separator(widget, space, space)
        else:
            separator(widget)


def separator(widget, width=4, height=4):
    """
    Add a separator of the given size into the widget.

    :param widget: the widget into whose layout the separator is added
    :type widget: PyQt4.QtGui.QWidget
    :param width: width of the separator
    :type width: int
    :param height: height of the separator
    :type height: int
    :return: separator
    :rtype: PyQt4.QtGui.QWidget
    """
    sep = QtGui.QWidget(widget)
    if widget.layout() is not None:
        widget.layout().addWidget(sep)
    sep.setFixedSize(width, height)
    return sep


def rubber(widget):
    """
    Insert a stretch 100 into the widget's layout
    """
    widget.layout().addStretch(100)


def widgetBox(widget, box=None, orientation='vertical', margin=None, spacing=4,
              **misc):
    """
    Construct a box with vertical or horizontal layout, and optionally,
    a border with an optional label.

    If the widget has a frame, the space after the widget is added unless
    explicitly disabled.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param orientation: orientation for the layout. If the argument is an
        instance of :obj:`~PyQt4.QtGui.QLayout`, it is used as a layout. If
        "horizontal" or false-ish, the layout is horizontal
        (:obj:`~PyQt4.QtGui.QHBoxLayout`), otherwise vertical
        (:obj:`~PyQt4.QtGui.QHBoxLayout`).
    :type orientation: str, int or :obj:`PyQt4.QtGui.QLayout`
    :param sizePolicy: The size policy for the widget (default: None)
    :type sizePolicy: :obj:`~PyQt4.QtGui.QSizePolicy`
    :param margin: The margin for the layout. Default is 7 if the widget has
        a border, and 0 if not.
    :type margin: int
    :param spacing: Spacing within the layout (default: 4)
    :type spacing: int
    :return: Constructed box
    :rtype: PyQt4.QtGui.QGroupBox or PyQt4.QtGui.QWidget
    """
    if box:
        b = QtGui.QGroupBox(widget)
        if isinstance(box, str):
            b.setTitle(" " + box.strip() + " ")
        if margin is None:
            margin = 7
    else:
        b = QtGui.QWidget(widget)
        b.setContentsMargins(0, 0, 0, 0)
        if margin is None:
            margin = 0
    setLayout(b, orientation)
    b.layout().setSpacing(spacing)
    b.layout().setMargin(margin)
    if not "addSpace" in misc:
        misc["addSpace"] = bool(box)
    miscellanea(b, None, widget, **misc)
    return b


def indentedBox(widget, sep=20, orientation="vertical", **misc):
    """
    Creates an indented box. The function can also be used "on the fly"::

        gui.checkBox(gui.indentedBox(box), self, "spam", "Enable spam")

    To align the control with a check box, use :obj:`checkButtonOffsetHint`::

        gui.hSlider(gui.indentedBox(self.interBox), self, "intervals")

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param sep: Indent size (default: 20)
    :type sep: int
    :param orientation: layout of the inserted box; see :obj:`widgetBox` for
        details
    :type orientation: str, int or PyQt4.QtGui.QLayout
    :return: Constructed box
    :rtype: PyQt4.QtGui.QGroupBox or PyQt4.QtGui.QWidget
    """
    outer = widgetBox(widget, orientation=False, spacing=0)
    separator(outer, sep, 0)
    indented = widgetBox(outer, orientation=orientation)
    miscellanea(indented, outer, widget, **misc)
    return indented


def widgetLabel(widget, label="", labelWidth=None, **misc):
    """
    Construct a simple, constant label.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param label: The text of the label (default: None)
    :type label: str
    :param labelWidth: The width of the label (default: None)
    :type labelWidth: int
    :return: Constructed label
    :rtype: PyQt4.QtGui.QLabel
    """
    lbl = QtGui.QLabel(label, widget)
    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    miscellanea(lbl, None, widget, **misc)
    return lbl



def label(widget, master, label, labelWidth=None, box=None,
          orientation="vertical", *misc):
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
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param label: The text of the label, including attribute names
    :type label: str
    :param labelWidth: The width of the label (default: None)
    :type labelWidth: int
    :return: label
    :rtype: PyQt4.QtGui.QLabel
    """
    if box:
        b = widgetBox(widget, box, orientation=None, addToLayout=False)
    else:
        b = widget

    lbl = QtGui.QLabel("", b)
    reprint = CallFrontLabel(lbl, label, master)
    for mo in __re_label.finditer(label):
        getattr(master, CONTROLLED_ATTRIBUTES)[mo.group("value")] = reprint
    reprint()
    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    miscellanea(lbl, b, widget, *misc)
    return lbl


class SpinBoxWFocusOut(QtGui.QSpinBox):
    """
    A class derived from QtGui.QSpinBox, which postpones the synchronization
    of the control's value with the master's attribute until the user presses
    Enter or clicks an icon that appears beside the spin box when the value
    is changed.

    The class overloads :obj:`onChange` event handler to show the commit button,
    and :obj:`onEnter` to commit the change when enter is pressed.

    .. attribute:: enterButton

        A widget (usually an icon) that is shown when the value is changed.

    .. attribute:: placeHolder

        A placeholder which is shown when the button is hidden

    .. attribute:: inSetValue

        A flag that is set when the value is being changed through
        :obj:`setValue` to prevent the programmatic changes from showing the
        commit button.
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
        :type parent: PyQt4.QtGui.QWidget
        """
        super().__init__(parent)
        self.setRange(minv, maxv)
        self.setSingleStep(step)
        self.inSetValue = False
        self.enterButton = None
        self.placeHolder = None

    def onChange(self, _):
        """
        Hides the place holder and shows the commit button unless
        :obj:`inSetValue` is set.
        """
        if not self.inSetValue:
            self.placeHolder.hide()
            self.enterButton.show()

    def onEnter(self):
        """
        If the commit button is visible, the overload event handler commits
        the change by calling the appropriate callbacks. It also hides the
        commit button and shows the placeHolder.
        """
        if self.enterButton.isVisible():
            self.enterButton.hide()
            self.placeHolder.show()
            if self.cback:
                self.cback(int(str(self.text())))
            if self.cfunc:
                self.cfunc()

    # doesn't work: it's probably LineEdit's focusOut that we should
    # (but can't) catch
    def focusOutEvent(self, *e):
        """
        This handler was intended to catch the focus out event and reintepret
        it as if enter was pressed. It does not work, though.
        """
        super().focusOutEvent(*e)
        if self.enterButton and self.enterButton.isVisible():
            self.onEnter()

    def setValue(self, value):
        """
        Set the :obj:`inSetValue` flag and call the inherited method.
        """
        self.inSetValue = True
        super().setValue(value)
        self.inSetValue = False


class DoubleSpinBoxWFocusOut(QtGui.QDoubleSpinBox):
    """
    Same as :obj:`SpinBoxWFocusOut`, except that it is derived from
    :obj:`~PyQt4.QtGui.QDoubleSpinBox`"""
    def __init__(self, minv, maxv, step, parent):
        super().__init__(parent)
        self.setDecimals(math.ceil(-math.log10(step)))
        self.setRange(minv, maxv)
        self.setSingleStep(step)
        self.inSetValue = False
        self.enterButton = None
        self.placeHolder = None

    def onChange(self, _):
        if not self.inSetValue:
            self.placeHolder.hide()
            self.enterButton.show()

    def onEnter(self):
        if self.enterButton.isVisible():
            self.enterButton.hide()
            self.placeHolder.show()
            if self.cback:
                self.cback(float(str(self.text()).replace(",", ".")))
            if self.cfunc:
                self.cfunc()

    # doesn't work: it's probably LineEdit's focusOut that we should
    # (and can't) catch
    def focusOutEvent(self, *e):
        super().focusOutEvent(*e)
        if self.enterButton and self.enterButton.isVisible():
            self.onEnter()

    def setValue(self, value):
        self.inSetValue = True
        super().setValue(value)
        self.inSetValue = False


def spin(widget, master, value, minv, maxv, step=1, box=None, label=None,
         labelWidth=None, orientation=None, callback=None,
         controlWidth=None, callbackOnReturn=False, checked=None,
         checkCallback=None, posttext=None, disabled=False,
         alignment=Qt.AlignLeft, keyboardTracking=True,
         decimals=None, spinType=int, **misc):
    """
    A spinbox with lots of bells and whistles, such as a checkbox and various
    callbacks. It constructs a control of type :obj:`SpinBoxWFocusOut` or
    :obj:`DoubleSpinBoxWFocusOut`.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param minv: minimal value
    :type minv: int
    :param maxv: maximal value
    :type maxv: int
    :param step: step (default: 1)
    :type step: int
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: label that is put in above or to the left of the spin box
    :type label: str
    :param labelWidth: optional label width (default: None)
    :type labelWidth: int
    :param orientation: tells whether to put the label above (`"vertical"` or
        `True`) or to the left (`"horizontal"` or `False`)
    :type orientation: int or bool
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
    :param alignment: alignment of the spin box (e.g. `QtCore.Qt.AlignLeft`)
    :type alignment: PyQt4.QtCore.Qt.Alignment
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
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False
    if not hasHBox and (checked or callback and callbackOnReturn or posttext):
        bi = widgetBox(b, orientation=0, addToLayout=False)
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
    sbox = bi.control = \
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

    cfront, sbox.cback, sbox.cfunc = connectControl(
        sbox, master, value, callback,
        not (callback and callbackOnReturn) and
        ("valueChanged(int)", "valueChanged(double)")[isDouble],
        (CallFrontSpin, CallFrontDoubleSpin)[isDouble](sbox))
    if checked:
        cbox.disables = [sbox]
        cbox.makeConsistent()
    if callback and callbackOnReturn:
        sbox.enterButton, sbox.placeHolder = _enterButton(bi, sbox)
        sbox.valueChanged[str].connect(sbox.onChange)
        sbox.editingFinished.connect(sbox.onEnter)
        sbox.enterButton.clicked.connect(sbox.onEnter)
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
               labelWidth=None, orientation=None, callback=None,
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


def checkBox(widget, master, value, label, box=None,
             callback=None, getwidget=False, id_=None, labelWidth=None,
             disables=None, **misc):
    """
    A simple checkbox.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
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
    :type disables: list or PyQt4.QtGui.QWidget or None
    :return: constructed check box; if is is placed within a box, the box is
        return in the attribute `box`
    :rtype: PyQt4.QtGui.QCheckBox
    """
    if box:
        b = widgetBox(widget, box, orientation=None, addToLayout=False)
    else:
        b = widget
    cbox = QtGui.QCheckBox(label, b)

    if labelWidth:
        cbox.setFixedSize(labelWidth, cbox.sizeHint().height())
    cbox.setChecked(getdeepattr(master, value))

    connectControl(cbox, master, value, None, "toggled(bool)",
                   CallFrontCheckBox(cbox),
                   cfunc=callback and FunctionCallback(
                       master, callback, widget=cbox, getwidget=getwidget,
                       id=id_))
    if isinstance(disables, QtGui.QWidget):
        disables = [disables]
    cbox.disables = disables or []
    cbox.makeConsistent = Disabler(cbox, master, value)
    cbox.toggled[bool].connect(cbox.makeConsistent)
    cbox.makeConsistent(value)
    miscellanea(cbox, b, widget, **misc)
    return cbox


class LineEditWFocusOut(QtGui.QLineEdit):
    """
    A class derived from QtGui.QLineEdit, which postpones the synchronization
    of the control's value with the master's attribute until the user leaves
    the line edit, presses Enter or clicks an icon that appears beside the
    line edit when the value is changed.

    The class also allows specifying a callback function for focus-in event.

    .. attribute:: enterButton

        A widget (usually an icon) that is shown when the value is changed.

    .. attribute:: placeHolder

        A placeholder which is shown when the button is hidden

    .. attribute:: inSetValue

        A flag that is set when the value is being changed through
        :obj:`setValue` to prevent the programmatic changes from showing the
        commit button.

    .. attribute:: callback

        Callback that is called when the change is confirmed

    .. attribute:: focusInCallback

        Callback that is called on the focus-in event
    """

    def __init__(self, parent, callback, focusInCallback=None,
                 placeholder=False):
        super().__init__(parent)
        if parent.layout() is not None:
            parent.layout().addWidget(self)
        self.callback = callback
        self.focusInCallback = focusInCallback
        self.enterButton, self.placeHolder = \
            _enterButton(parent, self, placeholder)
        self.enterButton.clicked.connect(self.returnPressedHandler)
        self.textChanged[str].connect(self.markChanged)
        self.returnPressed.connect(self.returnPressedHandler)

    def markChanged(self, *_):
        if self.placeHolder:
            self.placeHolder.hide()
        self.enterButton.show()

    def markUnchanged(self, *_):
        self.enterButton.hide()
        if self.placeHolder:
            self.placeHolder.show()

    def returnPressedHandler(self):
        if self.enterButton.isVisible():
            self.markUnchanged()
            if hasattr(self, "cback") and self.cback:
                self.cback(self.text())
            if self.callback:
                self.callback()

    def setText(self, t):
        super().setText(t)
        if self.enterButton:
            self.markUnchanged()

    def focusOutEvent(self, *e):
        super().focusOutEvent(*e)
        self.returnPressedHandler()

    def focusInEvent(self, *e):
        if self.focusInCallback:
            self.focusInCallback()
        return super().focusInEvent(*e)


def lineEdit(widget, master, value, label=None, labelWidth=None,
             orientation='vertical', box=None, callback=None,
             valueType=str, validator=None, controlWidth=None,
             callbackOnType=False, focusInCallback=None,
             enterPlaceholder=False, **misc):
    """
    Insert a line edit.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param label: label
    :type label: str
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param orientation: tells whether to put the label above (`"vertical"` or
        `True`) or to the left (`"horizontal"` or `False`)
    :type orientation: int or bool
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param callback: a function that is called when the check box state is
        changed
    :type callback: function
    :param valueType: the type into which the entered string is converted
        when synchronizing to `value`
    :type valueType: type
    :param validator: the validator for the input
    :type validator: PyQt4.QtGui.QValidator
    :param controlWidth: the width of the line edit
    :type controlWidth: int
    :param callbackOnType: if set to `True`, the callback is called at each
        key press (default: `False`)
    :type callbackOnType: bool
    :param focusInCallback: a function that is called when the line edit
        receives focus
    :type focusInCallback: function
    :param enterPlaceholder: if set to `True`, space of appropriate width is
        left empty to the right for the icon that shows that the value is
        changed but has not been committed yet
    :type enterPlaceholder: bool
    :rtype: PyQt4.QtGui.QLineEdit or a box
    """
    if box or label:
        b = widgetBox(widget, box, orientation, addToLayout=False)
        widgetLabel(b, label, labelWidth)
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False

    baseClass = misc.pop("baseClass", None)
    if baseClass:
        ledit = baseClass(b)
        ledit.enterButton = None
        if b is not widget:
            b.layout().addWidget(ledit)
    elif focusInCallback or callback and not callbackOnType:
        if not hasHBox:
            outer = widgetBox(b, "", 0, addToLayout=(b is not widget))
        else:
            outer = b
        ledit = LineEditWFocusOut(outer, callback, focusInCallback,
                                  enterPlaceholder)
    else:
        ledit = QtGui.QLineEdit(b)
        ledit.enterButton = None
        if b is not widget:
            b.layout().addWidget(ledit)

    if value:
        ledit.setText(str(getdeepattr(master, value)))
    if controlWidth:
        ledit.setFixedWidth(controlWidth)
    if validator:
        ledit.setValidator(validator)
    if value:
        ledit.cback = connectControl(
            ledit, master, value,
            callbackOnType and callback, "textChanged(const QString &)",
            CallFrontLineEdit(ledit), fvcb=value and valueType)[1]

    miscellanea(ledit, b, widget, **misc)
    return ledit


def button(widget, master, label, callback=None, width=None, height=None,
           toggleButton=False, value="", default=False, autoDefault=True,
           buttonType=QtGui.QPushButton, **misc):
    """
    Insert a button (QPushButton, by default)

    :param widget: the widget into which the button is inserted
    :type widget: PyQt4.QtGui.QWidget
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
    :type buttonType: PyQt4.QtGui.QAbstractButton
    :rtype: PyQt4.QtGui.QAbstractButton
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
    if buttonType == QtGui.QPushButton:
        button.setDefault(default)
        button.setAutoDefault(autoDefault)

    if value:
        button.setChecked(getdeepattr(master, value))
        connectControl(
            button, master, value, None, "toggled(bool)",
            CallFrontButton(button),
            cfunc=callback and FunctionCallback(master, callback,
                                                widget=button))
    elif callback:
        button.clicked.connect(callback)

    miscellanea(button, None, widget, **misc)
    return button


def toolButton(widget, master, label="", callback=None,
               width=None, height=None, tooltip=None):
    """
    Insert a tool button. Calls :obj:`button`

    :param widget: the widget into which the button is inserted
    :type widget: PyQt4.QtGui.QWidget
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
    :rtype: PyQt4.QtGui.QToolButton
    """
    return button(widget, master, label, callback, width, height,
                  buttonType=QtGui.QToolButton, tooltip=tooltip)


def createAttributePixmap(char, background=Qt.black, color=Qt.white):
    """
    Create a QIcon with a given character. The icon is 13 pixels high and wide.

    :param char: The character that is printed in the icon
    :type char: str
    :param background: the background color (default: black)
    :type background: PyQt4.QtGui.QColor
    :param color: the character color (default: white)
    :type color: PyQt4.QtGui.QColor
    :rtype: PyQt4.QtGui.QIcon
    """
    pixmap = QtGui.QPixmap(13, 13)
    pixmap.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter()
    painter.begin(pixmap)
    painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing |
                           painter.SmoothPixmapTransform)
    painter.setPen(background)
    painter.setBrush(background)
    rect = QtCore.QRectF(0, 0, 13, 13)
    painter.drawRoundedRect(rect, 4, 4)
    painter.setPen(color)
    painter.drawText(2, 11, char)
    painter.end()
    return QtGui.QIcon(pixmap)


class __AttributeIconDict(dict):
    def __getitem__(self, key):
        if not self:
            for tpe, char, col in ((vartype(ContinuousVariable()), 
                                        "C", (202, 0, 32)),
                                  (vartype(DiscreteVariable()), 
                                        "D", (26, 150, 65)),
                                  (vartype(StringVariable()), 
                                        "S", (0, 0, 0)),
                                  (-1, "?", (128, 128, 128))):
                self[tpe] = createAttributePixmap(char, QtGui.QColor(*col))
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
    :rtype: tuple with PyQt4.QtGui.QIcon and str
    """
    return attributeIconDict[var], var.name


def listBox(widget, master, value=None, labels=None, box=None, callback=None,
            selectionMode=QtGui.QListWidget.SingleSelection,
            enableDragDrop=False, dragDropCallback=None,
            dataValidityCallback=None, sizeHint=None, **misc):
    """
    Insert a list box.

    The value with which the box's value synchronizes (`master.<value>`)
    is a list of indices of selected items.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the name of the master's attribute with which the value is
        synchronized (list of ints - indices of selected items)
    :type value: str
    :param labels: the name of the master's attribute with the list of items
        (as strings or tuples with icon and string)
    :type labels: str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param callback: a function that is called when the selection state is
        changed
    :type callback: function
    :param selectionMode: selection mode - single, multiple etc
    :type selectionMode: PyQt4.QtGui.QAbstractItemView.SelectionMode
    :param enableDragDrop: flag telling whether drag and drop is available
    :type enableDragDrop: bool
    :param dragDropCallback: callback function on drop event
    :type dragDropCallback: function
    :param dataValidityCallback: function that check the validity on enter
        and move event; it should return either `ev.accept()` or `ev.ignore()`.
    :type dataValidityCallback: function
    :param sizeHint: size hint
    :type sizeHint: PyQt4.QtGui.QSize
    :rtype: OrangeListBox
    """
    if box:
        bg = widgetBox(widget, box,
                       orientation="horizontal", addToLayout=False)
    else:
        bg = widget
    lb = OrangeListBox(master, enableDragDrop, dragDropCallback,
                       dataValidityCallback, sizeHint, bg)
    lb.setSelectionMode(selectionMode)
    lb.ogValue = value
    lb.ogLabels = labels
    lb.ogMaster = master

    if value is not None:
        clist = getdeepattr(master, value)
        if not isinstance(clist, ControlledList):
            clist = ControlledList(clist, lb)
            master.__setattr__(value, clist)
    if labels is not None:
        setattr(master, labels, getdeepattr(master, labels))
        if hasattr(master, CONTROLLED_ATTRIBUTES):
            getattr(master, CONTROLLED_ATTRIBUTES)[labels] = CallFrontListBoxLabels(lb)
    if value is not None:
        setattr(master, value, getdeepattr(master, value))
    connectControl(lb, master, value, callback, "itemSelectionChanged()",
                   CallFrontListBox(lb), CallBackListBox(lb, master))

    miscellanea(lb, bg, widget, **misc)
    return lb


# btnLabels is a list of either char strings or pixmaps
def radioButtons(widget, master, value, btnLabels=(), tooltips=None,
                      box=None, label=None, orientation='vertical',
                      callback=None, **misc):
    """
    Construct a button group and add radio buttons, if they are given.
    The value with which the buttons synchronize is the index of selected
    button.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
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
    :param orientation: orientation of the layout in the box
    :type orientation: int or str or PyQt4.QtGui.QLayout
    :rtype: PyQt4.QtQui.QButtonGroup
    """
    bg = widgetBox(widget, box, orientation, addToLayout=False)
    if not label is None:
        widgetLabel(bg, label)

    rb = QtGui.QButtonGroup(bg)
    if bg is not widget:
        bg.group = rb
    bg.buttons = []
    bg.ogValue = value
    bg.ogMaster = master
    for i, lab in enumerate(btnLabels):
        appendRadioButton(bg, lab, tooltip=tooltips and tooltips[i])
    connectControl(bg.group, master, value, callback, "buttonClicked(int)",
                   CallFrontRadioButtons(bg), CallBackRadioButton(bg, master))

    miscellanea(bg.group, bg, widget, **misc)
    return bg


radioButtonsInBox = radioButtons

def appendRadioButton(group, label, insertInto=None,
                      disabled=False, tooltip=None, sizePolicy=None,
                      addToLayout=True, stretch=0, addSpace=False):
    """
    Construct a radio button and add it to the group. The group must be
    constructed with :obj:`radioButtonsInBox` since it adds additional
    attributes need for the call backs.

    The radio button is inserted into `insertInto` or, if omitted, into the
    button group. This is useful for more complex groups, like those that have
    radio buttons in several groups, divided by labels and inside indented
    boxes.

    :param group: the button group
    :type group: PyQt4.QtCore.QButtonGroup
    :param label: string label or a pixmap for the button
    :type label: str or PyQt4.QtGui.QPixmap
    :param insertInto: the widget into which the radio button is inserted
    :type insertInto: PyQt4.QtGui.QWidget
    :rtype: PyQt4.QtGui.QRadioButton
    """
    i = len(group.buttons)
    if isinstance(label, str):
        w = QtGui.QRadioButton(label)
    else:
        w = QtGui.QRadioButton(str(i))
        w.setIcon(QtGui.QIcon(label))
    if not hasattr(group, "buttons"):
        group.buttons = []
    group.buttons.append(w)
    group.group.addButton(w)
    w.setChecked(getdeepattr(group.ogMaster, group.ogValue) == i)

    # miscellanea for this case is weird, so we do it here
    if disabled:
        w.setDisabled(disabled)
    if tooltip is not None:
        w.setToolTip(tooltip)
    if sizePolicy:
        w.setSizePolicy(sizePolicy)
    if addToLayout:
        dest = insertInto or group
        dest.layout().addWidget(w, stretch)
        _addSpace(dest, addSpace)
    return w


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1,
            callback=None, label=None, labelFormat=" %d", ticks=False,
            divideFactor=1.0, vertical=False, createLabel=True, width=None,
            intOnly=True, **misc):
    """
    Construct a slider.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: a label that is inserted into the box
    :type label: str
    :param callback: a function that is called when the value is changed
    :type callback: function

    :param minValue: minimal value
    :type minValue: int or float
    :param maxValue: maximal value
    :type maxValue: int or float
    :param step: step size
    :type step: int or float
    :param labelFormat: the label format; default is `" %d"`
    :type labelFormat: str
    :param ticks: if set to `True`, ticks are added below the slider
    :type ticks: bool
    :param divideFactor: a factor with which the displayed value is divided
    :type divideFactor: float
    :param vertical: if set to `True`, the slider is vertical
    :type vertical: bool
    :param createLabel: unless set to `False`, labels for minimal, maximal
        and the current value are added to the widget
    :type createLabel: bool
    :param width: the width of the slider
    :type width: int
    :param intOnly: if `True`, the slider value is integer (the slider is
        of type :obj:`PyQt4.QtGui.QSlider`) otherwise it is float
        (:obj:`FloatSlider`, derived in turn from :obj:`PyQt4.QtQui.QSlider`).
    :type intOnly: bool
    :rtype: :obj:`PyQt4.QtGui.QSlider` or :obj:`FloatSlider`
    """
    sliderBox = widgetBox(widget, box, orientation="horizontal",
                          addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    sliderOrient = Qt.Vertical if vertical else Qt.Horizontal
    if intOnly:
        slider = QtGui.QSlider(sliderOrient, sliderBox)
        slider.setRange(minValue, maxValue)
        if step:
            slider.setSingleStep(step)
            slider.setPageStep(step)
            slider.setTickInterval(step)
        signal_signature = "valueChanged(int)"
    else:
        slider = FloatSlider(sliderOrient, minValue, maxValue, step)
        signal_signature = "valueChangedFloat(double)"
    sliderBox.layout().addWidget(slider)
    slider.setValue(getdeepattr(master, value))
    if width:
        slider.setFixedWidth(width)
    if ticks:
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    if createLabel:
        label = QtGui.QLabel(sliderBox)
        sliderBox.layout().addWidget(label)
        label.setText(labelFormat % minValue)
        width1 = label.sizeHint().width()
        label.setText(labelFormat % maxValue)
        width2 = label.sizeHint().width()
        label.setFixedSize(max(width1, width2), label.sizeHint().height())
        txt = labelFormat % (getdeepattr(master, value) / divideFactor)
        label.setText(txt)
        label.setLbl = lambda x: \
            label.setText(labelFormat % (x / divideFactor))
        QtCore.QObject.connect(slider, QtCore.SIGNAL(signal_signature),
                               label.setLbl)

    connectControl(slider, master, value, callback, signal_signature,
                   CallFrontHSlider(slider))

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


def labeledSlider(widget, master, value, box=None,
                 label=None, labels=(), labelFormat=" %d", ticks=False,
                 callback=None, vertical=False, width=None, **misc):
    """
    Construct a slider with labels instead of numbers.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: a label that is inserted into the box
    :type label: str
    :param labels: labels shown at different slider positions
    :type labels: tuple of str
    :param callback: a function that is called when the value is changed
    :type callback: function

    :param ticks: if set to `True`, ticks are added below the slider
    :type ticks: bool
    :param vertical: if set to `True`, the slider is vertical
    :type vertical: bool
    :param width: the width of the slider
    :type width: int
    :rtype: :obj:`PyQt4.QtGui.QSlider`
    """
    sliderBox = widgetBox(widget, box, orientation="horizontal",
                          addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    sliderOrient = Qt.Vertical if vertical else Qt.Horizontal
    slider = QtGui.QSlider(sliderOrient, sliderBox)
    slider.ogValue = value
    slider.setRange(0, len(labels) - 1)
    slider.setSingleStep(1)
    slider.setPageStep(1)
    slider.setTickInterval(1)
    sliderBox.layout().addWidget(slider)
    slider.setValue(labels.index(getdeepattr(master, value)))
    if width:
        slider.setFixedWidth(width)
    if ticks:
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    max_label_size = 0
    slider.value_label = value_label = QtGui.QLabel(sliderBox)
    value_label.setAlignment(Qt.AlignRight)
    sliderBox.layout().addWidget(value_label)
    for lb in labels:
        value_label.setText(labelFormat % lb)
        max_label_size = max(max_label_size, value_label.sizeHint().width())
    value_label.setFixedSize(max_label_size, value_label.sizeHint().height())
    value_label.setText(getdeepattr(master, value))
    if isinstance(labelFormat, str):
        value_label.set_label = lambda x: \
            value_label.setText(labelFormat % x)
    else:
        value_label.set_label = lambda x: value_label.setText(labelFormat(x))
    QtCore.QObject.connect(slider, QtCore.SIGNAL("valueChanged(int)"),
                           value_label.set_label)

    connectControl(slider, master, value, callback, "valueChanged(int)",
                   CallFrontLabeledSlider(slider, labels),
                   CallBackLabeledSlider(slider, master, labels))

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


def valueSlider(widget, master, value, box=None, label=None,
                values=(), labelFormat=" %d", ticks=False,
                callback=None, vertical=False, width=None, **misc):
    """
    Construct a slider with different values.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: a label that is inserted into the box
    :type label: str
    :param values: values at different slider positions
    :type values: list of int
    :param labelFormat: label format; default is `" %d"`; can also be a function
    :type labelFormat: str or func
    :param callback: a function that is called when the value is changed
    :type callback: function

    :param ticks: if set to `True`, ticks are added below the slider
    :type ticks: bool
    :param vertical: if set to `True`, the slider is vertical
    :type vertical: bool
    :param width: the width of the slider
    :type width: int
    :rtype: :obj:`PyQt4.QtGui.QSlider`
    """
    if isinstance(labelFormat, str):
        labelFormat = lambda x, f=labelFormat: f(x)

    sliderBox = widgetBox(widget, box, orientation="horizontal",
                          addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    slider_orient = Qt.Vertical if vertical else Qt.Horizontal
    slider = QtGui.QSlider(slider_orient, sliderBox)
    slider.ogValue = value
    slider.setRange(0, len(values) - 1)
    slider.setSingleStep(1)
    slider.setPageStep(1)
    slider.setTickInterval(1)
    sliderBox.layout().addWidget(slider)
    slider.setValue(values.index(getdeepattr(master, value)))
    if width:
        slider.setFixedWidth(width)
    if ticks:
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    max_label_size = 0
    slider.value_label = value_label = QtGui.QLabel(sliderBox)
    value_label.setAlignment(Qt.AlignRight)
    sliderBox.layout().addWidget(value_label)
    for lb in values:
        value_label.setText(labelFormat(lb))
        max_label_size = max(max_label_size, value_label.sizeHint().width())
    value_label.setFixedSize(max_label_size, value_label.sizeHint().height())
    value_label.setText(labelFormat(getdeepattr(master, value)))
    value_label.set_label = lambda x: value_label.setText(labelFormat(values[x]))
    QtCore.QObject.connect(slider, QtCore.SIGNAL("valueChanged(int)"),
                           value_label.set_label)

    connectControl(slider, master, value, callback, "valueChanged(int)",
                   CallFrontLabeledSlider(slider, values),
                   CallBackLabeledSlider(slider, master, values))

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


# TODO comboBox looks overly complicated:
# - is the argument control2attributeDict needed? doesn't emptyString do the
#    job?
# - can valueType be anything else than str?
# - sendSelectedValue is not a great name
def comboBox(widget, master, value, box=None, label=None, labelWidth=None,
             orientation='vertical', items=None, callback=None,
             sendSelectedValue=False, valueType=str,
             control2attributeDict=None, emptyString=None, editable=False,
             **misc):
    """
    Construct a combo box.

    The `value` attribute of the `master` contains either the index of the
    selected row (if `sendSelected` is left at default, `False`) or a value
    converted to `valueType` (`str` by default).

    Furthermore, the value is converted by looking up into dictionary
    `control2attributeDict`.

    :param widget: the widget into which the box is inserted
    :type widget: PyQt4.QtGui.QWidget
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param orientation: orientation of the layout in the box
    :type orientation: str
    :param label: a label that is inserted into the box
    :type label: str
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param callback: a function that is called when the value is changed
    :type callback: function
    :param items: items that are put into the box
    :type items: list of ints
    :param sendSelectedValue: flag telling whether to store/retrieve indices
        or string values from `value`
    :type sendSelectedValue: bool
    :param valueType: the type into which the selected value is converted
        if sentSelectedValue is `False`
    :type valueType: type
    :param control2attributeDict: a dictionary through which the value is
        converted
    :type control2attributeDict: dict or None
    :param emptyString: the string value in the combo box that gets stored as
        an empty string in `value`
    :type emptyString: str
    :param editable: a flag telling whether the combo is editable
    :type editable: bool
    :rtype: PyQt4.QtGui.QComboBox
    """
    if box or label:
        hb = widgetBox(widget, box, orientation, addToLayout=False)
        if label is not None:
            widgetLabel(hb, label, labelWidth)
    else:
        hb = widget
    combo = QtGui.QComboBox(hb)
    combo.setEditable(editable)
    combo.box = hb
    if items:
        combo.addItems([str(i) for i in items])

    if value:
        cindex = getdeepattr(master, value)
        if isinstance(cindex, str):
            if items and cindex in items:
                cindex = items.index(getdeepattr(master, value))
            else:
                cindex = 0
        if cindex > combo.count() - 1:
            cindex = 0
        combo.setCurrentIndex(cindex)

        if sendSelectedValue:
            if control2attributeDict is None:
                control2attributeDict = {}
            if emptyString:
                control2attributeDict[emptyString] = ""
            connectControl(combo, master, value, callback,
                           "activated(const QString &)",
                           CallFrontComboBox(combo, valueType,
                                             control2attributeDict),
                           ValueCallbackCombo(master, value, valueType,
                                              control2attributeDict))
        else:
            connectControl(combo, master, value, callback, "activated(int)",
                           CallFrontComboBox(combo, None,
                                             control2attributeDict))
    miscellanea(combo, hb, widget, **misc)
    return combo


class OrangeListBox(QtGui.QListWidget):
    """
    List box with drag and drop functionality. Function :obj:`listBox`
    constructs instances of this class; do not use the class directly.

    .. attribute:: master

        The widget into which the listbox is inserted.

    .. attribute:: ogLabels

        The name of the master's attribute that holds the strings with items
        in the list box.

    .. attribute:: ogValue

        The name of the master's attribute that holds the indices of selected
        items.

    .. attribute:: enableDragDrop

        A flag telling whether drag-and-drop is enabled.

    .. attribute:: dragDropCallback

        A callback that is called at the end of drop event.

    .. attribute:: dataValidityCallback

        A callback that is called on dragEnter and dragMove events and returns
        either `ev.accept()` or `ev.ignore()`.

    .. attribute:: defaultSizeHint

        The size returned by the `sizeHint` method.
    """
    def __init__(self, master, enableDragDrop=False, dragDropCallback=None,
                 dataValidityCallback=None, sizeHint=None, *args):
        """
        :param master: the master widget
        :type master: OWWidget or OWComponent
        :param enableDragDrop: flag telling whether drag and drop is enabled
        :type enableDragDrop: bool
        :param dragDropCallback: callback for the end of drop event
        :type dragDropCallback: function
        :param dataValidityCallback: callback that accepts or ignores dragEnter
            and dragMove events
        :type dataValidityCallback: function with one argument (event)
        :param sizeHint: size hint
        :type sizeHint: PyQt4.QtGui.QSize
        :param args: optional arguments for the inherited constructor
        """
        self.master = master
        super().__init__(*args)
        self.drop_callback = dragDropCallback
        self.valid_data_callback = dataValidityCallback
        if not sizeHint:
            self.size_hint = QtCore.QSize(150, 100)
        else:
            self.size_hint = sizeHint
        if enableDragDrop:
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(True)

    def sizeHint(self):
        return self.size_hint

    def dragEnterEvent(self, ev):
        super().dragEnterEvent(ev)
        if self.valid_data_callback:
            self.valid_data_callback(ev)
        elif isinstance(ev.source(), OrangeListBox):
            ev.setDropAction(Qt.MoveAction)
            ev.accept()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        ev.setDropAction(Qt.MoveAction)
        super().dropEvent(ev)

        items = self.update_master()
        if ev.source() is not self:
            ev.source().update_master(exclude=items)

        if self.drop_callback:
            self.drop_callback()

    def update_master(self, exclude=()):
        control_list = [self.item(i).data(Qt.UserRole) for i in range(self.count()) if self.item(i).data(Qt.UserRole) not in exclude]
        if self.ogLabels:
            master_list = getattr(self.master, self.ogLabels)

            if master_list != control_list:
                setattr(self.master, self.ogLabels, control_list)
        return control_list

    def updateGeometries(self):
        # A workaround for a bug in Qt
        # (see: http://bugreports.qt.nokia.com/browse/QTBUG-14412)
        if getattr(self, "_updatingGeometriesNow", False):
            return
        self._updatingGeometriesNow = True
        try:
            return super().updateGeometries()
        finally:
            self._updatingGeometriesNow = False


# TODO: SmallWidgetButton is used only in OWkNNOptimization.py. (Re)Move.
# eliminated?
class SmallWidgetButton(QtGui.QPushButton):
    def __init__(self, widget, text="", pixmap=None, box=None,
                 orientation='vertical', autoHideWidget=None, **misc):
        #self.parent = parent
        if pixmap is not None:
            iconDir = os.path.join(os.path.dirname(__file__), "icons")
            name = ""
            if isinstance(pixmap, str):
                if os.path.exists(pixmap):
                    name = pixmap
                elif os.path.exists(os.path.join(iconDir, pixmap)):
                    name = os.path.join(iconDir, pixmap)
            elif isinstance(pixmap, (QtGui.QPixmap, QtGui.QIcon)):
                name = pixmap
            name = name or os.path.join(iconDir, "arrow_down.png")
            super().__init__(QtGui.QIcon(name), text, widget)
        else:
            super().__init__(text, widget)
        if widget.layout() is not None:
            widget.layout().addWidget(self)
        # create autohide widget and set a layout
        self.widget = self.autohideWidget = \
            (autoHideWidget or AutoHideWidget)(None, Qt.Popup)
        setLayout(self.widget, orientation)
        if box:
            self.widget = widgetBox(self.widget, box, orientation)
        self.autohideWidget.hide()
        miscellanea(self, self.widget, widget, **misc)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            self.autohideWidget.move(
                self.mapToGlobal(QtCore.QPoint(0, self.height())))
            self.autohideWidget.show()


class SmallWidgetLabel(QtGui.QLabel):
    def __init__(self, widget, text="", pixmap=None, box=None,
                 orientation='vertical', **misc):
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
            elif isinstance(pixmap, (QtGui.QPixmap, QtGui.QIcon)):
                name = pixmap
            name = name or os.path.join(iconDir, "arrow_down.png")
            self.setPixmap(QtGui.QPixmap(name))
        self.autohideWidget = self.widget = AutoHideWidget(None, Qt.Popup)
        setLayout(self.widget, orientation)
        if box:
            self.widget = widgetBox(self.widget, box, orientation)
        self.autohideWidget.hide()
        miscellanea(self, self.widget, widget, **misc)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            self.autohideWidget.move(
                self.mapToGlobal(QtCore.QPoint(0, self.height())))
            self.autohideWidget.show()


class AutoHideWidget(QtGui.QWidget):
    def leaveEvent(self, _):
        self.hide()


# TODO Class SearchLineEdit: it doesn't seem to be used anywhere
# see widget DataDomain
class SearchLineEdit(QtGui.QLineEdit):
    """
    QLineEdit for quick searches
    """
    def __init__(self, t, searcher):
        super().__init__(self, t)
        self.searcher = searcher

    def keyPressEvent(self, e):
        """
        Handles keys up and down by selecting the previous and the next item
        in the list, and the escape key, which hides the searcher.
        """
        k = e.key()
        if k == Qt.Key_Down:
            curItem = self.searcher.lb.currentItem()
            if curItem + 1 < self.searcher.lb.count():
                self.searcher.lb.setCurrentItem(curItem + 1)
        elif k == Qt.Key_Up:
            curItem = self.searcher.lb.currentItem()
            if curItem:
                self.searcher.lb.setCurrentItem(curItem - 1)
        elif k == Qt.Key_Escape:
            self.searcher.window.hide()
        else:
            return super().keyPressEvent(e)


# TODO Class Searcher: it doesn't seem to be used anywhere
# see widget DataDomain
class Searcher:
    """
    The searcher class for :obj:`SearchLineEdit`.
    """
    def __init__(self, control, master):
        self.control = control
        self.master = master

    def __call__(self):
        _s = QtGui.QStyle
        self.window = t = QtGui.QFrame(
            self.master,
            _s.WStyle_Dialog + _s.WStyle_Tool + _s.WStyle_Customize +
            _s.WStyle_NormalBorder)
        QtGui.QVBoxLayout(t).setAutoAdd(1)
        gs = self.master.mapToGlobal(QtCore.QPoint(0, 0))
        gl = self.control.mapToGlobal(QtCore.QPoint(0, 0))
        t.move(gl.x() - gs.x(), gl.y() - gs.y())
        self.allItems = [self.control.text(i)
                         for i in range(self.control.count())]
        le = SearchLineEdit(t, self)
        self.lb = QtGui.QListWidget(t)
        for i in self.allItems:
            self.lb.insertItem(i)
        t.setFixedSize(self.control.width(), 200)
        t.show()
        le.setFocus()
        le.textChanged.connect(self.textChanged)
        le.returnPressed.connect(self.returnPressed)
        self.lb.itemClicked.connect(self.mouseClicked)

    def textChanged(self, s):
        s = str(s)
        self.lb.clear()
        for i in self.allItems:
            if s.lower() in i.lower():
                self.lb.insertItem(i)

    def returnPressed(self):
        if self.lb.count():
            self.conclude(self.lb.text(max(0, self.lb.currentItem())))
        else:
            self.window.hide()

    def mouseClicked(self, item):
        self.conclude(item.text())

    def conclude(self, value):
        index = self.allItems.index(value)
        self.control.setCurrentItem(index)
        if self.control.cback:
            if self.control.sendSelectedValue:
                self.control.cback(value)
            else:
                self.control.cback(index)
        if self.control.cfunc:
            self.control.cfunc()
        self.window.hide()


# creates a widget box with a button in the top right edge that shows/hides all
# widgets in the box and collapse the box to its minimum height
# TODO collapsableWidgetBox is used only in OWMosaicDisplay.py; (re)move
class collapsableWidgetBox(QtGui.QGroupBox):
    def __init__(self, widget, box="", master=None, value="",
                 orientation="vertical", callback=None):
        super().__init__(widget)
        self.setFlat(1)
        setLayout(self, orientation)
        if widget.layout() is not None:
            widget.layout().addWidget(self)
        if isinstance(box, str):
            self.setTitle(" " + box.strip() + " ")
        self.setCheckable(True)
        self.master = master
        self.value = value
        self.callback = callback
        self.clicked.connect(self.toggled)

    def toggled(self, _=0):
        if self.value:
            self.master.__setattr__(self.value, self.isChecked())
            self.updateControls()
        if self.callback is not None:
            self.callback()

    def updateControls(self):
        val = getdeepattr(self.master, self.value)
        width = self.width()
        self.setChecked(val)
        self.setFlat(not val)
        self.setMinimumSize(QtCore.QSize(width if not val else 0, 0))
        for c in self.children():
            if isinstance(c, QtGui.QLayout):
                continue
            if val:
                c.show()
            else:
                c.hide()


# creates an icon that allows you to show/hide the widgets in the widgets list
# TODO Class widgetHider doesn't seem to be used anywhere; remove?
class widgetHider(QtGui.QWidget):
    def __init__(self, widget, master, value, _=(19, 19), widgets=None,
                 tooltip=None):
        super().__init__(widget)
        if widget.layout() is not None:
            widget.layout().addWidget(self)
        self.value = value
        self.master = master
        if tooltip:
            self.setToolTip(tooltip)
        iconDir = os.path.join(os.path.dirname(__file__), "icons")
        icon1 = os.path.join(iconDir, "arrow_down.png")
        icon2 = os.path.join(iconDir, "arrow_up.png")
        self.pixmaps = [QtGui.QPixmap(icon1), QtGui.QPixmap(icon2)]
        self.setFixedSize(self.pixmaps[0].size())
        self.disables = list(widgets or [])
        self.makeConsistent = Disabler(self, master, value, type=HIDER)
        if widgets:
            self.setWidgets(widgets)

    def mousePressEvent(self, ev):
        self.master.__setattr__(self.value,
                                not getdeepattr(self.master, self.value))
        self.makeConsistent()

    def setWidgets(self, widgets):
        self.disables = list(widgets)
        self.makeConsistent()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self.pixmaps:
            pix = self.pixmaps[getdeepattr(self.master, self.value)]
            painter = QtGui.QPainter(self)
            painter.drawPixmap(0, 0, pix)


##############################################################################
# callback handlers

def setStopper(master, sendButton, stopCheckbox, changedFlag, callback):
    """
    Arrange the mechanics needed for a typical combination of the check box
    "Commit on change" and push button "Commit".

    The function tells the check box to disable the send button when the box is
    checked (this is done by adding `(-1, sendButton)` to the checkbox's list
    `disables`; the already disables the button if the box is checked now.

    The function connects a new callback to the checkbox's signal `toggled`
    to call the `callback` when the box is checked and the data has been
    changed, as indicated by the value of `changedFlag`.

    To set up the Commit-on-change---Commit interface, do the following. In
    the widget add something like::

        commitButton = gui.button(box, self, "Commit", callback=self.apply)
        autoCommit = gui.checkBox(box, self, "autoCommit", "Commit on change")
        gui.setStopper(self, commitButton, autoCommit, "dataDirty", self.apply)

    Whenever the data is changed and could be commited, call a method like::

        def applyIf(self):
        if self.autoApply:
            self.apply()
        else:
            self.dataDirty = True

    The method can have any name, not necessarily `applyIf`. Method `apply`
    sends the necessary data to signal manager.

    Used like this, `setStopper` tells `autoCommit` checkbox to disable the
    `commitButton`, and when the check box is checked, it will call
    `self.apply` if `dataDirty` is `True`.

    :param master: the master widget (used only to get the `changedFlag`)
    :type master: OWWidget or OWComponent
    :param sendButton: the button for committing the data
    :type sendButton: PyQt4.QtGui.QPushButton
    :param stopCheckbox: the check box
    :type stopCheckbox: PyQt4.QtGui.QCheckBox
    :param changedFlag: the name of the flag in the master that tells whether
        the data is changed
    :type changedFlag: str
    :param callback: the method (typically of the `master`) that commits the
        data
    :type callback: function
    """
    stopCheckbox.disables.append((-1, sendButton))
    sendButton.setDisabled(stopCheckbox.isChecked())
    stopCheckbox.toggled.connect(
        lambda x: x and getdeepattr(master, changedFlag, True) and callback())


class ControlledList(list):
    """
    A class derived from a list that is connected to a
    :obj:`PyQt4.QtGui.QListBox`: the list contains indices of items that are
    selected in the list box. Changing the list content changes the
    selection in the list box.
    """
    def __init__(self, content, listBox=None):
        super().__init__(content)
        self.listBox = listBox

    def __reduce__(self):
        # cannot pickle self.listBox, but can't discard it
        # (ControlledList may live on)
        import copyreg
        return copyreg._reconstructor, (list, list, ()), None, self.__iter__()

    # TODO ControllgedList.item2name is probably never used
    def item2name(self, item):
        item = self.listBox.labels[item]
        if type(item) is tuple:
            return item[1]
        else:
            return item

    def __setitem__(self, index, item):
        if isinstance(index, int):
            self.listBox.item(self[index]).setSelected(0)
            item.setSelected(1)
        else:
            for i in self[index]:
                self.listBox.item(i).setSelected(0)
            for i in item:
                self.listBox.item(i).setSelected(1)
        super().__setitem__(index, item)

    def __delitem__(self, index):
        if isinstance(index, int):
            self.listBox.item(self[index]).setSelected(0)
        else:
            for i in self[index]:
                self.listBox.item(i).setSelected(0)
        super().__delitem__(index)

    def append(self, item):
        super().append(item)
        item.setSelected(1)

    def extend(self, items):
        super().extend(items)
        for i in items:
            self.listBox.item(i).setSelected(1)

    def insert(self, index, item):
        item.setSelected(1)
        super().insert(index, item)

    def pop(self, index=-1):
        i = super().pop(index)
        self.listBox.item(i).setSelected(0)

    def remove(self, item):
        item.setSelected(0)
        super().remove(item)


def connectControlSignal(control, signal, f):
    if type(signal) is tuple:
        control, signal = signal
    QtCore.QObject.connect(control, QtCore.SIGNAL(signal), f)


def connectControl(control, master, value, f, signal,
                   cfront, cback=None, cfunc=None, fvcb=None):
    cback = cback or value and ValueCallback(master, value, fvcb)
    if cback:
        if signal:
            connectControlSignal(control, signal, cback)
        cback.opposite = cfront
        if value and cfront and hasattr(master, CONTROLLED_ATTRIBUTES):
            getattr(master, CONTROLLED_ATTRIBUTES)[value] = cfront

    cfunc = cfunc or f and FunctionCallback(master, f)
    if cfunc:
        if signal:
            connectControlSignal(control, signal, cfunc)
        cfront.opposite = cback, cfunc
    else:
        cfront.opposite = (cback,)

    return cfront, cback, cfunc


class ControlledCallback:
    def __init__(self, widget, attribute, f=None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        self.disabled = 0
        if isinstance(widget, dict):
            return  # we can't assign attributes to dict
        if not hasattr(widget, "callbackDeposit"):
            widget.callbackDeposit = []
        widget.callbackDeposit.append(self)

    def acyclic_setattr(self, value):
        if self.disabled:
            return
        if self.f:
            if self.f in (int, float) and (
                    not value or isinstance(value, str) and value in "+-"):
                value = self.f(0)
            else:
                value = self.f(value)
        opposite = getattr(self, "opposite", None)
        if opposite:
            try:
                opposite.disabled += 1
                if type(self.widget) is dict:
                    self.widget[self.attribute] = value
                else:
                    setattr(self.widget, self.attribute, value)
            finally:
                opposite.disabled -= 1
        else:
            if isinstance(self.widget, dict):
                self.widget[self.attribute] = value
            else:
                setattr(self.widget, self.attribute, value)


class ValueCallback(ControlledCallback):
    # noinspection PyBroadException
    def __call__(self, value):
        if value is None:
            return
        try:
            self.acyclic_setattr(value)
        except:
            print("gui.ValueCallback: %s" % value)
            import traceback
            import sys
            traceback.print_exception(*sys.exc_info())


class ValueCallbackCombo(ValueCallback):
    def __init__(self, widget, attribute, f=None, control2attributeDict=None):
        super().__init__(widget, attribute, f)
        self.control2attributeDict = control2attributeDict or {}

    def __call__(self, value):
        value = str(value)
        return super().__call__(self.control2attributeDict.get(value, value))


class ValueCallbackLineEdit(ControlledCallback):
    def __init__(self, control, widget, attribute, f=None):
        ControlledCallback.__init__(self, widget, attribute, f)
        self.control = control

    # noinspection PyBroadException
    def __call__(self, value):
        if value is None:
            return
        try:
            pos = self.control.cursorPosition()
            self.acyclic_setattr(value)
            self.control.setCursorPosition(pos)
        except:
            print("invalid value ", value, type(value))
            import traceback
            import sys
            traceback.print_exception(*sys.exc_info())


class SetLabelCallback:
    def __init__(self, widget, label, format="%5.2f", f=None):
        self.widget = widget
        self.label = label
        self.format = format
        self.f = f
        if hasattr(widget, "callbackDeposit"):
            widget.callbackDeposit.append(self)
        self.disabled = 0

    def __call__(self, value):
        if not self.disabled and value is not None:
            if self.f:
                value = self.f(value)
            self.label.setText(self.format % value)


class FunctionCallback:
    def __init__(self, master, f, widget=None, id=None, getwidget=False):
        self.master = master
        self.widget = widget
        self.f = f
        self.id = id
        self.getwidget = getwidget
        if hasattr(master, "callbackDeposit"):
            master.callbackDeposit.append(self)
        self.disabled = 0

    def __call__(self, *value):
        if not self.disabled and value is not None:
            kwds = {}
            if self.id is not None:
                kwds['id'] = self.id
            if self.getwidget:
                kwds['widget'] = self.widget
            if isinstance(self.f, list):
                for f in self.f:
                    f(**kwds)
            else:
                self.f(**kwds)


class CallBackListBox:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = 0

    def __call__(self, *_):  # triggered by selectionChange()
        if not self.disabled and self.control.ogValue is not None:
            clist = getdeepattr(self.widget, self.control.ogValue)
             # skip the overloaded method to avoid a cycle
            list.__delitem__(clist, slice(0, len(clist)))
            control = self.control
            for i in range(control.count()):
                if control.item(i).isSelected():
                    list.append(clist, i)
            self.widget.__setattr__(self.control.ogValue, clist)


class CallBackRadioButton:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = False

    def __call__(self, *_):  # triggered by toggled()
        if not self.disabled and self.control.ogValue is not None:
            arr = [butt.isChecked() for butt in self.control.buttons]
            self.widget.__setattr__(self.control.ogValue, arr.index(1))


class CallBackLabeledSlider:
    def __init__(self, control, widget, lookup):
        self.control = control
        self.widget = widget
        self.lookup = lookup
        self.disabled = False

    def __call__(self, *_):
        if not self.disabled and self.control.ogValue is not None:
            self.widget.__setattr__(self.control.ogValue,
                                    self.lookup[self.control.value()])


##############################################################################
# call fronts (change of the attribute value changes the related control)


class ControlledCallFront:
    def __init__(self, control):
        self.control = control
        self.disabled = 0

    def action(self, *_):
        pass

    def __call__(self, *args):
        if not self.disabled:
            opposite = getattr(self, "opposite", None)
            if opposite:
                try:
                    for op in opposite:
                        op.disabled += 1
                    self.action(*args)
                finally:
                    for op in opposite:
                        op.disabled -= 1
            else:
                self.action(*args)


class CallFrontSpin(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setValue(value)


class CallFrontDoubleSpin(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setValue(value)


class CallFrontCheckBox(ControlledCallFront):
    def action(self, value):
        if value is not None:
            values = [Qt.Unchecked, Qt.Checked, Qt.PartiallyChecked]
            self.control.setCheckState(values[value])


class CallFrontButton(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setChecked(bool(value))


class CallFrontComboBox(ControlledCallFront):
    def __init__(self, control, valType=None, control2attributeDict=None):
        super().__init__(control)
        self.valType = valType
        if control2attributeDict is None:
            self.attribute2controlDict = {}
        else:
            self.attribute2controlDict = \
                {y: x for x, y in control2attributeDict.items()}

    def action(self, value):
        if value is not None:
            value = self.attribute2controlDict.get(value, value)
            if self.valType:
                for i in range(self.control.count()):
                    if self.valType(str(self.control.itemText(i))) == value:
                        self.control.setCurrentIndex(i)
                        return
                values = ""
                for i in range(self.control.count()):
                    values += str(self.control.itemText(i)) + \
                        (i < self.control.count() - 1 and ", " or ".")
                print("unable to set %s to value '%s'. Possible values are %s"
                      % (self.control, value, values))
            else:
                if value < self.control.count():
                    self.control.setCurrentIndex(value)


class CallFrontHSlider(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setValue(value)


class CallFrontLabeledSlider(ControlledCallFront):
    def __init__(self, control, lookup):
        super().__init__(control)
        self.lookup = lookup

    def action(self, value):
        if value is not None:
            self.control.setValue(self.lookup.index(value))


class CallFrontLogSlider(ControlledCallFront):
    def action(self, value):
        if value is not None:
            if value < 1e-30:
                print("unable to set %s to %s (value too small)" %
                      (self.control, value))
            else:
                self.control.setValue(math.log10(value))


class CallFrontLineEdit(ControlledCallFront):
    def action(self, value):
        self.control.setText(str(value))


class CallFrontRadioButtons(ControlledCallFront):
    def action(self, value):
        if value < 0 or value >= len(self.control.buttons):
            value = 0
        self.control.buttons[value].setChecked(1)


class CallFrontListBox(ControlledCallFront):
    def action(self, value):
        if value is not None:
            if not isinstance(value, ControlledList):
                setattr(self.control.ogMaster, self.control.ogValue,
                        ControlledList(value, self.control))
            for i in range(self.control.count()):
                shouldBe = i in value
                if shouldBe != self.control.item(i).isSelected():
                    self.control.item(i).setSelected(shouldBe)


class CallFrontListBoxLabels(ControlledCallFront):
    unknownType = None

    def action(self, values):
        self.control.clear()
        if values:
            for value in values:
                if isinstance(value, tuple):
                    text, icon = value
                    if isinstance(icon, int):
                        item = QtGui.QListWidgetItem(attributeIconDict[icon], text)
                    else:
                        item = QtGui.QListWidgetItem(icon, text)
                else:
                    item = QtGui.QListWidgetItem(value)

                item.setData(Qt.UserRole, value)
                self.control.addItem(item)


class CallFrontLabel:
    def __init__(self, control, label, master):
        self.control = control
        self.label = label
        self.master = master

    def __call__(self, *_):
        self.control.setText(self.label % self.master.__dict__)

##############################################################################
## Disabler is a call-back class for check box that can disable/enable other
## widgets according to state (checked/unchecked, enabled/disable) of the
## given check box
##
## Tricky: if self.propagateState is True (default), then if check box is
## disabled the related widgets will be disabled (even if the checkbox is
## checked). If self.propagateState is False, the related widgets will be
## disabled/enabled if check box is checked/clear, disregarding whether the
## check box itself is enabled or not. (If you don't understand, see the
## code :-)
DISABLER = 1
HIDER = 2


# noinspection PyShadowingBuiltins
class Disabler:
    def __init__(self, widget, master, valueName, propagateState=True,
                 type=DISABLER):
        self.widget = widget
        self.master = master
        self.valueName = valueName
        self.propagateState = propagateState
        self.type = type

    def __call__(self, *value):
        currState = self.widget.isEnabled()
        if currState or not self.propagateState:
            if len(value):
                disabled = not value[0]
            else:
                disabled = not getdeepattr(self.master, self.valueName)
        else:
            disabled = 1
        for w in self.widget.disables:
            if type(w) is tuple:
                if isinstance(w[0], int):
                    i = 1
                    if w[0] == -1:
                        disabled = not disabled
                else:
                    i = 0
                if self.type == DISABLER:
                    w[i].setDisabled(disabled)
                elif self.type == HIDER:
                    if disabled:
                        w[i].hide()
                    else:
                        w[i].show()
                if hasattr(w[i], "makeConsistent"):
                    w[i].makeConsistent()
            else:
                if self.type == DISABLER:
                    w.setDisabled(disabled)
                elif self.type == HIDER:
                    if disabled:
                        w.hide()
                    else:
                        w.show()

##############################################################################
# some table related widgets


# noinspection PyShadowingBuiltins
class tableItem(QtGui.QTableWidgetItem):
    def __init__(self, table, x, y, text, editType=None, backColor=None,
                 icon=None, type=QtGui.QTableWidgetItem.Type):
        super().__init__(type)
        if icon:
            self.setIcon(QtGui.QIcon(icon))
        if editType is not None:
            self.setFlags(editType)
        else:
            self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable |
                          Qt.ItemIsSelectable)
        if backColor is not None:
            self.setBackground(QtGui.QBrush(backColor))
        # we add it this way so that text can also be int and sorting will be
        # done properly (as integers and not as text)
        self.setData(Qt.DisplayRole, text)
        table.setItem(x, y, self)


TableValueRole = next(OrangeUserRole)  # Role to retrieve orange.Value
TableClassValueRole = next(OrangeUserRole)  # Retrieve class value for the row
TableDistribution = next(OrangeUserRole)  # Retrieve distribution of the column
TableVariable = next(OrangeUserRole)  # Role to retrieve the column's variable

BarRatioRole = next(OrangeUserRole)  # Ratio for drawing distribution bars
BarBrushRole = next(OrangeUserRole)  # Brush for distribution bar

SortOrderRole = next(OrangeUserRole)  # Used for sorting


class TableBarItem(QtGui.QItemDelegate):
    BarRole = next(OrangeUserRole)
    ColorRole = next(OrangeUserRole)

    def __init__(self, widget, table=None, color=QtGui.QColor(255, 170, 127),
                 color_schema=None):
        """
        :param widget: OWWidget instance
        :type widget: :class:`OWWidget.OWWidget
        :param table: Table
        :type table: :class:`Orange.data.Table`
        :param color: Color of the distribution bar.
        :type color: :class:`PyQt4.QtCore.QColor`
        :param color_schema: If not None it must be an instance of
            :class:`OWColorPalette.ColorPaletteGenerator` (note: this
            parameter, if set, overrides the ``color``)
        :type color_schema: :class:`OWColorPalette.ColorPaletteGenerator`

        """
        super().__init__(widget)
        self.color = color
        self.color_schema = color_schema
        self.widget = widget
        self.table = table

    def paint(self, painter, option, index):
        from Orange.data import DiscreteVariable
        painter.save()
        self.drawBackground(painter, option, index)
        if self.table is None:
            table = getattr(index.model(), "examples", None)
        else:
            table = self.table
        ratio = index.data(TableBarItem.BarRole)
        if isinstance(ratio, float):
            if math.isnan(ratio):
                ratio = None
        elif table is not None and getattr(self.widget, "show_bars", False):
            value = index.data(Qt.DisplayRole)
            if isinstance(value, float):
                col = index.column()
                if col < len(table.normalizers):
                    maxv, span = table.normalizers[col]
                    ratio = (maxv - value) / span

        color = self.color
        if (self.color_schema is not None and table is not None and
                isinstance(table.domain.class_var, DiscreteVariable)):
            class_ = index.data(TableClassValueRole)
            if not math.isnan(class_):
                color = self.color_schema[int(class_)]
        else:
            color = self.color

        if ratio is not None:
            painter.save()
            painter.setPen(QtGui.QPen(QtGui.QBrush(color), 5,
                                      Qt.SolidLine, Qt.RoundCap))
            rect = option.rect.adjusted(3, 0, -3, -5)
            x, y = rect.x(), rect.y() + rect.height()
            painter.drawLine(x, y, x + rect.width() * ratio, y)
            painter.restore()
            text_rect = option.rect.adjusted(0, 0, 0, -3)
        else:
            text_rect = option.rect
        text = index.data(Qt.DisplayRole)
        self.drawDisplay(painter, option, text_rect, text)
        painter.restore()


class BarItemDelegate(QtGui.QStyledItemDelegate):
    def __init__(self, parent, brush=QtGui.QBrush(QtGui.QColor(255, 170, 127)),
                 scale=(0.0, 1.0)):
        super().__init__(parent)
        self.brush = brush
        self.scale = scale

    def paint(self, painter, option, index):
        QSt = QtGui.QStyle
        QtGui.qApp.style().drawPrimitive(
            QSt.PE_PanelItemViewRow, option, painter)
        QtGui.qApp.style().drawPrimitive(
            QSt.PE_PanelItemViewItem, option, painter)
        rect = option.rect
        val = index.data(Qt.DisplayRole)
        if isinstance(val, float):
            minv, maxv = self.scale
            val = (val - minv) / (maxv - minv)
            painter.save()
            if option.state & QSt.State_Selected:
                painter.setOpacity(0.75)
            painter.setBrush(self.brush)
            painter.drawRect(
                rect.adjusted(1, 1, - rect.width() * (1.0 - val) - 2, -2))
            painter.restore()


class IndicatorItemDelegate(QtGui.QStyledItemDelegate):
    IndicatorRole = next(OrangeUserRole)

    def __init__(self, parent, role=IndicatorRole, indicatorSize=2):
        super().__init__(parent)
        self.role = role
        self.indicatorSize = indicatorSize

    def paint(self, painter, option, index):
        super().paint(self, painter, option, index)
        rect = option.rect
        indicator, valid = index.data(self.role).toString(), True
        indicator = False if indicator == "false" else indicator
        if valid and indicator:
            painter.save()
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(Qt.black))
            painter.drawEllipse(rect.center(),
                                self.indicatorSize, self.indicatorSize)
            painter.restore()


class LinkStyledItemDelegate(QtGui.QStyledItemDelegate):
    LinkRole = next(OrangeUserRole)

    def __init__(self, parent):
        super().__init__(parent)
        self.mousePressState = QtCore.QModelIndex(), QtCore.QPoint()
        parent.entered.connect(self.onEntered)


    def sizeHint(self, option, index):
        size = super().sizeHint(self, option, index)
        return QtCore.QSize(size.width(), max(size.height(), 20))


    def linkRect(self, option, index):
        style = self.parent().style()
        text = self.displayText(index.data(Qt.DisplayRole),
                                QtCore.QLocale.system())
        self.initStyleOption(option, index)
        textRect = style.subElementRect(QtGui.QStyle.SE_ItemViewItemText,
                                        option)
        if not textRect.isValid():
            textRect = option.rect
        margin = style.pixelMetric(QtGui.QStyle.PM_FocusFrameHMargin,
                                   option) + 1
        textRect = textRect.adjusted(margin, 0, -margin, 0)
        font = index.data(Qt.FontRole)
        font = QtGui.QFont(font) if font.isValid() else option.font
        metrics = QtGui.QFontMetrics(font)
        elideText = metrics.elidedText(text, option.textElideMode,
                                       textRect.width())
        return metrics.boundingRect(textRect, option.displayAlignment,
                                    elideText)


    def editorEvent(self, event, model, option, index):
        if event.type() == QtCore.QEvent.MouseButtonPress and \
                self.linkRect(option, index).contains(event.pos()):
            self.mousePressState = (QtCore.QPersistentModelIndex(index),
                                    QtCore.QPoint(event.pos()))

        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            link = index.data(LinkRole)
            pressedIndex, pressPos = self.mousePressState
            if pressedIndex == index and \
                    (pressPos - event.pos()).manhattanLength() < 5 and \
                    link.isValid():
                import webbrowser
                webbrowser.open(link.toString())
            self.mousePressState = QtCore.QModelIndex(), event.pos()

        elif event.type() == QtCore.QEvent.MouseMove:
            link = index.data(LinkRole)
            if link.isValid() and \
                    self.linkRect(option, index).contains(event.pos()):
                self.parent().viewport().setCursor(Qt.PointingHandCursor)
            else:
                self.parent().viewport().setCursor(Qt.ArrowCursor)

        return super().editorEvent(self, event, model, option, index)


    def onEntered(self, index):
        link = index.data(LinkRole)
        if not link.isValid():
            self.parent().viewport().setCursor(Qt.ArrowCursor)


    def paint(self, painter, option, index):
        QSt = QtGui.QStyle
        if index.data(LinkRole).isValid():
            style = QtGui.qApp.style()
            style.drawPrimitive(QSt.PE_PanelItemViewRow, option, painter)
            style.drawPrimitive(QSt.PE_PanelItemViewItem, option, painter)
            text = self.displayText(index.data(Qt.DisplayRole),
                                    QtCore.QLocale.system())
            textRect = style.subElementRect(QSt.SE_ItemViewItemText, option)
            if not textRect.isValid():
                textRect = option.rect
            margin = style.pixelMetric(QSt.PM_FocusFrameHMargin, option) + 1
            textRect = textRect.adjusted(margin, 0, -margin, 0)
            elideText = QtGui.QFontMetrics(option.font).elidedText(
                text, option.textElideMode, textRect.width())
            painter.save()
            font = index.data(Qt.FontRole)
            if font.isValid():
                painter.setFont(QtGui.QFont(font))
            else:
                painter.setFont(option.font)
            painter.setPen(QtGui.QPen(Qt.blue))
            painter.drawText(textRect, option.displayAlignment, elideText)
            painter.restore()
        else:
            super().paint(self, painter, option, index)


LinkRole = LinkStyledItemDelegate.LinkRole


class ColoredBarItemDelegate(QtGui.QStyledItemDelegate):
    """ Item delegate that can also draws a distribution bar
    """
    def __init__(self, parent=None, decimals=3, color=Qt.red):
        super().__init__(parent)
        self.decimals = decimals
        self.float_fmt = "%%.%if" % decimals
        self.color = QtGui.QColor(color)

    def displayText(self, value, locale):
        if isinstance(value, float):
            return self.float_fmt % value
        elif isinstance(value, str):
            return value
        elif value is None:
            return "NA"
        else:
            return str(value)

    def sizeHint(self, option, index):
        font = self.get_font(option, index)
        metrics = QtGui.QFontMetrics(font)
        height = metrics.lineSpacing() + 8  # 4 pixel margin
        width = metrics.width(self.displayText(index.data(Qt.DisplayRole),
                                               QtCore.QLocale())) + 8
        return QtCore.QSize(width, height)

    def paint(self, painter, option, index):
        QSt = QtGui.QStyle
        self.initStyleOption(option, index)
        text = self.displayText(index.data(Qt.DisplayRole), QtCore.QLocale())
        ratio, have_ratio = self.get_bar_ratio(option, index)

        rect = option.rect
        if have_ratio:
            # The text is raised 3 pixels above the bar.
            # TODO: Style dependent margins?
            text_rect = rect.adjusted(4, 1, -4, -4)
        else:
            text_rect = rect.adjusted(4, 4, -4, -4)

        painter.save()
        font = self.get_font(option, index)
        painter.setFont(font)

        QtGui.qApp.style().drawPrimitive(QSt.PE_PanelItemViewRow, option, painter)
        QtGui.qApp.style().drawPrimitive(QSt.PE_PanelItemViewItem, option, painter)

        # TODO: Check ForegroundRole.
        if option.state & QSt.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
        painter.setPen(QtGui.QPen(color))

        align = self.get_text_align(option, index)

        metrics = QtGui.QFontMetrics(font)
        elide_text = metrics.elidedText(
            text, option.textElideMode, text_rect.width())
        painter.drawText(text_rect, align, elide_text)

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if have_ratio:
            brush = self.get_bar_brush(option, index)

            painter.setBrush(brush)
            painter.setPen(QtGui.QPen(brush, 1))
            bar_rect = QtCore.QRect(text_rect)
            bar_rect.setTop(bar_rect.bottom() - 1)
            bar_rect.setBottom(bar_rect.bottom() + 1)
            w = text_rect.width()
            bar_rect.setWidth(max(0, min(w * ratio, w)))
            painter.drawRoundedRect(bar_rect, 2, 2)
        painter.restore()

    def get_font(self, option, index):
        font = index.data(Qt.FontRole)
        if not isinstance(font, QtGui.QFont):
            font = option.font
        return font

    def get_text_align(self, _, index):
        align = index.data(Qt.TextAlignmentRole)
        if not isinstance(align, int):
            align = Qt.AlignLeft | Qt.AlignVCenter

        return align

    def get_bar_ratio(self, _, index):
        ratio = index.data(BarRatioRole)
        return ratio, isinstance(ratio, float)

    def get_bar_brush(self, _, index):
        bar_brush = index.data(BarBrushRole)
        if not isinstance(bar_brush, (QtGui.QColor, QtGui.QBrush)):
            bar_brush = self.color
        return QtGui.QBrush(bar_brush)

##############################################################################
# progress bar management


class ProgressBar:
    def __init__(self, widget, iterations):
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()

    def advance(self, count=1):
        self.count += count
        self.widget.progressBarSet(int(self.count * 100 / max(1, self.iter)))

    def finish(self):
        self.widget.progressBarFinished()



##############################################################################

def tabWidget(widget):
    w = QtGui.QTabWidget(widget)
    if widget.layout() is not None:
        widget.layout().addWidget(w)
    return w


def createTabPage(tabWidget, name, widgetToAdd=None, canScroll=False):
    if widgetToAdd is None:
        widgetToAdd = widgetBox(tabWidget, addToLayout=0, margin=4)
    if canScroll:
        scrollArea = QtGui.QScrollArea()
        tabWidget.addTab(scrollArea, name)
        scrollArea.setWidget(widgetToAdd)
        scrollArea.setWidgetResizable(1)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    else:
        tabWidget.addTab(widgetToAdd, name)
    return widgetToAdd


def table(widget, rows=0, columns=0, selectionMode=-1, addToLayout=True):
    w = QtGui.QTableWidget(rows, columns, widget)
    if widget and addToLayout and widget.layout() is not None:
        widget.layout().addWidget(w)
    if selectionMode != -1:
        w.setSelectionMode(selectionMode)
    w.setHorizontalScrollMode(QtGui.QTableWidget.ScrollPerPixel)
    w.horizontalHeader().setMovable(True)
    return w


class VisibleHeaderSectionContextEventFilter(QtCore.QObject):
    def __init__(self, parent, itemView=None):
        super().__init__(parent)
        self.itemView = itemView

    def eventFilter(self, view, event):
        if not isinstance(event, QtGui.QContextMenuEvent):
            return False

        model = view.model()
        headers = [(view.isSectionHidden(i),
                    model.headerData(i, view.orientation(), Qt.DisplayRole)
                    ) for i in range(view.count())]
        menu = QtGui.QMenu("Visible headers", view)

        for i, (checked, name) in enumerate(headers):
            action = QtGui.QAction(name, menu)
            action.setCheckable(True)
            action.setChecked(not checked)
            menu.addAction(action)

            def toogleHidden(b, section=i):
                view.setSectionHidden(section, not b)
                if not b:
                    return
                if self.itemView:
                    self.itemView.resizeColumnToContents(section)
                else:
                    view.resizeSection(section,
                                       max(view.sectionSizeHint(section), 10))

            action.toggled.connect(toogleHidden)
        menu.exec_(event.globalPos())
        return True


def checkButtonOffsetHint(button, style=None):
    option = QtGui.QStyleOptionButton()
    option.initFrom(button)
    if style is None:
        style = button.style()
    if isinstance(button, QtGui.QCheckBox):
        pm_spacing = QtGui.QStyle.PM_CheckBoxLabelSpacing
        pm_indicator_width = QtGui.QStyle.PM_IndicatorWidth
    else:
        pm_spacing = QtGui.QStyle.PM_RadioButtonLabelSpacing
        pm_indicator_width = QtGui.QStyle.PM_ExclusiveIndicatorWidth
    space = style.pixelMetric(pm_spacing, option, button)
    width = style.pixelMetric(pm_indicator_width, option, button)
    # TODO: add other styles (Maybe load corrections from .cfg file?)
    style_correction = {"macintosh (aqua)": -2, "macintosh(aqua)": -2,
                        "plastique": 1, "cde": 1, "motif": 1}
    return space + width + \
        style_correction.get(QtGui.qApp.style().objectName().lower(), 0)


def toolButtonSizeHint(button=None, style=None):
    if button is None and style is None:
        style = QtGui.qApp.style()
    elif style is None:
        style = button.style()

    button_size = \
        style.pixelMetric(QtGui.QStyle.PM_SmallIconSize) + \
        style.pixelMetric(QtGui.QStyle.PM_ButtonMargin)
    return button_size


class FloatSlider(QtGui.QSlider):
    valueChangedFloat = Signal(float)

    def __init__(self, orientation, min_value, max_value, step, parent=None):
        super().__init__(orientation, parent)
        self.setScale(min_value, max_value, step)
        self.valueChanged[int].connect(self.sendValue)

    def update(self):
        self.setSingleStep(1)
        if self.min_value != self.max_value:
            self.setEnabled(True)
            self.setMinimum(int(self.min_value / self.step))
            self.setMaximum(int(self.max_value / self.step))
        else:
            self.setEnabled(False)

    def sendValue(self, slider_value):
        value = min(max(slider_value * self.step, self.min_value),
                    self.max_value)
        self.valueChangedFloat.emit(value)

    def setValue(self, value):
        super().setValue(value // self.step)

    def setScale(self, minValue, maxValue, step=0):
        if minValue >= maxValue:
            ## It would be more logical to disable the slider in this case
            ## (self.setEnabled(False))
            ## However, we do nothing to keep consistency with Qwt
            # TODO If it's related to Qwt, remove it
            return
        if step <= 0 or step > (maxValue - minValue):
            if isinstance(maxValue, int) and isinstance(minValue, int):
                step = 1
            else:
                step = float(minValue - maxValue) / 100.0
        self.min_value = float(minValue)
        self.max_value = float(maxValue)
        self.step = step
        self.update()

    def setRange(self, minValue, maxValue, step=1.0):
        # For compatibility with qwtSlider
        # TODO If it's related to Qwt, remove it
        self.setScale(minValue, maxValue, step)

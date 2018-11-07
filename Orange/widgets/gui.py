"""
Wrappers for controls used in widgets
"""
import contextlib
import math
import os
import re
import itertools
import warnings
import logging
import weakref
from types import LambdaType
from collections import defaultdict, Sequence

import pkg_resources

from AnyQt import QtWidgets, QtCore, QtGui
# pylint: disable=unused-import
from AnyQt.QtCore import (
    Qt, QObject, QEvent, QSize, QItemSelection, QTimer, pyqtSignal as Signal
)
from AnyQt.QtGui import QCursor, QColor
from AnyQt.QtWidgets import (
    QApplication, QStyle, QSizePolicy, QWidget, QLabel, QGroupBox, QSlider,
    QComboBox, QTableWidgetItem, QItemDelegate, QStyledItemDelegate,
    QTableView, QHeaderView, QListView
)

try:
    # Some Orange widgets might expect this here
    from Orange.widgets.utils.webview import WebviewWidget  # pylint: disable=unused-import
except ImportError:
    pass  # Neither WebKit nor WebEngine are available

import Orange.data
from Orange.widgets.utils import getdeepattr
from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, DiscreteVariable, Variable
from Orange.widgets.utils import vartype
from Orange.widgets.utils.buttons import VariableTextPushButton
from Orange.util import namegen

YesNo = NoYes = ("No", "Yes")
_enter_icon = None
__re_label = re.compile(r"(^|[^%])%\((?P<value>[a-zA-Z]\w*)\)")

log = logging.getLogger(__name__)

OrangeUserRole = itertools.count(Qt.UserRole)

LAMBDA_NAME = namegen('_lambda_')


class TableView(QTableView):
    """An auxilliary table view for use with PyTableModel in control areas"""
    def __init__(self, parent=None, **kwargs):
        kwargs = dict(
            dict(showGrid=False,
                 sortingEnabled=True,
                 cornerButtonEnabled=False,
                 alternatingRowColors=True,
                 selectionBehavior=self.SelectRows,
                 selectionMode=self.ExtendedSelection,
                 horizontalScrollMode=self.ScrollPerPixel,
                 verticalScrollMode=self.ScrollPerPixel,
                 editTriggers=self.DoubleClicked | self.EditKeyPressed),
            **kwargs)
        super().__init__(parent, **kwargs)
        h = self.horizontalHeader()
        h.setCascadingSectionResizes(True)
        h.setMinimumSectionSize(-1)
        h.setStretchLastSection(True)
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        v = self.verticalHeader()
        v.setVisible(False)
        v.setSectionResizeMode(QHeaderView.ResizeToContents)

    class BoldFontDelegate(QStyledItemDelegate):
        """Paints the text of associated cells in bold font.

        Can be used e.g. with QTableView.setItemDelegateForColumn() to make
        certain table columns bold, or if callback is provided, the item's
        model index is passed to it, and the item is made bold only if the
        callback returns true.

        Parameters
        ----------
        parent: QObject
            The parent QObject.
        callback: callable
            Accepts model index and returns True if the item is to be
            rendered in bold font.
        """
        def __init__(self, parent=None, callback=None):
            super().__init__(parent)
            self._callback = callback

        def paint(self, painter, option, index):
            """Paint item text in bold font"""
            if not callable(self._callback) or self._callback(index):
                option.font.setWeight(option.font.Bold)
            super().paint(painter, option, index)

        def sizeHint(self, option, index):
            """Ensure item size accounts for bold font width"""
            if not callable(self._callback) or self._callback(index):
                option.font.setWeight(option.font.Bold)
            return super().sizeHint(option, index)


def resource_filename(path):
    """
    Return a resource filename (package data) for path.
    """
    return pkg_resources.resource_filename(__name__, path)


class OWComponent:
    """
    Mixin for classes that contain settings and/or attributes that trigger
    callbacks when changed.

    The class initializes the settings handler, provides `__setattr__` that
    triggers callbacks, and provides `control` attribute for access to
    Qt widgets controling particular attributes.

    Callbacks are exploited by controls (e.g. check boxes, line edits,
    combo boxes...) that are synchronized with attribute values. Changing
    the value of the attribute triggers a call to a function that updates
    the Qt widget accordingly.

    The class is mixed into `widget.OWWidget`, and must also be mixed into
    all widgets not derived from `widget.OWWidget` that contain settings or
    Qt widgets inserted by function in `Orange.widgets.gui` module. See
    `OWScatterPlotGraph` for an example.
    """
    def __init__(self, widget=None):
        self.controlled_attributes = defaultdict(list)
        self.controls = ControlGetter(self)
        if widget is not None and widget.settingsHandler:
            widget.settingsHandler.initialize(self)

    def connect_control(self, name, func):
        """
        Add `func` to the list of functions called when the value of the
        attribute `name` is set.

        If the name includes a dot, it is assumed that the part the before the
        first dot is a name of an attribute containing an instance of a
        component, and the call is transferred to its `conntect_control`. For
        instance, `calling `obj.connect_control("graph.attr_x", f)` is
        equivalent to `obj.graph.connect_control("attr_x", f)`.

        Args:
            name (str): attribute name
            func (callable): callback function
        """
        if "." in name:
            name, rest = name.split(".", 1)
            sub = getattr(self, name)
            sub.connect_control(rest, func)
        else:
            self.controlled_attributes[name].append(func)

    def __setattr__(self, name, value):
        """Set the attribute value and trigger any attached callbacks.

        For backward compatibility, the name can include dots, e.g.
        `graph.attr_x`. `obj.__setattr__('x.y', v)` is equivalent to
        `obj.x.__setattr__('x', v)`.

        Args:
            name (str): attribute name
            value (object): value to set to the member.
        """
        if "." in name:
            name, rest = name.split(".", 1)
            sub = getattr(self, name)
            setattr(sub, rest, value)
        else:
            super().__setattr__(name, value)
            # First check that the widget is not just being constructed
            if hasattr(self, "controlled_attributes"):
                for callback in self.controlled_attributes.get(name, ()):
                    callback(value)


def miscellanea(control, box, parent,
                addToLayout=True, stretch=0, sizePolicy=None, addSpace=False,
                disabled=False, tooltip=None, **kwargs):
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

    Unused keyword arguments are assumed to be properties; with this `gui`
    function mimic the behaviour of PyQt's constructors. For instance, if
    `gui.lineEdit` is called with keyword argument `sizePolicy=some_policy`,
    `miscallenea` will call `control.setSizePolicy(some_policy)`.

    :param control: the control, e.g. a `QCheckBox`
    :type control: QWidget
    :param box: the box into which the widget was inserted
    :type box: QWidget or None
    :param parent: the parent into whose layout the box or the control will be
        inserted
    :type parent: QWidget
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
    :type sizePolicy: QSizePolicy
    """
    for prop, val in kwargs.items():
        if prop == "sizePolicy":
            control.setSizePolicy(QSizePolicy(*val))
        else:
            getattr(control, "set" + prop[0].upper() + prop[1:])(val)
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
            isinstance(control, QtWidgets.QWidget) and \
            box.layout().indexOf(control) == -1:
        box.layout().addWidget(control)
    if sizePolicy is not None:
        if isinstance(sizePolicy, tuple):
            sizePolicy = QSizePolicy(*sizePolicy)
        (box or control).setSizePolicy(sizePolicy)
    if addToLayout and parent and parent.layout() is not None:
        parent.layout().addWidget(box or control, stretch)
        _addSpace(parent, addSpace)


def _is_horizontal(orientation):
    if isinstance(orientation, str):
        warnings.warn("string literals for orientation are deprecated",
                      DeprecationWarning)
    elif isinstance(orientation, bool):
        warnings.warn("boolean values for orientation are deprecated",
                      DeprecationWarning)
    return (orientation == Qt.Horizontal or
            orientation == 'horizontal' or
            not orientation)


def setLayout(widget, layout):
    """
    Set the layout of the widget.

    If `layout` is given as `Qt.Vertical` or `Qt.Horizontal`, the function
    sets the layout to :obj:`~QVBoxLayout` or :obj:`~QVBoxLayout`.

    :param widget: the widget for which the layout is being set
    :type widget: QWidget
    :param layout: layout
    :type layout: `Qt.Horizontal`, `Qt.Vertical` or instance of `QLayout`
    """
    if not isinstance(layout, QtWidgets.QLayout):
        if _is_horizontal(layout):
            layout = QtWidgets.QHBoxLayout()
        else:
            layout = QtWidgets.QVBoxLayout()
    widget.setLayout(layout)


def _addSpace(widget, space):
    """
    A helper function that adds space into the widget, if requested.
    The function is called by functions that have the `addSpace` argument.

    :param widget: Widget into which to insert the space
    :type widget: QWidget
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
    :type widget: QWidget
    :param width: width of the separator
    :type width: int
    :param height: height of the separator
    :type height: int
    :return: separator
    :rtype: QWidget
    """
    sep = QtWidgets.QWidget(widget)
    if widget is not None and widget.layout() is not None:
        widget.layout().addWidget(sep)
    sep.setFixedSize(width, height)
    return sep


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
        b = QtWidgets.QGroupBox(widget)
        if isinstance(box, str):
            b.setTitle(" " + box.strip() + " ")
        if margin is None:
            margin = 7
    else:
        b = QtWidgets.QWidget(widget)
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
    lbl = QtWidgets.QLabel(label, widget)
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
    if box:
        b = hBox(widget, box, addToLayout=False)
    else:
        b = widget

    lbl = QtWidgets.QLabel("", b)
    reprint = CallFrontLabel(lbl, label, master)
    for mo in __re_label.finditer(label):
        master.connect_control(mo.group("value"), reprint)
    reprint()
    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    miscellanea(lbl, b, widget, **misc)
    return lbl


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
        hasHBox = _is_horizontal(orientation)
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

    cfront, sbox.cback, sbox.cfunc = connectControl(
        master, value, callback,
        not (callback and callbackOnReturn) and
        sbox.valueChanged[(int, float)[isDouble]],
        (CallFrontSpin, CallFrontDoubleSpin)[isDouble](sbox))
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


def checkBox(widget, master, value, label, box=None,
             callback=None, getwidget=False, id_=None, labelWidth=None,
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
    cbox = QtWidgets.QCheckBox(label, b)

    if labelWidth:
        cbox.setFixedSize(labelWidth, cbox.sizeHint().height())
    cbox.setChecked(getdeepattr(master, value))

    connectControl(master, value, None, cbox.toggled[bool],
                   CallFrontCheckBox(cbox),
                   cfunc=callback and FunctionCallback(
                       master, callback, widget=cbox, getwidget=getwidget,
                       id=id_))
    if isinstance(disables, QtWidgets.QWidget):
        disables = [disables]
    cbox.disables = disables or []
    cbox.makeConsistent = Disabler(cbox, master, value)
    cbox.toggled[bool].connect(cbox.makeConsistent)
    cbox.makeConsistent(value)
    miscellanea(cbox, b, widget, **misc)
    return cbox


class LineEditWFocusOut(QtWidgets.QLineEdit):
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
        if parent is not None and parent.layout() is not None:
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
        self.selectAll()
        self.__callback_if_changed()

    def __callback_if_changed(self):
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
        self.__callback_if_changed()

    def focusInEvent(self, *e):
        self.__changed = False
        if self.focusInCallback:
            self.focusInCallback()
        return super().focusInEvent(*e)


def lineEdit(widget, master, value, label=None, labelWidth=None,
             orientation=Qt.Vertical, box=None, callback=None,
             valueType=str, validator=None, controlWidth=None,
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
        when synchronizing to `value`
    :type valueType: type
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
        ledit = QtWidgets.QLineEdit(b)
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
            master, value,
            callbackOnType and callback, ledit.textChanged[str],
            CallFrontLineEdit(ledit), fvcb=value and valueType)[1]

    miscellanea(ledit, b, widget, **misc)
    return ledit


def button(widget, master, label, callback=None, width=None, height=None,
           toggleButton=False, value="", default=False, autoDefault=True,
           buttonType=QtWidgets.QPushButton, **misc):
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
    if buttonType == QtWidgets.QPushButton:
        button.setDefault(default)
        button.setAutoDefault(autoDefault)

    if value:
        button.setChecked(getdeepattr(master, value))
        connectControl(
            master, value, None, button.toggled[bool],
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
                  buttonType=QtWidgets.QToolButton, tooltip=tooltip)


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
    icon = QtGui.QIcon()
    for size in (13, 16, 18, 20, 22, 24, 28, 32, 64):
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QtGui.QPainter()
        painter.begin(pixmap)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing |
                               painter.SmoothPixmapTransform)
        painter.setPen(background)
        painter.setBrush(background)
        margin = 1 + size // 16
        text_margin = size // 20
        rect = QtCore.QRectF(margin, margin,
                             size - 2 * margin, size - 2 * margin)
        painter.drawRoundedRect(rect, 30.0, 30.0, Qt.RelativeSize)
        painter.setPen(color)
        font = painter.font()  # type: QtGui.QFont
        font.setPixelSize(size - 2 * margin - 2 * text_margin)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, char)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class __AttributeIconDict(dict):
    def __getitem__(self, key):
        if not self:
            for tpe, char, col in ((vartype(ContinuousVariable()),
                                    "N", (202, 0, 32)),
                                   (vartype(DiscreteVariable()),
                                    "C", (26, 150, 65)),
                                   (vartype(StringVariable()),
                                    "S", (0, 0, 0)),
                                   (vartype(TimeVariable()),
                                    "T", (68, 170, 255)),
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
    :rtype: tuple with QIcon and str
    """
    return attributeIconDict[var], var.name


class ListViewWithSizeHint(QListView):
    def __init__(self, *args, preferred_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(preferred_size, tuple):
            preferred_size = QSize(*preferred_size)
        self.preferred_size = preferred_size

    def sizeHint(self):
        return self.preferred_size if self.preferred_size is not None \
            else super().sizeHint()


def listView(widget, master, value=None, model=None, box=None, callback=None,
             sizeHint=None, **misc):
    if box:
        bg = vBox(widget, box, addToLayout=False)
    else:
        bg = widget
    view = ListViewWithSizeHint(preferred_size=sizeHint)
    view.setModel(model)
    if value is not None:
        connectControl(master, value, callback,
                       view.selectionModel().selectionChanged,
                       CallFrontListView(view),
                       CallBackListView(model, view, master, value))
    misc.setdefault('addSpace', True)
    misc.setdefault('uniformItemSizes', True)
    miscellanea(view, bg, widget, **misc)
    return view


def listBox(widget, master, value=None, labels=None, box=None, callback=None,
            selectionMode=QtWidgets.QListWidget.SingleSelection,
            enableDragDrop=False, dragDropCallback=None,
            dataValidityCallback=None, sizeHint=None, **misc):
    """
    Insert a list box.

    The value with which the box's value synchronizes (`master.<value>`)
    is a list of indices of selected items.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
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
    :type selectionMode: QAbstractItemView.SelectionMode
    :param enableDragDrop: flag telling whether drag and drop is available
    :type enableDragDrop: bool
    :param dragDropCallback: callback function on drop event
    :type dragDropCallback: function
    :param dataValidityCallback: function that check the validity on enter
        and move event; it should return either `ev.accept()` or `ev.ignore()`.
    :type dataValidityCallback: function
    :param sizeHint: size hint
    :type sizeHint: QSize
    :rtype: OrangeListBox
    """
    if box:
        bg = hBox(widget, box, addToLayout=False)
    else:
        bg = widget
    lb = OrangeListBox(master, enableDragDrop, dragDropCallback,
                       dataValidityCallback, sizeHint, bg)
    lb.setSelectionMode(selectionMode)
    lb.ogValue = value
    lb.ogLabels = labels
    lb.ogMaster = master

    if labels is not None:
        setattr(master, labels, getdeepattr(master, labels))
        master.connect_control(labels, CallFrontListBoxLabels(lb))
    if value is not None:
        clist = getdeepattr(master, value)
        if not isinstance(clist, (int, ControlledList)):
            clist = ControlledList(clist, lb)
            master.__setattr__(value, clist)
        setattr(master, value, clist)
        connectControl(master, value, callback, lb.itemSelectionChanged,
                       CallFrontListBox(lb), CallBackListBox(lb, master))

    misc.setdefault('addSpace', True)
    miscellanea(lb, bg, widget, **misc)
    return lb


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

    rb = QtWidgets.QButtonGroup(bg)
    if bg is not widget:
        bg.group = rb
    bg.buttons = []
    bg.ogValue = value
    bg.ogMaster = master
    for i, lab in enumerate(btnLabels):
        appendRadioButton(bg, lab, tooltip=tooltips and tooltips[i], id=i + 1)
    connectControl(master, value, callback, bg.group.buttonClicked[int],
                   CallFrontRadioButtons(bg), CallBackRadioButton(bg, master))
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
        w = QtWidgets.QRadioButton(label)
    else:
        w = QtWidgets.QRadioButton(str(i))
        w.setIcon(QtGui.QIcon(label))
    if not hasattr(group, "buttons"):
        group.buttons = []
    group.buttons.append(w)
    if id is None:
        group.group.addButton(w)
    else:
        group.group.addButton(w, id)
    w.setChecked(getdeepattr(group.ogMaster, group.ogValue) == i)

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


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1,
            callback=None, label=None, labelFormat=" %d", ticks=False,
            divideFactor=1.0, vertical=False, createLabel=True, width=None,
            intOnly=True, **misc):
    """
    Construct a slider.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
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
        of type :obj:`QSlider`) otherwise it is float
        (:obj:`FloatSlider`, derived in turn from :obj:`QSlider`).
    :type intOnly: bool
    :rtype: :obj:`QSlider` or :obj:`FloatSlider`
    """
    sliderBox = hBox(widget, box, addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    sliderOrient = Qt.Vertical if vertical else Qt.Horizontal
    if intOnly:
        slider = QSlider(sliderOrient, sliderBox)
        slider.setRange(minValue, maxValue)
        if step:
            slider.setSingleStep(step)
            slider.setPageStep(step)
            slider.setTickInterval(step)
        signal = slider.valueChanged[int]
    else:
        slider = FloatSlider(sliderOrient, minValue, maxValue, step)
        signal = slider.valueChangedFloat[float]
    sliderBox.layout().addWidget(slider)
    slider.setValue(getdeepattr(master, value))
    if width:
        slider.setFixedWidth(width)
    if ticks:
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    if createLabel:
        label = QLabel(sliderBox)
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
        signal.connect(label.setLbl)

    connectControl(master, value, callback, signal, CallFrontHSlider(slider))

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


def labeledSlider(widget, master, value, box=None,
                  label=None, labels=(), labelFormat=" %d", ticks=False,
                  callback=None, vertical=False, width=None, **misc):
    """
    Construct a slider with labels instead of numbers.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
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
    :rtype: :obj:`QSlider`
    """
    sliderBox = hBox(widget, box, addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    sliderOrient = Qt.Vertical if vertical else Qt.Horizontal
    slider = QSlider(sliderOrient, sliderBox)
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
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    max_label_size = 0
    slider.value_label = value_label = QLabel(sliderBox)
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
    slider.valueChanged[int].connect(value_label.set_label)

    connectControl(master, value, callback, slider.valueChanged[int],
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
    :type widget: QWidget or None
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
    :rtype: :obj:`QSlider`
    """
    if isinstance(labelFormat, str):
        labelFormat = lambda x, f=labelFormat: f % x

    sliderBox = hBox(widget, box, addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    slider_orient = Qt.Vertical if vertical else Qt.Horizontal
    slider = QSlider(slider_orient, sliderBox)
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
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    max_label_size = 0
    slider.value_label = value_label = QLabel(sliderBox)
    value_label.setAlignment(Qt.AlignRight)
    sliderBox.layout().addWidget(value_label)
    for lb in values:
        value_label.setText(labelFormat(lb))
        max_label_size = max(max_label_size, value_label.sizeHint().width())
    value_label.setFixedSize(max_label_size, value_label.sizeHint().height())
    value_label.setText(labelFormat(getdeepattr(master, value)))
    value_label.set_label = lambda x: value_label.setText(labelFormat(values[x]))
    slider.valueChanged[int].connect(value_label.set_label)

    connectControl(master, value, callback, slider.valueChanged[int],
                   CallFrontLabeledSlider(slider, values),
                   CallBackLabeledSlider(slider, master, values))

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


class OrangeComboBox(QtWidgets.QComboBox):
    """
    A QComboBox subclass extended to support bounded contents width hint.

    Prefer to use this class in place of plain QComboBox when the used
    model will possibly contain many items.
    """
    def __init__(self, parent=None, maximumContentsLength=-1, **kwargs):
        # Forward-declared for sizeHint()
        self.__maximumContentsLength = maximumContentsLength
        super().__init__(parent, **kwargs)

        self.__in_mousePressEvent = False
        # Yet Another Mouse Release Ignore Timer
        self.__yamrit = QTimer(self, singleShot=True)
        view = self.view()
        # optimization for displaying large models
        if isinstance(view, QListView):
            view.setUniformItemSizes(True)
            view.viewport().installEventFilter(self)

    def setMaximumContentsLength(self, length):
        """
        Set the maximum contents length hint.

        The hint specifies the upper bound on the `sizeHint` and
        `minimumSizeHint` width specified in character length.
        Set to 0 or negative value to disable.

        .. note::
             This property does not affect the widget's `maximumSize`.
             The widget can still grow depending on its `sizePolicy`.

        Parameters
        ----------
        length : int
            Maximum contents length hint.
        """
        if self.__maximumContentsLength != length:
            self.__maximumContentsLength = length
            self.updateGeometry()

    def maximumContentsLength(self):
        """
        Return the maximum contents length hint.
        """
        return self.__maximumContentsLength

    def sizeHint(self):
        # reimplemented
        sh = super().sizeHint()
        if self.__maximumContentsLength > 0:
            width = (self.fontMetrics().width("X") * self.__maximumContentsLength
                     + self.iconSize().width() + 4)
            sh = sh.boundedTo(QtCore.QSize(width, sh.height()))
        return sh

    def minimumSizeHint(self):
        # reimplemented
        sh = super().minimumSizeHint()
        if self.__maximumContentsLength > 0:
            width = (self.fontMetrics().width("X") * self.__maximumContentsLength
                     + self.iconSize().width() + 4)
            sh = sh.boundedTo(QtCore.QSize(width, sh.height()))
        return sh

    # workaround for QTBUG-67583
    def mousePressEvent(self, event):
        # reimplemented
        self.__in_mousePressEvent = True
        super().mousePressEvent(event)
        self.__in_mousePressEvent = False

    def showPopup(self):
        # reimplemented
        super().showPopup()
        if self.__in_mousePressEvent:
            self.__yamrit.start(QApplication.doubleClickInterval())

    def eventFilter(self, obj, event):
        # type: (QObject, QEvent) -> bool
        if event.type() == QEvent.MouseButtonRelease \
                and event.button() == Qt.LeftButton \
                and obj is self.view().viewport() \
                and self.__yamrit.isActive():
            return True
        else:
            return super().eventFilter(obj, event)


# TODO comboBox looks overly complicated:
# - can valueType be anything else than str?
# - sendSelectedValue is not a great name
def comboBox(widget, master, value, box=None, label=None, labelWidth=None,
             orientation=Qt.Vertical, items=(), callback=None,
             sendSelectedValue=False, valueType=str,
             emptyString=None, editable=False,
             contentsLength=None, maximumContentsLength=25,
             **misc):
    """
    Construct a combo box.

    The `value` attribute of the `master` contains either the index of the
    selected row (if `sendSelected` is left at default, `False`) or a value
    converted to `valueType` (`str` by default).

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param orientation: tells whether to put the label above or to the left
    :type orientation: `Qt.Horizontal` (default), `Qt.Vertical` or
        instance of `QLayout`
    :param label: a label that is inserted into the box
    :type label: str
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param callback: a function that is called when the value is changed
    :type callback: function
    :param items: items (optionally with data) that are put into the box
    :type items: tuple of str or tuples
    :param sendSelectedValue: flag telling whether to store/retrieve indices
        or string values from `value`
    :type sendSelectedValue: bool
    :param valueType: the type into which the selected value is converted
        if sentSelectedValue is `False`
    :type valueType: type
    :param emptyString: the string value in the combo box that gets stored as
        an empty string in `value`
    :type emptyString: str
    :param editable: a flag telling whether the combo is editable
    :type editable: bool
    :param int contentsLength: Contents character length to use as a
        fixed size hint. When not None, equivalent to::

            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(contentsLength)
    :param int maximumContentsLength: Specifies the upper bound on the
        `sizeHint` and `minimumSizeHint` width specified in character
        length (default: 25, use 0 to disable)
    :rtype: QComboBox
    """

    # Local import to avoid circular imports
    from Orange.widgets.utils.itemmodels import VariableListModel

    if box or label:
        hb = widgetBox(widget, box, orientation, addToLayout=False)
        if label is not None:
            widgetLabel(hb, label, labelWidth)
    else:
        hb = widget

    combo = OrangeComboBox(
        hb, maximumContentsLength=maximumContentsLength,
        editable=editable)

    if contentsLength is not None:
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(contentsLength)

    combo.box = hb
    for item in items:
        if isinstance(item, (tuple, list)):
            combo.addItem(*item)
        else:
            combo.addItem(str(item))

    if value:
        cindex = getdeepattr(master, value)
        model = misc.pop("model", None)
        if model is not None:
            combo.setModel(model)
        if isinstance(model, VariableListModel):
            callfront = CallFrontComboBoxModel(combo, model)
            callfront.action(cindex)
        else:
            if isinstance(cindex, str):
                if items and cindex in items:
                    cindex = items.index(cindex)
                else:
                    cindex = 0
            if cindex > combo.count() - 1:
                cindex = 0
            combo.setCurrentIndex(cindex)

        if isinstance(model, VariableListModel):
            connectControl(
                master, value, callback, combo.activated[int],
                callfront,
                ValueCallbackComboModel(master, value, model)
            )
        elif sendSelectedValue:
            connectControl(
                master, value, callback, combo.activated[str],
                CallFrontComboBox(combo, valueType, emptyString),
                ValueCallbackCombo(master, value, valueType, emptyString))
        else:
            connectControl(
                master, value, callback, combo.activated[int],
                CallFrontComboBox(combo, None, emptyString))
    miscellanea(combo, hb, widget, **misc)
    combo.emptyString = emptyString
    return combo


class OrangeListBox(QtWidgets.QListWidget):
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
        :type sizeHint: QSize
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

    def dragEnterEvent(self, event):
        super().dragEnterEvent(event)
        if self.valid_data_callback:
            self.valid_data_callback(event)
        elif isinstance(event.source(), OrangeListBox):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        event.setDropAction(Qt.MoveAction)
        super().dropEvent(event)

        items = self.update_master()
        if event.source() is not self:
            event.source().update_master(exclude=items)

        if self.drop_callback:
            self.drop_callback()

    def update_master(self, exclude=()):
        control_list = [self.item(i).data(Qt.UserRole)
                        for i in range(self.count())
                        if self.item(i).data(Qt.UserRole) not in exclude]
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
class SmallWidgetButton(QtWidgets.QPushButton):
    def __init__(self, widget, text="", pixmap=None, box=None,
                 orientation=Qt.Vertical, autoHideWidget=None, **misc):
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

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            self.autohideWidget.move(
                self.mapToGlobal(QtCore.QPoint(0, self.height())))
            self.autohideWidget.show()


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

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            self.autohideWidget.move(
                self.mapToGlobal(QtCore.QPoint(0, self.height())))
            self.autohideWidget.show()


class AutoHideWidget(QWidget):
    def leaveEvent(self, _):
        self.hide()

# creates a widget box with a button in the top right edge that shows/hides all
# widgets in the box and collapse the box to its minimum height
# TODO collapsableWidgetBox is used only in OWMosaicDisplay.py; (re)move
class collapsableWidgetBox(QGroupBox):
    def __init__(self, widget, box="", master=None, value="",
                 orientation=Qt.Vertical, callback=None):
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
            if isinstance(c, QtWidgets.QLayout):
                continue
            if val:
                c.show()
            else:
                c.hide()


# creates an icon that allows you to show/hide the widgets in the widgets list
# TODO Class widgetHider doesn't seem to be used anywhere; remove?
class widgetHider(QWidget):
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

    def mousePressEvent(self, event):
        self.master.__setattr__(self.value,
                                not getdeepattr(self.master, self.value))
        self.makeConsistent()

    def setWidgets(self, widgets):
        self.disables = list(widgets)
        self.makeConsistent()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmaps:
            pix = self.pixmaps[getdeepattr(self.master, self.value)]
            painter = QtGui.QPainter(self)
            painter.drawPixmap(0, 0, pix)


##############################################################################
# callback handlers


def auto_commit(widget, master, value, label, auto_label=None, box=True,
                checkbox_label=None, orientation=None, commit=None,
                callback=None, **misc):
    """
    Add a commit button with auto-commit check box.

    The widget must have a commit method and a setting that stores whether
    auto-commit is on.

    The function replaces the commit method with a new commit method that
    checks whether auto-commit is on. If it is, it passes the call to the
    original commit, otherwise it sets the dirty flag.

    The checkbox controls the auto-commit. When auto-commit is switched on, the
    checkbox callback checks whether the dirty flag is on and calls the original
    commit.

    Important! Do not connect any signals to the commit before calling
    auto_commit.

    :param widget: the widget into which the box with the button is inserted
    :type widget: QWidget or None
    :param value: the master's attribute which stores whether the auto-commit
        is on
    :type value:  str
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param label: The button label
    :type label: str
    :param auto_label: The label used when auto-commit is on; default is
        `label + " Automatically"`
    :type auto_label: str
    :param commit: master's method to override ('commit' by default)
    :type commit: function
    :param callback: function to call whenever the checkbox's statechanged
    :type callback: function
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :return: the box
    """
    def checkbox_toggled():
        if getattr(master, value):
            btn.setText(auto_label)
            btn.setEnabled(False)
            if dirty:
                do_commit()
        else:
            btn.setText(label)
            btn.setEnabled(True)
        if callback:
            callback()

    def unconditional_commit():
        nonlocal dirty
        if getattr(master, value):
            do_commit()
        else:
            dirty = True

    def do_commit():
        nonlocal dirty
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            commit()
            dirty = False
        finally:
            QApplication.restoreOverrideCursor()

    dirty = False
    commit = commit or getattr(master, 'commit')
    commit_name = next(LAMBDA_NAME) if isinstance(commit, LambdaType) else commit.__name__
    setattr(master, 'unconditional_' + commit_name, commit)

    if not auto_label:
        if checkbox_label:
            auto_label = label
        else:
            auto_label = label.title() + " Automatically"
    if isinstance(box, QWidget):
        b = box
    else:
        if orientation is None:
            orientation = Qt.Vertical if checkbox_label else Qt.Horizontal
        b = widgetBox(widget, box=box, orientation=orientation,
                      addToLayout=False)
        b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    b.checkbox = cb = checkBox(b, master, value, checkbox_label,
                               callback=checkbox_toggled, tooltip=auto_label)
    if _is_horizontal(orientation):
        b.layout().addSpacing(10)
    cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    b.button = btn = VariableTextPushButton(
        b, text=label, textChoiceList=[label, auto_label], clicked=do_commit)
    if b.layout() is not None:
        b.layout().addWidget(b.button)

    if not checkbox_label:
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    checkbox_toggled()
    setattr(master, commit_name, unconditional_commit)
    misc['addToLayout'] = misc.get('addToLayout', True) and \
                          not isinstance(box, QtWidgets.QWidget)
    miscellanea(b, widget, widget, **misc)
    return b


class ControlledList(list):
    """
    A class derived from a list that is connected to a
    :obj:`QListBox`: the list contains indices of items that are
    selected in the list box. Changing the list content changes the
    selection in the list box.
    """
    def __init__(self, content, listBox=None):
        super().__init__(content if content is not None else [])
        # Controlled list is created behind the back by gui.listBox and
        # commonly used as a setting which gets synced into a GLOBAL
        # SettingsHandler and which keeps the OWWidget instance alive via a
        # reference in listBox (see gui.listBox)
        if listBox is not None:
            self.listBox = weakref.ref(listBox)
        else:
            self.listBox = lambda: None

    def __reduce__(self):
        # cannot pickle self.listBox, but can't discard it
        # (ControlledList may live on)
        import copyreg
        return copyreg._reconstructor, (list, list, ()), None, self.__iter__()

    # TODO ControllgedList.item2name is probably never used
    def item2name(self, item):
        item = self.listBox().labels[item]
        if isinstance(item, tuple):
            return item[1]
        else:
            return item

    def __setitem__(self, index, item):
        def unselect(i):
            try:
                item = self.listBox().item(i)
            except RuntimeError:  # Underlying C/C++ object has been deleted
                item = None
            if item is None:
                # Labels changed before clearing the selection: clear everything
                self.listBox().selectionModel().clear()
            else:
                item.setSelected(0)

        if isinstance(index, int):
            unselect(self[index])
            item.setSelected(1)
        else:
            for i in self[index]:
                unselect(i)
            for i in item:
                self.listBox().item(i).setSelected(1)
        super().__setitem__(index, item)

    def __delitem__(self, index):
        if isinstance(index, int):
            self.listBox().item(self[index]).setSelected(0)
        else:
            for i in self[index]:
                self.listBox().item(i).setSelected(0)
        super().__delitem__(index)

    def append(self, item):
        super().append(item)
        item.setSelected(1)

    def extend(self, items):
        super().extend(items)
        for i in items:
            self.listBox().item(i).setSelected(1)

    def insert(self, index, item):
        item.setSelected(1)
        super().insert(index, item)

    def pop(self, index=-1):
        i = super().pop(index)
        self.listBox().item(i).setSelected(0)

    def remove(self, item):
        item.setSelected(0)
        super().remove(item)


def connectControl(master, value, f, signal,
                   cfront, cback=None, cfunc=None, fvcb=None):
    cback = cback or value and ValueCallback(master, value, fvcb)
    if cback:
        if signal:
            signal.connect(cback)
        cback.opposite = cfront
        if value and cfront:
            master.connect_control(value, cfront)
    cfunc = cfunc or f and FunctionCallback(master, f)
    if cfunc:
        if signal:
            signal.connect(cfunc)
        cfront.opposite = tuple(x for x in (cback, cfunc) if x)
    return cfront, cback, cfunc


@contextlib.contextmanager
def disable_opposite(obj):
    opposite = getattr(obj, "opposite", None)
    if opposite:
        opposite.disabled += 1
        try:
            yield
        finally:
            if opposite:
                opposite.disabled -= 1


class ControlledCallback:
    def __init__(self, widget, attribute, f=None):
        self.widget = widget
        self.attribute = attribute
        self.func = f
        self.disabled = 0
        if isinstance(widget, dict):
            return  # we can't assign attributes to dict
        if not hasattr(widget, "callbackDeposit"):
            widget.callbackDeposit = []
        widget.callbackDeposit.append(self)

    def acyclic_setattr(self, value):
        if self.disabled:
            return
        if self.func:
            if self.func in (int, float) and (
                    not value or isinstance(value, str) and value in "+-"):
                value = self.func(0)
            else:
                value = self.func(value)
        with disable_opposite(self):
            if isinstance(self.widget, dict):
                self.widget[self.attribute] = value
            else:
                setattr(self.widget, self.attribute, value)


class ValueCallback(ControlledCallback):
    # noinspection PyBroadException
    def __call__(self, value):
        if value is None:
            return
        self.acyclic_setattr(value)


class ValueCallbackCombo(ValueCallback):
    def __init__(self, widget, attribute, f=None, emptyString=""):
        super().__init__(widget, attribute, f)
        self.emptyString = emptyString

    def __call__(self, value):
        value = str(value)
        return super().__call__("" if value == self.emptyString else value)


class ValueCallbackComboModel(ValueCallback):
    def __init__(self, widget, attribute, model):
        super().__init__(widget, attribute)
        self.model = model

    def __call__(self, index):
        # Can't use super here since, it doesn't set `None`'s?!
        return self.acyclic_setattr(self.model[index])


class ValueCallbackLineEdit(ControlledCallback):
    def __init__(self, control, widget, attribute, f=None):
        ControlledCallback.__init__(self, widget, attribute, f)
        self.control = control

    # noinspection PyBroadException
    def __call__(self, value):
        if value is None:
            return
        pos = self.control.cursorPosition()
        self.acyclic_setattr(value)
        self.control.setCursorPosition(pos)


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
        self.func = f
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
            if isinstance(self.func, list):
                for func in self.func:
                    func(**kwds)
            else:
                self.func(**kwds)


class CallBackListView(ControlledCallback):
    def __init__(self, model, view, widget, attribute):
        super().__init__(widget, attribute)
        self.model = model
        self.view = view

    # triggered by selectionModel().selectionChanged()
    def __call__(self, *_):
        # This must be imported locally to avoid circular imports
        from Orange.widgets.utils.itemmodels import PyListModel
        values = [i.row()
                  for i in self.view.selectionModel().selection().indexes()]
        if values:
            # FIXME: irrespective of PyListModel check, this might/should always
            # callback with values!
            if isinstance(self.model, PyListModel):
                values = [self.model[i] for i in values]
            if self.view.selectionMode() == self.view.SingleSelection:
                values = values[0]
            self.acyclic_setattr(values)


class CallBackListBox:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = 0

    def __call__(self, *_):  # triggered by selectionChange()
        if not self.disabled and self.control.ogValue is not None:
            clist = getdeepattr(self.widget, self.control.ogValue)
            control = self.control
            selection = [i for i in range(control.count())
                         if control.item(i).isSelected()]
            if isinstance(clist, int):
                self.widget.__setattr__(
                    self.control.ogValue, selection[0] if selection else None)
            else:
                list.__setitem__(clist, slice(0, len(clist)), selection)
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
    def __init__(self, control, valType=None, emptyString=""):
        super().__init__(control)
        self.valType = valType
        self.emptyString = emptyString

    def action(self, value):
        if value in ('', None):
            value = self.emptyString
        if self.valType:
            for i in range(self.control.count()):
                if self.valType(self.control.itemText(i)) == value:
                    self.control.setCurrentIndex(i)
                    return
            if value:
                log.warning("Unable to set %s to '%s'. Possible values are: %s",
                            self.control, value,
                            ', '.join(self.control.itemText(i)
                                      for i in range(self.control.count())))
        else:
            if value < self.control.count():
                self.control.setCurrentIndex(value)


class CallFrontComboBoxModel(ControlledCallFront):
    def __init__(self, control, model):
        super().__init__(control)
        self.model = model

    def action(self, value):
        if value == "":  # the latter accomodates PyListModel
            value = None
        if value is None and None not in self.model:
            return  # e.g. attribute x in uninitialized scatter plot
        if value in self.model:
            self.control.setCurrentIndex(self.model.indexOf(value))
            return
        elif isinstance(value, str):
            for i, val in enumerate(self.model):
                if value == str(val):
                    self.control.setCurrentIndex(i)
                    return
        raise ValueError("Combo box does not contain item " + repr(value))


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


class CallFrontListView(ControlledCallFront):
    def action(self, values):
        view = self.control
        model = view.model()
        sel_model = view.selectionModel()

        if not isinstance(values, Sequence):
            values = [values]

        selection = QItemSelection()
        for value in values:
            index = None
            if not isinstance(value, int):
                if isinstance(value, Variable):
                    search_role = TableVariable
                else:
                    search_role = Qt.DisplayRole
                    value = str(value)
                for i in range(model.rowCount()):
                    if model.data(model.index(i), search_role) == value:
                        index = i
                        break
            else:
                index = value
            if index is not None:
                selection.select(model.index(index), model.index(index))
        sel_model.select(selection, sel_model.ClearAndSelect)


class CallFrontListBox(ControlledCallFront):
    def action(self, value):
        if value is not None:
            if isinstance(value, int):
                for i in range(self.control.count()):
                    self.control.item(i).setSelected(i == value)
            else:
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
                        item = QtWidgets.QListWidgetItem(attributeIconDict[icon], text)
                    else:
                        item = QtWidgets.QListWidgetItem(icon, text)
                elif isinstance(value, Variable):
                    item = QtWidgets.QListWidgetItem(*attributeItem(value))
                else:
                    item = QtWidgets.QListWidgetItem(value)

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
            disabled = True
        for w in self.widget.disables:
            if isinstance(w, tuple):
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
class tableItem(QTableWidgetItem):
    def __init__(self, table, x, y, text, editType=None, backColor=None,
                 icon=None, type=QTableWidgetItem.Type):
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


class TableBarItem(QItemDelegate):
    BarRole = next(OrangeUserRole)
    BarColorRole = next(OrangeUserRole)

    def __init__(self, parent=None, color=QtGui.QColor(255, 170, 127),
                 color_schema=None):
        """
        :param QObject parent: Parent object.
        :param QColor color: Default color of the distribution bar.
        :param color_schema:
            If not None it must be an instance of
            :class:`OWColorPalette.ColorPaletteGenerator` (note: this
            parameter, if set, overrides the ``color``)
        :type color_schema: :class:`OWColorPalette.ColorPaletteGenerator`
        """
        super().__init__(parent)
        self.color = color
        self.color_schema = color_schema

    def paint(self, painter, option, index):
        painter.save()
        self.drawBackground(painter, option, index)
        ratio = index.data(TableBarItem.BarRole)
        if isinstance(ratio, float):
            if math.isnan(ratio):
                ratio = None

        color = None
        if ratio is not None:
            if self.color_schema is not None:
                class_ = index.data(TableClassValueRole)
                if isinstance(class_, Orange.data.Value) and \
                        class_.variable.is_discrete and \
                        not math.isnan(class_):
                    color = self.color_schema[int(class_)]
            else:
                color = index.data(self.BarColorRole)
        if color is None:
            color = self.color
        rect = option.rect
        if ratio is not None:
            pw = 5
            hmargin = 3 + pw / 2  # + half pen width for the round line cap
            vmargin = 1
            textoffset = pw + vmargin * 2
            baseline = rect.bottom() - textoffset / 2
            width = (rect.width() - 2 * hmargin) * ratio
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setPen(QtGui.QPen(QtGui.QBrush(color), pw,
                                      Qt.SolidLine, Qt.RoundCap))
            line = QtCore.QLineF(
                rect.left() + hmargin, baseline,
                rect.left() + hmargin + width, baseline
            )
            painter.drawLine(line)
            painter.restore()
            text_rect = rect.adjusted(0, 0, 0, -textoffset)
        else:
            text_rect = rect
        text = str(index.data(Qt.DisplayRole))
        self.drawDisplay(painter, option, text_rect, text)
        painter.restore()


class BarItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent, brush=QtGui.QBrush(QtGui.QColor(255, 170, 127)),
                 scale=(0.0, 1.0)):
        super().__init__(parent)
        self.brush = brush
        self.scale = scale

    def paint(self, painter, option, index):
        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter,
            option.widget)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter,
            option.widget)

        rect = option.rect
        val = index.data(Qt.DisplayRole)
        if isinstance(val, float):
            minv, maxv = self.scale
            val = (val - minv) / (maxv - minv)
            painter.save()
            if option.state & QStyle.State_Selected:
                painter.setOpacity(0.75)
            painter.setBrush(self.brush)
            painter.drawRect(
                rect.adjusted(1, 1, - rect.width() * (1.0 - val) - 2, -2))
            painter.restore()


class IndicatorItemDelegate(QtWidgets.QStyledItemDelegate):
    IndicatorRole = next(OrangeUserRole)

    def __init__(self, parent, role=IndicatorRole, indicatorSize=2):
        super().__init__(parent)
        self.role = role
        self.indicatorSize = indicatorSize

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        rect = option.rect
        indicator = index.data(self.role)

        if indicator:
            painter.save()
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(Qt.black))
            painter.drawEllipse(rect.center(),
                                self.indicatorSize, self.indicatorSize)
            painter.restore()


class LinkStyledItemDelegate(QStyledItemDelegate):
    LinkRole = next(OrangeUserRole)

    def __init__(self, parent):
        super().__init__(parent)
        self.mousePressState = QtCore.QModelIndex(), QtCore.QPoint()
        parent.entered.connect(self.onEntered)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QtCore.QSize(size.width(), max(size.height(), 20))

    def linkRect(self, option, index):
        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        text = self.displayText(index.data(Qt.DisplayRole),
                                QtCore.QLocale.system())
        self.initStyleOption(option, index)
        textRect = style.subElementRect(
            QStyle.SE_ItemViewItemText, option, option.widget)

        if not textRect.isValid():
            textRect = option.rect
        margin = style.pixelMetric(
            QStyle.PM_FocusFrameHMargin, option, option.widget) + 1
        textRect = textRect.adjusted(margin, 0, -margin, 0)
        font = index.data(Qt.FontRole)
        if not isinstance(font, QtGui.QFont):
            font = option.font

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
            if not isinstance(link, str):
                link = None

            pressedIndex, pressPos = self.mousePressState
            if pressedIndex == index and \
                    (pressPos - event.pos()).manhattanLength() < 5 and \
                    link is not None:
                import webbrowser
                webbrowser.open(link)
            self.mousePressState = QtCore.QModelIndex(), event.pos()

        elif event.type() == QtCore.QEvent.MouseMove:
            link = index.data(LinkRole)
            if not isinstance(link, str):
                link = None

            if link is not None and \
                    self.linkRect(option, index).contains(event.pos()):
                self.parent().viewport().setCursor(Qt.PointingHandCursor)
            else:
                self.parent().viewport().setCursor(Qt.ArrowCursor)

        return super().editorEvent(event, model, option, index)

    def onEntered(self, index):
        link = index.data(LinkRole)
        if not isinstance(link, str):
            link = None
        if link is None:
            self.parent().viewport().setCursor(Qt.ArrowCursor)

    def paint(self, painter, option, index):
        link = index.data(LinkRole)
        if not isinstance(link, str):
            link = None

        if link is not None:
            if option.widget is not None:
                style = option.widget.style()
            else:
                style = QApplication.style()
            style.drawPrimitive(
                QStyle.PE_PanelItemViewRow, option, painter,
                option.widget)
            style.drawPrimitive(
                QStyle.PE_PanelItemViewItem, option, painter,
                option.widget)

            text = self.displayText(index.data(Qt.DisplayRole),
                                    QtCore.QLocale.system())
            textRect = style.subElementRect(
                QStyle.SE_ItemViewItemText, option, option.widget)
            if not textRect.isValid():
                textRect = option.rect
            margin = style.pixelMetric(
                QStyle.PM_FocusFrameHMargin, option, option.widget) + 1
            textRect = textRect.adjusted(margin, 0, -margin, 0)
            elideText = QtGui.QFontMetrics(option.font).elidedText(
                text, option.textElideMode, textRect.width())
            painter.save()
            font = index.data(Qt.FontRole)
            if not isinstance(font, QtGui.QFont):
                font = option.font
            painter.setFont(font)
            if option.state & QStyle.State_Selected:
                color = option.palette.highlightedText().color()
            else:
                color = option.palette.link().color()
            painter.setPen(QtGui.QPen(color))
            painter.drawText(textRect, option.displayAlignment, elideText)
            painter.restore()
        else:
            super().paint(painter, option, index)


LinkRole = LinkStyledItemDelegate.LinkRole


class ColoredBarItemDelegate(QtWidgets.QStyledItemDelegate):
    """ Item delegate that can also draws a distribution bar
    """
    def __init__(self, parent=None, decimals=3, color=Qt.red):
        super().__init__(parent)
        self.decimals = decimals
        self.float_fmt = "%%.%if" % decimals
        self.color = QtGui.QColor(color)

    def displayText(self, value, locale=QtCore.QLocale()):
        if value is None or isinstance(value, float) and math.isnan(value):
            return "NA"
        if isinstance(value, float):
            return self.float_fmt % value
        return str(value)

    def sizeHint(self, option, index):
        font = self.get_font(option, index)
        metrics = QtGui.QFontMetrics(font)
        height = metrics.lineSpacing() + 8  # 4 pixel margin
        width = metrics.width(self.displayText(index.data(Qt.DisplayRole),
                                               QtCore.QLocale())) + 8
        return QtCore.QSize(width, height)

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        text = self.displayText(index.data(Qt.DisplayRole))
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

        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter,
            option.widget)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter,
            option.widget)

        # TODO: Check ForegroundRole.
        if option.state & QStyle.State_Selected:
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


class HorizontalGridDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        painter.setPen(QColor(212, 212, 212))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        painter.restore()
        QStyledItemDelegate.paint(self, painter, option, index)


class VerticalLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setMaximumWidth(self.sizeHint().width() + 2)
        self.setMargin(4)

    def sizeHint(self):
        metrics = QtGui.QFontMetrics(self.font())
        rect = metrics.boundingRect(self.text())
        size = QtCore.QSize(rect.height() + self.margin(),
                            rect.width() + self.margin())
        return size

    def setGeometry(self, rect):
        super().setGeometry(rect)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.geometry()
        text_rect = QtCore.QRect(0, 0, rect.width(), rect.height())

        painter.translate(text_rect.bottomLeft())
        painter.rotate(-90)
        painter.drawText(
            QtCore.QRect(QtCore.QPoint(0, 0),
                         QtCore.QSize(rect.height(), rect.width())),
            Qt.AlignCenter, self.text())
        painter.end()


class VerticalItemDelegate(QStyledItemDelegate):
    # Extra text top/bottom margin.
    Margin = 6

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        return QtCore.QSize(sh.height() + self.Margin * 2, sh.width())

    def paint(self, painter, option, index):
        option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(option, index)

        if not option.text:
            return

        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()
        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter,
            option.widget)
        cell_rect = option.rect
        itemrect = QtCore.QRect(0, 0, cell_rect.height(), cell_rect.width())
        opt = QtWidgets.QStyleOptionViewItem(option)
        opt.rect = itemrect
        textrect = style.subElementRect(
            QStyle.SE_ItemViewItemText, opt, opt.widget)

        painter.save()
        painter.setFont(option.font)

        if option.displayAlignment & (Qt.AlignTop | Qt.AlignBottom):
            brect = painter.boundingRect(
                textrect, option.displayAlignment, option.text)
            diff = textrect.height() - brect.height()
            offset = max(min(diff / 2, self.Margin), 0)
            if option.displayAlignment & Qt.AlignBottom:
                offset = -offset

            textrect.translate(0, offset)

        painter.translate(option.rect.x(), option.rect.bottom())
        painter.rotate(-90)
        painter.drawText(textrect, option.displayAlignment, option.text)
        painter.restore()

##############################################################################
# progress bar management


class ProgressBar:
    def __init__(self, widget, iterations):
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()
        self.finished = False

    def __del__(self):
        if not self.finished:
            self.widget.progressBarFinished(processEvents=False)

    def advance(self, count=1):
        self.count += count
        self.widget.progressBarSet(int(self.count * 100 / max(1, self.iter)))

    def finish(self):
        self.finished = True
        self.widget.progressBarFinished()


##############################################################################

def tabWidget(widget):
    w = QtWidgets.QTabWidget(widget)
    if widget.layout() is not None:
        widget.layout().addWidget(w)
    return w


def createTabPage(tab_widget, name, widgetToAdd=None, canScroll=False):
    if widgetToAdd is None:
        widgetToAdd = vBox(tab_widget, addToLayout=0, margin=4)
    if canScroll:
        scrollArea = QtWidgets.QScrollArea()
        tab_widget.addTab(scrollArea, name)
        scrollArea.setWidget(widgetToAdd)
        scrollArea.setWidgetResizable(1)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    else:
        tab_widget.addTab(widgetToAdd, name)
    return widgetToAdd


def table(widget, rows=0, columns=0, selectionMode=-1, addToLayout=True):
    w = QtWidgets.QTableWidget(rows, columns, widget)
    if widget and addToLayout and widget.layout() is not None:
        widget.layout().addWidget(w)
    if selectionMode != -1:
        w.setSelectionMode(selectionMode)
    w.setHorizontalScrollMode(QtWidgets.QTableWidget.ScrollPerPixel)
    w.horizontalHeader().setSectionsMovable(True)
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
                    model.headerData(i, view.orientation(), Qt.DisplayRole))
                   for i in range(view.count())]
        menu = QtWidgets.QMenu("Visible headers", view)

        for i, (checked, name) in enumerate(headers):
            action = QtWidgets.QAction(name, menu)
            action.setCheckable(True)
            action.setChecked(not checked)
            menu.addAction(action)

            def toogleHidden(visible, section=i):
                view.setSectionHidden(section, not visible)
                if not visible:
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
    option = QtWidgets.QStyleOptionButton()
    option.initFrom(button)
    if style is None:
        style = button.style()
    if isinstance(button, QtWidgets.QCheckBox):
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


def toolButtonSizeHint(button=None, style=None):
    if button is None and style is None:
        style = QApplication.style()
    elif style is None:
        style = button.style()

    button_size = \
        style.pixelMetric(QStyle.PM_SmallIconSize) + \
        style.pixelMetric(QStyle.PM_ButtonMargin)
    return button_size


class FloatSlider(QSlider):
    """
    Slider for continuous values.

    The slider is derived from `QtGui.QSlider`, but maps from its discrete
    numbers to the desired continuous interval.
    """
    valueChangedFloat = Signal(float)

    def __init__(self, orientation, min_value, max_value, step, parent=None):
        super().__init__(orientation, parent)
        self.setScale(min_value, max_value, step)
        self.valueChanged[int].connect(self._send_value)

    def _update(self):
        self.setSingleStep(1)
        if self.min_value != self.max_value:
            self.setEnabled(True)
            self.setMinimum(int(round(self.min_value / self.step)))
            self.setMaximum(int(round(self.max_value / self.step)))
        else:
            self.setEnabled(False)

    def _send_value(self, slider_value):
        value = min(max(slider_value * self.step, self.min_value),
                    self.max_value)
        self.valueChangedFloat.emit(value)

    def setValue(self, value):
        """
        Set current value. The value is divided by `step`

        Args:
            value: new value
        """
        super().setValue(int(round(value / self.step)))

    def setScale(self, minValue, maxValue, step=0):
        """
        Set slider's ranges (compatibility with qwtSlider).

        Args:
            minValue (float): minimal value
            maxValue (float): maximal value
            step (float): step
        """
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
        self._update()

    def setRange(self, minValue, maxValue, step=1.0):
        """
        Set slider's ranges (compatibility with qwtSlider).

        Args:
            minValue (float): minimal value
            maxValue (float): maximal value
            step (float): step
        """
        # For compatibility with qwtSlider
        # TODO If it's related to Qwt, remove it
        self.setScale(minValue, maxValue, step)


class ControlGetter:
    """
    Provide access to GUI elements based on their corresponding attributes
    in widget.

    Every widget has an attribute `controls` that is an instance of this
    class, which uses the `controlled_attributes` dictionary to retrieve the
    control (e.g. `QCheckBox`, `QComboBox`...) corresponding to the attribute.
    For `OWComponents`, it returns its controls so that subsequent
    `__getattr__` will retrieve the control.
    """
    def __init__(self, widget):
        self.widget = widget

    def __getattr__(self, name):
        widget = self.widget
        callfronts = widget.controlled_attributes.get(name, None)
        if callfronts is None:
            # This must be an OWComponent
            try:
                return getattr(widget, name).controls
            except AttributeError:
                raise AttributeError(
                    "'{}' is not an attribute related to a gui element or "
                    "component".format(name))
        else:
            return callfronts[0].control

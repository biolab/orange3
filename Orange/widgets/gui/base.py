"""
Wrappers for controls used in widgets
"""
import itertools
import logging
import warnings

import pkg_resources
from AnyQt import QtWidgets
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QSizePolicy, QWidget, QLayout, QHBoxLayout, QVBoxLayout)

try:
    # Some Orange widgets might expect this here
    # pylint: disable=unused-import
    from Orange.widgets.utils.webview import WebviewWidget
except ImportError:
    pass  # Neither WebKit nor WebEngine are available

__all__ = ["OrangeUserRole", "resource_filename", "separator"]

log = logging.getLogger(__name__)

OrangeUserRole = itertools.count(Qt.UserRole)

def resource_filename(path):
    """
    Return a resource filename (package data) for path.
    """
    return pkg_resources.resource_filename("Orange.widgets", path)


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
            isinstance(control, QWidget) and \
            box.layout().indexOf(control) == -1:
        box.layout().addWidget(control)
    if sizePolicy is not None:
        if isinstance(sizePolicy, tuple):
            sizePolicy = QSizePolicy(*sizePolicy)
        (box or control).setSizePolicy(sizePolicy)
    if addToLayout and parent and parent.layout() is not None:
        parent.layout().addWidget(box or control, stretch)
        _addSpace(parent, addSpace)


def is_horizontal(orientation):
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
    if not isinstance(layout, QLayout):
        if is_horizontal(layout):
            layout = QHBoxLayout()
        else:
            layout = QVBoxLayout()
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
        # distinguish between int and bool
        if type(space) == int:  # pylint: disable=unidiomatic-typecheck
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
    if widget.layout() is not None:
        widget.layout().addWidget(sep)
    sep.setFixedSize(width, height)
    return sep

.. currentmodule:: Orange.widgets.gui

###################################
Library of Common GUI Controls
###################################

:obj:`gui` is a library of functions which allow constructing a control (like
check box, line edit or a combo), inserting it into the parent's layout,
setting tooltips, callbacks and so forth, establishing synchronization with a
Python object's attribute (including saving and retrieving when the widgets is
closed and reopened) ... in a single call.

Almost all functions need three arguments:

* the `widget` into which the control is inserted,
* the `master` widget with one whose attributes the control's value is
  synchronized,
* the name of that attribute (`value`).

All other arguments should be given as keyword arguments for clarity and also
for allowing the potential future compatibility-breaking changes in the module.
Several arguments that are common to all functions must always be given as
keyword arguments.


**************
Common options
**************

All controls accept the following arguments that can only be given as
keyword arguments.

:tooltip:
    A string for a tooltip that appears when mouse is over the control.

:disabled:
    Tells whether the control be disabled upon the initialization.

:addSpace:
    Gives the amount of space that is inserted after the control (or the box
    that encloses it). If `True`, a space of 8 pixels is inserted. Default is
    0.

:addToLayout:

    The control is added to the parent's layout unless this flag is set to
    `False`.

:stretch:

    The stretch factor for this widget, used when adding to the layout.
    Default is 0.

:sizePolicy:

    The size policy for the box or the control.


****************
Common Arguments
****************

Many functions share common arguments.

:widget:
    Widget on which control will be drawn.

:master:
    Object which includes the attribute that is used to store the control's
    value; most often the `self` of the widget into which the control is
    inserted.

:value:
    String with the name of the master's attribute that synchronizes with the
    the control's value..

:box:
    Indicates if there should be a box that around the control. If :obj:`box`
    is `False` (default), no box is drawn; if it is a string, it is also used
    as the label for the box's name; if :obj:`box` is any other true value
    (such as :obj:`True` :), an unlabeled box is drawn.

:callback:
    A function that is called when the state of the control is changed. This
    can be a single function or a list of functions that will be called in the
    given order. The callback function should not change the value of the
    controlled attribute (the one given as the :obj:`value` argument described
    above) to avoid a cycle (a workaround is shown in the description of
    :obj:`listBox` function.

:label:
    A string that is displayed as control's label.

:labelWidth:
    The width of the label. This is useful for aligning the controls.

:orientation:
    When the label is given used, this argument determines the relative
    placement of the label and the control. Label can be above the control,
    (`"vertical"` or `True` - this is the default) or in the same line with
    control, (`"horizontal"` or `False`). Orientation can also be an instance
    of :obj:`~PyQt4.QtGui.QLayout`.


*****************
Common Attributes
*****************

:box:
    If the constructed widget is enclosed into a box, the attribute `box`
    refers to the box.

*******
Widgets
*******

This section describes the wrappers for controls like check boxes, buttons
and similar. Using them is preferred over calling Qt directly, for convenience,
readability, ease of porting to newer versions of Qt and, in particular,
because they set up a lot of things that happen in behind.

.. autofunction:: checkBox
.. autofunction:: lineEdit
.. autofunction:: listBox
.. autofunction:: comboBox
.. autofunction:: radioButtonsInBox
.. autofunction:: appendRadioButton
.. autofunction:: spin
.. autofunction:: doubleSpin
.. autofunction:: hSlider
.. autofunction:: button
.. autofunction:: toolButton
.. autofunction:: widgetLabel
.. autofunction:: label

*************
Other widgets
*************

.. autofunction:: widgetBox
.. autofunction:: indentedBox
.. autofunction:: separator
.. autofunction:: rubber

*****************
Utility functions
*****************

.. autodata:: attributeIconDict
.. autofunction:: attributeItem

.. autofunction:: setStopper

******************************
Internal functions and classes
******************************

This part of documentation describes some classes and functions that are used
internally. The classes will likely maintain compatibility in the future,
while the functions may be changed.

Wrappers for Qt classes
=======================

.. autoclass:: SpinBoxWFocusOut
.. autoclass:: DoubleSpinBoxWFocusOut
.. autoclass:: LineEditWFocusOut
.. autoclass:: OrangeListBox

Wrappers for Python classes
===========================

.. autoclass:: ControlledList

Other functions
===============

.. autofunction:: miscellanea
.. autofunction:: setLayout
.. autofunction:: _enterButton
.. autofunction:: _addSpace
.. autofunction:: createAttributePixmap

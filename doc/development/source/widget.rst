.. currentmodule:: Orange.widgets.widget

OWWidget
########


The :class:`~OWWidget` is the main component for implementing a widget in
the Orange Canvas workflow. It both defines the widget input/output
capabilities and implements it's functionality within the canvas.


Widget Meta Description
-----------------------

Every widget in the canvas framework needs to define it's meta definition.
This includes the widget's name and text descriptions but more
importantly also its input/output specification. This is done by defining
constants in the widget's class namespace:

.. code-block:: python

    class IntConstant(OWWidget):
        name = "Integer Constant"
        description = "A simple integer constant"

        outputs = [("Constant", int)]

        ...

        def commit(self):
            """
            Commit/send the outputs.
            """
            self.send("Constant", 42)

Omitting the implementation details, this defines a simple node named
*Integer Constant* which outputs (on a signal called *Constant*) a single
object of type :class:`int`.

The node's inputs are defined similarly but with and extra field naming
the widget instance method which accepts the inputs at runtime:

.. code-block:: python

    class Adder(OWWidget):
        name = "Add two integers"
        description = "Add two numbers"

        inputs = [("A", int, "set_A"),
                  ("B", int, "set_B")]

        outputs = [("A + B", int")]

        ...

        def set_A(self, a):
            """Set the `A` input."""
            self.A = a

        def set_B(self, b):
            """Set the `B` input."""
            self.B = b

        def handleNewSignals(self):
            """Coalescing update."""
            self.commit()

        def commit(self):
            """
            Commit/send the outputs.
            """
            sef.send("result", self.A + self.B)


.. seealso:: :doc:`Getting Started Tutorial <tutorial>`


Input/Output Signal Definitions
-------------------------------

Widgets specify their input/output capabilities in their class definitions
by means of a ``inputs`` and ``outputs`` class attributes which are lists of
tuples or lists of :class:`InputSignal`/:class:`OutputSignal`.

   * An input is defined by a ``(name, type, methodname [, flags])`` tuple
     or an :class:`InputSignal`.

     The `name` is the input's descriptive name, `type` the type of objects
     received, `methodname` a `str` naming a widget member method that will
     receive the input, and optional `flags`.

   * An output is defined by a ``(name, type, flags)`` tuple or an
     :class:`OutputSignal`

Input/Output flags:

.. attribute:: Default

   This input is the default for it's type.
   When there are multiple IO signals with the same type the
   one with the default flag takes precedence when adding a new
   link in the canvas.

.. attribute:: Multiple

   Multiple signal (more then one input on the channel). Input with this
   flag receive a second parameter `id`.

.. attribute:: Dynamic

   Only applies to output. Specifies that the instances on the output
   will in general be subtypes of the declared type and that the output
   can be connected to any input which can accept a subtype of the
   declared output type.

.. attribute:: Explicit

   Outputs with an explicit flag are never selected (auto connected) among
   candidate connections. Connections are only established when there is
   one unambiguous connection possible or through a dedicated 'Links'
   dialog.

.. autoclass:: InputSignal

.. autoclass:: OutputSignal


Sending/Receiving
-----------------

The widgets receive inputs at runtime with the designated handler method
(specified in the :const:`OWWidget.inputs` class member).

If a widget defines multiple inputs it can coalesce updates by reimplementing
:func:`OWWidget.handleNewSignals` method.

.. code-block:: python

   def set_foo(self, foo):
      self.foo = foo

   def set_bar(self, bar):
      self.bar = bar

   def handleNewSignals(self)
      dosomething(self.foo, self.bar)


If an input is defined with the :const:`Multiple`, then the input handler
method also receives an connection `id` uniquely identifying a
connection/link on which the value was sent (see also :doc:`tutorial-channels`)

The widgets publish their outputs using :meth:`OWWidget.send` method.

Accessing Controls though Attribute Names
-----------------------------------------

The preferred way for constructing the user interface is to use functions from
module :obj:`Orange.widgets.gui` that insert a Qt widget and establish the
signals for synchronization with the widget's attributes.

     gui.checkBox(box, self, "binary_trees", "Induce binary tree")

This inserts a `QCheckBox` into the layout of `box`, and make it reflect and
changes the attriubte `self.binary_trees`. The instance of `QCheckbox`
can be accessed through the name it controls. E.g. we can disable the check box
by calling

   self.controls.binary_trees.setDisabled(True)

This may be more practical than having to store the attribute and the Qt
widget that controls it, e.g. with

     self.binarization_cb = gui.checkBox(
         box, self, "binary_trees", "Induce binary tree")

Class Member Documentation
--------------------------

.. autoclass:: Orange.widgets.widget.OWWidget
   :members:
   :member-order: bysource


.. autoclass:: Orange.widgets.widget.Message

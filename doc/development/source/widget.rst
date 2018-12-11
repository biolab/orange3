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
importantly also its input/output specification.

.. code-block:: python

    class IntConstant(OWWidget):
        name = "Integer Constant"
        description = "A simple integer constant"

        class Outputs:
            constant = Output("Constant", int)

        ...

        def commit(self):
            """Commit/send the outputs."""
            self.Outputs.constant.send(42)

Omitting the implementation details, this defines a simple node named
*Integer Constant* which outputs (on a signal called *Constant*) a single
object of type :class:`int`.

The node's inputs are defined similarly. Each input is then used as a decorator
of its corresponding handler method, which accepts the inputs at runtime:

.. code-block:: python

    class Adder(OWWidget):
        name = "Add two integers"
        description = "Add two numbers"

        class Inputs:
            a = Input("A", int)
            b = Input("B", int)

        class Outputs:
            sum = Input("A + B", int)

        ...

        @Inputs.a
        def set_A(self, a):
            """Set the `A` input."""
            self.A = a

        @Inputs.b
        def set_B(self, b):
            """Set the `B` input."""
            self.B = b

        def handleNewSignals(self):
            """Coalescing update."""
            self.commit()

        def commit(self):
            """Commit/send the outputs"""
            sef.Outputs.sum.send("self.A + self.B)


.. seealso:: :doc:`Getting Started Tutorial <tutorial>`


Input/Output Signal Definitions
-------------------------------

Widgets specify their input/output capabilities in their class definitions
by defining classes named `Inputs` and `Outputs`, which contain class
attributes of type `Input` and `Output`, correspondingly. `Input` and `Output`
require at least two arguments, the signal's name (as shown in canvas) and
type. Optional arguments further define the behaviour of the signal.

**Note**: old-style signals define the input and output signals using class
attributes `inputs` and `outputs` instead of classes `Input` and `Output`.
The two attributes contain a list of tuples with the name and type and,
for inputs, the name of the handler method. The optional last argument
is an integer constant giving the flags. This style of signal definition
is deprecated.

.. autoclass:: Input

.. autoclass:: Output

Sending/Receiving
-----------------

The widgets receive inputs at runtime with the handler method decorated with
the signal, as shown in the above examples.

If an input is defined with the flag `multiple` set, the input handler
method also receives a connection `id` uniquely identifying a
connection/link on which the value was sent (see also :doc:`tutorial-channels`)

The widget sends an output by calling the signal's `send` method, as shown
above.

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


.. autoclass:: Orange.widgets.widget.StateInfo
   :members: Summary, Empty, Partial, input_summary_changed,
       output_summary_changed
   :exclude-members: Summary, Empty, Partial,
      input_summary_changed, output_summary_changed,
      set_input_summary, set_output_summary,
      NoInput, NoOutput

   .. autoclass:: Orange.widgets.widget::StateInfo.Summary
      :members:

   .. autoclass:: Orange.widgets.widget::StateInfo.Empty
      :show-inheritance:

   .. autoclass:: Orange.widgets.widget::StateInfo.Partial
      :show-inheritance:

   .. autoattribute:: NoInput
      :annotation: Empty()

   .. autoattribute:: NoOutput
      :annotation: Empty()

   .. function:: set_input_summary(summary: Optional[StateInfo.Summary]])

      Set the input summary description.

      :parameter summary:
      :type summary: Optional[StateInfo.Summary]

   .. function:: set_input_summary(brief:str, detailed:str="", \
                    icon:QIcon=QIcon, format:Qt.TextFormat=Qt.PlainText)

   .. function:: set_output_summary(summary: Optional[StateInfo.Summary]])

      Set the output summary description.

      :parameter summary:
      :type summary: Optional[StateInfo.Summary]

   .. function:: set_output_summary(brief:str, detailed:str="", \
                    icon:QIcon=QIcon, format:Qt.TextFormat=Qt.PlainText)

   .. autoattribute:: input_summary_changed(message: StateInfo.Message)

   .. autoattribute:: output_summary_changed(message: StateInfo.Message)
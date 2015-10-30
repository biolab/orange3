.. _getting started:

###############
Getting Started
###############


Orange Widgets are components in and Orange Canvas visual programming
environment. They represent some self contained functionality and
provide graphical user interface (GUI). Widgets communicate, and
pass objects through communication channels to interact with other
widgets.

On this page, we will start with some simple essentials, and then
show how to build a simple widget that will be ready to run within
Orange Canvas.


Prerequisites
*************

Each Orange widget belongs to a category and within a
category has an associated priority. Opening Orange Canvas, a visual
programming environment that comes with Orange, widgets are listed in
a toolbox on the left:

.. image:: images/widgettoolbox.png

Each widget has a name description and a set of input/outputs
(referred to as widget's meta description).

.. 
   This meta data is discovered at Orange Canvas application startup
   leveraging setuptools/distribute and it's `entry points`_ protocol.
   Orange Canvas looks for widgets using a `orange.widgets` entry point.

   .. _`entry points`: http://pythonhosted.org/distribute/setuptools.html#dynamic-discovery-of-services-and-plugins


Defining a widget
*****************

.. Here we shall explore true facts about the OWWidget.

:class:`~Orange.widgets.widget.OWWidget` is the base class of a widget
in the Orange Canvas workflow.

Every widget in the canvas framework needs to define it's meta data.
This includes the widget's name and text descriptions and more
importantly its input/output specification. This is done by
defining constants in the widget's class namespace.

We will start with a very simple example. A widget that will output
a single integer specified by the user.

.. code-block:: python

    from Orange.widgets import widget, gui

    class IntNumber(widget.OWWidget):
        # Widget's name as displayed in the canvas
        name = "Integer Number"
        # Short widget description
        description "Lets the user input a number"

        # An icon resource file path for this widget
        # (a path relative to the module where this widget is defined)
        icon = "icons/number.svg"

        # A list of output definitions (here on output named "Number"
        # of type int)
        outputs = [("Number", int)]


By design principle, in an interface Orange widgets are most
often split to control and main area. Control area appears on the left
and should include any controls for settings or options that your widget
will use. Main area would most often include a graph, table or some
drawing that will be based on the inputs to the widget and current
options/setting in the control area.
:class:`~Orange.widgets.widget.OWWidget` make these two areas available
through its attributes :obj:`self.controlArea` and :obj:`self.mainArea`.
Notice that while it would be nice for all widgets to have this common
visual look, you can use these areas in any way you want to, even
disregarding one and composing your widget completely unlike the
others in Orange.

We specify the default layout with class attribute flags.
Here we will only be using a single column (controlArea) GUI.

.. code-block:: python

       # Basic (convenience) GUI definition:
       #   a simple 'single column' GUI layout
       want_main_area = False
       #   with a fixed non resizable geometry.
       resizing_enabled = False

We want the current number entered by the user to be saved and restored
when saving/loading a workflow. We can achieve this by declaring a
special property/member in the widget's class definition like so:

.. code-block:: python

       number = Setting(42)


And finally the actual code to define the GUI and the associated
widget functionality

.. code-block:: python

       def __init__(self) 
           super().__init__()

           gui.lineEdit(self.controlArea, self, "number", "Enter a number",
                        orientation="horizontal", box="Number",
                        callback=self.number_changed,
                        valueType=int, validator=QIntValidator())
           self.number_changed()

       def number_changed(self):
           # Send the entered number on "Number" output
           self.send("Number", self.number)

.. seealso:: :func:`Orange.widgets.gui.lineEdit`, :func:`Orange.widgets.widget.OWWidget.send`

By itself this widget seems uninteresting. We need some thing more.
How about displaying a number.

.. code-block:: python

   from Orange.widgets widget, gui

   class Print(widget.OWWidget):
       name = "Print"
       description = "Print out a number"
       icon = "icons/print.svg"

       inputs = [("Number", int, "set_number")
       outputs = []

       want_main_area = False

       def __init__(self):
           super().__init__()
           self.number = None

           self.label = gui.widgetLabel(self.controlArea, "The number is: ??")

       def set_number(self, number):
           """Set the input number."""
           self.number = number
           if self.number is None:
               self.label.setText("The number is: ??")
           else:
               self.label.setText("The number is {}".format(self.number))

Notice how in the `set_number` method we check if number is `None`.
`None` is sent to the widget when a connection between the widgets is removed
or if the sending widget to which we are connected intentionally emptied
the channel.

Now we can use one widget to input a number and another to display it.

One more 

.. code-block:: python

   from Orange.widgets import widget
   class Adder(widget.OWWidget):
       name = "Add two integers"
       description = "Add two numbers"
       icon = "icons/add.svg"

       inputs = [("A", int, "set_A"),
                 ("B", int, "set_B")]
       outputs = [("A + B", int)]

       want_main_area = False

       def __init__(self):
           super().__init__()
           self.a = None
           self.b = None

       def set_A(self, a)
           """Set input 'A'."""
           self.a = a

       def set_B(self, b):
           """Set input 'B'."""
           self.b = b

       def handleNewSignals(self):
           """Reimplemeted from OWWidget."""
           if self.a is not None and self.b is not None:
               self.send("A + B", self.a + self.b)
           else:
               # Clear the channel by sending `None`
               self.send("A + B", None)

.. seealso:: :func:`~Orange.widgets.widget.OWWidget.handleNewSignals`

.. currentmodule:: Orange.widgets.tests.base


Debugging and testing
=====================

Running widgets as scripts
--------------------------

To run a widget without canvas - for debugging and for nitpicking about the GUI
- the widget module must be executable as a script. The base widget class
contains a mixin with a method
:obj:`~Orange.widgets.utils.test_run.WidgetTestRunMixin.test_run`
that must be called at the end of the widget module, i.e. ::

    if __name__ == "__main__":
        OWMyWidgetName.test_run()

where :obj:`OWMyWidgetName` is the widget's class.

We can also pass the data to the widget. For instance, ::

   if __name__ == "__main__":
       OWMyWidgetName.test_run(Orange.data.Table("iris"))

passes the Iris data set to the widget. Passing the data in this way requires
that there is a single or default signal for the argument's data type. Multiple
signals can be passed as keyword arguments in which the names correspond to
signal handlers::

    if __name__ == "__main__":
        data = Orange.data.Table("iris")
        OWScatterPlot.test_run(set_data=data,
                               set_subset_data=data[:30])

For more complex scenarios, define a method
:obj:`~Orange.widgets.utils.test_run.WidgetTestRunMixin.test_run_signals`,
which is called by
:obj:`~Orange.widgets.utils.test_run.WidgetTestRunMixin.test_run`
after sending any signals given as arguments to
:obj:`~Orange.widgets.utils.test_run.WidgetTestRunMixin.test_run`
This can be used for sending multiple-input signals, as in
the following method in the Table widget. ::

    def test_run_signals(self):
        iris = Table("iris")
        brown = Table("brown-selected")
        housing = Table("housing")
        self.set_dataset(iris, iris.name)
        self.set_dataset(brown, brown.name)
        self.set_dataset(housing, housing.name)

Method
:obj:`~Orange.widgets.utils.test_run.WidgetTestRunMixin.test_run`
can also be used for debugging various
sequences of signal connections and disconnections. For example, the
following code can be used to see what happens if the data signal is
disconnected before the data subset signal. ::

    def test_run_signals(self):
        data = Orange.data.Table("iris")
        self.set_data(data)
        self.set_subset_data(data[::10])
        self.handleNewSignals()
        self.set_data(None)
        self.handleNewSignals()

Note that while such code is useful for debugging purposes, it must eventually
be moved to unit tests instead of being kept in the finished code.

.. autoclass:: Orange.widgets.utils.test_run.WidgetTestRunMixin
   :members:
   :member-order: bysource

Unit-testing Widgets
--------------------

Orange provides a base class :class:`WidgetTest` with helper methods for unit
testing.


.. autoclass:: WidgetTest
   :members:
   :member-order: bysource

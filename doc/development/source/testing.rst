.. currentmodule:: Orange.widgets.tests.base


Debugging and testing
=====================

Running widgets as scripts
--------------------------

To run a widget without canvas - for debugging and for nitpicking about the GUI
- the widget module must be executable as a script. This is handled by
:obj:`~Orange.widgets.utils.widgetpreview.WidgetPreview`. It is typically
used as follows ::

    if __name__ == "__main__":
        WidgetPreview(OWMyWidgetName).run()

where :obj:`OWMyWidgetName` is the widget's class.

We can also pass the data to the widget. For instance, ::

   if __name__ == "__main__":
       WidgetPreview(OWMyWidgetName).run(Orange.data.Table("iris"))

passes the Iris data set to the widget. Passing data in this way requires
that there is a single or default signal for the argument's data type. Multiple
signals can be passed as keyword arguments in which the names correspond to
signal handlers::

    if __name__ == "__main__":
        data = Orange.data.Table("iris")
        WidgetPreview(OWScatterPlot).run(
            set_data=data,
            set_subset_data=data[:30]
        )

If the signal handler accepts multiple inputs, they can be passed as a list,
like in the following method in the Table widget. ::

    if __name__ == "__main__":
        WidgetPreview(OWDataTable).run(
            [(Table("iris"), "iris"),
            (Table("brown-selected"), "brown-selected"),
            (Table("housing"), "housing")
            ]
        )

Preview ends by tearing down the widget and calling :obj:`sys.exit` with the
widget's exit code. This can be prevented by adding a `no_exit=True` argument.
We can also prevent showing the widget and starting the event loop by using
`no_exec=True`. This, together with some previewers method described below,
can be used for debugging the widget. For example, `OWRank`'s preview, ::

    if __name__ == "__main__":
        from Orange.classification import RandomForestLearner
        WidgetPreview(OWRank).run(
            set_learner=(RandomForestLearner(), (3, 'Learner', None)),
            set_data=Table("heart_disease.tab"))

can be temporarily modified to ::

    if __name__ == "__main__":
        from Orange.classification import RandomForestLearner
        previewer = WidgetPreview(OWRank)
        previewer.run(Table("heart_disease.tab"), no_exit=True)
        previewer.send_signals(
            set_learner=(RandomForestLearner(), (3, 'Learner', None)))
        previewer.run()

which shows the widget twice, allows us a finer control of signal passing,
and offers adding some breakpoints.

.. autoclass:: Orange.widgets.utils.widgetpreview.WidgetPreview
   :members:
   :member-order: bysource

Unit-testing Widgets
--------------------

Orange provides a base class :class:`WidgetTest` with helper methods for unit
testing.


.. autoclass:: WidgetTest
   :members:
   :member-order: bysource

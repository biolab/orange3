##############
Responsive GUI
##############

And now for the hard part of making the widget responsive. We will do this
by offloading the learner evaluations into a separate thread.

First read up on `threading basics`_ in Qt and in particular the subject
of `threads and qobjects`_ and how they interact with the Qt's event loop.

.. _threading basics:
    http://doc.qt.io/qt-5/thread-basics.html

.. _threads and qobjects:
    http://doc.qt.io/qt-5/threads-qobject.html

We must also take special care that we can cancel/interrupt our task when
the user changes algorithm parameters or removes the widget from the canvas.
For that we use a strategy known as cooperative cancellation where we 'ask'
the pending task to stop executing (in the GUI thread), then in the worker
thread periodically check (at known predetermined points) whether we should
continue, and if not return early (in our case by raising an exception).


**********
Setting up
**********

We use :class:`Orange.widgets.utils.concurrent.ThreadExecutor` for thread
allocation/management (but could easily replace it with stdlib's
:class:`concurrent.futures.ThreadPoolExecutor`).

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-1
   :end-before: end-snippet-1


We will reorganize our code to make the learner evaluation an explicit
task as we will need to track its progress and state. For this we define
a `Task` class.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-2
   :end-before: end-snippet-2


In the widget's ``__init__`` we create an instance of the `ThreadExector`
and initialize the task field.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-3
   :end-before: end-snippet-3

All code snippets are from :download:`OWLearningCurveC.py <orange-demo/orangedemo/OWLearningCurveC.py>`.


***************************
Starting a task in a thread
***************************

In `handleNewSignals` we call `_update`.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-4
   :end-before: end-snippet-4


And finally the `_update` function (from :download:`OWLearningCurveC.py <orange-demo/orangedemo/OWLearningCurveC.py>`)
that will start/schedule all updates.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-5
   :end-before: end-snippet-5


At the start we cancel pending tasks if they are not yet completed. It is
important to do this, we cannot allow the widget to schedule tasks and
then just forget about them. Next we make some checks and return early if
there is nothing to be done.

Continue by setting up the learner evaluations as a partial function
capturing the necessary arguments:

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-6
   :end-before: end-snippet-6


Setup the task state and the communication between the main and worker thread.
The only `state` flowing from the GUI to the worker thread is the
`task.cancelled` field which is a simple trip wire causing the
`learning_curve`'s callback argument to raise an exception. In the other
direction we report the `percent` of work done.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-7
   :end-before: end-snippet-7

.. seealso::
   :func:`~Orange.widgets.widget.OWWidget.progressBarInit`,
   :func:`~Orange.widgets.widget.OWWidget.progressBarSet`,
   :func:`~Orange.widgets.widget.OWWidget.progressBarFinished`


Next, we submit the function to be run in a worker thread and instrument
a FutureWatcher instance to notify us when the task completes (via a
`_task_finished` slot).

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-8
   :end-before: end-snippet-8

For the above code to work, the `setProgressValue` needs defined as a pyqtSlot.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-progress
   :end-before: end-snippet-progress


******************
Collecting results
******************

In `_task_finished` (from :download:`OWLearningCurveC.py <orange-demo/orangedemo/OWLearningCurveC.py>`)
we handle the completed task (either success or failure) and then update the displayed score table.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-9
   :end-before: end-snippet-9


********
Stopping
********

Also of interest is the `cancel` method. Note that we also disconnect the
`_task_finished` slot so that `_task_finished` does not receive stale
results.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-10
   :end-before: end-snippet-10

We also use cancel in :func:`~Orange.widgets.widget.OWWidget.onDeleteWidget`
to stop if/when the widget is removed from the canvas.

.. literalinclude:: orange-demo/orangedemo/OWLearningCurveC.py
   :start-after: start-snippet-11
   :end-before: end-snippet-11

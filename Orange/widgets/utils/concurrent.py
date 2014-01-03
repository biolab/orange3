"""\
OWConcurent
===========

General helper functions and classes for Orange Canvas
concurrent programming

"""


import sys
import threading
import logging

from functools import partial
from contextlib import contextmanager

from PyQt4.QtGui import qApp

from PyQt4.QtCore import (
    Qt, QObject, QMetaObject, QTimer, QThreadPool, QThread, QMutex,
    QRunnable, QEventLoop, QCoreApplication, QEvent,
    Q_ARG, pyqtSignature,
)

from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

_log = logging.getLogger(__name__)


@contextmanager
def locked(mutex):
    """
    A context manager for locking an instance of a QMutex.
    """
    mutex.lock()
    try:
        yield
    finally:
        mutex.unlock()


class _TaskDepotThread(QThread):
    """
    A special 'depot' thread used to transfer Task instance into threads
    started by a QThreadPool.

    """

    def start(self):
        """
        Reimplemented from `QThread.start`
        """
        QThread.start(self)
        # Need to also handle method invoke from this thread
        self.moveToThread(self)

    def run(self):
        """
        Reimplemented from `QThread.run`
        """
        # Start the event loop.
        # On some old Qt4/PyQt4 installations base QThread.run does not seem
        # to enter the loop, despite being documented to do so.
        self.exec_()

    @Slot(object, object)
    def transfer(self, obj, thread):
        """
        Transfer `obj` (:class:`QObject`) instance from this thread to the
        target `thread` (a :class:`QThread`).

        """
        assert obj.thread() is self
        assert QThread.currentThread() is self
        obj.moveToThread(thread)


class _TaskRunnable(QRunnable):
    """
    A QRunnable for running a :class:`Task` by a :class:`ThreadExecuter`.
    """

    def __init__(self, future, task, args, kwargs):
        QRunnable.__init__(self)
        self.future = future
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.eventLoop = None

    def run(self):
        """
        Reimplemented from `QRunnable.run`
        """
        self.eventLoop = QEventLoop()
        self.eventLoop.processEvents()

        # Move the task to the current thread so it's events, signals, slots
        # are triggered from this thread.
        assert isinstance(self.task.thread(), _TaskDepotThread)
        QMetaObject.invokeMethod(
            self.task.thread(), "transfer", Qt.BlockingQueuedConnection,
            Q_ARG(object, self.task),
            Q_ARG(object, QThread.currentThread())
        )

        self.eventLoop.processEvents()

        # Schedule task.run from the event loop.
        self.task.start()

        # Quit the loop and exit when task finishes or is cancelled.
        # TODO: If the task encounters an critical error it might not emit
        # these signals and this Runnable will never complete.
        self.task.finished.connect(self.eventLoop.quit)
        self.task.cancelled.connect(self.eventLoop.quit)
        self.eventLoop.exec_()


class _Runnable(QRunnable):
    """
    A QRunnable for running plain functions by a :class:`ThreadExecuter`.
    """

    def __init__(self, future, func, args, kwargs):
        QRunnable.__init__(self)
        self.future = future
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """
        Reimplemented from QRunnable.run
        """
        try:
            if not self.future.set_running_or_notify_cancel():
                # Was cancelled
                return
            try:
                result = self.func(*self.args, **self.kwargs)
            except BaseException as ex:
                self.future.set_exception(ex)
            else:
                self.future.set_result(result)
        except BaseException:
            _log.critical("Exception in worker thread.", exc_info=True)


class ThreadExecutor(QObject):
    """
    ThreadExceuter object class provides an interface for running tasks
    in a thread pool.

    :param QObject parent:
        Executor's parent instance.

    :param QThreadPool threadPool:
        Thread pool to be used by the instance of the Executor. If `None`
        then ``QThreadPool.globalInstance()`` will be used.

    """

    def __init__(self, parent=None, threadPool=None):
        QObject.__init__(self, parent)
        if threadPool is None:
            threadPool = QThreadPool.globalInstance()
        self._threadPool = threadPool
        self._depot_thread = None

    def _get_depot_thread(self):
        if self._depot_thread is None:
            self._depot_thread = _TaskDepotThread()
            self._depot_thread.start()

        return self._depot_thread

    def submit(self, func, *args, **kwargs):
        """
        Schedule the `func(*args, **kwargs)` to be executed and return an
        :class:`Future` instance representing the result of the computation.

        """
        if isinstance(func, Task):
            if func.thread() is not QThread.currentThread():
                raise ValueError("Can only submit Tasks from it's own thread.")

            if func.parent() is not None:
                raise ValueError("Can not submit Tasks with a parent.")

            func.moveToThread(self._get_depot_thread())
            assert func.thread() is self._get_depot_thread()
            # Use the Task's own Future object
            f = func.future()
            runnable = _TaskRunnable(f, func, args, kwargs)
        else:
            f = Future()
            runnable = _Runnable(f, func, args, kwargs)
        self._threadPool.start(runnable)

        return f

    def map(self, func, *iterables):
        futures = [self.submit(func, *args) for args in zip(*iterables)]

        for f in futures:
            yield f.result()

    def shutdown(self, wait=True):
        """
        Shutdown the executor and free all resources. If `wait` is True then
        wait until all pending futures are executed or cancelled.
        """
        if self._depot_thread is not None:
            QMetaObject.invokeMethod(
                self._depot_thread, "quit", Qt.AutoConnection)

        if wait:
            self._threadPool.waitForDone()
            if self._depot_thread:
                self._depot_thread.wait()
                self._depot_thread = None


class ExecuteCallEvent(QEvent):
    """
    Represents an function call from the event loop (used by :class:`Task`
    to schedule the :func:`Task.run` method to be invoked)

    """
    ExecuteCall = QEvent.registerEventType()

    def __init__(self):
        QEvent.__init__(self, ExecuteCallEvent.ExecuteCall)


class Task(QObject):
    """
    """
    started = Signal()
    finished = Signal()
    cancelled = Signal()
    resultReady = Signal(object)
    exceptionReady = Signal(Exception)

    def __init__(self, parent=None, function=None):
        QObject.__init__(self, parent)
        self.function = function

        self._future = Future()

    def run(self):
        if self.function is None:
            raise NotImplementedError
        else:
            return self.function()

    def start(self):
        QCoreApplication.postEvent(self, ExecuteCallEvent())

    def future(self):
        return self._future

    def result(self, timeout=None):
        return self._future.result(timeout)

    def _execute(self):
        try:
            if not self._future.set_running_or_notify_cancel():
                self.cancelled.emit()
                return

            self.started.emit()
            try:
                result = self.run()
            except BaseException as ex:
                self._future.set_exception(ex)
                self.exceptionReady.emit(ex)
            else:
                self._future.set_result(result)
                self.resultReady.emit(result)

            self.finished.emit()
        except BaseException:
            _log.critical("Exception in Task", exc_info=True)

    def customEvent(self, event):
        if event.type() == ExecuteCallEvent.ExecuteCall:
            self._execute()
        else:
            QObject.customEvent(self, event)


def futures_iter(futures):
    for f in futures:
        yield f.result()


class TimeoutError(Exception):
    pass


class CancelledError(Exception):
    pass


class Future(object):
    """
    A :class:`Future` class represents a result of an asynchronous
    computation.

    """
    Pending, Canceled, Running, Finished = 1, 2, 4, 8

    def __init__(self):
        self._watchers = []
        self._state = Future.Pending
        self._condition = threading.Condition()
        self._result = None
        self._exception = None

    def _set_state(self, state):
        if self._state != state:
            self._state = state
            for watcher in self._watchers:
                watcher(self, state)

    def cancel(self):
        """
        Attempt to cancel the the call. Return `False` if the call is
        already in progress and cannot be canceled, otherwise return `True`.

        """
        with self._condition:
            if self._state in [Future.Running, Future.Finished]:
                return False
            elif self._state == Future.Canceled:
                return True
            else:
                self._set_state(Future.Canceled)
                self._condition.notify_all()

        return True

    def cancelled(self):
        """
        Return `True` if call was successfully cancelled.
        """
        with self._condition:
            return self._state == Future.Canceled

    def done(self):
        """
        Return `True` if the call was successfully cancelled or finished
        running.

        """
        with self._condition:
            return self._state in [Future.Canceled, Future.Finished]

    def running(self):
        """
        Return True if the call is currently being executed.
        """
        with self._condition:
            return self._state == Future.Running

    def _get_result(self):
        if self._exception:
            raise self._exception
        else:
            return self._result

    def result(self, timeout=None):
        """
        Return the result of the :class:`Futures` computation. If `timeout`
        is `None` the call will block until either the computation finished
        or is cancelled.
        """
        with self._condition:
            if self._state == Future.Finished:
                return self._get_result()
            elif self._state == Future.Canceled:
                raise CancelledError()

            self._condition.wait(timeout)

            if self._state == Future.Finished:
                return self._get_result()
            elif self._state == Future.Canceled:
                raise CancelledError()
            else:
                raise TimeoutError()

    def exception(self, timeout=None):
        """
        Return the exception instance (if any) resulting from the execution
        of the :class:`Future`. Can raise a :class:`CancelledError` if the
        computation was cancelled.

        """
        with self._condition:
            if self._state == Future.Finished:
                return self._exception
            elif self._state == Future.Canceled:
                raise CancelledError()

            self._condition.wait(timeout)

            if self._state == Future.Finished:
                return self._exception
            elif self._state == Future.Canceled:
                raise CancelledError()
            else:
                raise TimeoutError()

    def set_result(self, result):
        """
        Set the result of the computation (called by the worker thread).
        """
        with self._condition:
            self._result = result
            self._set_state(Future.Finished)
            self._condition.notify_all()

    def set_exception(self, exception):
        """
        Set the exception instance that was raised by the computation
        (called by the worker thread).

        """
        with self._condition:
            self._exception = exception
            self._set_state(Future.Finished)
            self._condition.notify_all()

    def set_running_or_notify_cancel(self):
        with self._condition:
            if self._state == Future.Canceled:
                return False
            elif self._state == Future.Pending:
                self._set_state(Future.Running)
                return True
            else:
                raise Exception()


class StateChangedEvent(QEvent):
    """
    Represents a change in the internal state of a :class:`Future`.
    """
    StateChanged = QEvent.registerEventType()

    def __init__(self, state):
        QEvent.__init__(self, StateChangedEvent.StateChanged)
        self._state = state

    def state(self):
        """
        Return the new state (Future.Pending, Future.Cancelled, ...).
        """
        return self._state


class FutureWatcher(QObject):
    """
    A `FutureWatcher` class provides a convenient interface to the
    :class:`Future` instance using Qt's signals.

    :param :class:`Future` future:
        A :class:`Future` instance to watch.
    :param :class:`QObject` parent:
        Object's parent instance.

    """
    #: The future was cancelled.
    cancelled = Signal()

    #: The future has finished.
    finished = Signal()

    #: The future has started computation.
    started = Signal()

    def __init__(self, future, parent=None):
        QObject.__init__(self, parent)
        self._future = future

        self._future._watchers.append(self._stateChanged)

    def isCancelled(self):
        """
        Was the future cancelled.
        """
        return self._future.cancelled()

    def isDone(self):
        """
        Is the future done (was cancelled or has finished).
        """
        return self._future.done()

    def isRunning(self):
        """
        Is the future running (i.e. has started).
        """
        return self._future.running()

    def isStarted(self):
        """
        Has the future computation started.
        """
        return self._future.running()

    def result(self):
        """
        Return the result of the computation.
        """
        return self._future.result()

    def exception(self):
        """
        Return the exception instance or `None` if no exception was raised.
        """
        return self._future.exception()

    def customEvent(self, event):
        """
        Reimplemented from `QObject.customEvent`.
        """
        if event.type() == StateChangedEvent.StateChanged:
            if event.state() == Future.Canceled:
                self.cancelled.emit()
            elif event.state() == Future.Running:
                self.started.emit()
            elif event.state() == Future.Finished:
                self.finished.emit()
            return

        return QObject.customEvent(self, event)

    def _stateChanged(self, future, state):
        """
        The `future` state has changed (called by :class:`Future`).
        """
        ev = StateChangedEvent(state)

        if self.thread() is QThread.currentThread():
            QCoreApplication.sendEvent(self, ev)
        else:
            QCoreApplication.postEvent(self, ev)


class methodinvoke(object):
    """
    Create an QObject method wrapper that invokes the method asynchronously
    in the object's own thread.

    :param obj:
        A QObject instance.
    :param str method:
        The method name.
    :param tuple arg_types:
        A tuple of positional argument types.

    """

    def __init__(self, obj, method, arg_types=()):
        self.obj = obj
        self.method = method
        self.arg_types = tuple(arg_types)

    def __call__(self, *args):
        args = [Q_ARG(atype, arg) for atype, arg in zip(self.arg_types, args)]
        QMetaObject.invokeMethod(
            self.obj, self.method, Qt.QueuedConnection,
            *args
        )


try:
    import unittest2 as unittest
except ImportError:
    import unittest


class TestFutures(unittest.TestCase):
    def test_futures(self):
        f = Future()
        self.assertEqual(f.done(), False)
        self.assertEqual(f.running(), False)

        self.assertTrue(f.cancel())
        self.assertTrue(f.cancelled())

        with self.assertRaises(CancelledError):
            f.result()

        with self.assertRaises(CancelledError):
            f.exception()

        f = Future()
        f.set_running_or_notify_cancel()

        with self.assertRaises(TimeoutError):
            f.result(0.1)

        with self.assertRaises(TimeoutError):
            f.exception(0.1)

        f = Future()
        f.set_running_or_notify_cancel()
        f.set_result("result")

        self.assertEqual(f.result(), "result")
        self.assertEqual(f.exception(), None)

        f = Future()
        f.set_running_or_notify_cancel()

        f.set_exception(Exception("foo"))

        with self.assertRaises(Exception):
            f.result()


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication([])

    def test_executor(self):
        executor = ThreadExecutor()
        f1 = executor.submit(pow, 100, 100)

        f2 = executor.submit(lambda: 1 / 0)

        f3 = executor.submit(QThread.currentThread)

        self.assertTrue(f1.result(), pow(100, 100))

        with self.assertRaises(ZeroDivisionError):
            f2.result()

        self.assertIsInstance(f2.exception(), ZeroDivisionError)

        self.assertIsNot(f3.result(), QThread.currentThread())

    def test_methodinvoke(self):
        executor = ThreadExecutor()
        state = [None, None]

        class StateSetter(QObject):
            @Slot(object)
            def set_state(self, value):
                state[0] = value
                state[1] = QThread.currentThread()

        def func(callback):
            callback(QThread.currentThread())

        obj = StateSetter()
        f1 = executor.submit(func, methodinvoke(obj, "set_state", (object,)))
        f1.result()

        # So invoked method can be called
        QCoreApplication.processEvents()

        self.assertIs(state[1], QThread.currentThread(),
                      "set_state was called from the wrong thread")

        self.assertIsNot(state[0], QThread.currentThread(),
                         "set_state was invoked in the main thread")

        executor.shutdown(wait=True)

    def test_executor_map(self):
        executor = ThreadExecutor()

        r = executor.map(pow, list(range(1000)), list(range(1000)))

        results = list(r)

        self.assertTrue(len(results) == 1000)


class TestFutureWatcher(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication([])

    def test_watcher(self):
        executor = ThreadExecutor()
        f = executor.submit(QThread.currentThread)
        watcher = FutureWatcher(f)

        if f.cancel():
            self.assertTrue(watcher.isCancelled())

        executor.shutdown()


class TestTask(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication([])

    def test_task(self):
        results = []

        task = Task(function=QThread.currentThread)
        task.resultReady.connect(results.append)

        task.start()
        self.app.processEvents()

        self.assertSequenceEqual(results, [QThread.currentThread()])

        results = []

        thread = QThread()
        thread.start()

        task = Task(function=QThread.currentThread)

        task.moveToThread(thread)

        self.assertIsNot(task.thread(), QThread.currentThread())
        self.assertIs(task.thread(), thread)

        task.resultReady.connect(results.append, Qt.DirectConnection)
        task.start()

        f = task.future()

        self.assertIsNot(f.result(3), QThread.currentThread())

        self.assertIs(f.result(3), results[-1])

    def test_executor(self):
        executor = ThreadExecutor()

        f = executor.submit(QThread.currentThread)

        self.assertIsNot(f.result(3), QThread.currentThread())

        f = executor.submit(lambda: 1 / 0)

        with self.assertRaises(ZeroDivisionError):
            f.result()

        results = []
        task = Task(function=QThread.currentThread)
        task.resultReady.connect(results.append, Qt.DirectConnection)

        f = executor.submit(task)

        self.assertIsNot(f.result(3), QThread.currentThread())

        executor.shutdown()

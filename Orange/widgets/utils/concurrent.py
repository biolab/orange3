"""
General helper functions and classes for PyQt concurrent programming
"""
# TODO: Rename the module to something that does not conflict with stdlib
# concurrent
import threading
import atexit
import logging
import warnings
import weakref
import concurrent.futures

from concurrent.futures import Future, CancelledError, TimeoutError
from contextlib import contextmanager

try:
    # pylint: disable=unused-import
    from typing import Callable, Optional, Any
except ImportError:
    pass

from AnyQt.QtCore import (
    Qt, QObject, QMetaObject, QThreadPool, QThread, QRunnable,
    QEventLoop, QCoreApplication, QEvent, Q_ARG
)

from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

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
    _lock = threading.Lock()
    _instance = None

    def __new__(cls):
        if _TaskDepotThread._instance is not None:
            raise RuntimeError("Already exists")
        return QThread.__new__(cls)

    def __init__(self):
        super().__init__()
        self.start()
        # Need to handle queued method calls from this thread.
        self.moveToThread(self)
        atexit.register(self._cleanup)

    def _cleanup(self):
        self.quit()
        self.wait()

    @staticmethod
    def instance():
        with _TaskDepotThread._lock:
            if _TaskDepotThread._instance is None:
                _TaskDepotThread._instance = _TaskDepotThread()
            return _TaskDepotThread._instance

    @Slot(object, object)
    def transfer(self, obj, thread):
        """
        Transfer `obj` (:class:`QObject`) instance from this thread to the
        target `thread` (a :class:`QThread`).

        """
        assert obj.thread() is self
        assert QThread.currentThread() is self
        obj.moveToThread(thread)

    def __del__(self):
        self._cleanup()


class _TaskRunnable(QRunnable):
    """
    A QRunnable for running a :class:`Task` by a :class:`ThreadExecutor`.
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
        assert self.task.thread() is _TaskDepotThread.instance()

        QMetaObject.invokeMethod(
            self.task.thread(), "transfer", Qt.BlockingQueuedConnection,
            Q_ARG(object, self.task),
            Q_ARG(object, QThread.currentThread())
        )

        self.eventLoop.processEvents()

        # Schedule task.run from the event loop.
        self.task.start()

        # Quit the loop and exit when task finishes or is cancelled.
        self.task.finished.connect(self.eventLoop.quit)
        self.task.cancelled.connect(self.eventLoop.quit)
        self.eventLoop.exec_()


class FutureRunnable(QRunnable):
    """
    A QRunnable to fulfil a `Future` in a QThreadPool managed thread.

    Parameters
    ----------
    future : concurrent.futures.Future
        Future whose contents will be set with the result of executing
        `func(*args, **kwargs)` after completion
    func : Callable
        Function to invoke in a thread
    args : tuple
        Positional arguments for `func`
    kwargs : dict
        Keyword arguments for `func`

    Example
    -------
    >>> f = concurrent.futures.Future()
    >>> task = FutureRunnable(f, int, (42,), {})
    >>> QThreadPool.globalInstance().start(task)
    >>> f.result()
    42
    """
    def __init__(self, future, func, args, kwargs):
        # type: (Future, Callable, tuple, dict) -> None
        super().__init__()
        self.future = future
        self.task = (func, args, kwargs)

    def run(self):
        """
        Reimplemented from `QRunnable.run`
        """
        try:
            if not self.future.set_running_or_notify_cancel():
                # future was cancelled
                return
            func, args, kwargs = self.task
            try:
                result = func(*args, **kwargs)
            except BaseException as ex: # pylint: disable=broad-except
                self.future.set_exception(ex)
            else:
                self.future.set_result(result)
        except BaseException:  # pylint: disable=broad-except
            log = logging.getLogger(__name__)
            log.critical("Exception in worker thread.", exc_info=True)


class ThreadExecutor(QObject, concurrent.futures.Executor):
    """
    ThreadExecutor object class provides an interface for running tasks
    in a QThreadPool.

    Parameters
    ----------
    parent : QObject
        Executor's parent instance.

    threadPool :  Optional[QThreadPool]
        Thread pool to be used by the instance of the Executor. If `None`
        then ``QThreadPool.globalInstance()`` will be used.

    """

    def __init__(self, parent=None, threadPool=None, **kwargs):
        super().__init__(parent, **kwargs)
        if threadPool is None:
            threadPool = QThreadPool.globalInstance()
        self._threadPool = threadPool
        self._depot_thread = None
        self._futures = []
        self._shutdown = False
        self._state_lock = threading.Lock()

    def _get_depot_thread(self):
        if self._depot_thread is None:
            self._depot_thread = _TaskDepotThread.instance()
        return self._depot_thread

    def submit(self, func, *args, **kwargs):
        """
        Reimplemented from :class:`concurrent.futures.Executor`

        Schedule the `func(*args, **kwargs)` to be executed and return an
        :class:`Future` instance representing the result of the computation.
        """
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after " +
                                   "shutdown.")

            if isinstance(func, Task):
                warnings.warn("Use `submit_task` to run `Task`s",
                              DeprecationWarning, stacklevel=2)
                f, runnable = self.__make_task_runnable(func)
            else:
                f = Future()
                runnable = FutureRunnable(f, func, args, kwargs)

            self._futures.append(f)
            f.add_done_callback(self._future_done)
            self._threadPool.start(runnable)
            return f

    def submit_task(self, task):
        # undocumented for a reason, should probably be deprecated and removed
        warnings.warn("`submit_task` will be deprecated",
                      PendingDeprecationWarning, stacklevel=2)
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after " +
                                   "shutdown.")

            f, runnable = self.__make_task_runnable(task)

            self._futures.append(f)
            f.add_done_callback(self._future_done)
            self._threadPool.start(runnable)
            return f

    def __make_task_runnable(self, task):
        if task.thread() is not QThread.currentThread():
            raise ValueError("Can only submit Tasks from it's own " +
                             "thread.")

        if task.parent() is not None:
            raise ValueError("Can not submit Tasks with a parent.")

        task.moveToThread(self._get_depot_thread())

        # Use the Task's own Future object
        f = task.future()
        runnable = _TaskRunnable(f, task, (), {})
        return f, runnable

    def shutdown(self, wait=True):
        """
        Shutdown the executor and free all resources. If `wait` is True then
        wait until all pending futures are executed or cancelled.
        """
        with self._state_lock:
            self._shutdown = True
            futures = list(self._futures)

        if wait:
            concurrent.futures.wait(futures)

    def _future_done(self, future):
        # Remove futures when finished.
        self._futures.remove(future)


class Task(QObject):
    started = Signal()
    finished = Signal()
    cancelled = Signal()
    resultReady = Signal(object)
    exceptionReady = Signal(Exception)

    __ExecuteCall = QEvent.registerEventType()

    def __init__(self, parent=None, function=None):
        super().__init__(parent)
        warnings.warn(
            "`Task` has been deprecated", PendingDeprecationWarning,
            stacklevel=2)
        self.function = function

        self._future = Future()

    def run(self):
        if self.function is None:
            raise NotImplementedError
        else:
            return self.function()

    def start(self):
        QCoreApplication.postEvent(self, QEvent(Task.__ExecuteCall))

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
        if event.type() == Task.__ExecuteCall:
            self._execute()
        else:
            super().customEvent(event)


class FutureWatcher(QObject):
    """
    An `QObject` watching the state changes of a `concurrent.futures.Future`

    Note
    ----
    The state change notification signals (`done`, `finished`, ...)
    are always emitted when the control flow reaches the event loop
    (even if the future is already completed when set).

    Note
    ----
    An event loop must be running, otherwise the notifier signals will
    not be emitted.

    Parameters
    ----------
    parent : QObject
        Parent object.
    future : Future
        The future instance to watch.

    Example
    -------
    >>> app = QCoreApplication.instance() or QCoreApplication([])
    >>> f = submit(lambda i, j: i ** j, 10, 3)
    >>> watcher = FutureWatcher(f)
    >>> watcher.resultReady.connect(lambda res: print("Result:", res))
    >>> watcher.done.connect(app.quit)
    >>> _ = app.exec()
    Result: 1000
    >>> f.result()
    1000
    """
    #: Signal emitted when the future is done (cancelled or finished)
    done = Signal(Future)

    #: Signal emitted when the future is finished (i.e. returned a result
    #: or raised an exception - but not if cancelled)
    finished = Signal(Future)

    #: Signal emitted when the future was cancelled
    cancelled = Signal(Future)

    #: Signal emitted with the future's result when successfully finished.
    resultReady = Signal(object)

    #: Signal emitted with the future's exception when finished with an
    #: exception.
    exceptionReady = Signal(BaseException)

    # A private event type used to notify the watcher of a Future's completion
    __FutureDone = QEvent.registerEventType()

    def __init__(self, future=None, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__future = None
        if future is not None:
            self.setFuture(future)

    def setFuture(self, future):
        # type: (Future) -> None
        """
        Set the future to watch.

        Raise a `RuntimeError` if a future is already set.

        Parameters
        ----------
        future : Future
        """
        if self.__future is not None:
            raise RuntimeError("Future already set")

        self.__future = future
        selfweakref = weakref.ref(self)

        def on_done(f):
            assert f is future
            selfref = selfweakref()

            if selfref is None:
                return

            try:
                QCoreApplication.postEvent(
                    selfref, QEvent(FutureWatcher.__FutureDone))
            except RuntimeError:
                # Ignore RuntimeErrors (when C++ side of QObject is deleted)
                # (? Use QObject.destroyed and remove the done callback ?)
                pass

        future.add_done_callback(on_done)

    def future(self):
        # type: () -> Future
        """
        Return the future instance.
        """
        return self.__future

    def isCancelled(self):
        warnings.warn("isCancelled is deprecated", DeprecationWarning,
                      stacklevel=2)
        return self.__future.cancelled()

    def isDone(self):
        warnings.warn("isDone is deprecated", DeprecationWarning,
                      stacklevel=2)
        return self.__future.done()

    def result(self):
        # type: () -> Any
        """
        Return the future's result.

        Note
        ----
        This method is non-blocking. If the future has not yet completed
        it will raise an error.
        """
        try:
            return self.__future.result(timeout=0)
        except TimeoutError:
            raise RuntimeError("Future is not yet done")

    def exception(self):
        # type: () -> Optional[BaseException]
        """
        Return the future's exception.

        Note
        ----
        This method is non-blocking. If the future has not yet completed
        it will raise an error.
        """
        try:
            return self.__future.exception(timeout=0)
        except TimeoutError:
            raise RuntimeError("Future is not yet done")

    def __emitSignals(self):
        assert self.__future is not None
        assert self.__future.done()
        if self.__future.cancelled():
            self.cancelled.emit(self.__future)
            self.done.emit(self.__future)
        elif self.__future.done():
            self.finished.emit(self.__future)
            self.done.emit(self.__future)
            if self.__future.exception():
                self.exceptionReady.emit(self.__future.exception())
            else:
                self.resultReady.emit(self.__future.result())
        else:
            assert False

    def customEvent(self, event):
        # Reimplemented.
        if event.type() == FutureWatcher.__FutureDone:
            self.__emitSignals()
        super().customEvent(event)


class methodinvoke(object):
    """
    A thin wrapper for invoking QObject's method through
    `QMetaObject.invokeMethod`.

    This can be used to invoke the method across thread boundaries (or even
    just for scheduling delayed calls within the same thread).

    Note
    ----
    An event loop MUST be running in the target QObject's thread.

    Parameters
    ----------
    obj : QObject
        A QObject instance.
    method : str
        The method name. This method must be registered with the Qt object
        meta system (e.g. decorated by a Slot decorator).
    arg_types : tuple
        A tuple of positional argument types.
    conntype : Qt.ConnectionType
        The connection/call type. Qt.QueuedConnection (the default) and
        Qt.BlockingConnection are the most interesting.

    See Also
    --------
    QMetaObject.invokeMethod

    Example
    -------
    >>> app = QCoreApplication.instance() or QCoreApplication([])
    >>> quit = methodinvoke(app, "quit", ())
    >>> t = threading.Thread(target=quit)
    >>> t.start()
    >>> app.exec()
    0
    """
    def __init__(self, obj, method, arg_types=(), *,
                 conntype=Qt.QueuedConnection):
        self.obj = obj
        self.method = method
        self.arg_types = tuple(arg_types)
        self.conntype = conntype

    def __call__(self, *args):
        args = [Q_ARG(atype, arg) for atype, arg in zip(self.arg_types, args)]
        return QMetaObject.invokeMethod(
            self.obj, self.method, self.conntype, *args)


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

        class Ref():
            def __init__(self, ref):
                self.ref = ref

            def set(self, ref):
                self.ref = ref

        # Test that done callbacks are called.
        called = Ref(False)
        f = Future()
        f.add_done_callback(lambda f: called.set(True))
        f.set_result(None)
        self.assertTrue(called.ref)

        # Test that callbacks are called when cancelled.
        called = Ref(False)
        f = Future()
        f.add_done_callback(lambda f: called.set(True))
        f.cancel()
        self.assertTrue(called.ref)

        # Test that callbacks are called immediately when the future is
        # already done.
        called = Ref(False)
        f = Future()
        f.set_result(None)
        f.add_done_callback(lambda f: called.set(True))
        self.assertTrue(called.ref)

        count = Ref(0)
        f = Future()
        f.add_done_callback(lambda f: count.set(count.ref + 1))
        f.add_done_callback(lambda f: count.set(count.ref + 1))
        f.set_result(None)
        self.assertEqual(count.ref, 2)

        # Test that the callbacks are called with the future as argument.
        done_future = Ref(None)
        f = Future()
        f.add_done_callback(lambda f: done_future.set(f))
        f.set_result(None)
        self.assertIs(f, done_future.ref)


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

        f = executor.submit_task(task)

        self.assertIsNot(f.result(3), QThread.currentThread())

        executor.shutdown()

"""
General helper functions and classes for PyQt concurrent programming
"""
# TODO: Rename the module to something that does not conflict with stdlib
# concurrent
import os
import threading
import atexit
import logging
import warnings
import weakref
from functools import partial
import concurrent.futures

from concurrent.futures import Future, TimeoutError
from contextlib import contextmanager
from typing import Callable, Optional, Any, List

from AnyQt.QtCore import (
    Qt, QObject, QMetaObject, QThreadPool, QThread, QRunnable, QSemaphore,
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
        then a private global thread pool will be used.

        .. versionchanged:: 3.15
            Before 3.15 a `QThreadPool.globalPool()` was used as the default.

        .. warning::
            If you pass a custom `QThreadPool` make sure it creates threads
            with sufficient stack size for the tasks submitted to the executor
            (see `QThreadPool.setStackSize`).
    """
    # A default thread pool. Replaced QThreadPool due to insufficient default
    # stack size for created threads (QTBUG-2568). Not using even on
    # Qt >= 5.10 just for consistency sake.
    class __global:
        __lock = threading.Lock()
        __instance = None

        @classmethod
        def instance(cls):
            # type: () -> concurrent.futures.ThreadPoolExecutor
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = concurrent.futures.ThreadPoolExecutor(
                        max_workers=(os.cpu_count() or 1)
                    )
                return cls.__instance

    def __init__(self, parent=None, threadPool=None, **kwargs):
        super().__init__(parent, **kwargs)

        if threadPool is None:
            threadPool = self.__global.instance()

        self._threadPool = threadPool
        if isinstance(threadPool, QThreadPool):
            def start(runnable):
                # type: (QRunnable) -> None
                threadPool.start(runnable)
        elif isinstance(threadPool, concurrent.futures.Executor):
            # adapt to Executor interface
            def start(runnable):
                # type: (QRunnable) -> None
                threadPool.submit(runnable.run)
        else:
            raise TypeError("Invalid `threadPool` type '{}'"
                            .format(type(threadPool).__name__))
        self.__start = start
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
            self.__start(runnable)
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
            self.__start(runnable)
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


class FutureSetWatcher(QObject):
    """
    An `QObject` watching the state changes of a list of
    `concurrent.futures.Future` instances

    Note
    ----
    The state change notification signals (`doneAt`, `finishedAt`, ...)
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
    futures : List[Future]
        A list of future instance to watch.

    Example
    -------
    >>> app = QCoreApplication.instance() or QCoreApplication([])
    >>> fs = [submit(lambda i, j: i ** j, 10, 3) for i in range(10)]
    >>> watcher = FutureSetWatcher(fs)
    >>> watcher.resultReadyAt.connect(
    ...     lambda i, res: print("Result at {}: {}".format(i, res))
    ... )
    >>> watcher.doneAll.connect(app.quit)
    >>> _ = app.exec()
    Result at 0: 1000
    ...
    """
    #: Signal emitted when the future at `index` is done (cancelled or
    #: finished)
    doneAt = Signal([int, Future])

    #: Signal emitted when the future at index is finished (i.e. returned
    #: a result)
    finishedAt = Signal([int, Future])

    #: Signal emitted when the future at `index` was cancelled.
    cancelledAt = Signal([int, Future])

    #: Signal emitted with the future's result when successfully
    #: finished.
    resultReadyAt = Signal([int, object])

    #: Signal emitted with the future's exception when finished with an
    #: exception.
    exceptionReadyAt = Signal([int, BaseException])

    #: Signal reporting the current completed count
    progressChanged = Signal([int, int])

    #: Signal emitted when all the futures have completed.
    doneAll = Signal()

    def __init__(self, futures=None, *args, **kwargs):
        # type: (List[Future], ...) -> None
        super().__init__(*args, **kwargs)
        self.__futures = None
        self.__semaphore = None
        self.__countdone = 0
        if futures is not None:
            self.setFutures(futures)

    def setFutures(self, futures):
        # type: (List[Future]) -> None
        """
        Set the future instances to watch.

        Raise a `RuntimeError` if futures are already set.

        Parameters
        ----------
        futures : List[Future]
        """
        if self.__futures is not None:
            raise RuntimeError("already set")
        self.__futures = []
        selfweakref = weakref.ref(self)
        schedule_emit = methodinvoke(self, "__emitpending", (int, Future))

        # Semaphore counting the number of future that have enqueued
        # done notifications. Used for the `wait` implementation.
        self.__semaphore = semaphore = QSemaphore(0)

        for i, future in enumerate(futures):
            self.__futures.append(future)

            def on_done(index, f):
                try:
                    selfref = selfweakref()  # not safe really
                    if selfref is None:  # pragma: no cover
                        return
                    try:
                        schedule_emit(index, f)
                    except RuntimeError:  # pragma: no cover
                        # Ignore RuntimeErrors (when C++ side of QObject is deleted)
                        # (? Use QObject.destroyed and remove the done callback ?)
                        pass
                finally:
                    semaphore.release()

            future.add_done_callback(partial(on_done, i))

        if not self.__futures:
            # `futures` was an empty sequence.
            methodinvoke(self, "doneAll", ())()

    @Slot(int, Future)
    def __emitpending(self, index, future):
        # type: (int, Future) -> None
        assert QThread.currentThread() is self.thread()
        assert self.__futures[index] is future
        assert future.done()
        assert self.__countdone < len(self.__futures)
        self.__futures[index] = None
        self.__countdone += 1

        if future.cancelled():
            self.cancelledAt.emit(index, future)
            self.doneAt.emit(index, future)
        elif future.done():
            self.finishedAt.emit(index, future)
            self.doneAt.emit(index, future)
            if future.exception():
                self.exceptionReadyAt.emit(index, future.exception())
            else:
                self.resultReadyAt.emit(index, future.result())
        else:
            assert False

        self.progressChanged.emit(self.__countdone, len(self.__futures))

        if self.__countdone == len(self.__futures):
            self.doneAll.emit()

    def flush(self):
        """
        Flush all pending signal emits currently enqueued.

        Must only ever be called from the thread this object lives in
        (:func:`QObject.thread()`).
        """
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("`flush()` called from a wrong thread.")
        # NOTE: QEvent.MetaCall is the event implementing the
        # `Qt.QueuedConnection` method invocation.
        QCoreApplication.sendPostedEvents(self, QEvent.MetaCall)

    def wait(self):
        """
        Wait for for all the futures to complete and *enqueue* notifications
        to this object, but do not emit any signals.

        Use `flush()` to emit all signals after a `wait()`
        """
        if self.__futures is None:
            raise RuntimeError("Futures were not set.")

        self.__semaphore.acquire(len(self.__futures))
        self.__semaphore.release(len(self.__futures))


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
    @staticmethod
    def from_method(method, arg_types=(), *, conntype=Qt.QueuedConnection):
        """
        Create and return a `methodinvoke` instance from a bound method.

        Parameters
        ----------
        method : Union[types.MethodType, types.BuiltinMethodType]
            A bound method of a QObject registered with the Qt meta object
            system (e.g. decorated by a Slot decorators)
        arg_types : Tuple[Union[type, str]]
            A tuple of positional argument types.
        conntype: Qt.ConnectionType
            The connection/call type (Qt.QueuedConnection and
            Qt.BlockingConnection are the most interesting)

        Returns
        -------
        invoker : methodinvoke
        """
        obj = method.__self__
        name = method.__name__
        return methodinvoke(obj, name, arg_types, conntype=conntype)

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

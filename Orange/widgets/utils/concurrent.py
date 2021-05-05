"""
General helper functions and classes for PyQt concurrent programming
"""
# TODO: Rename the module to something that does not conflict with stdlib
# concurrent
from typing import Callable, Any
import os
import threading
import atexit
import logging
import warnings
from functools import partial
import concurrent.futures
from concurrent.futures import Future, TimeoutError
from contextlib import contextmanager

from AnyQt.QtCore import (
    Qt, QObject, QMetaObject, QThreadPool, QThread, QRunnable,
    QEventLoop, QCoreApplication, QEvent, Q_ARG,
    pyqtSignal as Signal, pyqtSlot as Slot
)

from orangewidget.utils.concurrent import (
    FutureWatcher, FutureSetWatcher, methodinvoke, PyOwned
)

__all__ = [
    "FutureWatcher", "FutureSetWatcher", "methodinvoke",
    "TaskState", "ConcurrentMixin", "ConcurrentWidgetMixin", "PyOwned"
]

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
        self.eventLoop.exec()


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


class TaskState(QObject, PyOwned):

    status_changed = Signal(str)
    _p_status_changed = Signal(str)

    progress_changed = Signal(float)
    _p_progress_changed = Signal(float)

    partial_result_ready = Signal(object)
    _p_partial_result_ready = Signal(object)

    def __init__(self, *args):
        super().__init__(*args)
        self.__future = None
        self.watcher = FutureWatcher()
        self.__interruption_requested = False
        self.__progress = 0
        # Helpers to route the signal emits via a this object's queue.
        # This ensures 'atomic' disconnect from signals for targets/slots
        # in the same thread. Requires that the event loop is running in this
        # object's thread.
        self._p_status_changed.connect(
            self.status_changed, Qt.QueuedConnection)
        self._p_progress_changed.connect(
            self.progress_changed, Qt.QueuedConnection)
        self._p_partial_result_ready.connect(
            self.partial_result_ready, Qt.QueuedConnection)

    @property
    def future(self) -> Future:
        return self.__future

    def set_status(self, text: str):
        self._p_status_changed.emit(text)

    def set_progress_value(self, value: float):
        if round(value, 1) > round(self.__progress, 1):
            # Only emit progress when it has changed sufficiently
            self._p_progress_changed.emit(value)
            self.__progress = value

    def set_partial_result(self, value: Any):
        self._p_partial_result_ready.emit(value)

    def is_interruption_requested(self) -> bool:
        return self.__interruption_requested

    def start(self, executor: concurrent.futures.Executor,
              func: Callable[[], Any] = None) -> Future:
        assert self.future is None
        assert not self.__interruption_requested
        self.__future = executor.submit(func)
        self.watcher.setFuture(self.future)
        return self.future

    def cancel(self) -> bool:
        assert not self.__interruption_requested
        self.__interruption_requested = True
        if self.future is not None:
            rval = self.future.cancel()
        else:
            # not even scheduled yet
            rval = True
        return rval


class ConcurrentMixin:
    """
    A base class for concurrent mixins. The class provides methods for running
    tasks in a separate thread.

    Widgets should use `ConcurrentWidgetMixin` rather than this class.
    """
    def __init__(self):
        self.__executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.__task = None  # type: Optional[TaskState]

    @property
    def task(self) -> TaskState:
        return self.__task

    def on_partial_result(self, result: Any) -> None:
        """ Invoked from runner (by state) to send the partial results
        The method should handle partial results, i.e. show them in the plot.

        :param result: any data structure to hold final result
        """
        raise NotImplementedError

    def on_done(self, result: Any) -> None:
        """ Invoked when task is done.
        The method should re-set the result (to double check it) and
        perform operations with obtained results, eg. send data to the output.

        :param result: any data structure to hold temporary result
        """
        raise NotImplementedError

    def on_exception(self, ex: Exception):
        """ Invoked when an exception occurs during the calculation.
        Override in order to handle exceptions, eg. show an error
        message in the widget.

        :param ex: exception
        """
        raise ex

    def start(self, task: Callable, *args, **kwargs):
        """ Call from derived class to start the task.
        :param task: runner - a method to run in a thread - should accept
        `state` parameter
        """
        self.__cancel_task(wait=False)
        assert callable(task), "`task` must be callable!"
        state = TaskState(self)
        task = partial(task, *(args + (state,)), **kwargs)
        self.__start_task(task, state)

    def cancel(self):
        """ Call from derived class to stop the task. """
        self.__cancel_task(wait=False)

    def shutdown(self):
        """ Call from derived class when the widget is deleted
         (in onDeleteWidget).
        """
        self.__cancel_task(wait=True)
        self.__executor.shutdown(True)

    def __start_task(self, task: Callable[[], Any], state: TaskState):
        assert self.__task is None
        self._connect_signals(state)
        state.start(self.__executor, task)
        state.setParent(self)
        self.__task = state

    def __cancel_task(self, wait: bool = True):
        if self.__task is not None:
            state, self.__task = self.__task, None
            state.cancel()
            self._disconnect_signals(state)
            if wait:
                concurrent.futures.wait([state.future])

    def _connect_signals(self, state: TaskState):
        state.partial_result_ready.connect(self.on_partial_result)
        state.watcher.done.connect(self._on_task_done)

    def _disconnect_signals(self, state: TaskState):
        state.partial_result_ready.disconnect(self.on_partial_result)
        state.watcher.done.disconnect(self._on_task_done)

    def _on_task_done(self, future: Future):
        assert future.done()
        assert self.__task is not None
        assert self.__task.future is future
        assert self.__task.watcher.future() is future
        self.__task = None
        ex = future.exception()
        if ex is not None:
            self.on_exception(ex)
        else:
            self.on_done(future.result())
        # This assert prevents user to start new task (call start) from either
        # on_done or on_exception
        assert self.__task is None, (
            "Starting new task from "
            f"{'on_done' if ex is None else 'on_exception'} is forbidden"
        )


class ConcurrentWidgetMixin(ConcurrentMixin):
    """
    A concurrent mixin to be used along with OWWidget.
    """
    def __set_state_ready(self):
        self.progressBarFinished()
        self.setInvalidated(False)
        self.setStatusMessage("")

    def __set_state_busy(self):
        self.progressBarInit()
        self.setInvalidated(True)

    def start(self, task: Callable, *args, **kwargs):
        self.__set_state_ready()
        super().start(task, *args, **kwargs)
        self.__set_state_busy()

    def cancel(self):
        super().cancel()
        self.__set_state_ready()

    def _connect_signals(self, state: TaskState):
        super()._connect_signals(state)
        state.status_changed.connect(self.setStatusMessage)
        state.progress_changed.connect(self.progressBarSet)

    def _disconnect_signals(self, state: TaskState):
        super()._disconnect_signals(state)
        state.status_changed.disconnect(self.setStatusMessage)
        state.progress_changed.disconnect(self.progressBarSet)

    def _on_task_done(self, future: Future):
        super()._on_task_done(future)
        self.__set_state_ready()

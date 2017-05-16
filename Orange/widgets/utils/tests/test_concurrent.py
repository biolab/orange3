import unittest
import threading

from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

from AnyQt.QtCore import Qt, QObject, QCoreApplication, QThread, pyqtSlot
from AnyQt.QtTest import QSignalSpy

from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, Task, methodinvoke
)


class CoreAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication.instance()
        if self.app is None:
            self.app = QCoreApplication([])

    def tearDown(self):
        self.app.processEvents()
        del self.app


class TestExecutor(CoreAppTestCase):
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
            @pyqtSlot(object)
            def set_state(self, value):
                state[0] = value
                state[1] = QThread.currentThread()

        def func(callback):
            callback(QThread.currentThread())

        obj = StateSetter()
        f1 = executor.submit(func, methodinvoke(obj, "set_state", (object,)))
        f1.result()
        # So invoked method can be called from the event loop
        self.app.processEvents()

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


class TestFutureWatcher(CoreAppTestCase):
    def test_watcher(self):
        executor = ThreadPoolExecutor(max_workers=1)
        f = executor.submit(lambda: 42)
        w = FutureWatcher(f)

        def spies(w):
            return SimpleNamespace(
                done=QSignalSpy(w.done),
                finished=QSignalSpy(w.finished),
                result=QSignalSpy(w.resultReady),
                error=QSignalSpy(w.exceptionReady),
                cancelled=QSignalSpy(w.cancelled)
            )

        spy = spies(w)
        self.assertTrue(spy.done.wait())

        self.assertEqual(list(spy.done), [[f]])
        self.assertEqual(list(spy.finished), [[f]])
        self.assertEqual(list(spy.result), [[42]])
        self.assertEqual(list(spy.error), [])
        self.assertEqual(list(spy.cancelled), [])

        f = executor.submit(lambda: 1/0)
        w = FutureWatcher(f)
        spy = spies(w)

        self.assertTrue(spy.done.wait())

        self.assertEqual(list(spy.done), [[f]])
        self.assertEqual(list(spy.finished), [[f]])
        self.assertEqual(len(spy.error), 1)
        self.assertIsInstance(spy.error[0][0], ZeroDivisionError)
        self.assertEqual(list(spy.result), [])
        self.assertEqual(list(spy.cancelled), [])

        ev = threading.Event()
        # block the executor to test cancellation
        executor.submit(lambda: ev.wait())
        f = executor.submit(lambda: 0)
        w = FutureWatcher(f)
        self.assertTrue(f.cancel())
        ev.set()

        spy = spies(w)

        self.assertTrue(spy.done.wait())

        self.assertEqual(list(spy.done), [[f]])
        self.assertEqual(list(spy.finished), [])
        self.assertEqual(list(spy.error), [])
        self.assertEqual(list(spy.result), [])
        self.assertEqual(list(spy.cancelled), [[f]])


class TestTask(CoreAppTestCase):

    def test_task(self):
        results = []

        task = Task(function=QThread.currentThread)
        task.resultReady.connect(results.append)

        task.start()
        self.app.processEvents()

        self.assertSequenceEqual(results, [QThread.currentThread()])

        thread = QThread()
        thread.start()
        try:
            task = Task(function=QThread.currentThread)
            task.moveToThread(thread)

            self.assertIsNot(task.thread(), QThread.currentThread())
            self.assertIs(task.thread(), thread)
            results = Future()

            def record(value):
                # record the result value and the calling thread
                results.set_result((QThread.currentThread(), value))

            task.resultReady.connect(record, Qt.DirectConnection)
            task.start()
            f = task.future()
            emit_thread, thread_ = results.result(3)
            self.assertIs(f.result(3), thread)
            self.assertIs(emit_thread, thread)
            self.assertIs(thread_, thread)
        finally:
            thread.quit()
            thread.wait()

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

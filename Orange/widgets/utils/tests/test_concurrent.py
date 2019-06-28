import unittest.mock
import warnings

from concurrent.futures import Future

from AnyQt.QtCore import Qt, QCoreApplication, QThread

from Orange.widgets.utils.concurrent import ThreadExecutor,  Task


class CoreAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication.instance()
        if self.app is None:
            self.app = QCoreApplication([])

    def tearDown(self):
        self.app.processEvents()
        del self.app


class TestTask(CoreAppTestCase):
    def setUp(self):
        # This test tests a deprecated class, so ... obviously
        warnings.filterwarnings(
            "ignore", "`Task` has been deprecated", PendingDeprecationWarning)
        warnings.filterwarnings(
            "ignore", "`submit_task` will be deprecated",
            PendingDeprecationWarning)
        super().setUp()

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

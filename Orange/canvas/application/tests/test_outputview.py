import sys
import multiprocessing.pool

from datetime import datetime
from threading import current_thread

from PyQt4.QtCore import Qt, QThread
from ...gui.test import QAppTestCase

from ..outputview import OutputView, TextStream, ExceptHook


class TestOutputView(QAppTestCase):
    def test_outputview(self):
        output = OutputView()
        output.show()

        line1 = "A line \n"
        line2 = "A different line\n"
        output.write(line1)
        self.assertEqual(str(output.toPlainText()), line1)

        output.write(line2)
        self.assertEqual(str(output.toPlainText()), line1 + line2)

        output.clear()
        self.assertEqual(str(output.toPlainText()), "")

        output.writelines([line1, line2])
        self.assertEqual(str(output.toPlainText()), line1 + line2)

        output.setMaximumLines(5)

        def advance():
            now = datetime.now().strftime("%c\n")
            output.write(now)

            text = str(output.toPlainText())
            self.assertLessEqual(len(text.splitlines()), 5)

            self.singleShot(500, advance)

        advance()

        self.app.exec_()

    def test_formated(self):
        output = OutputView()
        output.show()

        output.write("A sword day, ")
        with output.formated(color=Qt.red) as f:
            f.write("a red day...\n")

            with f.formated(color=Qt.green) as f:
                f.write("Actually sir, orcs bleed green.\n")

        bold = output.formated(weight=100, underline=True)
        bold.write("Shutup")

        self.app.exec_()

    def test_threadsafe(self):
        output = OutputView()
        output.resize(500, 300)
        output.show()

        blue_formater = output.formated(color=Qt.blue)
        red_formater = output.formated(color=Qt.red)

        correct = []

        def check_thread(*args):
            correct.append(QThread.currentThread() == self.app.thread())

        blue = TextStream()
        blue.stream.connect(blue_formater.write)
        blue.stream.connect(check_thread)

        red = TextStream()
        red.stream.connect(red_formater.write)
        red.stream.connect(check_thread)

        def printer(i):
            if i % 12 == 0:
                fizzbuz = "fizzbuz"
            elif i % 4 == 0:
                fizzbuz = "buz"
            elif i % 3 == 0:
                fizzbuz = "fizz"
            else:
                fizzbuz = str(i)

            if i % 2:
                writer = blue
            else:
                writer = red

            writer.write("Greetings from thread {0}. "
                         "This is {1}\n".format(current_thread().name,
                                                fizzbuz))

        pool = multiprocessing.pool.ThreadPool(100)
        res = pool.map_async(printer, list(range(10000)))

        self.app.exec_()

        res.wait()

        self.assertTrue(all(correct))
        self.assertTrue(len(correct) == 10000)

    def test_excepthook(self):
        output = OutputView()
        output.resize(500, 300)
        output.show()

        red_formater = output.formated(color=Qt.red)

        red = TextStream()
        red.stream.connect(red_formater.write)

        hook = ExceptHook(stream=red)

        def raise_exception(i):
            try:
                if i % 2 == 0:
                    raise ValueError("odd")
                else:
                    raise ValueError("even")
            except Exception:
                # explicitly call hook (Thread class has it's own handler)
                hook(*sys.exc_info())

        pool = multiprocessing.pool.ThreadPool(10)
        res = pool.map_async(raise_exception, list(range(100)))

        self.app.exec_()

        res.wait()

import sys
import unittest
from unittest.mock import MagicMock

from AnyQt.QtWidgets import QApplication

from Orange.widgets.widget import OWWidget, Input
from Orange.widgets.utils import test_run

# prevent test_run from calling sys.exit
sys.exit = MagicMock()

# prevent multiple initializations of QApplication
app = QApplication([])
test_run.QApplication = lambda *_: app

# prevent entering the application's main loop
app.exec_ = MagicMock()


# MagicMocks have no names, so we patch Input.__call__ to use them
# as input handlers
class MockInput(Input):
    def __call__(self, method):
        self.handler = self.name
        return method


class TestRunTest(unittest.TestCase):
    def setUp(self):
        class MockWidget(OWWidget):
            name = "foo"

            class Inputs:
                input_int = MockInput("int1", int)
                input_float1 = MockInput("float1", float, default=True)
                input_float2 = MockInput("float2", float)
                input_str1 = MockInput("float1", str)
                input_str2 = MockInput("float2", str)

            int1 = Inputs.input_int(MagicMock())
            float1 = Inputs.input_float1(MagicMock())
            float2 = Inputs.input_float2(MagicMock())
            str1 = Inputs.input_str1(MagicMock())
            str2 = Inputs.input_str2(MagicMock())

            # Don't show the widget
            show = MagicMock()

        self.widgetClass = MockWidget

    def test_widget_is_shown_and_ran(self):
        w = self.widgetClass
        app.exec_.reset_mock()

        w.test_run()
        self.assertEqual(w.show.call_count, 1)
        self.assertEqual(app.exec_.call_count, 1)

    def test_no_data(self):
        self.widgetClass.test_run()  # just don't crash

    def test_single_signal(self):
        w = self.widgetClass

        w.test_run(42)
        w.int1.assert_called_with(42)

        w.test_run(3.14)
        w.float1.assert_called_with(3.14)
        self.assertEqual(w.float2.call_count, 0)

        with self.assertRaises(ValueError):
            w.test_run("foo")

        with self.assertRaises(ValueError):
            w.test_run([])

    def test_named_signals(self):
        w = self.widgetClass
        w.test_run(42, float2=2.7, str1="foo")
        w.int1.assert_called_with(42)
        self.assertEqual(w.float1.call_count, 0)
        w.float2.assert_called_with(2.7)
        w.str1.assert_called_with("foo")
        self.assertEqual(w.str2.call_count, 0)


if __name__ == "__main__":
    unittest.main()

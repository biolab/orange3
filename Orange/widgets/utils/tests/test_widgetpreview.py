import sys
import unittest
from unittest.mock import MagicMock, call

from AnyQt.QtWidgets import QApplication

from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input
from Orange.widgets.utils import widgetpreview

app = QApplication([])


# MagicMocks have no names, so we patch Input.__call__ to use them
# as input handlers
class MockInput(Input):
    def __call__(self, method):
        self.handler = self.name
        return method


class TestWidgetPreviewBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_sys_exit = sys.exit
        cls.orig_app_exec = app.exec
        cls.orig_qapplication = widgetpreview.QApplication

        sys.exit = MagicMock()
        widgetpreview.QApplication = MagicMock(return_value=app)
        app.exec_ = MagicMock()

    @classmethod
    def tearDownClass(cls):
        sys.exit = cls.orig_sys_exit
        app.exec_ = cls.orig_app_exec
        widgetpreview.QApplication = cls.orig_qapplication

    def setUp(self):
        # Class is defined within setUp to reset mocks before each test
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

            show = MagicMock()
            saveSettings = MagicMock()

        self.widgetClass = MockWidget
        sys.exit.reset_mock()
        widgetpreview.QApplication.reset_mock()
        app.exec_.reset_mock()


class TestWidgetPreview(TestWidgetPreviewBase):
    def test_widget_is_shown_and_ran(self):
        w = self.widgetClass
        app.exec_.reset_mock()

        previewer = WidgetPreview(w)

        previewer.run()
        w.show.assert_called()
        w.show.reset_mock()
        app.exec_.assert_called()
        app.exec_.reset_mock()
        w.saveSettings.assert_called()
        w.saveSettings.reset_mock()
        sys.exit.assert_called()
        sys.exit.reset_mock()
        self.assertIsNone(previewer.widget)

        previewer.run(no_exit=True)
        w.show.assert_called()
        w.show.reset_mock()
        app.exec_.assert_called()
        app.exec_.reset_mock()
        w.saveSettings.assert_not_called()
        sys.exit.assert_not_called()
        self.assertIsNotNone(previewer.widget)
        widget = previewer.widget

        previewer.run(no_exec=True, no_exit=True)
        w.show.assert_not_called()
        app.exec_.assert_not_called()
        w.saveSettings.assert_not_called()
        sys.exit.assert_not_called()
        self.assertIs(widget, previewer.widget)

        previewer.run(no_exec=True)
        w.show.assert_not_called()
        app.exec_.assert_not_called()
        w.saveSettings.assert_called()
        sys.exit.assert_called()
        self.assertIsNone(previewer.widget)

    def test_single_signal(self):
        w = self.widgetClass

        WidgetPreview(w).run(42)
        w.int1.assert_called_with(42)

        WidgetPreview(w).run(3.14)
        w.float1.assert_called_with(3.14)
        self.assertEqual(w.float2.call_count, 0)

        with self.assertRaises(ValueError):
            WidgetPreview(w).run("foo")

        with self.assertRaises(ValueError):
            WidgetPreview(w).run([])

    def test_named_signals(self):
        w = self.widgetClass
        WidgetPreview(w).run(42, float2=2.7, str1="foo")
        w.int1.assert_called_with(42)
        self.assertEqual(w.float1.call_count, 0)
        w.float2.assert_called_with(2.7)
        w.str1.assert_called_with("foo")
        self.assertEqual(w.str2.call_count, 0)

    def test_multiple_runs(self):
        w = self.widgetClass
        previewer = WidgetPreview(w)
        previewer.run(42, no_exit=True)
        w.int1(43)
        previewer.send_signals([(44, 1), (45, 2)])
        previewer.run(46, no_exit=True)
        w.int1.assert_has_calls(
            [call(42), call(43), call(44, 1), call(45, 2), call(46)])


class TestWidgetPreviewInternal(TestWidgetPreviewBase):
    def test_find_handler_name(self):
        previewer = WidgetPreview(self.widgetClass)
        previewer.create_widget()
        find_name = previewer._find_handler_name
        self.assertEqual(find_name(42), "int1")
        self.assertEqual(find_name(3.14), "float1")
        self.assertRaises(ValueError, find_name, "foo")
        self.assertRaises(ValueError, find_name, [])
        self.assertRaises(ValueError, find_name, [42])

        self.assertEqual(find_name([(42, 1)]), "int1")
        self.assertEqual(find_name([(2, 1), (3, 2)]), "int1")

        self.assertEqual(find_name([(42.4, 1)]), "float1")
        self.assertEqual(find_name([(42.4, 1), (5.1, 1)]), "float1")

        self.assertRaises(ValueError, find_name, [("foo", 1)])
        self.assertRaises(ValueError, find_name, [])

    def test_data_chunks(self):
        self.assertEqual(
            list(WidgetPreview._data_chunks(42)),
            [(42, )])
        self.assertEqual(
            list(WidgetPreview._data_chunks((42, 1))),
            [(42, 1)])
        self.assertEqual(
            list(WidgetPreview._data_chunks([(42, 1), (65, 3)])),
            [(42, 1), (65, 3)])

    def test_create_widget(self):
        previewer = WidgetPreview(self.widgetClass)
        self.assertIsNone(previewer.widget)
        previewer.create_widget()
        self.assertIsInstance(previewer.widget, self.widgetClass)

    def test_send_signals(self):
        previewer = WidgetPreview(self.widgetClass)
        previewer.create_widget()
        widget = previewer.widget
        previewer.send_signals(42)
        widget.int1.assert_called_with(42)
        widget.int1.reset_mock()
        previewer.send_signals(
            [(42, 1), (40, 2)],
            str2="foo",
            float1=[(3.14, 1), (5.1, 8)])
        widget.int1.assert_has_calls([call(42, 1), call(40, 2)])
        widget.str2.assert_called_with("foo")
        widget.float1.assert_has_calls([call(3.14, 1), call(5.1, 8)])


if __name__ == "__main__":
    unittest.main()

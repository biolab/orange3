# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, unused-wildcard-import
# pylint: disable=wildcard-import, protected-access
import sys
import unittest

from AnyQt.QtCore import QMimeData, QUrl, QPoint, Qt
from AnyQt.QtGui import QDragEnterEvent

from Orange.data import Table
from Orange.classification import LogisticRegressionLearner
from Orange.tests import named_file
from Orange.widgets.data.owpythonscript import OWPythonScript, \
    read_file_content, Script, OWPythonScriptDropHandler
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget, Input

# import tests for python editor
from Orange.widgets.data.utils.pythoneditor.tests.test_api import *
from Orange.widgets.data.utils.pythoneditor.tests.test_bracket_highlighter import *
from Orange.widgets.data.utils.pythoneditor.tests.test_draw_whitespace import *
from Orange.widgets.data.utils.pythoneditor.tests.test_edit import *
from Orange.widgets.data.utils.pythoneditor.tests.test_indent import *
from Orange.widgets.data.utils.pythoneditor.tests.test_indenter.test_python import *
from Orange.widgets.data.utils.pythoneditor.tests.test_rectangular_selection import *
from Orange.widgets.data.utils.pythoneditor.tests.test_vim import *


class TestOWPythonScript(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPythonScript)
        self.iris = Table("iris")
        self.learner = LogisticRegressionLearner()
        self.model = self.learner(self.iris)

    def tearDown(self):
        # clear sys.last_*, these are set/used by interactive interpreter
        sys.last_type = sys.last_value = sys.last_traceback = None
        super().tearDown()

    def test_inputs(self):
        """Check widget's inputs"""
        for input_, data in (("Data", self.iris),
                             ("Learner", self.learner),
                             ("Classifier", self.model),
                             ("Object", "object")):
            self.assertEqual(getattr(self.widget, input_.lower()), [])
            self.send_signal(input_, data, 1)
            self.assertEqual(getattr(self.widget, input_.lower()), [data])
            self.send_signal(input_, None, 1)
            self.assertEqual(getattr(self.widget, input_.lower()), [None])
            self.send_signal(input_, Input.Closed, 1)
            self.assertEqual(getattr(self.widget, input_.lower()), [])

    def test_outputs(self):
        """Check widget's outputs"""
        for signal, data in (
                ("Data", self.iris),
                ("Learner", self.learner),
                ("Classifier", self.model)):
            lsignal = signal.lower()
            self.widget.text.setPlainText("out_{0} = in_{0}".format(lsignal))
            self.send_signal(signal, data, 1)
            self.assertIs(self.get_output(signal), data)
            self.send_signal(signal, None, 1)
            self.widget.text.setPlainText("print(in_{})".format(lsignal))
            self.widget.execute_button.click()
            self.assertIsNone(self.get_output(signal))

    def test_local_variable(self):
        """Check if variable remains in locals after removed from script"""
        self.widget.text.setPlainText("temp = 42\nprint(temp)")
        self.widget.execute_button.click()
        self.assertIn("42", self.widget.console.toPlainText())
        self.widget.text.setPlainText("print(temp)")
        self.widget.execute_button.click()
        self.assertNotIn("NameError: name 'temp' is not defined",
                         self.widget.console.toPlainText())

    def test_wrong_outputs(self):
        """
        Error is shown when output variables are filled with wrong variable
        types and also output variable is set to None. (GH-2308)
        """
        self.assertEqual(len(self.widget.Error.active), 0)
        for signal, data in (
                ("Data", self.iris),
                ("Learner", self.learner),
                ("Classifier", self.model)):
            lsignal = signal.lower()
            self.send_signal(signal, data, 1)
            self.widget.text.setPlainText("out_{} = 42".format(lsignal))
            self.widget.execute_button.click()
            self.assertEqual(self.get_output(signal), None)
            self.assertTrue(hasattr(self.widget.Error, lsignal))
            self.assertTrue(getattr(self.widget.Error, lsignal).is_shown())

            self.widget.text.setPlainText("out_{0} = in_{0}".format(lsignal))
            self.widget.execute_button.click()
            self.assertIs(self.get_output(signal), data)
            self.assertFalse(getattr(self.widget.Error, lsignal).is_shown())

    def test_owns_errors(self):
        self.assertIsNot(self.widget.Error, OWWidget.Error)

    def test_multiple_signals(self):
        click = self.widget.execute_button.click
        console_locals = self.widget.console.locals

        titanic = Table("titanic")

        click()
        self.assertIsNone(console_locals["in_data"])
        self.assertEqual(console_locals["in_datas"], [])

        self.send_signal("Data", self.iris, 1)
        click()
        self.assertIs(console_locals["in_data"], self.iris)
        datas = console_locals["in_datas"]
        self.assertEqual(len(datas), 1)
        self.assertIs(datas[0], self.iris)

        self.send_signal("Data", titanic, 2)
        click()
        self.assertIsNone(console_locals["in_data"])
        self.assertEqual({id(obj) for obj in console_locals["in_datas"]},
                         {id(self.iris), id(titanic)})

        self.send_signal("Data", None, 2)
        click()
        datas = console_locals["in_datas"]
        self.assertEqual(len(datas), 2)
        self.assertIs(datas[0], self.iris)
        self.assertIs(datas[1], None)

        self.send_signal("Data", Input.Closed, 2)
        click()
        self.assertIs(console_locals["in_data"], self.iris)
        datas = console_locals["in_datas"]
        self.assertEqual(len(datas), 1)
        self.assertIs(datas[0], self.iris)

        self.send_signal("Data", Input.Closed, 1)
        click()
        self.assertIsNone(console_locals["in_data"])
        self.assertEqual(console_locals["in_datas"], [])

    def test_store_new_script(self):
        self.widget.text.setPlainText("42")
        self.widget.onAddScript()
        script = self.widget.text.toPlainText()
        self.assertEqual("42", script)

    def test_restore_from_library(self):
        before = self.widget.text.toPlainText()
        self.widget.text.setPlainText("42")
        self.widget.restoreSaved()
        script = self.widget.text.toPlainText()
        self.assertEqual(before, script)

    def test_store_current_script(self):
        self.widget.text.setPlainText("42")
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.widget = self.create_widget(OWPythonScript)
        script = self.widget.text.toPlainText()
        self.assertNotEqual("42", script)
        self.widget = self.create_widget(OWPythonScript, stored_settings=settings)
        script = self.widget.text.toPlainText()
        self.assertEqual("42", script)

    def test_read_file_content(self):
        with named_file("Content", suffix=".42") as fn:
            # valid file opens
            content = read_file_content(fn)
            self.assertEqual("Content", content)
            # invalid utf-8 file does not
            with open(fn, "wb") as f:
                f.write(b"\xc3\x28")
            content = read_file_content(fn)
            self.assertIsNone(content)

    def test_script_insert_mime_text(self):
        current = self.widget.text.toPlainText()
        insert = "test\n"
        cursor = self.widget.text.cursor()
        cursor.setPos(0, 0)
        mime = QMimeData()
        mime.setText(insert)
        self.widget.text.insertFromMimeData(mime)
        self.assertEqual(insert + current, self.widget.text.toPlainText())

    def test_script_insert_mime_file(self):
        with named_file("test", suffix=".42") as fn:
            previous = self.widget.text.toPlainText()
            mime = QMimeData()
            url = QUrl.fromLocalFile(fn)
            mime.setUrls([url])
            self.widget.text.insertFromMimeData(mime)
            text = self.widget.text.toPlainText().split("print('Hello world')")[0]
            self.assertTrue(
                "'" + fn + "'",
                text
            )
            self.widget.text.undo()
            self.assertEqual(previous, self.widget.text.toPlainText())

    def test_dragEnterEvent_accepts_text(self):
        with named_file("Content", suffix=".42") as fn:
            event = self._drag_enter_event(QUrl.fromLocalFile(fn))
            self.widget.dragEnterEvent(event)
            self.assertTrue(event.isAccepted())

    def test_dragEnterEvent_rejects_binary(self):
        with named_file("", suffix=".42") as fn:
            with open(fn, "wb") as f:
                f.write(b"\xc3\x28")
            event = self._drag_enter_event(QUrl.fromLocalFile(fn))
            self.widget.dragEnterEvent(event)
            self.assertFalse(event.isAccepted())

    def _drag_enter_event(self, url):
        # make sure data does not get garbage collected before it used
        # pylint: disable=attribute-defined-outside-init
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])
        return QDragEnterEvent(
            QPoint(0, 0), Qt.MoveAction, data,
            Qt.NoButton, Qt.NoModifier)

    def test_migrate(self):
        w = self.create_widget(OWPythonScript, {
            "libraryListSource": [Script("A", "1")],
            "__version__": 0
        })
        self.assertEqual(w.libraryListSource[0].name, "A")

    def test_restore(self):
        w = self.create_widget(OWPythonScript, {
            "scriptLibrary": [dict(name="A", script="1", filename=None)],
            "__version__": 2
        })
        self.assertEqual(w.libraryListSource[0].name, "A")

    def test_no_shared_namespaces(self):
        """
        Previously, Python Script widgets in the same schema shared a namespace.
        I (irgolic) think this is just a way to encourage users in writing
        messy workflows with race conditions, so I encourage them to share
        between Python Script widgets with Object signals.
        """
        widget1 = self.create_widget(OWPythonScript)
        widget2 = self.create_widget(OWPythonScript)

        click1 = widget1.execute_button.click
        click2 = widget2.execute_button.click

        widget1.text.text = "x = 42"
        click1()

        widget2.text.text = "y = 2 * x"
        click2()
        self.assertIn("NameError: name 'x' is not defined",
                      widget2.console.toPlainText())


class TestOWPythonScriptDropHandler(unittest.TestCase):
    def test_canDropFile(self):
        handler = OWPythonScriptDropHandler()
        self.assertTrue(handler.canDropFile(__file__))
        self.assertFalse(handler.canDropFile("test.tab"))

    def test_parametersFromFile(self):
        handler = OWPythonScriptDropHandler()
        r = handler.parametersFromFile(__file__)
        item = r["scriptLibrary"][0]
        self.assertEqual(item["filename"], __file__)


if __name__ == '__main__':
    unittest.main()

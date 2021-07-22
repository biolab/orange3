# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, unused-wildcard-import
# pylint: disable=wildcard-import, protected-access
from AnyQt.QtCore import QMimeData, QUrl, QPoint, Qt
from AnyQt.QtGui import QDragEnterEvent

from Orange.data import Table
from Orange.classification import LogisticRegressionLearner
from Orange.tests import named_file
from Orange.widgets.data.owpythonscript import OWPythonScript, read_file_content, Script
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget

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
    iris = Table("iris")
    learner = LogisticRegressionLearner()
    model = learner(iris)
    default_settings = None
    python_widget = None

    def setUp(self):
        if type(self).python_widget is None:
            type(self).python_widget = self.create_widget(
                OWPythonScript, stored_settings=self.default_settings)
        self.widget = self.python_widget
        self.wait_execute_script('clear')

    def wait_execute_script(self, script=None):
        """
        Tests that invoke scripts take longer,
        because they wait for the IPython kernel.
        """
        done = False

        def results_ready_callback():
            nonlocal done
            done = True

        def execution_finished_callback(success):
            if not success:
                nonlocal done
                done = True

        self.widget.console.execution_finished.connect(execution_finished_callback)
        self.widget.console.results_ready.connect(results_ready_callback)

        def is_done():
            return done

        if script is not None:
            self.widget.editor.text = script
        self.widget.execute_button.click()

        def is_ready_and_clear():
            return self.widget.console._OrangeConsoleWidget__is_ready and \
                   self.widget.console._OrangeConsoleWidget__queued_execution is None and \
                   not self.widget.console._OrangeConsoleWidget__executing and \
                   self.widget.console._OrangeConsoleWidget__queued_broadcast is None and \
                   not self.widget.console._OrangeConsoleWidget__broadcasting

        if not is_ready_and_clear():
            self.process_events(until=is_ready_and_clear, timeout=30000)
        self.process_events(until=is_done)

        self.widget.console.results_ready.disconnect(results_ready_callback)
        self.widget.console.execution_finished.disconnect(execution_finished_callback)

    def test_inputs(self):
        """Check widget's inputs"""
        for input_, data in (("Data", self.iris),
                             ("Learner", self.learner),
                             ("Classifier", self.model),
                             ("Object", "object")):
            self.assertEqual(getattr(self.widget, input_.lower()), {})
            self.send_signal(input_, data, 1)
            self.assertEqual(getattr(self.widget, input_.lower()), {1: data})
            self.send_signal(input_, None, 1)
            self.assertEqual(getattr(self.widget, input_.lower()), {})

    def test_outputs(self):
        """Check widget's outputs"""
        # The type equation method for learners and classifiers probably isn't ideal,
        # but it's something. The problem is that the console runs in a separate
        # python process, so identity is broken when the objects are sent from
        # process to process. If python3.8 shared memory is implemented for
        # main process <-> IPython kernel communication,
        # change this test back to checking identity equality.
        for signal, data, assert_method in (
                ("Data", self.iris, self.assert_table_equal),
                ("Learner", self.learner, lambda a, b: self.assertEqual(type(a), type(b))),
                ("Classifier", self.model, lambda a, b: self.assertEqual(type(a), type(b)))):
            lsignal = signal.lower()
            self.send_signal(signal, data, (1,))
            self.wait_execute_script("out_{0} = in_{0}".format(lsignal))
            assert_method(self.get_output(signal), data)
            self.wait_execute_script("print(5)")
            assert_method(self.get_output(signal), data)
            self.send_signal(signal, None, (1,))
            assert_method(self.get_output(signal), data)
            self.wait_execute_script("print(5)")
            self.assertIsNone(self.get_output(signal))

    def test_local_variable(self):
        """Check if variable remains in locals after removed from script"""
        self.wait_execute_script("temp = 42\nprint(temp)")
        self.assertIn('42', self.widget.console._control.toPlainText())

        # after a successful execution, previous outputs are cleared
        self.wait_execute_script("print(temp)")
        self.assertNotIn("NameError: name 'temp' is not defined",
                         self.widget.console._control.toPlainText())

    def test_wrong_outputs(self):
        """
        Warning is shown when output variables are filled with wrong variable
        types and also output variable is set to None. (GH-2308)
        """
        self.widget.orangeDataTablesEnabled = True
        # see comment in test_outputs()
        for signal, data, assert_method in (
                ("Data", self.iris, self.assert_table_equal),
                ("Learner", self.learner, lambda a, b: self.assertEqual(type(a), type(b))),
                ("Classifier", self.model, lambda a, b: self.assertEqual(type(a), type(b)))):
            lsignal = signal.lower()
            self.send_signal(signal, data, (1,))
            self.wait_execute_script("out_{} = 42".format(lsignal))
            assert_method(self.get_output(signal), None)
            self.assertTrue(self.widget.Warning.illegal_var_type.is_shown())

            self.wait_execute_script("out_{0} = in_{0}".format(lsignal))
            assert_method(self.get_output(signal), data)
            self.assertFalse(self.widget.Warning.illegal_var_type.is_shown())

    def test_owns_errors(self):
        self.assertIsNot(self.widget.Error, OWWidget.Error)

    def test_multiple_signals(self):
        titanic = Table("titanic")

        self.wait_execute_script('clear')

        # if no data input signal, in_data is None
        self.wait_execute_script("print(in_data)")
        self.assertIn("None",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if no data input signal, in_datas is empty list
        self.wait_execute_script("print(in_datas)")
        self.assertIn("[]",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if one data input signal, in_data is iris
        self.send_signal("Data", self.iris, (1,))
        self.wait_execute_script("in_data")
        self.assertIn(repr(self.iris),
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if one data input signal, in_datas is of len 1
        self.wait_execute_script("'in_datas len: ' + str(len(in_datas))")
        self.assertIn("in_datas len: 1",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if two data input signals, in_data is defined
        self.send_signal("Data", titanic, (2,))
        self.wait_execute_script("print(in_data)")
        self.assertNotIn("None",
                         self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if two data input signals, in_datas is of len 2
        self.wait_execute_script("'in_datas len: ' + str(len(in_datas))")
        self.assertIn("in_datas len: 2",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if two data signals, in_data == in_datas[0]
        self.wait_execute_script('in_data == in_datas[0]')
        self.assertIn("True",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to one data signal, in_data is titanic
        self.send_signal("Data", None, (1,))

        self.wait_execute_script("in_data")
        self.assertIn(repr(titanic),
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to one data signal after removing first signal, in_data == in_datas[0]
        self.wait_execute_script('in_data == in_datas[0]')
        self.assertIn("True",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to no data signal, in_data is None
        self.send_signal("Data", None, (2,))

        self.wait_execute_script("print(in_data)")
        self.assertIn("None",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to no data signal, in_datas is undefined
        self.wait_execute_script("print(in_datas)")
        self.assertIn("[]",
                      self.widget.console._control.toPlainText())

    def test_store_new_script(self):
        self.widget.editor.text = "42"
        self.widget.onAddScript()
        script = self.widget.editor.toPlainText()
        self.assertEqual("42", script)

    def test_restore_from_library(self):
        self.widget.restoreSaved()
        before = self.widget.editor.text
        self.widget.editor.text = "42"
        self.widget.restoreSaved()
        script = self.widget.editor.text
        self.assertEqual(before, script)

    def test_store_current_script(self):
        self.widget.editor.text = "42"
        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWPythonScript)
        script = widget.editor.text
        self.assertNotEqual("42", script)
        widget2 = self.create_widget(OWPythonScript, stored_settings=settings)
        script = widget2.editor.text
        self.assertEqual("42", script)
        widget.onDeleteWidget()
        widget2.onDeleteWidget()

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
        current = self.widget.editor.text
        insert = "test\n"
        cursor = self.widget.editor.cursor()
        cursor.setPos(0, 0)
        mime = QMimeData()
        mime.setText(insert)
        self.widget.editor.insertFromMimeData(mime)
        self.assertEqual(insert + current, self.widget.editor.text)

    def test_script_insert_mime_file(self):
        with named_file("test", suffix=".42") as fn:
            previous = self.widget.editor.text
            mime = QMimeData()
            url = QUrl.fromLocalFile(fn)
            mime.setUrls([url])
            self.widget.editor.insertFromMimeData(mime)
            text = self.widget.editor.text.split("print('Hello world')")[0]
            self.assertTrue(
                "'" + fn + "'",
                text
            )
            self.widget.editor.undo()
            self.assertEqual(previous, self.widget.editor.text)

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

    def test_migrate_2(self):
        w = self.create_widget(OWPythonScript, {
            '__version__': 2
        })
        self.assertTrue(w.useInProcessKernel)

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

        self.send_signal(widget1.Inputs.data, self.iris, (1,), widget=widget1)
        self.widget = widget1
        self.wait_execute_script("x = 42")

        self.widget = widget2
        self.wait_execute_script("y = 2 * x")
        self.assertIn("NameError: name 'x' is not defined",
                      self.widget.console._control.toPlainText())

    def test_unreferencible(self):
        self.wait_execute_script('out_object = 14')
        self.assertEqual(self.get_output("Object"), 14)
        self.wait_execute_script('out_object = ("a",14)')
        self.assertEqual(self.get_output("Object"), ('a', 14))


class TestInProcessOWPythonScript(TestOWPythonScript):
    default_settings = {'useInProcessKernel': True}
    python_widget = None


if __name__ == '__main__':
    unittest.main()

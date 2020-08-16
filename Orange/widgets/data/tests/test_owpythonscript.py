# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
# pylint: disable=protected-access
import os
import shutil

from AnyQt.QtCore import QMimeData, QUrl, QPoint, Qt
from AnyQt.QtGui import QDragEnterEvent, QDropEvent

from Orange.data import Table
from Orange.classification import LogisticRegressionLearner
from Orange.tests import named_file
from Orange.widgets.data.owpythonscript import OWPythonScript, read_file_content, Script, \
    DEFAULT_FILENAME, SCRIPTS_FOLDER_PATH
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget


class TestOWPythonScript(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWPythonScript)
        self.iris = Table("iris")
        self.learner = LogisticRegressionLearner()
        self.model = self.learner(self.iris)

    def tearDown(self):
        super().tearDown()
        self.widget.onDeleteWidget()
        shutil.rmtree(SCRIPTS_FOLDER_PATH)

    def wait_execute_script(self, script):
        """
        Tests that invoke scripts take longer,
        because they wait for the IPython kernel.
        """
        if not self.widget.console._OrangeConsoleWidget__is_ready:
            self.process_events(until=lambda: self.widget.console._OrangeConsoleWidget__is_ready,
                                timeout=30000)

        done = False

        def results_ready_callback():
            self.widget.console.results_ready.disconnect(results_ready_callback)
            nonlocal done
            done = True

        def execution_finished_callback(success):
            self.widget.console.execution_finished.disconnect(execution_finished_callback)
            if success:
                self.widget.console.results_ready.connect(results_ready_callback)
            else:
                nonlocal done
                done = True

        self.widget.console.execution_finished.connect(execution_finished_callback)

        def is_done():
            return done

        self.widget.editor.text = script
        self.widget.execute_button.click()
        self.process_events(until=is_done)

    def test_inputs(self):
        """Check widget's inputs"""
        for input_, data in (("Data", self.iris),
                             ("Learner", self.learner),
                             ("Classifier", self.model),
                             ("Object", "object")):
            self.assertEqual(getattr(self.widget, input_.lower()), {})
            self.send_signal(input_, data, (1,))
            self.assertEqual(getattr(self.widget, input_.lower()), {1: data})
            self.send_signal(input_, None, (1,))
            self.assertEqual(getattr(self.widget, input_.lower()), {})

    def test_outputs(self):
        """Check widget's outputs"""
        self.widget.orangeDataTablesEnabled = True
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
            self.send_signal(signal, data, (1, ))
            self.wait_execute_script("out_{} = 42".format(lsignal))
            assert_method(self.get_output(signal), None)
            self.assertTrue(self.widget.Warning.illegal_var_type.is_shown())

            self.wait_execute_script("out_{0} = in_{0}".format(lsignal))
            assert_method(self.get_output(signal), data)
            self.assertFalse(self.widget.Warning.illegal_var_type.is_shown())

    def test_owns_errors(self):
        self.assertIsNot(self.widget.Error, OWWidget.Error)

    def test_multiple_signals(self):
        self.widget.orangeDataTablesEnabled = True
        titanic = Table("titanic")

        # if no data input signal, in_data is undefined
        self.wait_execute_script("in_data")
        self.assertIn("NameError: name 'in_data' is not defined",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if no data input signal, in_datas is undefined
        self.wait_execute_script("in_datas")
        self.assertIn("NameError: name 'in_datas' is not defined",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # if one data input signal, in_data is iris
        self.send_signal("Data", self.iris, (1, ))
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
        self.send_signal("Data", titanic, (2, ))
        self.wait_execute_script("in_data")
        self.assertNotIn("NameError: name 'in_data' is not defined",
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
        self.send_signal("Data", None, (1, ))

        self.wait_execute_script("in_data")
        self.assertIn(repr(titanic),
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to one data signal after removing first signal, in_data == in_datas[0]
        self.wait_execute_script('in_data == in_datas[0]')
        self.assertIn("True",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to no data signal, in_data is undefined
        self.send_signal("Data", None, (2, ))

        self.wait_execute_script("in_data")
        self.assertIn("NameError: name 'in_data' is not defined",
                      self.widget.console._control.toPlainText())

        self.wait_execute_script('clear')

        # back to no data signal, in_datas is undefined
        self.wait_execute_script("in_datas")
        self.assertIn("NameError: name 'in_datas' is not defined",
                      self.widget.console._control.toPlainText())

    def test_add_new_script(self):
        self.widget.editor.text = "42"
        self.widget.onAddScript()
        script = self.widget.editor.text
        self.assertEqual("", script)

    def test_save_restore_script(self):
        self.widget.editor.text = "42"
        self.widget.commitChangesToLibrary()
        self.widget.editor.text = "53"
        self.widget.restoreSaved()
        self.assertEqual("42", self.widget.editor.text)

    def test_save_restore_multiple_widgets(self):
        self.widget.editor.text = "42"
        self.widget.commitChangesToLibrary()
        i = self.widget.selectedScriptIndex()
        widget2 = self.create_widget(OWPythonScript)
        widget2.setSelectedScript(i)
        widget2.editor.text = "53"
        widget2.restoreSaved()
        self.assertEqual("42", widget2.editor.text)

    def test_rename_multiple_widgets(self):
        self.widget.editor.text = "42"
        self.widget.commitChangesToLibrary()
        i = self.widget.selectedScriptIndex()
        widget2 = self.create_widget(OWPythonScript)
        self.widget.libraryList[i].filename = 'baba.py'
        self.assertEqual('baba.py', widget2.libraryList[i].filename)

    def test_save_name_collision_multiple_widgets(self):
        widget2 = self.create_widget(OWPythonScript)
        self.widget.onAddScript()
        widget2.onAddScript()
        s1 = self.widget.libraryList[self.widget.selectedScriptIndex()]
        s2 = widget2.libraryList[self.widget.selectedScriptIndex()]
        s1.filename = s2.filename
        self.widget.commitChangesToLibrary()
        self.assertNotEqual(s1.filename, s2.filename)

    def test_save_delete_multiple_widgets(self):
        widget2 = self.create_widget(OWPythonScript)
        self.widget.onAddScript()
        s1 = self.widget.libraryList[self.widget.selectedScriptIndex()]
        self.assertTrue(s1.flags & Script.MissingFromFilesystem)
        self.assertNotIn(s1.filename, [s.filename for s in widget2.libraryList])
        self.widget.commitChangesToLibrary()
        self.assertFalse(s1.flags & Script.MissingFromFilesystem)
        self.assertIn(s1.filename, [s.filename for s in widget2.libraryList])
        self.widget.removeScript(self.widget.selectedScriptIndex())
        self.assertNotIn(s1.filename, [s.filename for s in self.widget.libraryList])
        self.assertNotIn(s1.filename, [s.filename for s in widget2.libraryList])

    def test_restore_from_library(self):
        self.widget.commitChangesToLibrary()
        before = self.widget.editor.text
        self.widget.editor.text = "42"
        self.widget.restoreSaved()
        script = self.widget.editor.text
        self.assertEqual(before, script)

    def test_store_current_script(self):
        self.widget.editor.text = "42"
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.widget = self.create_widget(OWPythonScript)
        script = self.widget.editor.text
        self.assertNotEqual("42", script)
        self.widget = self.create_widget(OWPythonScript, stored_settings=settings)
        script = self.widget.editor.text
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
            self.assertIn(fn, self.widget.editor.text)
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

    def test_dropEvent_adds_file_to_library_if_py(self):
        with named_file("test", suffix=".py") as fn:
            event = self._drop_event(QUrl.fromLocalFile(fn))
            libLen = len(self.widget.libraryList)
            self.widget.dropEvent(event)
            self.assertEqual(libLen + 1, len(self.widget.libraryList))
            self.assertEqual("test", self.widget.libraryList[-1].script)
        with named_file("test", suffix=".42") as fn:
            event = self._drop_event(QUrl.fromLocalFile(fn))
            libLen = len(self.widget.libraryList)
            self.widget.dropEvent(event)
            self.assertEqual(libLen, len(self.widget.libraryList))

    def _drop_event(self, url):
        # make sure data does not get garbage collected before it used
        # pylint: disable=attribute-defined-outside-init
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])

        return QDropEvent(
            QPoint(0, 0), Qt.MoveAction, data,
            Qt.NoButton, Qt.NoModifier, QDropEvent.Drop)

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

    def test_edit_file_externally(self):
        self.widget.commitChangesToLibrary()
        filename = os.path.join(SCRIPTS_FOLDER_PATH, DEFAULT_FILENAME + '.py')
        with open(filename, 'w') as f:
            f.write('52')
        self.widget.restoreSaved()
        self.assertEqual('52', self.widget.editor.text)
        index = self.widget.selectedScriptIndex()
        script = self.widget.libraryList[index]
        os.remove(filename)
        self.assertFalse(script.flags & Script.MissingFromFilesystem)
        self.widget.editor.text = '562'
        self.widget.restoreSaved()
        self.assertEqual('52', self.widget.editor.text, )
        self.assertTrue(script.flags & Script.MissingFromFilesystem)

    def test_migrate_0(self):
        class _Script:
            def __init__(self, name, script):
                self.name = name
                self.script = script
        w = self.create_widget(OWPythonScript, {
            "libraryListSource": [_Script('A', '1')],
            "__version__": 0
        })
        self.assertEqual("1", w.editor.text)
        self.assertEqual('A.py', w.libraryList[w.selectedScriptIndex()].filename)

    def test_migrate_2(self):
        w = self.create_widget(OWPythonScript, {
            "currentScriptIndex": 1,
            "scriptLibrary": [dict(name="A", script="1", filename=None),
                              dict(name="B", script="2", filename=None)],
            "__version__": 2
        })
        self.assertEqual(w.editor.text, '2')
        self.assertEqual(w.libraryList[w.selectedScriptIndex()].filename, 'B.py')
        w = self.create_widget(OWPythonScript, {
            "currentScriptIndex": 0,
            "scriptLibrary": [dict(name="C", script="1", filename=None),
                              dict(name="D", script="2", filename=None)],
            "scriptText": 'haha',
            "__version__": 2
        })
        self.assertEqual(w.editor.text, 'haha')
        self.assertEqual(w.libraryList[w.selectedScriptIndex()].filename, 'C.py')

    def test_restore(self):
        w = self.create_widget(OWPythonScript, {
            "current_script": ('hi', 'm.py'),
            "other_scratch_scripts": [('no', 'n.py'), ('way', 'y.py')],
            "__version__": 3
        })
        self.assertEqual(w.editor.text, 'hi')
        script_names = [s.filename for s in w.libraryList]
        for filename in ('m.py', 'n.py', 'y.py'):
            self.assertIn(filename, script_names)

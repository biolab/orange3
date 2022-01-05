# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,protected-access
from os import path, remove, getcwd
from os.path import dirname
import unittest
from unittest.mock import Mock, patch
import pickle
import tempfile
import warnings
import time

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QMimeData, QPoint, Qt, QUrl, QThread, QObject
from AnyQt.QtGui import QDragEnterEvent, QDropEvent
from AnyQt.QtWidgets import QComboBox

import Orange

from Orange.data import FileFormat, dataset_dirs, StringVariable, Table, \
    Domain, DiscreteVariable, ContinuousVariable
from Orange.util import OrangeDeprecationWarning

from Orange.data.io import TabReader
from Orange.tests import named_file
from Orange.widgets.data.owfile import OWFile, OWFileDropHandler, DEFAULT_READER_TEXT
from Orange.widgets.utils.filedialogs import dialog_formats, format_filter, RecentPath
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.domaineditor import ComboDelegate, VarTypeDelegate, VarTableModel

TITANIC_PATH = path.join(path.dirname(Orange.__file__), 'datasets', 'titanic.tab')
orig_path_exists = path.exists


class FailedSheetsFormat(FileFormat):
    EXTENSIONS = ('.failed_sheet',)
    DESCRIPTION = "Make a sheet function that fails"

    def read(self):
        pass

    def sheets(self):
        raise Exception("Not working")


class WithWarnings(FileFormat):
    EXTENSIONS = ('.with_warning',)
    DESCRIPTION = "Warning"

    @staticmethod
    def read():
        warnings.warn("Some warning")
        return Orange.data.Table("iris")


class MyCustomTabReader(FileFormat):
    EXTENSIONS = ('.tab',)
    DESCRIPTION = "Always return iris"
    PRIORITY = 999999

    @staticmethod
    def read():
        return Orange.data.Table("iris")


class TestOWFile(WidgetTest):
    # Attribute used to store event data so it does not get garbage
    # collected before event is processed.
    event_data = None

    def setUp(self):
        self.widget = self.create_widget(OWFile)  # type: OWFile
        dataset_dirs.append(dirname(__file__))

    def tearDown(self):
        dataset_dirs.pop()

    def test_dragEnterEvent_accepts_urls(self):
        event = self._drag_enter_event(QUrl.fromLocalFile(TITANIC_PATH))
        self.widget.dragEnterEvent(event)
        self.assertTrue(event.isAccepted())

    def test_dragEnterEvent_skips_osx_file_references(self):
        event = self._drag_enter_event(QUrl.fromLocalFile('/.file/id=12345'))
        self.widget.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())

    def test_dragEnterEvent_skips_usupported_files(self):
        event = self._drag_enter_event(QUrl.fromLocalFile('file.unsupported'))
        self.widget.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())

    def _drag_enter_event(self, url):
        # make sure data does not get garbage collected before it used
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])
        return QDragEnterEvent(
            QPoint(0, 0), Qt.MoveAction, data,
            Qt.NoButton, Qt.NoModifier)

    def test_dropEvent_selects_file(self):
        self.widget.load_data = Mock()
        self.widget.source = OWFile.URL

        event = self._drop_event(QUrl.fromLocalFile(TITANIC_PATH))
        self.widget.dropEvent(event)

        self.assertEqual(self.widget.source, OWFile.LOCAL_FILE)
        self.assertTrue(path.samefile(self.widget.last_path(), TITANIC_PATH))
        self.widget.load_data.assert_called_with()

    def _drop_event(self, url):
        # make sure data does not get garbage collected before it used
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])

        return QDropEvent(
            QPoint(0, 0), Qt.MoveAction, data,
            Qt.NoButton, Qt.NoModifier, QDropEvent.Drop)

    def test_check_file_size(self):
        self.assertFalse(self.widget.Warning.file_too_big.is_shown())
        self.widget.SIZE_LIMIT = 4000
        self.widget.__init__()
        self.assertTrue(self.widget.Warning.file_too_big.is_shown())

    def test_domain_changes_are_stored(self):
        assert isinstance(self.widget, OWFile)

        self.open_dataset("iris")
        idx = self.widget.domain_editor.model().createIndex(4, 1)
        self.widget.domain_editor.model().setData(idx, "text", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(data.domain["iris"], StringVariable)

        self.open_dataset("zoo")
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(data.name, "zoo")

        self.open_dataset("iris")
        data = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(data.domain["iris"], StringVariable)

    def test_rename_duplicates(self):
        self.open_dataset("iris")

        idx = self.widget.domain_editor.model().createIndex(3, 0)
        self.assertFalse(self.widget.Warning.renamed_vars.is_shown())
        self.widget.domain_editor.model().setData(idx, "iris", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn("iris (1)", data.domain)
        self.assertIn("iris (2)", data.domain)
        self.assertTrue(self.widget.Warning.renamed_vars.is_shown())

        self.widget.domain_editor.model().setData(idx, "different iris", Qt.EditRole)
        self.widget.apply_button.click()
        self.assertFalse(self.widget.Warning.renamed_vars.is_shown())

    def test_variable_name_change(self):
        """
        Test whether the name of the variable is changed correctly by
        the domaineditor.
        """
        self.open_dataset("iris")

        # just rename
        idx = self.widget.domain_editor.model().createIndex(4, 0)
        self.widget.domain_editor.model().setData(idx, "a", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn("a", data.domain)

        idx = self.widget.domain_editor.model().createIndex(3, 0)
        self.widget.domain_editor.model().setData(idx, "d", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn("d", data.domain)

        # rename and change to text
        idx = self.widget.domain_editor.model().createIndex(4, 0)
        self.widget.domain_editor.model().setData(idx, "b", Qt.EditRole)
        idx = self.widget.domain_editor.model().createIndex(4, 1)
        self.widget.domain_editor.model().setData(idx, "text", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn("b", data.domain)
        self.assertIsInstance(data.domain["b"], StringVariable)

        # rename and change to discrete
        idx = self.widget.domain_editor.model().createIndex(4, 0)
        self.widget.domain_editor.model().setData(idx, "c", Qt.EditRole)
        idx = self.widget.domain_editor.model().createIndex(4, 1)
        self.widget.domain_editor.model().setData(
            idx, "categorical", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn("c", data.domain)
        self.assertIsInstance(data.domain["c"], DiscreteVariable)

        # rename and change to continuous
        self.open_dataset("zoo")
        idx = self.widget.domain_editor.model().createIndex(0, 0)
        self.widget.domain_editor.model().setData(idx, "c", Qt.EditRole)
        idx = self.widget.domain_editor.model().createIndex(0, 1)
        self.widget.domain_editor.model().setData(idx, "numeric", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn("c", data.domain)
        self.assertIsInstance(data.domain["c"], ContinuousVariable)

    def open_dataset(self, name):
        filename = FileFormat.locate(name, dataset_dirs)
        self.widget.add_path(filename)
        self.widget.load_data()

    def test_no_last_path(self):
        self.widget =\
            self.create_widget(OWFile, stored_settings={"recent_paths": []})
        # Doesn't crash and contains a single item, (none).
        self.assertEqual(self.widget.file_combo.count(), 1)

    def test_file_not_found(self):
        # Create a dummy file
        file_name = "test_owfile_data.tab"
        domainA = Domain([DiscreteVariable("d1", values=("a", "b"))],
                         DiscreteVariable("c1", values=("aaa", "bbb")))
        dataA = Table(domainA, np.array([[0], [1], [0], [np.nan]]),
                      np.array([0, 1, 0, 1]))
        dataA.save(file_name)

        # Open the file with the widget
        self.open_dataset(file_name)
        self.assertEqual(self.get_output(self.widget.Outputs.data).domain, dataA.domain)

        # Delete the file and try to reload it
        remove(file_name)
        self.widget.load_data()
        self.assertEqual(file_name, path.basename(self.widget.last_path()))
        self.assertTrue(self.widget.Error.file_not_found.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual(self.widget.infolabel.text(), "No data.")

        # Open a sample dataset
        self.open_dataset("iris")
        self.assertFalse(self.widget.Error.file_not_found.is_shown())

    def test_nothing_selected(self):
        # pylint: disable=protected-access
        widget = self.widget = \
            self.create_widget(OWFile, stored_settings={"recent_paths": []})

        widget.Outputs.data.send = Mock()
        widget.load_data()
        self.assertTrue(widget.Information.no_file_selected.is_shown())
        widget.Outputs.data.send.assert_called_with(None)

        widget.Outputs.data.send.reset_mock()
        widget.source = widget.URL
        widget.load_data()
        self.assertTrue(widget.Information.no_file_selected.is_shown())
        widget.Outputs.data.send.assert_called_with(None)

    def test_check_column_noname(self):
        """
        Column name cannot be changed to an empty string or a string with whitespaces.
        GH-2039
        """
        self.open_dataset("iris")
        idx = self.widget.domain_editor.model().createIndex(1, 0)
        temp = self.widget.domain_editor.model().data(idx, Qt.DisplayRole)
        self.widget.domain_editor.model().setData(idx, "   ", Qt.EditRole)
        self.assertEqual(self.widget.domain_editor.model().data(idx, Qt.DisplayRole), temp)
        self.widget.domain_editor.model().setData(idx, "", Qt.EditRole)
        self.assertEqual(self.widget.domain_editor.model().data(idx, Qt.DisplayRole), temp)

    def test_invalid_role_mode(self):
        self.open_dataset("iris")
        model = self.widget.domain_editor.model()
        idx = model.createIndex(1, 0)
        self.assertFalse(model.setData(idx, Qt.StatusTipRole, ""))
        self.assertIsNone(model.data(idx, Qt.StatusTipRole))

    def test_context_match_includes_variable_values(self):
        file1 = """\
var
a b

a
"""
        file2 = """\
var
a b c

a
"""
        editor = self.widget.domain_editor
        idx = self.widget.domain_editor.model().createIndex(0, 3)

        with named_file(file1, suffix=".tab") as filename:
            self.open_dataset(filename)
            self.assertEqual(editor.model().data(idx, Qt.DisplayRole), "a, b")

        with named_file(file2, suffix=".tab") as filename:
            self.open_dataset(filename)
            self.assertEqual(editor.model().data(idx, Qt.DisplayRole), "a, b, c")

    def test_check_datetime_disabled(self):
        """
        Datetime option is disable if numerical is disabled as well.
        GH-2050 (code fixes)
        GH-2120
        """
        dat = """\
            01.08.16\t42.15\tneumann\t2017-02-20
            03.08.16\t16.08\tneumann\t2017-02-21
            04.08.16\t23.04\tneumann\t2017-02-22
            03.09.16\t48.84\tturing\t2017-02-23
            02.02.17\t23.16\tturing\t2017-02-24"""
        with named_file(dat, suffix=".tab") as filename:
            self.open_dataset(filename)
            domain_editor = self.widget.domain_editor
            idx = lambda x: self.widget.domain_editor.model().createIndex(x, 1)

            qcombobox = QComboBox()
            combo = ComboDelegate(domain_editor,
                                  VarTableModel.typenames).createEditor(qcombobox, None, idx(2))
            vartype_delegate = VarTypeDelegate(domain_editor, VarTableModel.typenames)

            vartype_delegate.setEditorData(combo, idx(2))
            counts = [4, 2, 4, 2]
            for i in range(4):
                vartype_delegate.setEditorData(combo, idx(i))
                self.assertEqual(combo.count(), counts[i])

    def test_reader_custom_tab(self):
        with named_file("", suffix=".tab") as fn:
            qname = MyCustomTabReader.qualified_name()
            reader = RecentPath(fn, None, None, file_format=qname)
            self.widget = self.create_widget(OWFile,
                                             stored_settings={"recent_paths": [reader]})
            self.widget.load_data()
        self.assertFalse(self.widget.Error.missing_reader.is_shown())
        outdata = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(outdata), 150)  # loaded iris

    def test_no_reader_extension(self):
        with named_file("", suffix=".xyz_unknown") as fn:
            no_reader = RecentPath(fn, None, None)
            self.widget = self.create_widget(OWFile,
                                             stored_settings={"recent_paths": [no_reader]})
            self.widget.load_data()
        self.assertTrue(self.widget.Error.missing_reader.is_shown())

    def test_fail_sheets(self):
        with named_file("", suffix=".failed_sheet") as fn:
            self.open_dataset(fn)
        self.assertTrue(self.widget.Error.sheet_error.is_shown())

    def test_with_warnings(self):
        with named_file("", suffix=".with_warning") as fn:
            self.open_dataset(fn)
        self.assertTrue(self.widget.Warning.load_warning.is_shown())

    def test_fail(self):
        with named_file("name\nc\n\nstring", suffix=".tab") as fn, \
                patch("Orange.widgets.data.owfile.log.exception") as log:
            self.open_dataset(fn)
            log.assert_called()
        self.assertTrue(self.widget.Error.unknown.is_shown())

    def test_read_format(self):
        iris = Table("iris")

        def open_iris_with_no_spec_format(_a, _b, _c, filters, _e):
            return iris.__file__, filters.split(";;")[0]

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   open_iris_with_no_spec_format):
            self.widget.browse_file()

        self.assertIsNone(self.widget.recent_paths[0].file_format)
        self.assertEqual(self.widget.reader_combo.currentText(), DEFAULT_READER_TEXT)

        def open_iris_with_tab(*_):
            return iris.__file__, format_filter(TabReader)

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   open_iris_with_tab):
            self.widget.browse_file()

        self.assertEqual(self.widget.recent_paths[0].file_format, "Orange.data.io.TabReader")
        self.assertTrue(self.widget.reader_combo.currentText().startswith("Tab-separated"))

    def test_no_specified_reader(self):
        with named_file("", suffix=".tab") as fn:
            no_class = RecentPath(fn, None, None, file_format="not.a.file.reader.class")
            self.widget = self.create_widget(OWFile,
                                             stored_settings={"recent_paths": [no_class]})
            self.widget.load_data()
        self.assertTrue(self.widget.Error.missing_reader.is_shown())
        self.assertEqual(self.widget.reader_combo.currentText(), "not.a.file.reader.class")

    def test_select_reader(self):
        filename = FileFormat.locate("iris.tab", dataset_dirs)

        # a setting which adds a new qualified name to the reader combo
        no_class = RecentPath(filename, None, None, file_format="not.a.file.reader.class")
        self.widget = self.create_widget(OWFile,
                                         stored_settings={"recent_paths": [no_class]})
        self.widget.load_data()
        len_with_qname = len(self.widget.reader_combo)
        self.assertEqual(self.widget.reader_combo.currentText(), "not.a.file.reader.class")
        self.assertEqual(self.widget.reader, None)

        # select the last option, the same reader
        self.widget.reader_combo.activated.emit(len_with_qname - 1)
        self.assertEqual(len(self.widget.reader_combo), len_with_qname)
        self.assertEqual(self.widget.reader_combo.currentText(), "not.a.file.reader.class")
        self.assertEqual(self.widget.reader, None)

        # select the tab reader
        for i in range(len_with_qname):
            text = self.widget.reader_combo.itemText(i)
            if text.startswith("Tab-separated"):
                break
        self.widget.reader_combo.activated.emit(i)
        self.assertEqual(len(self.widget.reader_combo), len_with_qname - 1)
        self.assertTrue(self.widget.reader_combo.currentText().startswith("Tab-separated"))
        self.assertIsInstance(self.widget.reader, TabReader)

        # select the default reader
        self.widget.reader_combo.activated.emit(0)
        self.assertEqual(len(self.widget.reader_combo), len_with_qname - 1)
        self.assertEqual(self.widget.reader_combo.currentText(), DEFAULT_READER_TEXT)
        self.assertIsInstance(self.widget.reader, TabReader)

    def test_select_reader_errors(self):
        filename = FileFormat.locate("iris.tab", dataset_dirs)

        no_class = RecentPath(filename, None, None, file_format="Orange.data.io.ExcelReader")
        self.widget = self.create_widget(OWFile,
                                         stored_settings={"recent_paths": [no_class]})
        self.widget.load_data()
        self.assertIn("Excel", self.widget.reader_combo.currentText())
        self.assertTrue(self.widget.Error.unknown.is_shown())
        self.assertFalse(self.widget.Error.missing_reader.is_shown())

    def test_domain_edit_no_changes(self):
        self.open_dataset("iris")
        data = self.get_output(self.widget.Outputs.data)
        # When no changes have been made in the domain editor,
        # output data should be the same object (and not recreated)
        self.assertTrue(data is self.widget.data)

    def test_domain_edit_on_sparse_data(self):
        iris = Table("iris").to_sparse()

        with named_file("", suffix='.pickle') as fn:
            with open(fn, "wb") as f:
                pickle.dump(iris, f)

            self.widget.add_path(fn)
            self.widget.load_data()

        output = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(output, Table)
        self.assertEqual(iris.X.shape, output.X.shape)
        self.assertTrue(sp.issparse(output.X))

    def test_drop_data_when_everything_skipped(self):
        """
        No data when everything is skipped. Otherwise Select Rows crashes.
        GH-2237
        """
        self.open_dataset("iris")
        data = self.get_output(self.widget.Outputs.data)
        self.assertTrue(len(data), 150)
        self.assertTrue(len(data.domain.variables), 5)
        for i in range(5):
            idx = self.widget.domain_editor.model().createIndex(i, 2)
            self.widget.domain_editor.model().setData(idx, "skip", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(data)

    def test_call_deprecated_dialog_formats(self):
        with self.assertWarns(OrangeDeprecationWarning):
            self.assertIn("Tab", dialog_formats())

    def test_add_new_format(self):
        # test adding file formats after registering the widget
        called = False
        with named_file("", suffix=".tab") as filename:
            def test_format(_sd, _sf, ff, **_):
                nonlocal called
                called = True
                self.assertIn(FailedSheetsFormat, ff)
                return filename, TabReader, ""
            with patch("Orange.widgets.data.owfile.open_filename_dialog", test_format):
                self.widget.browse_file()
        self.assertTrue(called)

    def test_domain_editor_conversions(self):
        dat = """V0\tV1\tV2\tV3\tV4\tV5\tV6
                 c\tc\td\td\tc\td\td
                  \t \t \t \t \t \t
                 3.0\t1.0\t4\ta\t0.0\tx\t1.0
                 1.0\t2.0\t4\tb\t0.0\ty\t2.0
                 2.0\t1.0\t7\ta\t0.0\ty\t2.0
                 0.0\t2.0\t7\ta\t0.0\tz\t2.0"""
        with named_file(dat, suffix=".tab") as filename:
            self.open_dataset(filename)
            data1 = self.get_output(self.widget.Outputs.data)
            model = self.widget.domain_editor.model()
            # check the ordering of attributes
            for i, a in enumerate(data1.domain.attributes):
                self.assertEqual(str(a), model.data(model.createIndex(i, 0), Qt.DisplayRole))
            # make conversions
            model.setData(model.createIndex(0, 1), "categorical", Qt.EditRole)
            model.setData(model.createIndex(1, 1), "text", Qt.EditRole)
            model.setData(model.createIndex(2, 1), "numeric", Qt.EditRole)
            model.setData(model.createIndex(3, 1), "numeric", Qt.EditRole)
            model.setData(model.createIndex(6, 1), "numeric", Qt.EditRole)
            self.widget.apply_button.click()
            data2 = self.get_output(self.widget.Outputs.data)
            # round continuous values should be converted to integers (3.0 -> 3, "3")
            self.assertEqual(len(data2.domain.attributes[0].values[0]), 1)
            self.assertEqual(len(data2[0].metas[0]), 1)
            # discrete integer values should stay the same after conversion to continuous
            self.assertAlmostEqual(float(data1[0][2].value), data2[0][1])
            # discrete round floats should stay the same after conversion to continuous
            self.assertAlmostEqual(float(data1[0][6].value), data2[0][5])

    def test_domaineditor_continuous_to_string(self):
        # GH 2744
        dat = """V0\nc\n\n1.0\nnan\n3.0"""
        with named_file(dat, suffix=".tab") as filename:
            self.open_dataset(filename)

            model = self.widget.domain_editor.model()
            model.setData(model.createIndex(0, 1), "text", Qt.EditRole)
            self.widget.apply_button.click()

            data = self.get_output(self.widget.Outputs.data)
            self.assertSequenceEqual(data.metas.ravel().tolist(), ['1', '', '3'])

    def test_domaineditor_makes_variables(self):
        # Variables created with domain editor should be interchangeable
        # with variables read from file.

        dat = """V0\tV1\nc\td\n\n1.0\t2"""
        v0 = StringVariable.make("V0")
        v1 = ContinuousVariable.make("V1")

        with named_file(dat, suffix=".tab") as filename:
            self.open_dataset(filename)

            model = self.widget.domain_editor.model()
            model.setData(model.createIndex(0, 1), "text", Qt.EditRole)
            model.setData(model.createIndex(1, 1), "numeric", Qt.EditRole)
            self.widget.apply_button.click()

            data = self.get_output(self.widget.Outputs.data)
            self.assertEqual(data.domain["V0"], v0)
            self.assertEqual(data.domain["V1"], v1)

    def test_url_no_scheme(self):
        mock_urlreader = Mock(side_effect=ValueError())
        url = 'foo.bar/xxx.csv'

        with patch('Orange.widgets.data.owfile.UrlReader', mock_urlreader):
            self.widget.url_combo.insertItem(0, url)
            self.widget.url_combo.activated.emit(0)

        mock_urlreader.assert_called_once_with('http://' + url)

    def test_adds_origin(self):
        self.open_dataset("origin1/images")
        data1 = self.get_output(self.widget.Outputs.data)
        attrs = data1.domain["image"].attributes
        self.assertIn("origin", attrs)
        self.assertIn("origin1", attrs["origin"])

        self.open_dataset("origin2/images")
        data2 = self.get_output(self.widget.Outputs.data)
        attrs = data2.domain["image"].attributes
        self.assertIn("origin", attrs)
        self.assertIn("origin2", attrs["origin"])

        # Make sure that variable in data1 still contains correct origin
        attrs = data1.domain["image"].attributes
        self.assertIn("origin", attrs)
        self.assertIn("origin1", attrs["origin"])

    @patch("Orange.widgets.widget.OWWidget.workflowEnv",
           Mock(return_value={"basedir": getcwd()}))
    def test_open_moved_workflow(self):
        """Test opening workflow that has been moved to another location
        (i.e. sent by email), considering data file is stored in the same
        directory as the workflow.
        """
        with tempfile.NamedTemporaryFile(dir=getcwd(), delete=False) as temp_file:
            file_name = temp_file.name
        base_name = path.basename(file_name)
        try:
            recent_path = RecentPath(
                path.join("temp/datasets", base_name), "",
                path.join("datasets", base_name)
            )
            stored_settings = {"recent_paths": [recent_path]}
            w = self.create_widget(OWFile, stored_settings=stored_settings)
            w.load_data()
            self.assertEqual(w.file_combo.count(), 1)
            self.assertFalse(w.Error.file_not_found.is_shown())
        finally:
            remove(file_name)

    @patch("Orange.widgets.widget.OWWidget.workflowEnv",
           Mock(return_value={"basedir": getcwd()}))
    def test_files_relocated(self):
        """
        This test testes if paths are relocated correctly
        """
        with tempfile.NamedTemporaryFile(dir=getcwd(), delete=False) as temp_file:
            file_name = temp_file.name
        base_name = path.basename(file_name)
        try:
            recent_path = RecentPath(
                path.join("temp/datasets", base_name), "",
                path.join("datasets", base_name)
            )
            stored_settings = {"recent_paths": [recent_path]}
            w = self.create_widget(OWFile, stored_settings=stored_settings)
            w.load_data()

            # relocation is called already
            # if it works correctly relative path should be same than base name
            self.assertEqual(w.recent_paths[0].relpath, base_name)

            w.workflowEnvChanged("basedir", base_name, base_name)
            self.assertEqual(w.recent_paths[0].relpath, base_name)
        finally:
            remove(file_name)

    def test_sheets(self):
        # pylint: disable=protected-access
        widget = self.widget
        combo = widget.sheet_combo
        widget.last_path = \
            lambda: path.join(path.dirname(__file__), '..', "..", '..',
                              'tests', 'xlsx_files', 'header_0_sheet.xlsx')
        widget._try_load()
        widget.reader.sheet = "my_sheet"
        widget._select_active_sheet()
        self.assertEqual(combo.itemText(0), "Sheet1")
        self.assertEqual(combo.itemText(1), "my_sheet")
        self.assertEqual(combo.itemText(2), "Sheet3")
        self.assertEqual(combo.currentIndex(), 1)

        widget.reader.sheet = "no such sheet"
        widget._select_active_sheet()
        self.assertEqual(combo.currentIndex(), 0)

    @patch("os.path.exists", new=lambda _: True)
    def test_warning_from_another_thread(self):
        class AnotherWidget(QObject):
            # This must be a method, not a staticmethod to run in the thread
            def issue_warning(self):  # pylint: disable=no-self-use
                time.sleep(0.1)
                warnings.warn("warning from another thread")
                warning_thread.quit()

        def read():
            warning_thread.start()
            time.sleep(0.2)
            return Table(TITANIC_PATH)

        warning_thread = QThread()
        another_widget = AnotherWidget()
        another_widget.moveToThread(warning_thread)
        warning_thread.started.connect(another_widget.issue_warning)

        reader = Mock()
        reader.read = read
        self.widget._get_reader = lambda: reader
        self.widget.last_path = lambda: "foo"
        self.widget._update_sheet_combo = Mock()

        # Warning must be caught by unit tests, but not the widget
        with self.assertWarns(UserWarning):
            self.widget._try_load()
            self.assertFalse(self.widget.Warning.load_warning.is_shown())


    @patch("os.path.exists", new=lambda _: True)
    def test_warning_from_this_thread(self):
        WARNING_MSG = "warning from this thread"

        def read():
            warnings.warn(WARNING_MSG)
            return Table(TITANIC_PATH)

        reader = Mock()
        reader.read = read
        self.widget._get_reader = lambda: reader
        self.widget.last_path = lambda: "foo"
        self.widget._update_sheet_combo = Mock()

        self.widget._try_load()
        self.assertTrue(self.widget.Warning.load_warning.is_shown())
        self.assertIn(WARNING_MSG, str(self.widget.Warning.load_warning))


class TestOWFileDropHandler(unittest.TestCase):
    def test_canDropUrl(self):
        handler = OWFileDropHandler()
        self.assertTrue(handler.canDropUrl(QUrl("https://example.com/test.tab")))
        self.assertTrue(handler.canDropUrl(QUrl.fromLocalFile("test.tab")))

    def test_parametersFromUrl(self):
        handler = OWFileDropHandler()
        r = handler.parametersFromUrl(QUrl("https://example.com/test.tab"))
        self.assertEqual(r["source"], OWFile.URL)
        self.assertEqual(r["recent_urls"], ["https://example.com/test.tab"])
        r = handler.parametersFromUrl(QUrl.fromLocalFile("test.tab"))
        self.assertEqual(r["source"], OWFile.LOCAL_FILE)
        self.assertEqual(r["recent_paths"][0].basename, "test.tab")


if __name__ == "__main__":
    unittest.main()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from os import path, remove
from unittest.mock import Mock
import pickle
import tempfile


import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QMimeData, QPoint, Qt, QUrl
from AnyQt.QtGui import QDragEnterEvent, QDropEvent
from AnyQt.QtWidgets import QComboBox

import Orange
from Orange.data import FileFormat, dataset_dirs, StringVariable, Table, \
    Domain, DiscreteVariable
from Orange.tests import named_file
from Orange.widgets.data.owfile import OWFile
from Orange.widgets.utils.filedialogs import dialog_formats
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.domaineditor import ComboDelegate, VarTypeDelegate, VarTableModel

TITANIC_PATH = path.join(path.dirname(Orange.__file__), 'datasets', 'titanic.tab')


class AddedFormat(FileFormat):
    EXTENSIONS = ('.123',)
    DESCRIPTION = "Test if a dialog format works after reading OWFile"

    def read(self):
        pass


class TestOWFile(WidgetTest):
    # Attribute used to store event data so it does not get garbage
    # collected before event is processed.
    event_data = None

    def setUp(self):
        self.widget = self.create_widget(OWFile)

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
        self.widget.domain_editor.model().setData(idx, "string", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output("Data")
        self.assertIsInstance(data.domain["iris"], StringVariable)

        self.open_dataset("zoo")
        data = self.get_output("Data")
        self.assertEqual(data.name, "zoo")

        self.open_dataset("iris")
        data = self.get_output("Data")
        self.assertIsInstance(data.domain["iris"], StringVariable)

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
        self.assertEqual(self.get_output("Data").domain, dataA.domain)

        # Delete the file and try to reload it
        remove(file_name)
        self.widget.load_data()
        self.assertEqual(file_name, path.basename(self.widget.last_path()))
        self.assertTrue(self.widget.Error.file_not_found.is_shown())
        self.assertIsNone(self.get_output("Data"))
        self.assertEqual(self.widget.info.text(), "No data.")

        # Open a sample dataset
        self.open_dataset("iris")
        self.assertFalse(self.widget.Error.file_not_found.is_shown())

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

    def test_domain_edit_on_sparse_data(self):
        iris = Table("iris")
        iris.X = sp.csr_matrix(iris.X)

        f = tempfile.NamedTemporaryFile(suffix='.pickle', delete=False)
        pickle.dump(iris, f)
        f.close()

        self.widget.add_path(f.name)
        self.widget.load_data()

        output = self.get_output("Data")
        self.assertIsInstance(output, Table)
        self.assertEqual(iris.X.shape, output.X.shape)
        self.assertTrue(sp.issparse(output.X))

    def test_drop_data_when_everything_skipped(self):
        """
        No data when everything is skipped. Otherwise Select Rows crashes.
        GH-2237
        """
        self.open_dataset("iris")
        data = self.get_output("Data")
        self.assertTrue(len(data), 150)
        self.assertTrue(len(data.domain), 5)
        for i in range(5):
            idx = self.widget.domain_editor.model().createIndex(i, 2)
            self.widget.domain_editor.model().setData(idx, "skip", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output("Data")
        self.assertIsNone(data)

    def test_add_new_format(self):
        # test adding file formats after registering the widget
        formats = dialog_formats()
        self.assertTrue(".123" in formats)

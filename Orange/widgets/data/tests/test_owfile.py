# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from os import path
from unittest.mock import Mock

from AnyQt.QtCore import QMimeData, QPoint, Qt, QUrl
from AnyQt.QtGui import QDragEnterEvent, QDropEvent

import Orange
from Orange.data import FileFormat, dataset_dirs, StringVariable
from Orange.widgets.data.owfile import OWFile
from Orange.widgets.tests.base import WidgetTest

TITANIC_PATH = path.join(path.dirname(Orange.__file__), 'datasets', 'titanic.tab')


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

    def test_check_column_noname(self):
        '''
        GH-2018
        '''
        self.assertTrue(True)

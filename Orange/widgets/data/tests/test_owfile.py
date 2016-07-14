# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from os import path
from unittest.mock import Mock

from PyQt4.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt4.QtGui import QDragEnterEvent, QDropEvent

import Orange
from Orange.widgets.data.owfile import OWFile
from Orange.widgets.tests.base import WidgetTest

TITANIC_URL = path.join(path.dirname(Orange.__file__), 'datasets', 'titanic.tab')


class TestOWFile(WidgetTest):
    # Attribute used to store event data so it does not get garbage
    # collected before event is processed.
    event_data = None

    def setUp(self):
        self.widget = self.create_widget(OWFile)

    def test_dragEnterEvent_accepts_urls(self):
        event = self._drag_enter_event(TITANIC_URL)
        self.widget.dragEnterEvent(event)
        self.assertTrue(event.isAccepted())

    def test_dragEnterEvent_skips_osx_file_references(self):
        event = self._drag_enter_event('/.file/id=12345')
        self.widget.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())

    def test_dragEnterEvent_skips_usupported_files(self):
        event = self._drag_enter_event('file.unsupported')
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

        event = self._drop_event(TITANIC_URL)
        self.widget.dropEvent(event)

        self.assertEqual(self.widget.source, OWFile.LOCAL_FILE)
        self.assertEqual(self.widget.last_path(), TITANIC_URL)
        self.widget.load_data.assert_called_with()

    def _drop_event(self, url):
        # make sure data does not get garbage collected before it used
        self.event_data = data = QMimeData()
        data.setUrls([QUrl(url)])

        return QDropEvent(
            QPoint(0, 0), Qt.MoveAction, data,
            Qt.NoButton, Qt.NoModifier, QDropEvent.Drop)



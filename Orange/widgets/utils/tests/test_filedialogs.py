from AnyQt.QtCore import QUrl, QMimeData

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import dragDrop
from Orange.widgets.utils.filedialogs import OWUrlDropBase


class TestOWUrlDropBase(WidgetTest):
    def test_drop(self):
        class TestW(OWUrlDropBase):
            path = None

            def canDropUrl(self, url: QUrl) -> bool:
                return url.toLocalFile().endswith(".foo")

            def handleDroppedUrl(self, url: QUrl) -> None:
                self.path = url.toLocalFile()

        w = self.create_widget(TestW)
        url = QUrl("file:///bar.foo")
        mime = QMimeData()
        mime.setUrls([url])
        self.assertTrue(dragDrop(w, mime))
        self.assertEqual(w.path, url.toLocalFile())
        url = QUrl("file:///bar.baz")
        mime.setUrls([url])
        self.assertFalse(dragDrop(w, mime))
        self.assertNotEqual(w.path, url.toLocalFile())

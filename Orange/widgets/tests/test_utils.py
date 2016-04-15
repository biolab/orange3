import unittest

from PyQt4.QtGui import QApplication, QWidget
from PyQt4.QtCore import Qt, QByteArray, QPoint, QRect, QSize

from Orange.widgets.utils import geom_state, geom_state_normal


class TestGeometryRestore(unittest.TestCase):
    def setUp(self):
        app = QApplication.instance()
        if not app:
            app = QApplication([])
        self.app = app

    def test_pack(self):
        w = QWidget()
        geom = geom_state(
            geom_state.MAGIC, 1, 0,
            50, 50, 99, 99,  # frame geom
            50, 70, 99, 99,  # normal geom
            0, 0, 0    # screen, maximized, fullscreen
        )
        w.restoreGeometry(QByteArray(geom.to_bytes()))
        self.assertEqual(w.frameGeometry().topLeft(), QPoint(50, 50))
        self.assertEqual(w.geometry().size(), QSize(50, 30))

        self.assertEqual(geom, geom_state.from_bytes(geom.to_bytes()))

    def test_unpack(self):
        w = QWidget()
        w.setGeometry(QRect(50, 50, 50, 30))
        w.move(50, 50)
        state = geom_state.from_bytes(bytes(w.saveGeometry()))
        self.assertEqual(state.frame_top, 50)
        self.assertEqual(state.frame_left, 50)
        self.assertEqual(state.maximized, 0)
        self.assertEqual(state.full_screen, 0)

        w.setWindowState(w.windowState() | Qt.WindowFullScreen)
        state = geom_state.from_bytes(bytes(w.saveGeometry()))
        self.assertEqual(state.full_screen, Qt.WindowFullScreen)
        self.assertEqual(state.maximized, 0)

        w.setWindowState(w.windowState() ^ Qt.WindowFullScreen)
        w.setWindowState(w.windowState() | Qt.WindowMaximized)
        state = geom_state.from_bytes(bytes(w.saveGeometry()))
        self.assertEqual(state.full_screen, 0)
        self.assertEqual(state.maximized, Qt.WindowMaximized)

    def test_restore_to_normal(self):
        geom = geom_state(
            geom_state.MAGIC, 1, 0,
            0, 0, 200, 200,
            50, 70, 99, 99,
            0, 0, Qt.WindowFullScreen
        )
        normal = geom_state_normal(geom, (0, 20, 0, 0))
        self.assertEqual(normal.full_screen, 0)
        self.assertEqual(normal.maximized, 0)
        self.assertEqual(normal.frame_left, 50)
        self.assertEqual(normal.frame_top, 50)
        self.assertEqual(normal.frame_bottom, 99)
        self.assertEqual(normal.frame_right, 99)

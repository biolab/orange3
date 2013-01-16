"""
Tests for DropShadowFrame wiget.

"""

from PyQt4.QtGui import (
    QMainWindow, QWidget, QListView, QTextEdit, QHBoxLayout, QToolBar,
    QVBoxLayout, QColor
)

from PyQt4.QtCore import Qt, QTimer
from .. import dropshadow

from .. import test


class TestDropShadow(test.QAppTestCase):
    def test_drop_shadow_old(self):
        w = dropshadow._DropShadowWidget()
        w.setContentsMargins(20, 20, 20, 20)
        w.setLayout(QHBoxLayout())
        w.layout().setContentsMargins(0, 0, 0, 0)
        w.layout().addWidget(QListView())
        w.show()
        QTimer.singleShot(1500, lambda: w.setRadius(w.radius + 5))
        self.app.exec_()

    def test(self):
        lv = QListView()
        mw = QMainWindow()
        # Add two tool bars, the shadow should extend over them.
        mw.addToolBar(Qt.BottomToolBarArea, QToolBar())
        mw.addToolBar(Qt.TopToolBarArea, QToolBar())
        mw.setCentralWidget(lv)

        f = dropshadow.DropShadowFrame(color=Qt.blue, radius=20)

        f.setWidget(lv)

        self.assertIs(f.parentWidget(), mw)
        self.assertIs(f.widget(), lv)

        mw.show()

        self.app.processEvents()

        self.singleShot(3000, lambda: f.setColor(Qt.red))
        self.singleShot(4000, lambda: f.setRadius(30))
        self.singleShot(5000, lambda: f.setRadius(40))
        self.app.exec_()

    def test1(self):
        class FT(QToolBar):
            def paintEvent(self, e):
                pass

        w = QMainWindow()
        ftt, ftb = FT(), FT()
        ftt.setFixedHeight(15)
        ftb.setFixedHeight(15)

        w.addToolBar(Qt.TopToolBarArea, ftt)
        w.addToolBar(Qt.BottomToolBarArea, ftb)

        f = dropshadow.DropShadowFrame()
        te = QTextEdit()
        c = QWidget()
        c.setLayout(QVBoxLayout())
        c.layout().setContentsMargins(20, 0, 20, 0)
        c.layout().addWidget(te)
        w.setCentralWidget(c)
        f.setWidget(te)
        f.radius = 15
        f.color = QColor(Qt.blue)
        w.show()

        self.singleShot(3000, lambda: f.setColor(Qt.red))
        self.singleShot(4000, lambda: f.setRadius(30))
        self.singleShot(5000, lambda: f.setRadius(40))

        self.app.exec_()


if __name__ == "__main__":
    test.unittest.main()

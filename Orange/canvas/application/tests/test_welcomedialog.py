"""
Test for welcome screen.
"""

from AnyQt.QtWidgets import QAction, QLabel
from AnyQt.QtTest import QSignalSpy

from ...resources import icon_loader

from ..welcomedialog import WelcomeDialog, PagedDialog, decorate_welcome_icon

from ...gui.test import QAppTestCase


class TestDialog(QAppTestCase):
    def test_dialog(self):
        d = WelcomeDialog()
        loader = icon_loader()
        icon = loader.get("icons/default-widget.svg")
        action1 = QAction(decorate_welcome_icon(icon, "light-green"),
                          "one", self.app)
        action2 = QAction(decorate_welcome_icon(icon, "orange"),
                          "two", self.app)
        d.addRow([action1, action2])

        action3 = QAction(decorate_welcome_icon(icon, "light-green"),
                          "three", self.app)
        d.addRow([action3])

        self.assertTrue(d.buttonAt(1, 0).defaultAction() == action3)

        d.show()
        action = [None]

        def p(a):
            print(str(a.text()))
            action[0] = a

        d.triggered.connect(p)
        self.app.exec_()
        self.assertIs(action[0], d.triggeredAction())


class TestPagedDialog(QAppTestCase):
    def test_dialog(self):
        d = PagedDialog()
        spy = QSignalSpy(d.currentIndexChanged)
        loader = icon_loader()
        icon = loader.get("icons/default-widget.svg")
        assert d.currentIndex() == -1
        d.addPage(icon, "Hello", QLabel("Hello"))
        assert d.currentIndex() == 0
        assert d.count() == 1
        assert len(spy) == 1 and spy[0] == [0]
        del spy[0]
        d.addPage(icon, "World", QLabel("world!"))
        d.addPage(icon, "Watch the ball, its going to move!", QLabel("A ball"))
        assert d.count() == 3
        d.setCurrentIndex(1)
        assert len(spy) == 1 and spy[0] == [1]
        del spy[0]

        d.removePage(0)
        assert len(spy) == 1 and spy[0] == [0]
        assert d.currentIndex() == 0
        del spy[0]
        d.insertPage(0, icon, "Hello", QLabel("Hello"))
        assert d.currentIndex() == 1

        d.show()
        self.app.exec()

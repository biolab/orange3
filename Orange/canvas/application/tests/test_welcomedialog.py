"""
Test for welcome screen.
"""

from AnyQt.QtWidgets import QAction

from ...resources import icon_loader

from ..welcomedialog import WelcomeDialog, decorate_welcome_icon

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

"""
Test for DynamicResizeToolbar

"""
import logging

from AnyQt.QtWidgets import QAction

from AnyQt.QtCore import Qt

from .. import test
from .. import toolbar


class ToolBoxTest(test.QAppTestCase):

    def test_dynamic_toolbar(self):
        logging.basicConfig(level=logging.DEBUG)

        w = toolbar.DynamicResizeToolBar(None)
        w.setStyleSheet("QToolButton { border: 1px solid red; }")

        w.addAction(QAction("1", w))
        w.addAction(QAction("2", w))
        w.addAction(QAction("A long name", w))
        actions = list(w.actions())

        self.assertSequenceEqual([str(action.text()) for action in actions],
                                 ["1", "2", "A long name"])

        w.resize(100, 30)
        w.show()

        w.raise_()

        w.removeAction(actions[1])
        w.insertAction(actions[2], actions[1])

        self.assertSequenceEqual(actions, list(w.actions()),
                                 msg="insertAction does not preserve "
                                     "action order")

        self.singleShot(2000, lambda: w.setOrientation(Qt.Vertical))
        self.singleShot(5000, lambda: w.removeAction(actions[1]))

        self.app.exec_()

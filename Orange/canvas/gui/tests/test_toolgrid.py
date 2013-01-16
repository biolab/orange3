from PyQt4.QtGui import QAction, QToolButton

from .. import test
from ..toolgrid import ToolGrid


class TestToolGrid(test.QAppTestCase):
    def test_tool_grid(self):
        w = ToolGrid()

        w.show()
        self.app.processEvents()

        def buttonsOrderedVisual():
            # Process layout events so the buttons have right positions
            self.app.processEvents()
            buttons = w.findChildren(QToolButton)
            return sorted(buttons, key=lambda b: (b.y(), b.x()))

        def buttonsOrderedLogical():
            return map(w.buttonForAction, w.actions())

        def assertOrdered():
            self.assertSequenceEqual(buttonsOrderedLogical(),
                                     buttonsOrderedVisual())

        action_a = QAction("A", w)
        action_b = QAction("B", w)
        action_c = QAction("C", w)
        action_d = QAction("D", w)

        w.addAction(action_b)
        w.insertAction(0, action_a)
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b])
        assertOrdered()

        w.addAction(action_d)
        w.insertAction(action_d, action_c)

        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b, action_c, action_d])
        assertOrdered()

        w.removeAction(action_c)
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b, action_d])

        assertOrdered()

        w.removeAction(action_a)
        self.assertSequenceEqual(w.actions(),
                                 [action_b, action_d])

        assertOrdered()

        w.insertAction(0, action_a)
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b, action_d])

        assertOrdered()

        w.setColumnCount(2)
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b, action_d])

        assertOrdered()

        w.insertAction(2, action_c)
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b, action_c, action_d])
        assertOrdered()

        w.clear()
        # test no 'before' action edge case
        w.insertAction(0, action_a)
        self.assertIs(action_a, w.actions()[0])
        w.insertAction(1, action_b)
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b])

        w.clear()
        w.setActions([action_a, action_b, action_c, action_d])
        self.assertSequenceEqual(w.actions(),
                                 [action_a, action_b, action_c, action_d])
        assertOrdered()

        triggered_actions = []

        def p(action):
            print action.text()

        w.actionTriggered.connect(p)
        w.actionTriggered.connect(triggered_actions.append)
        action_a.trigger()

        w.show()
        self.app.exec_()

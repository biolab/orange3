import unittest
import collections

from AnyQt.QtWidgets import QMenu
from AnyQt.QtCore import QPoint

from Orange.widgets.evaluate.utils import ScoreTable
from Orange.widgets.tests.base import GuiTest


class TestScoreTable(GuiTest):
    def test_show_column_chooser(self):
        score_table = ScoreTable(None)
        view = score_table.view
        all, shown = "MABDEFG", "ABDF"
        header = view.horizontalHeader()
        score_table.shown_scores = set(shown)
        score_table.model.setHorizontalHeaderLabels(list(all))
        score_table._update_shown_columns()

        actions = collections.OrderedDict()
        menu_add_action = QMenu.addAction

        def addAction(menu, a):
            action = menu_add_action(menu, a)
            actions[a] = action
            return action

        def execmenu(*_):
            self.assertEqual(list(actions), list(all)[1:])
            for name, action in actions.items():
                self.assertEqual(action.isChecked(), name in shown)
            actions["E"].triggered.emit(True)
            self.assertEqual(score_table.shown_scores, set("ABDEF"))
            actions["B"].triggered.emit(False)
            self.assertEqual(score_table.shown_scores, set("ADEF"))
            for i, name in enumerate(all):
                self.assertEqual(name == "M" or name in "ADEF",
                                 not header.isSectionHidden(i),
                                 msg="error in section {}({})".format(i, name))

        # We must patch `QMenu.exec` because the Qt would otherwise (invisibly)
        # show the popup and wait for the user.
        # Assertions are made within `menuexec` since they check the
        # instances of `QAction`, which are invalid (destroyed by Qt?) after
        # `menuexec` finishes.
        with unittest.mock.patch("AnyQt.QtWidgets.QMenu.addAction", addAction), \
             unittest.mock.patch("AnyQt.QtWidgets.QMenu.exec", execmenu):
            score_table.show_column_chooser(QPoint(0, 0))

    def test_update_shown_columns(self):
        score_table = ScoreTable(None)
        view = score_table.view
        all, shown = "MABDEFG", "ABDF"
        header = view.horizontalHeader()
        score_table.shown_scores = set(shown)
        score_table.model.setHorizontalHeaderLabels(list(all))
        score_table._update_shown_columns()
        for i, name in enumerate(all):
            self.assertEqual(name == "M" or name in shown,
                             not header.isSectionHidden(i),
                             msg="error in section {}({})".format(i, name))

        score_table.shown_scores = set()
        score_table._update_shown_columns()
        for i, name in enumerate(all):
            self.assertEqual(i == 0,
                             not header.isSectionHidden(i),
                             msg="error in section {}({})".format(i, name))

if __name__ == "__main__":
    unittest.main()

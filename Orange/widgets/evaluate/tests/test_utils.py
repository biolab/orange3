# pylint: disable=protected-access

import unittest
import collections
from distutils.version import LooseVersion

import numpy as np

from AnyQt.QtWidgets import QMenu
from AnyQt.QtGui import QStandardItem
from AnyQt.QtCore import QPoint, Qt

import Orange
from Orange.widgets.evaluate.utils import ScoreTable, usable_scorers, BUILTIN_SCORERS_ORDER
from Orange.widgets.tests.base import GuiTest
from Orange.data import Table, DiscreteVariable, ContinuousVariable
from Orange.util import OrangeDeprecationWarning
from Orange.evaluation import scoring


class TestUsableScorers(unittest.TestCase):
    def setUp(self):
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.registered_scorers = {scorer.name for scorer in scoring.Score.registry.values()}

    def validate_scorer_candidates(self, scorers, class_type):
        built_in_scorers = set(BUILTIN_SCORERS_ORDER[class_type])
        # scorer candidates are not all registered scorers
        self.assertNotEqual(scorers, self.registered_scorers)
        # scorer candidates are subset of registered scorers
        self.assertTrue(scorers.issubset(self.registered_scorers))
        # builtins scorers are in fact a subset of valid candidates
        self.assertTrue(built_in_scorers.issubset(scorers))

    def test_usable_scores(self):
        classification_scorers = {scorer.name for scorer in usable_scorers(self.iris.domain)}
        regression_scorers = {scorer.name for scorer in usable_scorers(self.housing.domain)}

        self.validate_scorer_candidates(classification_scorers, class_type=DiscreteVariable)
        self.validate_scorer_candidates(regression_scorers, class_type=ContinuousVariable)

    def test_if_deprecation_warning_is_raised(self):
        with self.assertWarns(OrangeDeprecationWarning):
            classification_scorers = {scorer.name for scorer in
                                      usable_scorers(self.iris.domain.class_var)}

        with self.assertWarns(OrangeDeprecationWarning):
            regression_scorers = {scorer.name for scorer in
                                  usable_scorers(self.housing.domain.class_var)}

        # check if we get valid scorers using deprecated approach
        self.validate_scorer_candidates(classification_scorers, class_type=DiscreteVariable)
        self.validate_scorer_candidates(regression_scorers, class_type=ContinuousVariable)

    def test_time_bomb(self):
        """This test is to be included in the 3.32.1 release and will fail in
        version 3.35. This serves as a reminder to remove the deprecated method
        and this test."""
        if LooseVersion(Orange.__version__) >= LooseVersion("3.35"):
            self.fail(
                "Passing Variable to usable_scorers was deprecated in "
                "version 3.32.1, and there have been three minor versions in "
                "between. Please remove the deprecated method."
            )


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

    def test_sorting(self):
        def order(n=5):
            return "".join(model.index(i, 0).data() for i in range(n))

        score_table = ScoreTable(None)

        data = [
            ["D", 11.0, 15.3],
            ["C", 5.0, -15.4],
            ["b", 20.0, np.nan],
            ["A", None, None],
            ["E", "", 0.0]
        ]
        for data_row in data:
            row = []
            for x in data_row:
                item = QStandardItem()
                if x is not None:
                    item.setData(x, Qt.DisplayRole)
                row.append(item)
            score_table.model.appendRow(row)

        model = score_table.view.model()

        model.sort(0, Qt.AscendingOrder)
        self.assertEqual(order(), "AbCDE")

        model.sort(0, Qt.DescendingOrder)
        self.assertEqual(order(), "EDCbA")

        model.sort(1, Qt.AscendingOrder)
        self.assertEqual(order(3), "CDb")

        model.sort(1, Qt.DescendingOrder)
        self.assertEqual(order(3), "bDC")

        model.sort(2, Qt.AscendingOrder)
        self.assertEqual(order(3), "CED")

        model.sort(2, Qt.DescendingOrder)
        self.assertEqual(order(3), "DEC")

    def test_column_settings_reminder(self):
        if LooseVersion(Orange.__version__) >= LooseVersion("3.34"):
            self.fail(
                "Orange 3.32 added a workaround to show C-Index into ScoreTable.__init__. "
                "This should have been properly fixed long ago."
            )


if __name__ == "__main__":
    unittest.main()

# pylint: disable=protected-access

import unittest
import collections
from itertools import count
from unittest.mock import patch

import numpy as np

from AnyQt.QtWidgets import QMenu
from AnyQt.QtGui import QStandardItem
from AnyQt.QtCore import QPoint, Qt

import Orange
from Orange.evaluation.scoring import Score, AUC, CA, F1, Specificity
from Orange.widgets.evaluate.utils import ScoreTable, usable_scorers
from Orange.widgets.tests.base import GuiTest
from Orange.data import Table, DiscreteVariable, ContinuousVariable
from Orange.evaluation import scoring


class TestUsableScorers(unittest.TestCase):
    def setUp(self):
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.registered_scorers = set(scoring.Score.registry.values())

    def validate_scorer_candidates(self, scorers, class_type):
        # scorer candidates are (a proper) subset of registered scorers
        self.assertTrue(set(scorers) < self.registered_scorers)
        # all scorers are adequate
        self.assertTrue(all(class_type in scorer.class_types
                            for scorer in scorers))
        # scorers are sorted
        self.assertTrue(all(s1.priority <= s2.priority
                            for s1, s2 in zip(scorers, scorers[1:])))

    def test_usable_scores(self):
        self.validate_scorer_candidates(
            usable_scorers(self.iris.domain), class_type=DiscreteVariable)
        self.validate_scorer_candidates(
            usable_scorers(self.housing.domain), class_type=ContinuousVariable)


class TestScoreTable(GuiTest):
    def setUp(self):
        class NewScore(Score):
            name = "new score"

        self.NewScore = NewScore  # pylint: disable=invalid-name

        self.orig_hints = ScoreTable.show_score_hints
        hints = ScoreTable.show_score_hints = self.orig_hints.default.copy()
        hints.update(dict(F1=True, CA=False, AUC=True, Recall=True,
                          Specificity=False, NewScore=True))
        self.score_table = ScoreTable(None)
        self.score_table.update_header([F1, CA, AUC, Specificity, NewScore])

    def tearDown(self):
        ScoreTable.show_score_hints = self.orig_hints
        del Score.registry["NewScore"]

    def test_show_column_chooser(self):
        hints = ScoreTable.show_score_hints
        actions = collections.OrderedDict()
        menu_add_action = QMenu.addAction

        def addAction(menu, a):
            action = menu_add_action(menu, a)
            actions[a] = action
            return action

        def execmenu(*_):
            # pylint: disable=unsubscriptable-object,unsupported-assignment-operation
            scorers = [F1, CA, AUC, Specificity, self.NewScore]
            self.assertEqual(list(actions)[3:], ['F1',
                                                 'Classification accuracy (CA)',
                                                 'Area under ROC curve (AUC)',
                                                 'Specificity (Spec)',
                                                 'new score'])
            header = self.score_table.view.horizontalHeader()
            for i, action, scorer in zip(count(), list(actions.values())[3:], scorers):
                self.assertEqual(action.isChecked(),
                                 hints[scorer.__name__],
                                 msg=f"error in section {scorer.name}")
                self.assertEqual(header.isSectionHidden(3 + i),
                                 not hints[scorer.__name__],
                                 msg=f"error in section {scorer.name}")
            actions["Classification accuracy (CA)"].triggered.emit(True)
            hints["CA"] = True
            for k, v in hints.items():
                self.assertEqual(self.score_table.show_score_hints[k], v,
                                 msg=f"error at {k}")
            actions["Area under ROC curve (AUC)"].triggered.emit(False)
            hints["AUC"] = False
            for k, v in hints.items():
                self.assertEqual(self.score_table.show_score_hints[k], v,
                                 msg=f"error at {k}")

        # We must patch `QMenu.exec` because the Qt would otherwise (invisibly)
        # show the popup and wait for the user.
        # Assertions are made within `menuexec` since they check the
        # instances of `QAction`, which are invalid (destroyed by Qt?) after
        # `menuexec` finishes.
        with patch("AnyQt.QtWidgets.QMenu.addAction", addAction), \
             patch("AnyQt.QtWidgets.QMenu.exec", execmenu):
            self.score_table.view.horizontalHeader().show_column_chooser(QPoint(0, 0))

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

    def test_shown_scores_backward_compatibility(self):
        self.assertEqual(self.score_table.shown_scores,
                         {"F1", "AUC", "new score"})

    def test_migration(self):
        settings = dict(foo=False, shown_scores={"Sensitivity"})
        ScoreTable.migrate_to_show_scores_hints(settings)
        self.assertTrue(settings["show_score_hints"]["Sensitivity"])


if __name__ == "__main__":
    unittest.main()

# pylint: disable=missing-docstring
# pylint: disable=protected-access
import collections
import unittest

import numpy as np
from AnyQt.QtWidgets import QMenu
from AnyQt.QtCore import QPoint

from Orange.classification import MajorityLearner
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import Results, TestOnTestData
from Orange.evaluation.scoring import ClassificationScore, RegressionScore, \
    Score
from Orange.modelling import ConstantLearner
from Orange.regression import MeanLearner
from Orange.widgets.evaluate.owtestlearners import (
    OWTestLearners, results_one_vs_rest)
from Orange.widgets.settings import (
    ClassValuesContextHandler, PerfectDomainContextHandler)
from Orange.widgets.tests.base import WidgetTest


class TestOWTestLearners(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWTestLearners)  # type: OWTestLearners

    def test_basic(self):
        data = Table("iris")[::3]
        self.send_signal(self.widget.Inputs.train_data, data)
        self.send_signal(self.widget.Inputs.learner, MajorityLearner(), 0, wait=5000)
        res = self.get_output(self.widget.Outputs.evaluations_results)
        self.assertIsInstance(res, Results)
        self.assertIsNotNone(res.domain)
        self.assertIsNotNone(res.data)
        self.assertIsNotNone(res.probabilities)

        self.send_signal(self.widget.Inputs.learner, None, 0, wait=5000)
        res = self.get_output(self.widget.Outputs.evaluations_results)
        self.assertIsNone(res)

        data = Table("housing")[::10]
        self.send_signal(self.widget.Inputs.train_data, data)
        self.send_signal(self.widget.Inputs.learner, MeanLearner(), 0, wait=5000)
        res = self.get_output(self.widget.Outputs.evaluations_results)
        self.assertIsInstance(res, Results)
        self.assertIsNotNone(res.domain)
        self.assertIsNotNone(res.data)

    def test_testOnTest(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.train_data, data)
        self.widget.resampling = OWTestLearners.TestOnTest
        self.send_signal(self.widget.Inputs.test_data, data)

    def test_CrossValidationByFeature(self):
        data = Table("iris")
        attrs = data.domain.attributes
        domain = Domain(attrs[:-1], attrs[-1], data.domain.class_vars)
        data_with_disc_metas = Table.from_table(domain, data)
        rb = self.widget.controls.resampling.buttons[OWTestLearners.FeatureFold]

        self.send_signal(self.widget.Inputs.learner, ConstantLearner(), 0)
        self.send_signal(self.widget.Inputs.train_data, data)
        self.assertFalse(rb.isEnabled())
        self.assertFalse(self.widget.features_combo.isEnabled())

        self.send_signal(self.widget.Inputs.train_data, data_with_disc_metas)
        self.assertTrue(rb.isEnabled())
        rb.click()
        self.assertEqual(self.widget.resampling, OWTestLearners.FeatureFold)
        self.assertTrue(self.widget.features_combo.isEnabled())
        self.assertEqual(self.widget.features_combo.currentText(), "iris")
        self.assertEqual(len(self.widget.features_combo.model()), 1)

        self.send_signal(self.widget.Inputs.train_data, None)
        self.assertFalse(rb.isEnabled())
        self.assertEqual(self.widget.resampling, OWTestLearners.KFold)
        self.assertFalse(self.widget.features_combo.isEnabled())

    def test_update_shown_columns(self):
        w = self.widget  #: OWTestLearners
        all, shown = "MABDEFG", "ABDF"
        header = w.view.horizontalHeader()
        w.shown_scores = set(shown)
        w.result_model.setHorizontalHeaderLabels(list(all))
        w._update_shown_columns()
        for i, name in enumerate(all):
            self.assertEqual(name == "M" or name in shown,
                             not header.isSectionHidden(i),
                             msg="error in section {}({})".format(i, name))

        w.shown_scores = set()
        w._update_shown_columns()
        for i, name in enumerate(all):
            self.assertEqual(i == 0,
                             not header.isSectionHidden(i),
                             msg="error in section {}({})".format(i, name))

    def test_show_column_chooser(self):
        w = self.widget  #: OWTestLearners
        all, shown = "MABDEFG", "ABDF"
        header = w.view.horizontalHeader()
        w.shown_scores = set(shown)
        w.result_model.setHorizontalHeaderLabels(list(all))
        w._update_shown_columns()

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
            self.assertEqual(w.shown_scores, set("ABDEF"))
            actions["B"].triggered.emit(False)
            self.assertEqual(w.shown_scores, set("ADEF"))
            for i, name in enumerate(all):
                self.assertEqual(name == "M" or name in "ADEF",
                                 not header.isSectionHidden(i),
                                 msg="error in section {}({})".format(i, name))

        # We must patch `QMenu.exec` because the Qt would otherwise (invisibly)
        # show the popup and wait for the user.
        # Assertions are made within `menuexec` since they check the
        # instances of `QAction`, which are invalid (destroyed by Qt?) after
        # `menuexec` finishes.
        with unittest.mock.patch("AnyQt.QtWidgets.QMenu.addAction", addAction),\
                unittest.mock.patch("AnyQt.QtWidgets.QMenu.exec", execmenu):
            w.show_column_chooser(QPoint(0, 0))

    def test_migrate_removes_invalid_contexts(self):
        context_invalid = ClassValuesContextHandler().new_context([0, 1, 2])
        context_valid = PerfectDomainContextHandler().new_context(*[[]] * 4)
        settings = {'context_settings': [context_invalid, context_valid]}
        self.widget.migrate_settings(settings, 2)
        self.assertEqual(settings['context_settings'], [context_valid])

    def test_memory_error(self):
        """
        Handling memory error.
        GH-2316
        """
        data = Table("iris")[::3]
        self.send_signal(self.widget.Inputs.train_data, data)
        self.assertFalse(self.widget.Error.memory_error.is_shown())

        with unittest.mock.patch(
            "Orange.evaluation.testing.Results.get_augmented_data",
            side_effect=MemoryError):
            self.send_signal(self.widget.Inputs.learner, MajorityLearner(), 0, wait=5000)
            self.assertTrue(self.widget.Error.memory_error.is_shown())

    def test_one_class_value(self):
        """
        Data with a class with one value causes widget to crash when that value
        is selected.
        GH-2351
        """
        table = Table(
            Domain(
                [ContinuousVariable("a"), ContinuousVariable("b")],
                [DiscreteVariable("c", values=["y"])]),
            list(zip(
                [42.48, 16.84, 15.23, 23.8],
                [1., 2., 3., 4.],
                "yyyy"))
        )
        self.widget.n_folds = 0
        self.widget.class_selection = "y"
        self.assertFalse(self.widget.Error.only_one_class_var_value.is_shown())
        self.send_signal("Data", table)
        self.send_signal("Learner", MajorityLearner(), 0, wait=1000)
        self.assertTrue(self.widget.Error.only_one_class_var_value.is_shown())

    def test_addon_scorers(self):
        try:
            class NewScore(Score):
                class_types = (DiscreteVariable, ContinuousVariable)

            class NewClassificationScore(ClassificationScore):
                pass

            class NewRegressionScore(RegressionScore):
                pass

            builtins = self.widget.BUILTIN_ORDER
            self.send_signal("Data", Table("iris"))
            scorer_names = [scorer.name for scorer in self.widget.scorers]
            self.assertEqual(
                tuple(scorer_names[:len(builtins[DiscreteVariable])]),
                builtins[DiscreteVariable])
            self.assertIn("NewScore", scorer_names)
            self.assertIn("NewClassificationScore", scorer_names)
            self.assertNotIn("NewRegressionScore", scorer_names)

            self.send_signal("Data", Table("housing"))
            scorer_names = [scorer.name for scorer in self.widget.scorers]
            self.assertEqual(
                tuple(scorer_names[:len(builtins[ContinuousVariable])]),
                builtins[ContinuousVariable])
            self.assertIn("NewScore", scorer_names)
            self.assertNotIn("NewClassificationScore", scorer_names)
            self.assertIn("NewRegressionScore", scorer_names)

            self.send_signal("Data", None)
            self.assertEqual(self.widget.scorers, [])
        finally:
            del Score.registry["NewScore"]
            del Score.registry["NewClassificationScore"]
            del Score.registry["NewRegressionScore"]

class TestHelpers(unittest.TestCase):
    def test_results_one_vs_rest(self):
        data = Table("lenses")
        learners = [MajorityLearner()]
        res = TestOnTestData(data[1::2], data[::2], learners=learners)
        r1 = results_one_vs_rest(res, pos_index=0)
        r2 = results_one_vs_rest(res, pos_index=1)
        r3 = results_one_vs_rest(res, pos_index=2)

        np.testing.assert_almost_equal(np.sum(r1.probabilities, axis=2), 1.0)
        np.testing.assert_almost_equal(np.sum(r2.probabilities, axis=2), 1.0)
        np.testing.assert_almost_equal(np.sum(r3.probabilities, axis=2), 1.0)

        np.testing.assert_almost_equal(
            r1.probabilities[:, :, 1] +
            r2.probabilities[:, :, 1] +
            r3.probabilities[:, :, 1],
            1.0
        )
        self.assertEqual(r1.folds, res.folds)
        self.assertEqual(r2.folds, res.folds)
        self.assertEqual(r3.folds, res.folds)

        np.testing.assert_equal(r1.row_indices, res.row_indices)
        np.testing.assert_equal(r2.row_indices, res.row_indices)
        np.testing.assert_equal(r3.row_indices, res.row_indices)


if __name__ == "__main__":
    unittest.main()

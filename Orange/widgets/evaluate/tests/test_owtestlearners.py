# pylint: disable=missing-docstring
# pylint: disable=protected-access
import unittest
import warnings

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtTest import QTest

from Orange.classification import MajorityLearner, LogisticRegressionLearner
from Orange.classification.majority import ConstantModel
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import Results, TestOnTestData
from Orange.evaluation.scoring import ClassificationScore, RegressionScore, \
    Score
from Orange.modelling import ConstantLearner
from Orange.regression import MeanLearner
from Orange.widgets.evaluate.owtestlearners import (
    OWTestLearners, results_one_vs_rest)
from Orange.widgets.evaluate.utils import BUILTIN_SCORERS_ORDER
from Orange.widgets.settings import (
    ClassValuesContextHandler, PerfectDomainContextHandler)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.tests import test_filename


class TestOWTestLearners(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWTestLearners)  # type: OWTestLearners

        self.scores_domain = Domain(
            [ContinuousVariable("a"), ContinuousVariable("b")],
            [DiscreteVariable("c", values=["y", "n"])])

        self.scores_table_values = [[1, 1, 1.23, 23.8], [1., 2., 3., 4.]]

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_basic(self):
        data = Table("iris")[::15]
        self.send_signal(self.widget.Inputs.train_data, data)
        self.send_signal(self.widget.Inputs.learner, MajorityLearner(), 0)
        res = self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        self.assertIsInstance(res, Results)
        self.assertIsNotNone(res.domain)
        self.assertIsNotNone(res.data)
        self.assertIsNotNone(res.probabilities)

        self.send_signal(self.widget.Inputs.learner, None, 0)
        res = self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        self.assertIsNone(res)

        data = Table("housing")[::10]
        self.send_signal(self.widget.Inputs.train_data, data)
        self.send_signal(self.widget.Inputs.learner, MeanLearner(), 0)
        res = self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        self.assertIsInstance(res, Results)
        self.assertIsNotNone(res.domain)
        self.assertIsNotNone(res.data)

    def test_more_learners(self):
        data = Table("iris")[::15]
        self.send_signal(self.widget.Inputs.train_data, data)
        self.send_signal(self.widget.Inputs.learner, MajorityLearner(), 0)
        self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        self.send_signal(self.widget.Inputs.learner, MajorityLearner(), 1)
        res = self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        np.testing.assert_equal(res.probabilities[0], res.probabilities[1])

    def test_testOnTest(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.train_data, data)
        self.widget.resampling = OWTestLearners.TestOnTest
        self.send_signal(self.widget.Inputs.test_data, data)

    def test_testOnTest_incompatible_domain(self):
        iris = Table("iris")
        self.send_signal(self.widget.Inputs.train_data, iris)
        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner(), 0)
        self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        self.assertFalse(self.widget.Error.test_data_incompatible.is_shown())
        self.widget.resampling = OWTestLearners.TestOnTest
        # test data with the same class (otherwise the widget shows a different error)
        # and a non-nan X
        iris_test = iris.transform(Domain([ContinuousVariable("x")],
                                          class_vars=iris.domain.class_vars))
        iris_test.X[:, 0] = 1
        self.send_signal(self.widget.Inputs.test_data, iris_test)
        self.get_output(self.widget.Outputs.evaluations_results, wait=5000)
        self.assertTrue(self.widget.Error.test_data_incompatible.is_shown())

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
        self.get_output(self.widget.Outputs.evaluations_results, wait=5000)

        self.send_signal(self.widget.Inputs.train_data, data_with_disc_metas)
        self.assertTrue(rb.isEnabled())
        rb.click()
        self.assertEqual(self.widget.resampling, OWTestLearners.FeatureFold)
        self.assertTrue(self.widget.features_combo.isEnabled())
        self.assertEqual(self.widget.features_combo.currentText(), "iris")
        self.assertEqual(len(self.widget.features_combo.model()), 1)
        self.get_output(self.widget.Outputs.evaluations_results, wait=5000)

        self.send_signal(self.widget.Inputs.train_data, None)
        self.assertFalse(rb.isEnabled())
        self.assertEqual(self.widget.resampling, OWTestLearners.KFold)
        self.assertFalse(self.widget.features_combo.isEnabled())

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
        data = Table("iris")[::15]
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
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a"), ContinuousVariable("b")],
                [DiscreteVariable("c", values=["y"])]),
            list(zip(
                [42.48, 16.84, 15.23, 23.8],
                [1., 2., 3., 4.],
                "yyyy"))
        )
        self.widget.n_folds = 0
        self.assertFalse(self.widget.Error.only_one_class_var_value.is_shown())
        self.send_signal("Data", table)
        self.send_signal("Learner", MajorityLearner(), 0, wait=1000)
        self.assertTrue(self.widget.Error.only_one_class_var_value.is_shown())

    def test_nan_class(self):
        """
        Do not crash on a data with only nan class values.
        GH-2751
        """
        def assertErrorShown(data, is_shown):
            self.send_signal("Data", data)
            self.assertEqual(is_shown, self.widget.Error.no_class_values.is_shown())

        data = Table("iris")[::30]
        data.Y[:] = np.nan

        for data, is_shown in zip([None, data, Table("iris")[:30]], [False, True, False]):
            assertErrorShown(data, is_shown)

    def test_addon_scorers(self):
        try:
            # These classes are registered, pylint: disable=unused-variable
            class NewScore(Score):
                class_types = (DiscreteVariable, ContinuousVariable)

            class NewClassificationScore(ClassificationScore):
                pass

            class NewRegressionScore(RegressionScore):
                pass

            builtins = BUILTIN_SCORERS_ORDER
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
            del Score.registry["NewScore"]  # pylint: disable=no-member
            del Score.registry["NewClassificationScore"]  # pylint: disable=no-member
            del Score.registry["NewRegressionScore"]  # pylint: disable=no-member

    def test_target_changing(self):
        data = Table("iris")
        w = self.widget  #: OWTestLearners
        model = w.score_table.model

        w.n_folds = 2
        self.send_signal(self.widget.Inputs.train_data, data)
        self.send_signal(self.widget.Inputs.learner,
                         LogisticRegressionLearner(), 0, wait=5000)

        average_auc = float(model.item(0, 3).text())

        simulate.combobox_activate_item(w.controls.class_selection, "Iris-setosa")
        setosa_auc = float(model.item(0, 3).text())

        simulate.combobox_activate_item(w.controls.class_selection, "Iris-versicolor")
        versicolor_auc = float(model.item(0, 3).text())

        simulate.combobox_activate_item(w.controls.class_selection, "Iris-virginica")
        virginica_auc = float(model.item(0, 3).text())

        self.assertGreater(average_auc, versicolor_auc)
        self.assertGreater(average_auc, virginica_auc)
        self.assertLess(average_auc, setosa_auc)
        self.assertGreater(setosa_auc, versicolor_auc)
        self.assertGreater(setosa_auc, virginica_auc)

    def test_resort_on_data_change(self):
        iris = Table("iris")
        # one example is included from the other class
        # to keep F1 from complaining
        setosa = iris[:51]
        versicolor = iris[49:100]

        class SetosaLearner:
            def __call__(self, data):
                model = ConstantModel([1., 0, 0])
                model.domain = iris.domain
                return model

        class VersicolorLearner:
            def __call__(self, data):
                model = ConstantModel([0, 1., 0])
                model.domain = iris.domain
                return model

        # this is done manually to avoid multiple computations
        self.widget.resampling = 5
        self.widget.set_train_data(iris)
        self.widget.set_learner(SetosaLearner(), 1)
        self.widget.set_learner(VersicolorLearner(), 2)

        self.send_signal(self.widget.Inputs.test_data, setosa, wait=5000)

        self.widget.show()
        view = self.widget.score_table.view
        header = view.horizontalHeader()
        QTest.mouseClick(header.viewport(), Qt.LeftButton)
        # Ensure that the click on header caused an ascending sort
        # Ascending sort means that wrong model should be listed first
        if header.sortIndicatorOrder() != Qt.AscendingOrder:
            QTest.mouseClick(header.viewport(), Qt.LeftButton)

        self.assertEqual(view.model().index(0, 0).data(), "VersicolorLearner")

        self.send_signal(self.widget.Inputs.test_data, versicolor, wait=5000)
        self.assertEqual(view.model().index(0, 0).data(), "SetosaLearner")

        self.widget.hide()

    def _retrieve_scores(self):
        w = self.widget
        model = w.score_table.model
        auc = model.item(0, 3).text()
        auc = float(auc) if auc != "" else None
        ca = float(model.item(0, 4).text())
        f1 = float(model.item(0, 5).text())
        precision = float(model.item(0, 6).text())
        recall = float(model.item(0, 7).text())
        return auc, ca, f1, precision, recall

    def _test_scores(self, train_data, test_data, learner, sampling, n_folds):
        w = self.widget  #: OWTestLearners
        w.controls.resampling.buttons[sampling].click()
        if n_folds is not None:
            w.n_folds = n_folds

        self.send_signal(self.widget.Inputs.train_data, train_data)
        if test_data is not None:
            self.send_signal(self.widget.Inputs.test_data, test_data)
        self.send_signal(self.widget.Inputs.learner, learner, 0, wait=5000)
        return self._retrieve_scores()

    def test_scores_constant_all_same(self):
        table = Table.from_list(
            self.scores_domain,
            list(zip(*self.scores_table_values + [list("yyyy")]))
        )

        self.assertTupleEqual(self._test_scores(
            table, table, ConstantLearner(), OWTestLearners.TestOnTest, None),
                              (None, 1, 1, 1, 1))

    def test_scores_log_reg_overfitted(self):
        table = Table.from_list(
            self.scores_domain,
            list(zip(*self.scores_table_values + [list("yyyn")]))
        )

        self.assertTupleEqual(self._test_scores(
            table, table, LogisticRegressionLearner(),
            OWTestLearners.TestOnTest, None),
                              (1, 1, 1, 1, 1))

    def test_scores_log_reg_bad(self):
        table_train = Table.from_list(
            self.scores_domain,
            list(zip(*self.scores_table_values + [list("nnny")]))
        )
        table_test = Table.from_list(
            self.scores_domain,
            list(zip(*self.scores_table_values + [list("yyyn")]))
        )

        self.assertTupleEqual(self._test_scores(
            table_train, table_test, LogisticRegressionLearner(),
            OWTestLearners.TestOnTest, None),
                              (0, 0, 0, 0, 0))

    def test_scores_log_reg_bad2(self):
        table_train = Table.from_list(
            self.scores_domain,
            list(zip(*(self.scores_table_values + [list("nnyy")]))))
        table_test = Table.from_list(
            self.scores_domain,
            list(zip(*(self.scores_table_values + [list("yynn")]))))
        self.assertTupleEqual(self._test_scores(
            table_train, table_test, LogisticRegressionLearner(),
            OWTestLearners.TestOnTest, None),
                              (0, 0, 0, 0, 0))

    def test_scores_log_reg_advanced(self):
        table_train = Table.from_list(
            self.scores_domain, list(zip(
                [1, 1, 1.23, 23.8, 5.], [1., 2., 3., 4., 3.], "yyynn"))
        )
        table_test = Table.from_list(
            self.scores_domain, list(zip(
                [1, 1, 1.23, 23.8, 5.], [1., 2., 3., 4., 3.], "yynnn"))
        )

        np.testing.assert_almost_equal(
            self._test_scores(table_train, table_test,
                              LogisticRegressionLearner(),
                              OWTestLearners.TestOnTest, None),
            (2 / 3, 0.8, 0.8, 13 / 15, 0.8))

    def test_scores_cross_validation(self):
        """
        Test more than two classes and cross-validation
        """
        self.assertTrue(
            all(x >= y for x, y in zip(
                self._test_scores(
                    Table("iris")[::15], None, LogisticRegressionLearner(),
                    OWTestLearners.KFold, 0),
                (0.8, 0.5, 0.5, 0.5, 0.5))))

    def test_no_pregressbar_warning(self):
        data = Table("iris")[::15]

        with warnings.catch_warnings(record=True) as w:
            self.send_signal(self.widget.Inputs.train_data, data)
            self.send_signal(self.widget.Inputs.learner, MajorityLearner(), 0)
            assert not w


class TestHelpers(unittest.TestCase):
    def test_results_one_vs_rest(self):
        data = Table(test_filename("datasets/lenses.tab"))
        learners = [MajorityLearner()]
        res = TestOnTestData()(data[1::2], data[::2], learners=learners)
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

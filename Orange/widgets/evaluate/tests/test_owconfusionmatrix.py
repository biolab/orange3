# pylint: disable=missing-docstring, protected-access
import unittest
import numpy as np

from Orange.data import Table, Domain
from Orange.classification import NaiveBayesLearner, TreeLearner
from Orange.regression import MeanLearner
from Orange.evaluation.testing import CrossValidation, TestOnTrainingData, \
    ShuffleSplit, Results
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.tests.utils import possible_duplicate_table, simulate


class TestOWConfusionMatrix(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        bayes = NaiveBayesLearner()
        tree = TreeLearner()
        # `data` is defined in WidgetOutputsTestMixin, pylint: disable=no-member
        cls.iris = cls.data
        titanic = Table("titanic")
        cv = CrossValidation(k=3, store_data=True)
        cls.results_1_iris = cv(cls.iris, [bayes])
        cls.results_2_iris = cv(cls.iris, [bayes, tree])
        cls.results_2_titanic = cv(titanic, [bayes, tree])

        cls.signal_name = OWConfusionMatrix.Inputs.evaluation_results
        cls.signal_data = cls.results_1_iris
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWConfusionMatrix,
                                         stored_settings={"auto_apply": False})

    def test_selected_learner(self):
        """Check learner and model for various values of all parameters
        when pruning parameters are not checked
        """
        self.send_signal(self.widget.Inputs.evaluation_results, self.results_2_iris)
        self.assertEqual(self.widget.selected_learner, [0])
        self.widget.selected_learner[:] = [1]
        self.send_signal(self.widget.Inputs.evaluation_results, self.results_2_titanic)
        self.widget.selected_learner[:] = [1]
        self.send_signal(self.widget.Inputs.evaluation_results, self.results_1_iris)
        self.widget.selected_learner[:] = [0]
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.send_signal(self.widget.Inputs.evaluation_results, self.results_1_iris)
        self.widget.selected_learner[:] = [0]

    def _select_data(self):
        self.widget.select_correct()
        indices = self.widget.tableview.selectedIndexes()
        indices = {(ind.row() - 2, ind.column() - 2) for ind in indices}
        selected = [i for i, t in enumerate(zip(
            self.widget.results.actual, self.widget.results.predicted[0]))
                    if t in indices]
        return self.widget.results.row_indices[selected]

    def test_show_error_on_regression(self):
        """On regression data, the widget must show error"""
        housing = Table("housing")
        results = TestOnTrainingData(store_data=True)(housing, [MeanLearner()])
        self.send_signal(self.widget.Inputs.evaluation_results, results)
        self.assertTrue(self.widget.Error.no_regression.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.assertFalse(self.widget.Error.no_regression.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, results)
        self.assertTrue(self.widget.Error.no_regression.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, self.results_1_iris)
        self.assertFalse(self.widget.Error.no_regression.is_shown())

    def test_row_indices(self):
        """Map data instances when using random shuffling"""
        results = ShuffleSplit(store_data=True
                               )(self.iris, [NaiveBayesLearner()])
        self.send_signal(self.widget.Inputs.evaluation_results, results)
        self.widget.select_correct()
        selected = self.get_output(self.widget.Outputs.selected_data)
        # pylint: disable=unsubscriptable-object
        correct = np.equal(results.actual, results.predicted)[0]
        correct_indices = results.row_indices[correct]
        self.assertSetEqual(set(self.iris[correct_indices].ids),
                            set(selected.ids))

    def test_empty_results(self):
        """Test on empty results."""
        res = Results(data=self.iris[:0], store_data=True)
        res.row_indices = np.array([], dtype=int)
        res.actual = np.array([])
        res.predicted = np.array([[]])
        res.probabilities = np.zeros((1, 0, 3))
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.select_correct()
        self.widget.select_wrong()

    def test_nan_results(self):
        """Test on results with nan values in actual/predicted"""
        res = Results(data=self.iris, nmethods=2, store_data=True)
        res.row_indices = np.array([0, 50, 100], dtype=int)
        res.actual = np.array([0., np.nan, 2.])
        res.predicted = np.array([[np.nan, 1, 2],
                                  [np.nan, np.nan, np.nan]])
        res.probabilities = np.zeros((1, 3, 3))
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.assertTrue(self.widget.Error.invalid_values.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.assertFalse(self.widget.Error.invalid_values.is_shown())

    def test_not_append_extra_meta_columns(self):
        """
        When a user does not want append extra meta column, the widget
        should not crash.
        GH-2386
        """
        self.widget.append_predictions = False
        self.send_signal(self.widget.Inputs.evaluation_results, self.results_1_iris)

    def test_unique_output_domain(self):
        bayes = NaiveBayesLearner()
        data = possible_duplicate_table('iris(Learner #1)')
        input_data = CrossValidation(k=3, store_data=True)(data, [bayes])
        self.send_signal(self.widget.Inputs.evaluation_results, input_data)
        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(output.domain.metas[0].name, 'iris(Learner #1) (1)')

    def test_unique_var_names(self):
        bayes = NaiveBayesLearner()
        domain = self.iris.domain
        results = CrossValidation(k=3, store_data=True)(self.iris, [bayes])
        self.widget.append_probabilities = True
        self.widget.append_predictions = True
        self.send_signal(self.widget.Inputs.evaluation_results, results)

        out_data = self.get_output(self.widget.Outputs.annotated_data)

        widget2 = self.create_widget(OWConfusionMatrix)
        data2 = out_data.transform(
            Domain(domain.attributes, domain.class_vars,
                   [meta for meta in out_data.domain.metas if "versicolor" not in meta.name]))
        results2 = CrossValidation(k=3, store_data=True)(data2, [bayes])
        widget2.append_probabilities = True
        widget2.append_predictions = True
        self.send_signal(widget2.Inputs.evaluation_results, results2)
        out_data2 = self.get_output(widget2.Outputs.annotated_data)
        self.assertEqual({meta.name for meta in out_data2.domain.metas},
                         {'Selected', 'Selected (1)',
                          'iris(Learner #1)', 'iris(Learner #1) (1)',
                          'p(Iris-setosa)', 'p(Iris-virginica)',
                          'p(Iris-setosa) (1)', 'p(Iris-versicolor) (1)',
                          'p(Iris-virginica) (1)'})

    def test_sum_of_probabilities(self):
        results: Results = self.results_1_iris
        self.send_signal(self.widget.Inputs.evaluation_results, results)

        model = self.widget.tablemodel
        n = model.rowCount() - 3
        matrix = np.zeros((n, n))
        probabilities = results.probabilities[0]
        for label_index in np.unique(results.actual).astype(int):
            mask = results.actual == label_index
            prob_sum = np.sum(probabilities[mask], axis=0)
            matrix[label_index] = prob_sum
        colsum = matrix.sum(axis=0)
        rowsum = matrix.sum(axis=1)

        simulate.combobox_activate_index(
            self.widget.controls.selected_quantity, 3)
        # matrix
        for i in range(n):
            for j in range(n):
                value = model.data(model.index(i + 2, j + 2))
                self.assertAlmostEqual(float(value), matrix[i, j], 1)
        # rowsum
        for i in range(n):
            value = model.data(model.index(i + 2, n + 2))
            self.assertAlmostEqual(float(value), rowsum[i], 0)
        # colsum
        for i in range(n):
            value = model.data(model.index(n + 2, i + 2))
            self.assertAlmostEqual(float(value), colsum[i], 0)
        # total
        value = model.data(model.index(n + 2, n + 2))
        self.assertAlmostEqual(float(value), colsum.sum(), 0)


if __name__ == "__main__":
    unittest.main()

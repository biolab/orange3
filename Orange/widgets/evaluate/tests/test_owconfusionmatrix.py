# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.classification import NaiveBayesLearner, TreeLearner
from Orange.regression import MeanLearner
from Orange.evaluation.testing import CrossValidation, TestOnTrainingData, \
    ShuffleSplit, Results
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWConfusionMatrix(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        bayes = NaiveBayesLearner()
        tree = TreeLearner()
        cls.iris = cls.data
        titanic = Table("titanic")
        common = dict(k=3, store_data=True)
        cls.results_1_iris = CrossValidation(cls.iris, [bayes], **common)
        cls.results_2_iris = CrossValidation(cls.iris, [bayes, tree], **common)
        cls.results_2_titanic = CrossValidation(titanic, [bayes, tree],
                                                **common)

        cls.signal_name = "Evaluation Results"
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
        results = TestOnTrainingData(housing, [MeanLearner()], store_data=True)
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
        results = ShuffleSplit(self.iris, [NaiveBayesLearner()],
                               store_data=True)
        self.send_signal(self.widget.Inputs.evaluation_results, results)
        self.widget.select_correct()
        selected = self.get_output(self.widget.Outputs.selected_data)
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

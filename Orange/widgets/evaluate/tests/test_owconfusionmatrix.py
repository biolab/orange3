# pylint: disable=missing-docstring

from Orange.data import Table
from Orange.classification import NaiveBayesLearner, TreeLearner
from Orange.regression import MeanLearner
from Orange.evaluation.testing import CrossValidation, TestOnTrainingData
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWConfusionMatrix(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        bayes = NaiveBayesLearner()
        tree = TreeLearner()
        iris = cls.data
        titanic = Table("titanic")
        common = dict(k=3, store_data=True)
        cls.results_1_iris = CrossValidation(iris, [bayes], **common)
        cls.results_2_iris = CrossValidation(iris, [bayes, tree], **common)
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
        self.send_signal("Evaluation Results", self.results_2_iris)
        self.assertEqual(self.widget.selected_learner, [0])
        self.widget.selected_learner[:] = [1]
        self.send_signal("Evaluation Results", self.results_2_titanic)
        self.widget.selected_learner[:] = [1]
        self.send_signal("Evaluation Results", self.results_1_iris)
        self.widget.selected_learner[:] = [0]
        self.send_signal("Evaluation Results", None)
        self.send_signal("Evaluation Results", self.results_1_iris)
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
        results = TestOnTrainingData(housing, [MeanLearner()])
        results.data = housing
        self.send_signal("Evaluation Results", results)
        self.assertTrue(self.widget.Error.no_regression.is_shown())
        self.send_signal("Evaluation Results", None)
        self.assertFalse(self.widget.Error.no_regression.is_shown())
        self.send_signal("Evaluation Results", results)
        self.assertTrue(self.widget.Error.no_regression.is_shown())
        self.send_signal("Evaluation Results", self.results_1_iris)
        self.assertFalse(self.widget.Error.no_regression.is_shown())

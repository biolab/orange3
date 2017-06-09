import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.modelling import RandomForestLearner, SGDLearner
from Orange.preprocess.score import Scorer
from Orange.classification import LogisticRegressionLearner
from Orange.regression import LinearRegressionLearner
from Orange.projection import PCA
from Orange.widgets.data.owrank import OWRank
from Orange.widgets.tests.base import WidgetTest

from AnyQt.QtCore import Qt


class TestOWRank(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWRank)
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.log_reg = LogisticRegressionLearner()
        self.lin_reg = LinearRegressionLearner()
        self.pca = PCA()

    def test_input_data(self):
        """Check widget's data with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)

    def test_input_data_disconnect(self):
        """Check widget's data after disconnecting data on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)

    def test_input_scorer(self):
        """Check widget's scorer with scorer on the input"""
        self.assertEqual(self.widget.learners, {})
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        value = self.widget.learners[1]
        self.assertEqual(self.log_reg, value.score)
        self.assertIsInstance(value.score, Scorer)

    def test_input_scorer_fitter(self):
        heart_disease = Table('heart_disease')
        self.assertEqual(self.widget.learners, {})

        for fitter, name in ((RandomForestLearner(), 'random forest'),
                             (SGDLearner(), 'sgd')):
            with self.subTest(fitter=fitter):
                self.send_signal("Scorer", fitter, 1)

                for data, model in ((self.housing, self.widget.contRanksModel),
                                    (heart_disease, self.widget.discRanksModel)):
                    with self.subTest(data=data.name):
                        self.send_signal('Data', data)
                        scores = [model.data(model.index(row, model.columnCount() - 1))
                                  for row in range(model.rowCount())]
                        self.assertEqual(len(scores), len(data.domain.attributes))
                        self.assertFalse(np.isnan(scores).any())

                        last_column = model.headerData(
                            model.columnCount() - 1, Qt.Horizontal).lower()
                        self.assertIn(name, last_column)

                self.send_signal("Scorer", None, 1)
                self.assertEqual(self.widget.learners, {})

    def test_input_scorer_disconnect(self):
        """Check widget's scorer after disconnecting scorer on the input"""
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.assertEqual(len(self.widget.learners), 1)
        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.assertEqual(self.widget.learners, {})

    def test_output_data(self):
        """Check data on the output after apply"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output.X), len(self.iris))
        self.assertEqual(output.domain.class_var, self.iris.domain.class_var)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))

    def test_output_scores(self):
        """Check scores on the output after apply"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(output.X.shape, (len(self.iris.domain.attributes), 2))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_output_scores_with_scorer(self):
        """Check scores on the output after apply with scorer on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(output.X.shape, (len(self.iris.domain.attributes), 5))

    def test_scoring_method_check_box(self):
        """Check scoring methods check boxes"""
        boxes = [self.widget.cls_scoring_box] * 7 + \
                [self.widget.reg_scoring_box] * 2
        for check_box, box in zip(self.widget.score_checks, boxes):
            self.assertEqual(check_box.parent(), box)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.score_stack.currentWidget(), boxes[0])
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(self.widget.score_stack.currentWidget(), boxes[7])
        data = Table.from_table(Domain(self.iris.domain.variables), self.iris)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertNotIn(self.widget.score_stack.currentWidget(), boxes)

    def test_scoring_method_default(self):
        """Check selected scoring methods with no data on the input"""
        self.send_signal(self.widget.Inputs.data, None)
        check_score = (False, True, True, False, False, False, False, False,
                       False)
        for check_box, checked in zip(self.widget.score_checks, check_score):
            self.assertEqual(check_box.isChecked(), checked)

    def test_scoring_method_classification(self):
        """Check selected scoring methods with classification data on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        check_score = (False, True, True, False, False, False, False, False,
                       False)
        for check_box, checked in zip(self.widget.score_checks, check_score):
            self.assertEqual(check_box.isChecked(), checked)

    def test_scoring_method_regression(self):
        """Check selected scoring methods with regression data on the input"""
        self.send_signal(self.widget.Inputs.data, self.housing)
        check_score = (False, False, False, False, False, False, False,
                       True, True)
        for check_box, checked in zip(self.widget.score_checks, check_score):
            self.assertEqual(check_box.isChecked(), checked)

    def test_cls_scorer_reg_data(self):
        """Check scores on the output with inadequate scorer"""
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.scorer, self.pca, 1)
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 2)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 16))

    def test_reg_scorer_cls_data(self):
        """Check scores on the output with inadequate scorer"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.pca, 1)
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 2)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 7))

    def test_scoring_method_visible(self):
        """Check which scoring box is visible according to data"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.score_stack.currentIndex(), 0)
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(self.widget.score_stack.currentIndex(), 1)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.score_stack.currentIndex(), 0)

    def test_scores_updates_cls(self):
        """Check arbitrary workflow with classification data"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self.widget.score_checks[2].setChecked(False)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 4))
        self.widget.score_checks[2].setChecked(True)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 2)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 8))
        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 8))
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 3)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 9))

    def test_scores_updates_reg(self):
        """Check arbitrary workflow with regression data"""
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 3))
        self.widget.score_checks[-2].setChecked(False)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 2))
        self.widget.score_checks[-2].setChecked(True)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 3))
        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 2))
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 3))

    def test_scores_updates_no_class(self):
        """Check arbitrary workflow with no class variable dataset"""
        data = Table.from_table(Domain(self.iris.domain.variables), self.iris)
        self.assertIsNone(data.domain.class_var)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 1))
        self.send_signal(self.widget.Inputs.scorer, self.pca, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 7))
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 2)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 8))

    def test_data_which_make_scorer_nan(self):
        """
        Tests if widget crashes due to too high (Infinite) calculated values.
        GH-2168
        """
        table = Table(
            Domain(
                [ContinuousVariable("c")],
                [DiscreteVariable("d", values="01")]
            ),
            list(zip(
                [-np.power(10, 10), 1, 1],
                [0, 1, 1]
            )))
        self.widget.score_checks[3].setChecked(True) #ANOVA
        self.send_signal(self.widget.Inputs.data, table)

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.modelling import RandomForestLearner, SGDLearner
from Orange.preprocess.score import Scorer
from Orange.classification import LogisticRegressionLearner
from Orange.regression import LinearRegressionLearner
from Orange.projection import PCA
from Orange.widgets.data.owrank import OWRank, ProblemType, CLS_SCORES, REG_SCORES
from Orange.widgets.tests.base import WidgetTest

from AnyQt.QtCore import Qt, QItemSelection
from AnyQt.QtWidgets import QCheckBox


class TestOWRank(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWRank)  # type: OWRank
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.log_reg = LogisticRegressionLearner()
        self.lin_reg = LinearRegressionLearner()
        self.pca = PCA()

    def _get_checkbox(self, method_shortname):
        return self.widget.controlArea.findChild(QCheckBox, method_shortname)

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
        self.assertEqual(self.widget.scorers, {})
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        value = self.widget.scorers[1]
        self.assertEqual(self.log_reg, value.scorer)
        self.assertIsInstance(value.scorer, Scorer)

    def test_input_scorer_fitter(self):
        heart_disease = Table('heart_disease')
        self.assertEqual(self.widget.scorers, {})

        model = self.widget.ranksModel

        for fitter, name in ((RandomForestLearner(), 'random forest'),
                             (SGDLearner(), 'sgd')):
            with self.subTest(fitter=fitter):
                self.send_signal("Scorer", fitter, 1)

                for data in (self.housing,
                             heart_disease):
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
                self.assertEqual(self.widget.scorers, {})

    def test_input_scorer_disconnect(self):
        """Check widget's scorer after disconnecting scorer on the input"""
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.assertEqual(len(self.widget.scorers), 1)
        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.assertEqual(self.widget.scorers, {})

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

    def test_scoring_method_problem_type(self):
        """Check scoring methods check boxes"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.problem_type_mode, ProblemType.CLASSIFICATION)
        self.assertEqual(self.widget.measuresStack.currentIndex(), ProblemType.CLASSIFICATION)

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(self.widget.problem_type_mode, ProblemType.REGRESSION)
        self.assertEqual(self.widget.measuresStack.currentIndex(), ProblemType.REGRESSION)

        data = Table.from_table(Domain(self.iris.domain.variables), self.iris)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(self.widget.problem_type_mode, ProblemType.UNSUPERVISED)
        self.assertEqual(self.widget.measuresStack.currentIndex(), ProblemType.UNSUPERVISED)

    def test_scoring_method_defaults(self):
        """Check default scoring methods are selected"""
        self.send_signal(self.widget.Inputs.data, None)
        for method in CLS_SCORES:
            checkbox = self._get_checkbox(method.shortname)
            self.assertEqual(checkbox.isChecked(), method.is_default)

        self.send_signal(self.widget.Inputs.data, self.housing)
        for method in REG_SCORES:
            checkbox = self._get_checkbox(method.shortname)
            self.assertEqual(checkbox.isChecked(), method.is_default)

        self.send_signal(self.widget.Inputs.data, self.iris)
        for method in CLS_SCORES:
            checkbox = self._get_checkbox(method.shortname)
            self.assertEqual(checkbox.isChecked(), method.is_default)

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

    def test_scores_updates_cls(self):
        """Check arbitrary workflow with classification data"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self._get_checkbox('Gini').setChecked(False)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 4))
        self._get_checkbox('Gini').setChecked(True)
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

        self._get_checkbox('Univar. reg.').setChecked(False)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 2))

        self._get_checkbox('Univar. reg.').setChecked(True)
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

    def test_scores_sorting(self):
        """Check clicking on header column orders scores in a different way"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        order1 = self.widget.ranksModel.mapToSourceRows(...).tolist()
        self._get_checkbox('FCBF').setChecked(True)
        self.widget.ranksView.horizontalHeader().setSortIndicator(3, Qt.DescendingOrder)
        order2 = self.widget.ranksModel.mapToSourceRows(...).tolist()
        self.assertNotEqual(order1, order2)

    def test_scores_nan_sorting(self):
        """Check NaNs are sorted last"""
        data = self.iris.copy()
        data.get_column_view('petal length')[0][:] = np.nan
        self.send_signal(self.widget.Inputs.data, data)

        # Assert last row is all nan
        for order in (Qt.AscendingOrder,
                      Qt.DescendingOrder):
            self.widget.ranksView.horizontalHeader().setSortIndicator(1, order)
            last_row = self.widget.ranksModel[self.widget.ranksModel.mapToSourceRows(...)[-1]]
            np.testing.assert_array_equal(last_row, np.repeat(np.nan, 3))

    def test_default_sort_indicator(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertNotEqual(
            0, self.widget.ranksView.horizontalHeader().sortIndicatorSection())

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
        self.widget.selected_methods.add('ANOVA')
        self.send_signal(self.widget.Inputs.data, table)

    def test_setting_migration_fixes_header_state(self):
        # Settings as of version 3.3.5
        settings = {
            '__version__': 1,
            'auto_apply': True,
            'headerState': (
                b'\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00'
                b'\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00'
                b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\xd0\x00'
                b'\x00\x00\x08\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00'
                b'\x00\x00\x00\x00\x00d\xff\xff\xff\xff\x00\x00\x00\x84\x00'
                b'\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x14\x00\x00\x00'
                b'\x01\x00\x00\x00\x00\x00\x00\x02\xbc\x00\x00\x00\x07\x00'
                b'\x00\x00\x00',
                b'\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00'
                b'\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00'
                b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdc\x00'
                b'\x00\x00\x03\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00'
                b'\x00\x00\x00\x00\x00d\xff\xff\xff\xff\x00\x00\x00\x84\x00'
                b'\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x14\x00\x00\x00'
                b'\x01\x00\x00\x00\x00\x00\x00\x00\xc8\x00\x00\x00\x02\x00'
                b'\x00\x00\x00'),
            'nSelected': 5,
            'selectMethod': 3
        }

        w = self.create_widget(OWRank, stored_settings=settings)

        self.assertEqual(w.sorting, (0, Qt.AscendingOrder))

    def test_auto_send(self):
        widget = self.widget
        model = widget.ranksModel
        selectionModel = widget.ranksView.selectionModel()

        # Auto-send disabled
        widget.controls.auto_apply.setChecked(False)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertIsNone(self.get_output(widget.Outputs.reduced_data))

        # Make selection, but auto-send disabled
        selection = QItemSelection(model.index(1, 0),
                                   model.index(1, model.columnCount() - 1))
        selectionModel.select(selection, selectionModel.ClearAndSelect)
        self.assertIsNone(self.get_output(widget.Outputs.reduced_data))

        # Enable auto-send: should output data
        widget.controls.auto_apply.setChecked(True)
        reduced_data = self.get_output(widget.Outputs.reduced_data)
        self.assertEqual(reduced_data.domain.attributes,
                         (self.iris.domain["petal width"], ))

        # Change selection: should change the output immediately
        selection = QItemSelection(model.index(0, 0),
                                   model.index(0, model.columnCount() - 1))
        selectionModel.select(selection, selectionModel.ClearAndSelect)
        reduced_data = self.get_output(widget.Outputs.reduced_data)
        self.assertEqual(reduced_data.domain.attributes,
                         (self.iris.domain["petal length"], ))

    def test_no_attributes(self):
        """
        Rank should not fail on data with no attributes.
        GH-2745
        """
        data = Table("iris")[::30]
        domain = Domain(attributes=[], class_vars=data.domain.class_vars)
        new_data = data.transform(domain)
        self.assertFalse(self.widget.Error.no_attributes.is_shown())
        self.send_signal(self.widget.Inputs.data, new_data)
        self.assertTrue(self.widget.Error.no_attributes.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(self.widget.Error.no_attributes.is_shown())

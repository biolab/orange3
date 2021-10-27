# pylint: disable=unsubscriptable-object
import time
import warnings
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from AnyQt.QtCore import Qt, QItemSelection, QItemSelectionModel
from AnyQt.QtWidgets import QCheckBox, QApplication

from orangewidget.settings import Context, IncompatibleContext
from orangewidget.tests.base import GuiTest

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.modelling import RandomForestLearner, SGDLearner
from Orange.preprocess.score import Scorer
from Orange.classification import LogisticRegressionLearner
from Orange.regression import LinearRegressionLearner
from Orange.projection import PCA
from Orange.widgets.data.owrank import OWRank, ProblemType, CLS_SCORES, \
    REG_SCORES, TableModel
from Orange.widgets.tests.base import WidgetTest, datasets
from Orange.widgets.widget import AttributeList


class SlowScorer(Scorer):
    name = "Slow scorer"

    def score_data(self, data, feature=None):
        time.sleep(1)
        return np.ones((1, len(data.domain.attributes)))


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
        self.wait_until_finished()
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)

    def test_input_scorer(self):
        """Check widget's scorer with scorer on the input"""
        self.assertEqual(self.widget.scorers, [])
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.wait_until_finished()
        value = self.widget.scorers[0]
        self.assertEqual(self.log_reg, value.scorer)
        self.assertIsInstance(value.scorer, Scorer)

    def test_input_scorer_fitter(self):
        heart_disease = Table('heart_disease')
        self.assertEqual(self.widget.scorers, [])

        model = self.widget.ranksModel

        for fitter, name in ((RandomForestLearner(), 'random forest'),
                             (SGDLearner(), 'sgd')):
            with self.subTest(fitter=fitter),\
                    warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*", ConvergenceWarning)
                self.send_signal("Scorer", fitter, 1)

                for data in (self.housing,
                             heart_disease):
                    with self.subTest(data=data.name):
                        self.send_signal('Data', data)
                        self.wait_until_finished()
                        scores = [model.data(model.index(row, model.columnCount() - 1))
                                  for row in range(model.rowCount())]
                        self.assertEqual(len(scores), len(data.domain.attributes))
                        self.assertFalse(np.isnan(scores).any())

                        last_column = model.headerData(
                            model.columnCount() - 1, Qt.Horizontal).lower()
                        self.assertIn(name, last_column)

                self.send_signal("Scorer", None, 1)
                self.assertEqual(self.widget.scorers, [])

    def test_input_scorer_disconnect(self):
        """Check widget's scorer after disconnecting scorer on the input"""
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.assertEqual(len(self.widget.scorers), 1)
        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.assertEqual(self.widget.scorers, [])

    def test_output_data(self):
        """Check data on the output after apply"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output.X), len(self.iris))
        self.assertEqual(output.domain.class_var, self.iris.domain.class_var)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))

    def test_output_scores(self):
        """Check scores on the output after apply"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(output.X.shape, (len(self.iris.domain.attributes), 2))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_output_scores_with_scorer(self):
        """Check scores on the output after apply with scorer on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(output.X.shape, (len(self.iris.domain.attributes), 5))

    def test_output_features(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.features)
        self.assertIsInstance(output, AttributeList)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

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
        self.wait_until_finished()
        with patch("Orange.widgets.data.owrank.log.error") as log:
            self.send_signal(self.widget.Inputs.scorer, self.log_reg, 2)
            self.wait_until_finished()
            log.assert_called()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 16))

    def test_reg_scorer_cls_data(self):
        """Check scores on the output with inadequate scorer"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.pca, 1)
        self.wait_until_finished()
        with patch("Orange.widgets.data.owrank.log.error") as log:
            self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 2)
            self.wait_until_finished()
            log.assert_called()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 7))

    def test_scores_updates_cls(self):
        """Check arbitrary workflow with classification data"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self._get_checkbox('Gini').setChecked(False)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 4))
        self._get_checkbox('Gini').setChecked(True)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 2)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 8))
        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 5))
        self.send_signal(self.widget.Inputs.scorer, self.log_reg, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 8))
        with patch("Orange.widgets.data.owrank.log.error") as log:
            self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 3)
            self.wait_until_finished()
            log.assert_called()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.attributes), 9))

    def test_scores_updates_reg(self):
        """Check arbitrary workflow with regression data"""
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 3))

        self._get_checkbox('Univar. reg.').setChecked(False)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 2))

        self._get_checkbox('Univar. reg.').setChecked(True)
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 3))

        self.send_signal(self.widget.Inputs.scorer, None, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 2))

        self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.housing.domain.attributes), 3))

    def test_scores_updates_no_class(self):
        """Check arbitrary workflow with no class variable dataset"""
        data = Table.from_table(Domain(self.iris.domain.variables), self.iris)
        self.assertIsNone(data.domain.class_var)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

        with patch("Orange.widgets.data.owrank.log.error") as log:
            self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 1)
            self.wait_until_finished()
            log.assert_called()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 1))

        self.send_signal(self.widget.Inputs.scorer, self.pca, 1)
        self.wait_until_finished()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 7))

        with patch("Orange.widgets.data.owrank.log.error") as log:
            self.send_signal(self.widget.Inputs.scorer, self.lin_reg, 2)
            self.wait_until_finished()
            log.assert_called()
        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 8))

    def test_no_class_data_learner_class_reg(self):
        """
        Check workflow with learners that can be both classifier
        or regressor and data have no class variable. This test should not
        fail.
        """
        data = Table.from_table(Domain(self.iris.domain.variables), self.iris)
        random_forest = RandomForestLearner()
        self.assertIsNone(data.domain.class_var)
        self.send_signal(self.widget.Inputs.data, data)

        with patch("Orange.widgets.data.owrank.log.error") as log:
            self.send_signal(self.widget.Inputs.scorer, random_forest, 1)
            self.wait_until_finished()
            log.assert_called()

        self.assertEqual(self.get_output(self.widget.Outputs.scores).X.shape,
                         (len(self.iris.domain.variables), 1))

    def test_scores_sorting(self):
        """Check clicking on header column orders scores in a different way"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        order1 = self.widget.ranksModel.mapToSourceRows(...).tolist()
        self._get_checkbox('FCBF').setChecked(True)
        self.wait_until_finished()
        self.widget.ranksView.horizontalHeader().setSortIndicator(4, Qt.DescendingOrder)
        order2 = self.widget.ranksModel.mapToSourceRows(...).tolist()
        self.assertNotEqual(order1, order2)

    def test_scores_nan_sorting(self):
        """Check NaNs are sorted last"""
        data = self.iris.copy()
        with data.unlocked():
            data.get_column_view('petal length')[0][:] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()

        # Assert last row is all nan
        for order in (Qt.AscendingOrder,
                      Qt.DescendingOrder):
            self.widget.ranksView.horizontalHeader().setSortIndicator(2, order)
            last_row = self.widget.ranksModel[self.widget.ranksModel.mapToSourceRows(...)[-1]]
            np.testing.assert_array_equal(last_row[1:], np.repeat(np.nan, 3))

    def test_default_sort_indicator(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertNotEqual(
            0, self.widget.ranksView.horizontalHeader().sortIndicatorSection())

    def test_data_which_make_scorer_nan(self):
        """
        Tests if widget crashes due to too high (Infinite) calculated values.
        GH-2168
        """
        table = Table.from_list(
            Domain(
                [ContinuousVariable("c")],
                [DiscreteVariable("d", values="01")]
            ),
            # false positive, pylint: disable=invalid-unary-operand-type
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

    def test_discard_settings_before_v3(self):
        for version in (None, 1, 2):
            self.assertRaises(IncompatibleContext, OWRank.migrate_context,
                              Context(), version=version)

    def test_auto_selection_manual(self):
        w = self.widget

        data = Table("heart_disease")
        dom = data.domain
        self.send_signal(w.Inputs.data, data)
        self.wait_until_finished()

        # Sort by number of values and set selection to attributes with most
        # values. This must select the top 4 rows.
        self.widget.ranksView.horizontalHeader().setSortIndicator(1, Qt.DescendingOrder)
        w.selectionMethod = w.SelectManual
        w.selected_attrs = [dom["chest pain"], dom["rest ECG"],
                            dom["slope peak exc ST"], dom["thal"]]
        w.autoSelection()
        self.assertEqual(
            sorted({idx.row() for idx in w.ranksView.selectedIndexes()}),
            [0, 1, 2, 3])

    def test_resorting_and_selection(self):
        def sortby(col):
            model.sort(col)
            view.horizontalHeader().sectionClicked.emit(col)
            QApplication.processEvents()

        Names, Values, Gain = range(3)

        w = self.widget

        data = Table("heart_disease")
        self.send_signal(w.Inputs.data, data)
        self.wait_until_finished()

        first4 = set(sorted((var.name for var in data.domain.attributes), key=str.lower)[:4])

        view = w.ranksView
        model = w.ranksModel
        selModel = view.selectionModel()
        columnCount = model.columnCount()

        w.selectionMethod = w.SelectNBest
        w.nSelected = 4

        # Sort by gain ratio, store selection
        sortby(Gain)
        gain_sel_4 = w.selected_attrs[:]
        self.assertEqual(len(gain_sel_4), 4)

        # Sort by names or number of values: selection unchanged
        sortby(Values)
        self.assertEqual(w.selected_attrs[:], gain_sel_4)

        sortby(Names)
        self.assertEqual(w.selected_attrs[:], gain_sel_4)

        # Select first four (alphabetically)
        w.selectionMethod = w.SelectManual
        selection = QItemSelection(
            model.index(0, 0),
            model.index(3, columnCount - 1)
        )
        selModel.select(selection, QItemSelectionModel.ClearAndSelect)
        # Sanity check
        self.assertEqual({var.name for var in w.selected_attrs}, first4)

        # Manual sorting: sorting by score does not change selection
        sortby(Gain)
        self.assertEqual({var.name for var in w.selected_attrs}, first4)

        # Sort by first four, again
        sortby(Names)
        # Sanity check
        self.assertEqual({var.name for var in w.selected_attrs}, first4)

        w.selectionMethod = w.SelectNBest
        # Sanity check
        self.assertEqual({var.name for var in w.selected_attrs}, first4)

        # Sorting by gain must change selection
        sortby(Gain)
        self.assertEqual(set(w.selected_attrs), set(gain_sel_4))

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

    def test_dataset(self):
        for method in CLS_SCORES + REG_SCORES:
            self._get_checkbox(method.shortname).setChecked(True)
        with patch("Orange.widgets.data.owrank.log.warning"), \
                patch("Orange.widgets.data.owrank.log.error"),\
                warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Features .* are constant",
                                    UserWarning)
            for ds in datasets.datasets():
                self.send_signal(self.widget.Inputs.data, ds)

    def test_selected_rows(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        # select first and second row
        w.selected_rows = [1, 2]
        output = self.get_output(w.Outputs.reduced_data)

        self.assertEqual(len(output), len(self.iris))

    def test_concurrent_cancel(self):
        """
        Send one signal after another. It test if the first process get
        correctly canceled when new signal comes.
        """
        sc = SlowScorer()
        self.send_signal(self.widget.Inputs.scorer, sc, 1)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertEqual(len(output), len(self.housing))


class TestRankModel(GuiTest):
    @staticmethod
    def test_argsort():
        func = TableModel()._argsortData  # pylint: disable=protected-access
        assert_equal = np.testing.assert_equal

        test_array = np.array([4.2, 7.2, np.nan, 1.3, np.nan])
        assert_equal(func(test_array, Qt.AscendingOrder)[:3], [3, 0, 1])
        assert_equal(func(test_array, Qt.DescendingOrder)[:3], [1, 0, 3])

        test_array = np.array([4, 7, 2])
        assert_equal(func(test_array, Qt.AscendingOrder), [2, 0, 1])
        assert_equal(func(test_array, Qt.DescendingOrder), [1, 0, 2])

        test_array = np.array(["Bertha", "daniela", "ann", "Cecilia"])
        assert_equal(func(test_array, Qt.AscendingOrder), [2, 0, 3, 1])
        assert_equal(func(test_array, Qt.DescendingOrder), [1, 3, 0, 2])


if __name__ == "__main__":
    unittest.main()

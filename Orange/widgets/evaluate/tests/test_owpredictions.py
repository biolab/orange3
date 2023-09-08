"""Tests for OWPredictions"""
# pylint: disable=protected-access
import io
import unittest
from functools import partial
from typing import Optional
from unittest.mock import Mock, patch

import numpy as np

from AnyQt.QtCore import QItemSelectionModel, QItemSelection, Qt, QRect
from AnyQt.QtWidgets import QToolTip

from Orange.base import Model
from Orange.classification import LogisticRegressionLearner, NaiveBayesLearner
from Orange.classification.majority import ConstantModel, MajorityLearner
from Orange.data.io import TabReader
from Orange.evaluation.scoring import TargetScore
from Orange.preprocess import Remove
from Orange.regression import LinearRegressionLearner, MeanLearner
from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.widgets.evaluate.owpredictions import (
    OWPredictions, SharedSelectionModel, SharedSelectionStore, DataModel,
    PredictionsModel,
    PredictionsItemDelegate, ClassificationItemDelegate, RegressionItemDelegate,
    NoopItemDelegate, RegressionErrorDelegate, ClassificationErrorDelegate,
    NO_ERR, DIFF_ERROR, ABSDIFF_ERROR, REL_ERROR, ABSREL_ERROR)
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.modelling import ConstantLearner, TreeLearner
from Orange.evaluation import Results
from Orange.widgets.tests.utils import excepthook_catch, \
    possible_duplicate_table, simulate
from Orange.widgets.utils.colorpalettes import LimitedDiscretePalette


class TestOWPredictions(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPredictions)  # type: OWPredictions
        self.iris = Table("iris")
        self.iris_classless = self.iris.transform(Domain(self.iris.domain.attributes, []))
        self.housing = Table("housing")

    def test_minimum_size(self):
        pass

    def test_rowCount_from_model(self):
        """Don't crash if the bottom row is visible"""
        self.send_signal(self.widget.Inputs.data, self.iris[:5])
        self.widget.dataview.sizeHintForColumn(0)

    def test_nan_target_input(self):
        data = self.iris[::10].copy()
        with data.unlocked():
            data.Y[1] = np.nan
        yvec = data.get_column(data.domain.class_var)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.predictors, ConstantLearner()(data), 1)
        pred = self.get_output(self.widget.Outputs.predictions)
        self.assertIsInstance(pred, Table)
        np.testing.assert_array_equal(
            yvec, pred.get_column(data.domain.class_var))

        evres = self.get_output(self.widget.Outputs.evaluation_results)
        self.assertIsInstance(evres, Results)
        self.assertIsInstance(evres.data, Table)
        ev_yvec = evres.data.get_column(data.domain.class_var)

        self.assertTrue(np.all(~np.isnan(ev_yvec)))
        self.assertTrue(np.all(~np.isnan(evres.actual)))

        with data.unlocked():
            data.Y[:] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        evres = self.get_output(self.widget.Outputs.evaluation_results)
        self.assertEqual(len(evres.data), 0)

    def test_no_values_target(self):
        train = Table("titanic")
        model = ConstantLearner()(train)
        self.send_signal(self.widget.Inputs.predictors, model)
        domain = Domain([DiscreteVariable("status", values=("first", "third")),
                         DiscreteVariable("age", values=("adult", "child")),
                         DiscreteVariable("sex", values=("female", "male"))],
                        [DiscreteVariable("survived", values=())])
        test = Table(domain, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                     np.full((3, 1), np.nan))
        self.send_signal(self.widget.Inputs.data, test)
        pred = self.get_output(self.widget.Outputs.predictions)
        self.assertEqual(len(pred), len(test))

        results = self.get_output(self.widget.Outputs.evaluation_results)

        cm_widget = self.create_widget(OWConfusionMatrix)
        self.send_signal(cm_widget.Inputs.evaluation_results, results,
                         widget=cm_widget)

        ra_widget = self.create_widget(OWROCAnalysis)
        self.send_signal(ra_widget.Inputs.evaluation_results, results,
                         widget=ra_widget)

        lc_widget = self.create_widget(OWLiftCurve)
        self.send_signal(lc_widget.Inputs.evaluation_results, results,
                         widget=lc_widget)

        cp_widget = self.create_widget(OWCalibrationPlot)
        self.send_signal(cp_widget.Inputs.evaluation_results, results,
                         widget=cp_widget)

    def test_mismatching_targets(self):
        warning = self.widget.Warning

        maj_iris = ConstantLearner()(self.iris)
        dom = self.iris.domain
        iris3 = self.iris.transform(Domain(dom[:3], dom[3]))
        maj_iris3 = ConstantLearner()(iris3)

        self.send_signal(self.widget.Inputs.predictors, maj_iris, 1)
        self.send_signal(self.widget.Inputs.predictors, maj_iris3, 2)
        self.assertFalse(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.predictors, None, 2)
        self.assertFalse(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.predictors, maj_iris3, 2)
        self.assertTrue(warning.wrong_targets.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(warning.wrong_targets.is_shown())

    def test_no_class_on_test(self):
        """Allow test data with no class"""
        titanic = Table("titanic")
        majority_titanic = ConstantLearner()(titanic)

        no_class = titanic.transform(Domain(titanic.domain.attributes, None))
        self.send_signal(self.widget.Inputs.predictors, majority_titanic, 1)
        self.send_signal(self.widget.Inputs.data, no_class)
        out = self.get_output(self.widget.Outputs.predictions)
        np.testing.assert_allclose(out.get_column("constant"), 0)

        predmodel = self.widget.predictionsview.model()
        self.assertTrue(np.isnan(
            predmodel.data(predmodel.index(0, 0), Qt.UserRole)))
        self.assertIn(predmodel.data(predmodel.index(0, 0))[0],
                      titanic.domain.class_var.values)
        self.widget.send_report()

        housing = self.housing[::5]
        mean_housing = ConstantLearner()(housing)
        no_target = housing.transform(Domain(housing.domain.attributes, None))
        self.send_signal(self.widget.Inputs.data, no_target)
        self.send_signal(self.widget.Inputs.predictors, mean_housing, 1)
        self.widget.send_report()

    def test_invalid_regression_target(self):
        widget = self.widget
        self.send_signal(widget.Inputs.predictors,
                         LinearRegressionLearner()(self.housing), 0)

        dom = self.housing.domain
        wrong_class = self.iris.transform(Domain(dom.attributes[:-1],
                                                 dom.attributes[-1]))
        self.send_signal(widget.Inputs.data, wrong_class)

        # can't make a prediction
        predmodel = self.widget.predictionsview.model()
        self.assertTrue(np.isnan(
            predmodel.data(predmodel.index(0, 0), Qt.UserRole)))
        # ... but model reports a value
        self.assertFalse(np.isnan(predmodel.data(predmodel.index(0, 0))[0]))

        no_class = self.iris.transform(Domain(dom.attributes[:-1],
                                              dom.attributes[-1]))
        self.send_signal(widget.Inputs.data, no_class)

        # can't make a prediction
        predmodel = self.widget.predictionsview.model()
        self.assertTrue(np.isnan(
            predmodel.data(predmodel.index(0, 0), Qt.UserRole)))
        # ... but model reports a value
        self.assertFalse(np.isnan(predmodel.data(predmodel.index(0, 0))[0]))

    def test_bad_data(self):
        """
        Firstly it creates predictions with TreeLearner. Then sends predictions and
        different data with different domain to Predictions widget. Those different
        data and domain are similar to original data and domain but they have three
        different target values instead of two.
        GH-2129
        """
        filestr1 = """\
        age\tsex\tsurvived
        d\td\td
        \t\tclass
        adult\tmale\tyes
        adult\tfemale\tno
        child\tmale\tyes
        child\tfemale\tyes
        """
        file1 = io.StringIO(filestr1)
        table = TabReader(file1).read()
        learner = TreeLearner()
        tree = learner(table)

        filestr2 = """\
        age\tsex\tsurvived
        d\td\td
        \t\tclass
        adult\tmale\tyes
        adult\tfemale\tno
        child\tmale\tyes
        child\tfemale\tunknown
        """
        file2 = io.StringIO(filestr2)
        bad_table = TabReader(file2).read()

        self.send_signal(self.widget.Inputs.predictors, tree, 1)

        with excepthook_catch():
            self.send_signal(self.widget.Inputs.data, bad_table)

    def test_continuous_class(self):
        data = Table("housing")
        cl_data = ConstantLearner()(data)
        self.send_signal(self.widget.Inputs.predictors, cl_data, 1)
        self.send_signal(self.widget.Inputs.data, data)

    def test_changed_class_var(self):
        def set_input(data, model):
            self.send_signals([
                (self.widget.Inputs.data, data),
                (self.widget.Inputs.predictors, model)
            ])

        iris = self.iris
        learner = ConstantLearner()
        heart_disease = Table("heart_disease")
        # catch exceptions in item delegates etc. during switching inputs
        with excepthook_catch():
            set_input(iris[:5], learner(iris))
            set_input(Table("housing"), None)
            set_input(heart_disease[:5], learner(heart_disease))

    def test_predictor_fails(self):
        titanic = Table("titanic")
        failing_model = ConstantLearner()(titanic)
        failing_model.predict = Mock(side_effect=ValueError("foo"))
        self.send_signal(self.widget.Inputs.predictors, failing_model, 1)
        self.send_signal(self.widget.Inputs.data, titanic)

        model2 = ConstantLearner()(titanic)
        self.send_signal(self.widget.Inputs.predictors, model2, 2)

    def test_sort_matching(self):
        def get_items_order(model):
            return model.mapToSourceRows(np.arange(model.rowCount()))

        w = self.widget

        titanic = Table("titanic")
        majority_titanic = LogisticRegressionLearner()(titanic)
        self.send_signal(self.widget.Inputs.predictors, majority_titanic)
        self.send_signal(self.widget.Inputs.data, titanic)

        pred_model = w.predictionsview.model()
        data_model = w.dataview.model()
        n = pred_model.rowCount()

        # no sort
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        np.testing.assert_array_equal(pred_order, np.arange(n))
        np.testing.assert_array_equal(data_order, np.arange(n))

        # sort by first column in prediction table
        pred_model.sort(0)
        w.predictionsview.horizontalHeader().sectionClicked.emit(0)
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        np.testing.assert_array_equal(pred_order, data_order)

        # sort by second column in data table
        data_model.sort(1)
        w.dataview.horizontalHeader().sectionClicked.emit(0)
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        np.testing.assert_array_equal(pred_order, data_order)

        # restore order
        w.reset_button.click()
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        np.testing.assert_array_equal(pred_order, np.arange(n))
        np.testing.assert_array_equal(data_order, np.arange(n))

    def test_sort_predictions(self):
        """
        Test whether sorting of probabilities by FilterSortProxy is correct.
        """

        def get_items_order(model):
            return model.mapToSourceRows(np.arange(model.rowCount()))

        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(self.widget.Inputs.predictors, log_reg_iris)
        self.send_signal(self.widget.Inputs.data, self.iris)
        _, log_reg_probs = log_reg_iris(self.iris, Model.ValueProbs)
        # here I assume that classes are in order 0, 1, 2 in widget

        pred_model = self.widget.predictionsview.model()
        pred_model.sort(0)
        widget_order = get_items_order(pred_model)

        # correct order first sort by probs[:,0] then probs[:,1], ...
        keys = tuple(
            log_reg_probs[:, i] for i in range(
                log_reg_probs.shape[1] - 1, -1, -1))
        sort_order = np.lexsort(keys)
        np.testing.assert_array_equal(widget_order, sort_order)

    def test_reset_no_data(self):
        """
        Check no error when resetting the view without model and data
        """
        self.widget.reset_button.click()

    def test_colors_same_domain(self):
        """
        Test whether the color selection for values is correct.
        """
        # pylint: disable=protected-access
        self.send_signal(self.widget.Inputs.data, self.iris)

        # case 1: one same model
        predictor_iris = ConstantLearner()(self.iris)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris)
        colors = self.widget._get_colors()
        np.testing.assert_array_equal(
            colors, self.iris.domain.class_var.colors)

        # case 2: two same models
        predictor_iris1 = ConstantLearner()(self.iris)
        predictor_iris2 = ConstantLearner()(self.iris)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris1)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris2, 1)
        colors = self.widget._get_colors()
        # assume that colors have same order since values have same order
        np.testing.assert_array_equal(
            colors, self.iris.domain.class_var.colors)

        # case 3: two same models - changed color order
        idom = self.iris.domain
        dom = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values)
        )
        dom.class_var.colors = dom.class_var.colors[::-1]
        iris2 = self.iris.transform(dom)

        predictor_iris1 = ConstantLearner()(iris2)
        predictor_iris2 = ConstantLearner()(iris2)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris1)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris2, 1)
        colors = self.widget._get_colors()
        np.testing.assert_array_equal(colors, iris2.domain.class_var.colors)

    def test_colors_diff_domain(self):
        """
        Test whether the color selection for values is correct.
        """
        # pylint: disable=protected-access
        self.send_signal(self.widget.Inputs.data, self.iris)

        # case 1: two domains one subset other
        idom = self.iris.domain
        dom1 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values)
        )
        dom2 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values[:2])
        )
        iris1 = self.iris[:100].transform(dom1)
        iris2 = self.iris[:100].transform(dom2)

        predictor_iris1 = ConstantLearner()(iris1)
        predictor_iris2 = ConstantLearner()(iris2)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris1)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris2, 1)
        colors = self.widget._get_colors()
        np.testing.assert_array_equal(colors, iris1.domain.class_var.colors)

        # case 2: two domains one subset other - different color order
        idom = self.iris.domain
        colors = idom.class_var.colors[::-1]
        dom1 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values)
        )
        dom2 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values[:2])
        )
        dom1.class_var.colors = colors
        dom2.class_var.colors = colors[:2]
        iris1 = self.iris[:100].transform(dom1)
        iris2 = self.iris[:100].transform(dom2)

        predictor_iris1 = ConstantLearner()(iris1)
        predictor_iris2 = ConstantLearner()(iris2)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris1)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris2, 1)
        colors = self.widget._get_colors()
        np.testing.assert_array_equal(colors, iris1.domain.class_var.colors)

        # case 3: domain color, values miss-match - use default colors
        idom = self.iris.domain
        dom1 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values)
        )
        dom2 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values)
        )
        dom1.class_var.colors = dom1.class_var.colors[::-1]
        iris1 = self.iris.transform(dom1)
        iris2 = self.iris.transform(dom2)

        predictor_iris1 = ConstantLearner()(iris1)
        predictor_iris2 = ConstantLearner()(iris2)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris1)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris2, 1)
        colors = self.widget._get_colors()
        np.testing.assert_array_equal(colors, LimitedDiscretePalette(3).palette)

        # case 4: two domains different values order, matching colors
        idom = self.iris.domain
        # this way we know that default colors are not used
        colors = LimitedDiscretePalette(5).palette[2:]
        dom1 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values)
        )
        dom2 = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values[::-1])
        )
        dom1.class_var.colors = colors
        dom2.class_var.colors = colors[::-1]  # colors mixed same than values
        iris1 = self.iris[:100].transform(dom1)
        iris2 = self.iris[:100].transform(dom2)

        predictor_iris1 = ConstantLearner()(iris1)
        predictor_iris2 = ConstantLearner()(iris2)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris1)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris2, 1)
        colors = self.widget._get_colors()
        np.testing.assert_array_equal(colors, iris1.domain.class_var.colors)

    def test_colors_continuous(self):
        """
        When only continuous variables in predictor no color should be selected
        we do not work with classes.
        When we add one classifier there should be colors.
        """
        # pylint: disable=protected-access
        data = Table("housing")
        cl_data = ConstantLearner()(data)
        self.send_signal(self.widget.Inputs.predictors, cl_data)
        self.send_signal(self.widget.Inputs.data, data)
        colors = self.widget._get_colors()
        self.assertListEqual([], colors)

        predictor_iris = ConstantLearner()(self.iris)
        self.send_signal(self.widget.Inputs.predictors, predictor_iris, 1)
        colors = self.widget._get_colors()
        self.assertEqual(3, len(colors))

        self.widget.send_report()  # just a quick check that it doesn't crash

    def test_unique_output_domain(self):
        data = possible_duplicate_table('constant')
        predictor = ConstantLearner()(data)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.predictors, predictor)

        output = self.get_output(self.widget.Outputs.predictions)
        self.assertEqual(output.domain.metas[0].name, 'constant (1)')

    def test_select(self):
        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(self.widget.Inputs.predictors, log_reg_iris)
        self.send_signal(self.widget.Inputs.data, self.iris)

        pred_model = self.widget.predictionsview.model()
        pred_model.sort(0)
        self.widget.predictionsview.selectRow(1)
        sel = {(index.row(), index.column())
               for index in self.widget.dataview.selectionModel().selectedIndexes()}
        self.assertEqual(sel, {(1, col) for col in range(5)})

    def test_select_data_first(self):
        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.predictors, log_reg_iris)

        pred_model = self.widget.predictionsview.model()
        pred_model.sort(0)
        self.widget.predictionsview.selectRow(1)
        sel = {(index.row(), index.column())
               for index in self.widget.dataview.selectionModel().selectedIndexes()}
        self.assertEqual(sel, {(1, col) for col in range(5)})

    def test_selection_in_setting(self):
        widget = self.create_widget(OWPredictions,
                                    stored_settings={"selection": [1, 3, 4]})
        self.send_signal(widget.Inputs.data, self.iris)
        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(widget.Inputs.predictors, log_reg_iris)
        sel = {(index.row(), index.column())
               for index in widget.dataview.selectionModel().selectedIndexes()}
        self.assertEqual(sel, {(row, col)
                               for row in [1, 3, 4] for col in range(5)})
        out = self.get_output(widget.Outputs.predictions)
        exp = self.iris[np.array([1, 3, 4])]
        np.testing.assert_equal(out.X, exp.X)

    def test_unregister_prediction_model(self):
        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(self.widget.Inputs.predictors, log_reg_iris)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.selection_store.unregister = Mock()
        prev_model = self.widget.predictionsview.model()
        self.send_signal(self.widget.Inputs.predictors, log_reg_iris)
        self.widget.selection_store.unregister.called_with(prev_model)

    def test_multi_inputs(self):
        w = self.widget
        data = self.iris[::5].copy()

        p1 = ConstantLearner()(data)
        p1.name = "P1"
        p2 = ConstantLearner()(data)
        p2.name = "P2"
        p3 = ConstantLearner()(data)
        p3.name = "P3"
        for i, p in enumerate([p1, p2, p3], 1):
            self.send_signal(w.Inputs.predictors, p, i)
        self.send_signal(w.Inputs.data, data)

        def check_evres(expected):
            out = self.get_output(w.Outputs.evaluation_results)
            self.assertSequenceEqual(out.learner_names, expected)
            self.assertEqual(out.folds, [...])
            self.assertEqual(out.models.shape, (1, len(out.learner_names)))
            self.assertIsInstance(out.models[0, 0], ConstantModel)

        check_evres(["P1", "P2", "P3"])

        self.send_signal(w.Inputs.predictors, None, 2)
        check_evres(["P1", "P3"])

        self.send_signal(w.Inputs.predictors, p2, 2)
        check_evres(["P1", "P2", "P3"])

        self.send_signal(w.Inputs.predictors,
                         w.Inputs.predictors.closing_sentinel, 2)
        check_evres(["P1", "P3"])

        self.send_signal(w.Inputs.predictors, p2, 2)
        check_evres(["P1", "P3", "P2"])

    def test_missing_target_cls(self):
        mask = np.zeros(len(self.iris), dtype=bool)
        mask[::2] = True
        train_data = self.iris[~mask]
        predict_data = self.iris[mask]
        model = LogisticRegressionLearner()(train_data)

        self.send_signal(self.widget.Inputs.predictors, model)
        self.send_signal(self.widget.Inputs.data, predict_data)
        self.assertFalse(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

        with predict_data.unlocked():
            predict_data.Y[0] = np.nan
        self.send_signal(self.widget.Inputs.data, predict_data)
        self.assertTrue(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

        with predict_data.unlocked():
            predict_data.Y[:] = np.nan
        self.send_signal(self.widget.Inputs.data, predict_data)
        self.assertTrue(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

        self.send_signal(self.widget.Inputs.predictors, None)
        self.assertFalse(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

    def test_missing_target_reg(self):
        mask = np.zeros(len(self.housing), dtype=bool)
        mask[::2] = True
        train_data = self.housing[~mask]
        predict_data = self.housing[mask]
        model = LinearRegressionLearner()(train_data)

        self.send_signal(self.widget.Inputs.predictors, model)
        self.send_signal(self.widget.Inputs.data, predict_data)
        self.assertFalse(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

        with predict_data.unlocked():
            predict_data.Y[0] = np.nan
        self.send_signal(self.widget.Inputs.data, predict_data)
        self.assertTrue(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

        with predict_data.unlocked():
            predict_data.Y[:] = np.nan
        self.send_signal(self.widget.Inputs.data, predict_data)
        self.assertTrue(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

        self.send_signal(self.widget.Inputs.predictors, None)
        self.assertFalse(self.widget.Warning.missing_targets.is_shown())
        self.assertFalse(self.widget.Error.scorer_failed.is_shown())

    def _mock_predictors(self):
        def pred(values):
            slot = Mock()
            slot.predictor.domain = Domain([], DiscreteVariable("c", tuple(values)))
            return slot

        def predc():
            slot = Mock()
            slot.predictor.domain = Domain([], ContinuousVariable("c"))
            return slot

        widget = self.widget
        model = Mock()
        model.setProbInd = Mock()
        widget.predictionsview.model = Mock(return_value=model)

        widget.predictors = \
            [pred(values) for values in ("abc", "ab", "cbd", "e")] + [predc()]

    def test_update_prediction_delegate_discrete(self):
        self._mock_predictors()

        widget = self.widget
        prob_combo = widget.controls.shown_probs
        set_prob_ind = widget.predictionsview.model().setProbInd
        widget._non_errored_predictors = lambda: widget.predictors[:4]

        widget.data = Table.from_list(
            Domain([], DiscreteVariable("c", values=tuple("abc"))), [])

        widget._update_control_visibility()
        self.assertFalse(prob_combo.isHidden())

        widget._set_class_values()
        self.assertEqual(widget.class_values, list("abcde"))

        widget._set_target_combos()
        self.assertEqual(
            [prob_combo.itemText(i) for i in range(prob_combo.count())],
            widget.PROB_OPTS + list("abc"))

        widget.shown_probs = widget.NO_PROBS
        widget._update_prediction_delegate()
        for delegate in widget._delegates:
            if isinstance(delegate, ClassificationItemDelegate):
                self.assertEqual(list(delegate.shown_probabilities), [])
                self.assertEqual(delegate.tooltip, "")
        set_prob_ind.assert_called_with([[], [], [], []])

        widget.shown_probs = widget.DATA_PROBS
        widget._update_prediction_delegate()
        self.assertEqual(widget._delegates[0].shown_probabilities, [0, 1, 2])
        self.assertEqual(widget._delegates[2].shown_probabilities, [0, 1, None])
        self.assertEqual(widget._delegates[2].shown_probabilities, [0, 1, None])
        self.assertEqual(widget._delegates[4].shown_probabilities, [None, 1, 2])
        self.assertEqual(widget._delegates[6].shown_probabilities, [None, None, None])
        for delegate in widget._delegates[:-1:2]:
            self.assertEqual(delegate.tooltip, "p(a, b, c)")
        set_prob_ind.assert_called_with([[0, 1, 2], [0, 1], [1, 2], []])

        widget.shown_probs = widget.MODEL_PROBS
        widget._update_prediction_delegate()
        self.assertEqual(widget._delegates[0].shown_probabilities, [0, 1, 2])
        self.assertEqual(widget._delegates[0].tooltip, "p(a, b, c)")
        self.assertEqual(widget._delegates[2].shown_probabilities, [0, 1])
        self.assertEqual(widget._delegates[2].tooltip, "p(a, b)")
        self.assertEqual(widget._delegates[4].shown_probabilities, [2, 1, 3])
        self.assertEqual(widget._delegates[4].tooltip, "p(c, b, d)")
        self.assertEqual(widget._delegates[6].shown_probabilities, [4])
        self.assertEqual(widget._delegates[6].tooltip, "p(e)")
        set_prob_ind.assert_called_with([[0, 1, 2], [0, 1], [2, 1, 3], [4]])

        widget.shown_probs = widget.BOTH_PROBS
        widget._update_prediction_delegate()
        self.assertEqual(widget._delegates[0].shown_probabilities, [0, 1, 2])
        self.assertEqual(widget._delegates[0].tooltip, "p(a, b, c)")
        self.assertEqual(widget._delegates[2].shown_probabilities, [0, 1])
        self.assertEqual(widget._delegates[2].tooltip, "p(a, b)")
        self.assertEqual(widget._delegates[4].shown_probabilities, [1, 2])
        self.assertEqual(widget._delegates[4].tooltip, "p(b, c)")
        self.assertEqual(widget._delegates[6].shown_probabilities, [])
        self.assertEqual(widget._delegates[6].tooltip, "")
        set_prob_ind.assert_called_with([[0, 1, 2], [0, 1], [1, 2], []])

        n_fixed = len(widget.PROB_OPTS)
        widget.shown_probs = n_fixed  # a
        widget._update_prediction_delegate()
        self.assertEqual(widget._delegates[0].shown_probabilities, [0])
        self.assertEqual(widget._delegates[2].shown_probabilities, [0])
        self.assertEqual(widget._delegates[4].shown_probabilities, [None])
        self.assertEqual(widget._delegates[6].shown_probabilities, [None])
        for delegate in widget._delegates[:-1:2]:
            self.assertEqual(delegate.tooltip, "p(a)")
        set_prob_ind.assert_called_with([[0], [0], [], []])

        n_fixed = len(widget.PROB_OPTS)
        widget.shown_probs = n_fixed + 1  # b
        widget._update_prediction_delegate()
        self.assertEqual(widget._delegates[0].shown_probabilities, [1])
        self.assertEqual(widget._delegates[2].shown_probabilities, [1])
        self.assertEqual(widget._delegates[4].shown_probabilities, [1])
        self.assertEqual(widget._delegates[6].shown_probabilities, [None])
        for delegate in widget._delegates[:-1:2]:
            self.assertEqual(delegate.tooltip, "p(b)")
        set_prob_ind.assert_called_with([[1], [1], [1], []])

        n_fixed = len(widget.PROB_OPTS)
        widget.shown_probs = n_fixed + 2  # c
        widget._update_prediction_delegate()
        self.assertEqual(widget._delegates[0].shown_probabilities, [2])
        self.assertEqual(widget._delegates[2].shown_probabilities, [None])
        self.assertEqual(widget._delegates[4].shown_probabilities, [2])
        self.assertEqual(widget._delegates[6].shown_probabilities, [None])
        for delegate in widget._delegates[:-1:2]:
            self.assertEqual(delegate.tooltip, "p(c)")
        set_prob_ind.assert_called_with([[2], [], [2], []])

    def test_update_delegates_continuous(self):
        self._mock_predictors()

        widget = self.widget
        widget.shown_probs = widget.DATA_PROBS

        widget.data = Table.from_list(Domain([], ContinuousVariable("c")), [])

        # only regression
        all_predictors = widget.predictors
        widget.predictors = [widget.predictors[-1]]
        widget._update_control_visibility()
        self.assertTrue(widget.controls.shown_probs.isHidden())
        self.assertTrue(widget.controls.target_class.isHidden())

        # regression and classification
        widget.predictors = all_predictors
        widget._update_control_visibility()
        self.assertFalse(widget.controls.shown_probs.isHidden())
        self.assertTrue(widget.controls.target_class.isHidden())

        widget._set_class_values()
        self.assertEqual(widget.class_values, list("abcde"))

        widget._set_target_combos()
        self.assertEqual(widget.shown_probs, widget.NO_PROBS)

        def is_enabled(prob_item):
            return widget.controls.shown_probs.model().item(prob_item).flags() & Qt.ItemIsEnabled
        self.assertTrue(is_enabled(widget.NO_PROBS))
        self.assertTrue(is_enabled(widget.MODEL_PROBS))
        self.assertFalse(is_enabled(widget.DATA_PROBS))
        self.assertFalse(is_enabled(widget.BOTH_PROBS))

    def test_delegate_ranges(self):
        widget = self.widget

        class Model1(Model):
            name = "foo"

            def predict(self, X):
                return X[:, 0] - 2

        class Model2(Model):
            name = "bar"

            def predict(self, X):
                return np.full(len(X), np.nan)

        domain = Domain([ContinuousVariable("x")], ContinuousVariable("y"))
        x = np.arange(12, 17, dtype=float)[:, None]
        y = np.array([12, 13, 14, 15, np.nan])
        data = Table(domain, x, y)

        ddomain = Domain(
            [ContinuousVariable("x")],
            DiscreteVariable("y", values=tuple("abcdefghijklmnopq")))
        self.send_signal(widget.Inputs.data, data)
        self.send_signal(widget.Inputs.predictors, Model1(domain), 1)
        self.send_signal(widget.Inputs.predictors, Model2(domain), 2)
        self.send_signal(widget.Inputs.predictors, Model1(ddomain), 3)

        delegate = widget.predictionsview.itemDelegateForColumn(0)
        # values for model are 10 to 14 (incl), Y goes from 12 to 15 (incl)
        self.assertIsInstance(delegate, RegressionItemDelegate)
        self.assertEqual(delegate.offset, 10)
        self.assertEqual(delegate.span, 5)

        delegate = widget.predictionsview.itemDelegateForColumn(2)
        # values for model are all-nan, Y goes from 12 to 15 (incl)
        self.assertIsInstance(delegate, RegressionItemDelegate)
        self.assertEqual(delegate.offset, 12)
        self.assertEqual(delegate.span, 3)

        delegate = widget.predictionsview.itemDelegateForColumn(4)
        self.assertIsInstance(delegate, ClassificationItemDelegate)

        data = Table(domain, x, np.full(5, np.nan))
        self.send_signal(widget.Inputs.data, data)
        delegate = widget.predictionsview.itemDelegateForColumn(0)
        # values for model are 10 to 14 (incl), Y is nan
        self.assertIsInstance(delegate, RegressionItemDelegate)
        self.assertEqual(delegate.offset, 10)
        self.assertEqual(delegate.span, 4)

        delegate = widget.predictionsview.itemDelegateForColumn(2)
        # values for model and y are nan
        self.assertIsInstance(delegate, RegressionItemDelegate)
        self.assertEqual(delegate.offset, 0)
        self.assertEqual(delegate.span, 1)

        delegate = widget.predictionsview.itemDelegateForColumn(4)
        self.assertIsInstance(delegate, ClassificationItemDelegate)

    class _Scorer(TargetScore):
        # pylint: disable=arguments-differ
        def compute_score(self, _, target, **__):
            return [42 if target is None else target]

    def test_output_wrt_shown_probs_1(self):
        """Data has one class less, models have same, different or one more"""
        widget = self.widget
        iris012 = self.iris
        purge = Remove(class_flags=Remove.RemoveUnusedValues)
        iris01 = purge(iris012[:100])
        iris12 = purge(iris012[50:])

        bayes01 = NaiveBayesLearner()(iris01)
        bayes12 = NaiveBayesLearner()(iris12)
        bayes012 = NaiveBayesLearner()(iris012)

        self.send_signal(widget.Inputs.data, iris01)
        self.send_signal(widget.Inputs.predictors, bayes01, 0)
        self.send_signal(widget.Inputs.predictors, bayes12, 1)
        self.send_signal(widget.Inputs.predictors, bayes012, 2)

        for i, pred in enumerate(widget.predictors):
            p = pred.results.unmapped_probabilities
            p[0] = 10 + 100 * i + np.arange(p.shape[1])
            pred.results.unmapped_predicted[:] = i

        widget.shown_probs = widget.NO_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 1, 2])

        widget.shown_probs = widget.DATA_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 1, 0, 110, 2, 210, 211])

        widget.shown_probs = widget.MODEL_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 1, 110, 111, 2, 210, 211, 212])

        widget.shown_probs = widget.BOTH_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 1, 110, 2, 210, 211])

        widget.shown_probs = widget.BOTH_PROBS + 1
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 1, 0, 2, 210])

        widget.shown_probs = widget.BOTH_PROBS + 2
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 11, 1, 110, 2, 211])

    def test_output_wrt_shown_probs_2(self):
        """One model misses one class"""
        widget = self.widget
        iris012 = self.iris
        purge = Remove(class_flags=Remove.RemoveUnusedValues)
        iris01 = purge(iris012[:100])

        bayes01 = NaiveBayesLearner()(iris01)
        bayes012 = NaiveBayesLearner()(iris012)

        self.send_signal(widget.Inputs.data, iris012)
        self.send_signal(widget.Inputs.predictors, bayes01, 0)
        self.send_signal(widget.Inputs.predictors, bayes012, 1)

        for i, pred in enumerate(widget.predictors):
            p = pred.results.unmapped_probabilities
            p[0] = 10 + 100 * i + np.arange(p.shape[1])
            pred.results.unmapped_predicted[:] = i

        widget.shown_probs = widget.NO_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 1])

        widget.shown_probs = widget.DATA_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 0, 1, 110, 111, 112])

        widget.shown_probs = widget.MODEL_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 1, 110, 111, 112])

        widget.shown_probs = widget.BOTH_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 1, 110, 111, 112])

        widget.shown_probs = widget.BOTH_PROBS + 1
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 1, 110])

        widget.shown_probs = widget.BOTH_PROBS + 2
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 11, 1, 111])

        widget.shown_probs = widget.BOTH_PROBS + 3
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 0, 1, 112])

    def test_output_regression(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.housing)
        self.send_signal(widget.Inputs.predictors,
                         LinearRegressionLearner()(self.housing), 0)
        self.send_signal(widget.Inputs.predictors,
                         MeanLearner()(self.housing), 1)
        out = self.get_output(widget.Outputs.predictions)
        np.testing.assert_equal(
            out.metas,
            np.hstack([pred.results.predicted.T for pred in widget.predictors]))

    def test_classless(self):
        widget = self.widget
        iris012 = self.iris
        purge = Remove(class_flags=Remove.RemoveUnusedValues)
        iris01 = purge(iris012[:100])
        iris12 = purge(iris012[50:])

        bayes01 = NaiveBayesLearner()(iris01)
        bayes12 = NaiveBayesLearner()(iris12)
        bayes012 = NaiveBayesLearner()(iris012)

        self.send_signal(widget.Inputs.data, self.iris_classless)
        self.send_signal(widget.Inputs.predictors, bayes01, 0)
        self.send_signal(widget.Inputs.predictors, bayes12, 1)
        self.send_signal(widget.Inputs.predictors, bayes012, 2)

        for i, pred in enumerate(widget.predictors):
            p = pred.results.unmapped_probabilities
            p[0] = 10 + 100 * i + np.arange(p.shape[1])
            pred.results.unmapped_predicted[:] = i

        widget.shown_probs = widget.NO_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 1, 2])

        widget.shown_probs = widget.MODEL_PROBS
        widget._commit_predictions()
        out = self.get_output(widget.Outputs.predictions)
        self.assertEqual(list(out.metas[0]), [0, 10, 11, 1, 110, 111, 2, 210, 211, 212])

    @patch("Orange.widgets.evaluate.owpredictions.usable_scorers",
           Mock(return_value=[_Scorer]))
    def test_change_target(self):
        widget = self.widget
        table = widget.score_table
        combo = widget.controls.target_class

        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(widget.Inputs.predictors, log_reg_iris)
        self.send_signal(widget.Inputs.data, self.iris)

        self.assertEqual(table.model.rowCount(), 1)
        self.assertEqual(table.model.columnCount(), 4)
        self.assertEqual(float(table.model.data(table.model.index(0, 3))), 42)

        for idx, value in enumerate(widget.class_var.values):
            simulate.combobox_activate_item(combo, value, Qt.DisplayRole)
            self.assertEqual(table.model.rowCount(), 1)
            self.assertEqual(table.model.columnCount(), 4)
            self.assertEqual(float(table.model.data(table.model.index(0, 3))),
                             idx)

    def test_multi_target_input(self):
        widget = self.widget

        domain = Domain([ContinuousVariable('var1')],
                        class_vars=[
                            ContinuousVariable('c1'),
                            DiscreteVariable('c2', values=('no', 'yes'))
                        ])
        data = Table.from_list(domain, [[1, 5, 0], [2, 10, 1]])

        mock_model = Mock(spec=Model, return_value=np.asarray([0.2, 0.1]))
        mock_model.name = 'Mockery'
        mock_model.domain = domain

        self.send_signal(widget.Inputs.data, data)
        self.send_signal(widget.Inputs.predictors, mock_model, 1)
        pred = self.get_output(widget.Outputs.predictions)
        self.assertIsInstance(pred, Table)

    def test_error_controls_visibility(self):
        widget = self.widget
        senddata = partial(self.send_signal, widget.Inputs.data)
        sendpredictor = partial(self.send_signal, widget.Inputs.predictors)
        clshidden = widget._cls_error_controls[0].isHidden
        reghidden = widget._reg_error_controls[0].isHidden
        colhidden = widget.predictionsview.isColumnHidden
        delegate = widget.predictionsview.itemDelegateForColumn

        iris = self.iris
        regiris = iris.transform(Domain(iris.domain.attributes[:3],
                                        iris.domain.attributes[3]))
        riris = MeanLearner()(regiris)
        ciris = MajorityLearner()(iris)

        self.assertFalse(clshidden())
        self.assertFalse(reghidden())

        senddata(self.housing)
        self.assertTrue(clshidden())
        self.assertFalse(reghidden())

        senddata(self.iris)
        self.assertFalse(clshidden())
        self.assertTrue(reghidden())

        senddata(None)
        self.assertTrue(clshidden())
        self.assertTrue(reghidden())

        senddata(self.iris_classless)
        self.assertTrue(clshidden())
        self.assertTrue(reghidden())

        sendpredictor(ciris, 0)
        sendpredictor(riris, 1)
        self.assertFalse(colhidden(0))
        self.assertTrue(colhidden(1))
        self.assertFalse(colhidden(2))
        self.assertTrue(colhidden(3))
        self.assertIsInstance(delegate(1), NoopItemDelegate)
        self.assertIsInstance(delegate(3), NoopItemDelegate)

        senddata(regiris)
        self.assertFalse(colhidden(0))
        self.assertTrue(colhidden(1))
        self.assertFalse(colhidden(2))
        self.assertFalse(colhidden(3))
        self.assertIsInstance(delegate(1), NoopItemDelegate)
        self.assertIsInstance(delegate(3), RegressionErrorDelegate)

        err_combo = self.widget.controls.show_reg_errors
        err_combo.setCurrentIndex(0)
        err_combo.activated.emit(0)
        self.assertTrue(colhidden(1))
        self.assertTrue(colhidden(3))
        self.assertIsInstance(delegate(1), NoopItemDelegate)
        self.assertIsInstance(delegate(3), (RegressionErrorDelegate,
                                            NoopItemDelegate))

        senddata(iris)
        self.assertFalse(colhidden(1))
        self.assertTrue(colhidden(3))
        self.assertIsInstance(delegate(1), ClassificationErrorDelegate)
        self.assertIsInstance(delegate(3), NoopItemDelegate)

        err_box = self.widget.controls.show_probability_errors
        err_box.click()
        self.assertTrue(colhidden(1))
        self.assertIsInstance(delegate(1), (ClassificationErrorDelegate,
                                            NoopItemDelegate))
        self.assertIsInstance(delegate(3), NoopItemDelegate)

    def test_regression_error_delegate_ranges(self):
        def set_type(tpe):
            combo = widget.controls.show_reg_errors
            combo.setCurrentIndex(tpe)
            combo.activated.emit(tpe)

        def get_delegate() -> Optional[RegressionErrorDelegate]:
            return widget.predictionsview.itemDelegateForColumn(1)

        widget = self.widget
        domain = Domain([ContinuousVariable("x")],
                        ContinuousVariable("y"))
        data = Table.from_numpy(domain, np.arange(2, 12)[:, None], np.arange(2, 12))
        model = MeanLearner()(data)
        model.mean = 5
        self.send_signal(widget.Inputs.data, data)
        self.send_signal(widget.Inputs.predictors, model, 0)

        set_type(NO_ERR)
        self.assertIsInstance(get_delegate(), NoopItemDelegate)

        set_type(DIFF_ERROR)
        delegate = get_delegate()
        self.assertEqual(delegate.span, 6)
        self.assertTrue(delegate.centered)

        set_type(ABSDIFF_ERROR)
        delegate = get_delegate()
        self.assertEqual(delegate.span, 6)
        self.assertFalse(delegate.centered)

        set_type(REL_ERROR)
        delegate = get_delegate()
        self.assertEqual(delegate.span, max(3 / 2, 6 / 11))
        self.assertTrue(delegate.centered)

        set_type(ABSREL_ERROR)
        delegate = get_delegate()
        self.assertEqual(delegate.span, max(3 / 2, 6 / 11))
        self.assertFalse(delegate.centered)

    def test_regression_error_no_model(self):
        data = self.housing[:5]
        self.send_signal(self.widget.Inputs.data, data)
        combo = self.widget.controls.show_reg_errors
        with excepthook_catch(raise_on_exit=True):
            simulate.combobox_activate_index(combo, 1)

    def test_report(self):
        widget = self.widget

        log_reg_iris = LogisticRegressionLearner()(self.iris)
        self.send_signal(widget.Inputs.predictors, log_reg_iris)
        self.send_signal(widget.Inputs.data, self.iris)

        widget.report_paragraph = Mock()
        reports = set()
        for widget.shown_probs in range(len(widget.PROB_OPTS)):
            widget.send_report()
            reports.add(widget.report_paragraph.call_args[0][1])
        self.assertEqual(len(reports), len(widget.PROB_OPTS))

        for widget.shown_probs, value in enumerate(
                widget.class_var.values, start=widget.shown_probs + 1):
            widget.send_report()
            self.assertIn(value, widget.report_paragraph.call_args[0][1])

    def test_migrate_shown_scores(self):
        settings = {"score_table": {"shown_scores": {"Sensitivity"}}}
        self.widget.migrate_settings(settings, 1)
        self.assertTrue(settings["score_table"]["show_score_hints"]["Sensitivity"])


class SelectionModelTest(unittest.TestCase):
    def setUp(self):
        iris = Table("iris")

        self.datamodel1 = DataModel(iris[:5, :2])
        self.store = SharedSelectionStore(self.datamodel1)
        self.model1 = SharedSelectionModel(self.store, self.datamodel1, None)

        self.datamodel2 = DataModel(iris[-5:, :3])
        self.model2 = SharedSelectionModel(self.store, self.datamodel2, None)

    def itsel(self, rows):
        sel = QItemSelection()
        for row in rows:
            index = self.store.model.index(row, 0)
            sel.select(index, index)
        return sel


class SharedSelectionStoreTest(SelectionModelTest):
    def test_registration(self):
        self.store.unregister(self.model1)
        self.store.unregister(self.model1)  # should have no effect and no error
        self.store.unregister(self.model2)

    def test_select_rows(self):
        emit1 = self.model1.emit_selection_rows_changed = Mock()
        emit2 = self.model2.emit_selection_rows_changed = Mock()

        store = self.store
        store.select_rows({1, 2}, QItemSelectionModel.Select)
        self.assertEqual(store.rows, {1, 2})
        emit1.assert_called_with([1, 2], [])
        emit2.assert_called_with([1, 2], [])

        store.select_rows({1, 2, 4}, QItemSelectionModel.Select)
        self.assertEqual(store.rows, {1, 2, 4})
        emit1.assert_called_with([4], [])
        emit2.assert_called_with([4], [])

        store.select_rows({3, 4}, QItemSelectionModel.Toggle)
        self.assertEqual(store.rows, {1, 2, 3})
        emit1.assert_called_with([3], [4])
        emit2.assert_called_with([3], [4])

        store.select_rows({0, 2}, QItemSelectionModel.Deselect)
        self.assertEqual(store.rows, {1, 3})
        emit1.assert_called_with([], [2])
        emit2.assert_called_with([], [2])

        store.select_rows({2, 3, 4}, QItemSelectionModel.ClearAndSelect)
        self.assertEqual(store.rows, {2, 3, 4})
        emit1.assert_called_with([2, 4], [1])
        emit2.assert_called_with([2, 4], [1])

        store.select_rows({2, 3, 4}, QItemSelectionModel.Clear)
        self.assertEqual(store.rows, set())
        emit1.assert_called_with([], [2, 3, 4])
        emit2.assert_called_with([], [2, 3, 4])

        store.select_rows({2, 3, 4}, QItemSelectionModel.ClearAndSelect)
        emit1.reset_mock()
        emit2.reset_mock()
        store.select_rows({2, 4}, QItemSelectionModel.Select)
        emit1.assert_not_called()
        emit2.assert_not_called()

    def test_select_maps_from_proxy(self):
        store = self.store
        store.select_rows = Mock()

        # Map QItemSelection
        store.model.setSortIndices(np.array([4, 0, 1, 2, 3]))
        store.select(self.itsel([1, 2]), QItemSelectionModel.Select)
        store.select_rows.assert_called_with({0, 1}, QItemSelectionModel.Select)

        # Map QModelIndex
        store.model.setSortIndices(np.array([4, 0, 1, 2, 3]))
        store.select(store.model.index(0, 0), QItemSelectionModel.Select)
        store.select_rows.assert_called_with({4}, QItemSelectionModel.Select)

        # Map empty selection
        store.select(self.itsel([]), QItemSelectionModel.Select)
        store.select_rows.assert_called_with(set(), QItemSelectionModel.Select)

    def test_clear(self):
        store = self.store

        store.select_rows({1, 2, 4}, QItemSelectionModel.Select)
        self.assertEqual(store.rows, {1, 2, 4})

        emit1 = self.model1.emit_selection_rows_changed = Mock()
        emit2 = self.model2.emit_selection_rows_changed = Mock()

        store.clear_selection()
        self.assertEqual(store.rows, set())
        emit1.assert_called_with([], [1, 2, 4])
        emit2.assert_called_with([], [1, 2, 4])

    def test_reset(self):
        store = self.store

        store.select_rows({1, 2, 4}, QItemSelectionModel.Select)
        self.assertEqual(store.rows, {1, 2, 4})

        emit1 = self.model1.emit_selection_rows_changed = Mock()
        emit2 = self.model2.emit_selection_rows_changed = Mock()

        store.reset()
        self.assertEqual(store.rows, set())
        emit1.assert_not_called()
        emit2.assert_not_called()

    def test_emit_changed_maps_to_proxy(self):
        store = self.store
        emit1 = self.model1.emit_selection_rows_changed = Mock()
        self.model2.emit_selection_rows_changed = Mock()

        def assert_called(exp_selected, exp_deselected):
            # pylint: disable=unsubscriptable-object
            selected, deselected = emit1.call_args[0]
            self.assertEqual(list(selected), exp_selected)
            self.assertEqual(list(deselected), exp_deselected)

        store.model.setSortIndices([4, 0, 1, 2, 3])
        store.select_rows({3, 4}, QItemSelectionModel.Select)
        assert_called([4, 0], [])

        store.model.setSortIndices(None)
        store.select_rows({4}, QItemSelectionModel.Deselect)
        assert_called([], [4])

        store.model.setSortIndices([1, 0, 3, 4, 2])
        store.model.setSortIndices(None)
        store.select_rows({2, 3}, QItemSelectionModel.Deselect)
        assert_called([], [3])


class SharedSelectionModelTest(SelectionModelTest):
    def test_select(self):
        self.store.select = Mock()
        sel = self.itsel({1, 2})
        self.model1.select(sel, QItemSelectionModel.Deselect)
        self.store.select.assert_called_with(sel, QItemSelectionModel.Deselect)

    def test_selection_from_rows(self):
        sel = self.model1.selection_from_rows({1, 2})
        self.assertEqual(len(sel), 2)
        ind1 = sel[0]
        self.assertIs(ind1.model(), self.model1.model())
        self.assertEqual(ind1.left(), 0)
        self.assertEqual(ind1.right(), self.model1.model().columnCount() - 1)
        self.assertEqual(ind1.top(), 1)
        self.assertEqual(ind1.bottom(), 1)

    def test_emit_selection_rows_changed(self):
        m1 = Mock()
        m2 = Mock()
        self.model1.selectionChanged.connect(m1)
        self.model2.selectionChanged.connect(m2)
        self.model1.model().setSortIndices(np.array([1, 0, 2, 4, 3]))
        self.model1.select(self.itsel({1, 3}), QItemSelectionModel.Select)
        self.assertEqual(self.store.rows, {0, 4})

        for model, m in zip((self.model1, self.model2), (m1, m2)):
            sel = m.call_args[0][0]
            self.assertEqual(len(sel), 2)
            for ind, row in zip(sel, (1, 3)):
                self.assertIs(ind.model(), model.model())
                self.assertEqual(ind.left(), 0)
                self.assertEqual(ind.right(), model.model().columnCount() - 1)
                self.assertEqual(ind.top(), row)
                self.assertEqual(ind.bottom(), row)

    def test_methods(self):
        def rowcol(sel):
            return {(index.row(), index.column()) for index in sel}

        self.model1.model().setSortIndices(np.array([1, 0, 2, 4, 3]))
        self.model2.model().setSortIndices(np.array([1, 0, 2, 4, 3]))

        self.assertFalse(self.model1.hasSelection())
        self.assertFalse(self.model1.isColumnSelected(1))
        self.assertFalse(self.model1.isRowSelected(1))
        self.assertEqual(self.model1.selectedColumns(1), [])
        self.assertEqual(self.model1.selectedRows(1), [])
        self.assertEqual(self.model1.selectedIndexes(), [])

        self.model1.select(self.itsel({1, 3}), QItemSelectionModel.Select)

        self.assertEqual(rowcol(self.model1.selection().indexes()),
                         {(1, 0), (1, 1), (3, 0), (3, 1)})
        self.assertEqual(rowcol(self.model2.selection().indexes()),
                         {(1, 0), (1, 1), (1, 2), (3, 0), (3, 1), (3, 2)})

        self.assertTrue(self.model1.hasSelection())
        self.assertFalse(self.model1.isColumnSelected(1))
        self.assertTrue(self.model1.isRowSelected(1))
        self.assertFalse(self.model1.isRowSelected(2))
        self.assertEqual(self.model1.selectedColumns(1), [])
        self.assertEqual(rowcol(self.model1.selectedRows(1)), {(1, 1), (3, 1)})
        self.assertEqual(rowcol(self.model1.selectedIndexes()),
                         {(1, 0), (1, 1), (3, 0), (3, 1)})

        self.model1.select(self.itsel({0, 1, 2, 4}), QItemSelectionModel.Select)
        self.assertTrue(self.model1.isColumnSelected(1))
        self.assertEqual(rowcol(self.model1.selectedColumns(1)),
                         {(1, 0), (1, 1)})

        self.model1.clearSelection()
        self.assertFalse(self.model1.hasSelection())
        self.assertFalse(self.model1.isColumnSelected(1))
        self.assertEqual(self.model1.selectedColumns(1), [])
        self.assertEqual(self.model1.selectedRows(1), [])
        self.assertEqual(self.model1.selectedIndexes(), [])

        self.model1.select(self.itsel({0, 1, 2, 3, 4}),
                           QItemSelectionModel.Select)
        self.model1.reset()
        self.assertFalse(self.model1.hasSelection())
        self.assertFalse(self.model1.isColumnSelected(1))
        self.assertEqual(self.model1.selectedColumns(1), [])
        self.assertEqual(self.model1.selectedRows(1), [])
        self.assertEqual(self.model1.selectedIndexes(), [])


class PredictionsModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = np.array([[0, 1, 1, 2, 0], [0, 0, 0, 1, 0]], dtype=float)
        cls.actual = np.array([0, 1, 2, 1, 0], dtype=float)
        cls.probs = [np.array([[80, 10, 10],
                               [30, 70, 0],
                               [15, 80, 5],
                               [0, 10, 90],
                               [55, 40, 5]]) / 100,
                     np.array([[80, 0, 20],
                               [90, 5, 5],
                               [70, 10, 20],
                               [10, 60, 30],
                               [50, 25, 25]]) / 100]
        cls.no_probs = [np.zeros((5, 0)), np.zeros((5, 0))]

    def test_model_classification(self):
        model = PredictionsModel(self.values, self.probs, self.actual)
        self.assertEqual(model.rowCount(), 5)
        self.assertEqual(model.columnCount(), 4)

        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 0)
        np.testing.assert_equal(prob, [0.8, 0, 0.2])

        val, prob = model.data(model.index(3, 2))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [0.1, 0.6, 0.3])

    def test_model_classification_errors(self):
        model = PredictionsModel(self.values, self.probs, self.actual)

        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            1 - np.array([80, 70, 5, 10, 55]) / 100)

        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)],
            1 - np.array([80, 5, 20, 60, 50]) / 100)

    def test_model_regression(self):
        model = PredictionsModel(self.values, self.no_probs, self.actual)
        self.assertEqual(model.rowCount(), 5)
        self.assertEqual(model.columnCount(), 4)

        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 0)
        np.testing.assert_equal(prob, [])

        val, prob = model.data(model.index(3, 2))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [])

    def test_model_regression_errors(self):
        actual = np.array([40, 0, 12, 0, -45])
        model = PredictionsModel(values=np.array([[0] * 5,
                                                  [30, 0, 12, -5, -40]]),
                                 probs=self.no_probs,
                                 actual=actual,
                                 reg_error_type=NO_ERR)

        self.assertIsNone(model.data(model.index(0, 1)))

        model.setRegErrorType(DIFF_ERROR)
        diff_error = np.array([-10, 0, 0, -5, 5])
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)],
            diff_error)
        np.testing.assert_almost_equal(model.errorColumn(1), diff_error)

        model.setRegErrorType(ABSDIFF_ERROR)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)],
            np.abs(diff_error))
        np.testing.assert_almost_equal(model.errorColumn(1), np.abs(diff_error))

        model.setRegErrorType(REL_ERROR)
        rel_error = [-10 / 40, 0, 0, -np.inf, 5 / 45]
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)], rel_error)
        np.testing.assert_almost_equal(model.errorColumn(1), rel_error)

        model.setRegErrorType(ABSREL_ERROR)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)], np.abs(rel_error))
        np.testing.assert_almost_equal(model.errorColumn(1), np.abs(rel_error))

        model.setRegErrorType(DIFF_ERROR)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)], -actual)
        np.testing.assert_almost_equal(model.errorColumn(0), -actual)

    def test_model_actual(self):
        model = PredictionsModel(self.values, self.no_probs, self.actual)
        self.assertEqual(model.data(model.index(2, 0), Qt.UserRole),
                         self.actual[2])

    def test_model_no_actual(self):
        model = PredictionsModel(self.values, self.no_probs, None)
        self.assertTrue(np.isnan(model.data(model.index(2, 0), Qt.UserRole)),
                         self.actual[2])

    def test_model_header(self):
        model = PredictionsModel(self.values, self.probs, self.actual)
        self.assertIsNone(model.headerData(0, Qt.Horizontal))
        self.assertEqual(model.headerData(3, Qt.Vertical), "4")

        model = PredictionsModel(self.values, self.probs, self.actual, ["a", "b"])
        self.assertEqual(model.headerData(0, Qt.Horizontal), "a")
        self.assertEqual(model.headerData(1, Qt.Horizontal), "error")
        self.assertEqual(model.headerData(2, Qt.Horizontal), "b")
        self.assertEqual(model.headerData(3, Qt.Horizontal), "error")
        self.assertIsNone(model.headerData(4, Qt.Horizontal))
        self.assertEqual(model.headerData(4, Qt.Vertical), "5")

        model = PredictionsModel(self.values, self.probs, self.actual, ["a"])
        self.assertEqual(model.headerData(0, Qt.Horizontal), "a")
        self.assertEqual(model.headerData(1, Qt.Horizontal), "error")
        self.assertIsNone(model.headerData(2, Qt.Horizontal))
        self.assertEqual(model.headerData(3, Qt.Vertical), "4")

    def test_model_empty(self):
        model = PredictionsModel()
        self.assertEqual(model.rowCount(), 0)
        self.assertEqual(model.columnCount(), 0)
        self.assertIsNone(model.headerData(1, Qt.Horizontal))

    def test_sorting_classification(self):
        model = PredictionsModel(self.values, self.probs, self.actual)

        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 0)
        np.testing.assert_equal(prob, [0.8, 0, 0.2])

        val, prob = model.data(model.index(3, 2))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [0.1, 0.6, 0.3])

        model.setProbInd([[2], [2]])
        model.sort(0, Qt.DescendingOrder)
        val, prob = model.data(model.index(0, 0))
        self.assertEqual(val, 2)
        np.testing.assert_equal(prob, [0, 0.1, 0.9])
        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [0.1, 0.6, 0.3])

        model.setProbInd([[2], [2]])
        model.sort(2, Qt.AscendingOrder)
        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 0)
        np.testing.assert_equal(prob, [0.9, 0.05, 0.05])
        val, prob = model.data(model.index(0, 0))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [0.3, 0.7, 0])

        model.setProbInd([[1, 0], [1, 0]])
        model.sort(0, Qt.AscendingOrder)
        np.testing.assert_equal(model.data(model.index(0, 0))[1], [0, .1, .9])
        np.testing.assert_equal(model.data(model.index(1, 0))[1], [0.8, .1, .1])

        model.setProbInd([[1, 2], [1, 2]])
        model.sort(0, Qt.AscendingOrder)
        np.testing.assert_equal(model.data(model.index(0, 0))[1], [0.8, .1, .1])
        np.testing.assert_equal(model.data(model.index(1, 0))[1], [0, .1, .9])

        model.setProbInd([[], []])
        model.sort(0, Qt.AscendingOrder)
        self.assertEqual([model.data(model.index(i, 0))[0]
                          for i in range(model.rowCount())], [0, 0, 1, 1, 2])

        model.setProbInd([[], []])
        model.sort(0, Qt.DescendingOrder)
        self.assertEqual([model.data(model.index(i, 0))[0]
                          for i in range(model.rowCount())], [2, 1, 1, 0, 0])

    def test_sorting_classification_error(self):
        model = PredictionsModel(self.values, self.probs, self.actual)

        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            1 - np.array([80, 70, 5, 10, 55]) / 100)

        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)],
            1 - np.array([80, 5, 20, 60, 50]) / 100)

        model.sort(1, Qt.AscendingOrder)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            1 - np.array(sorted([80, 70, 5, 10, 55], reverse=True)) / 100)

        model.sort(3, Qt.DescendingOrder)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 3)) for row in range(5)],
            1 - np.array(sorted([80, 5, 20, 60, 50])) / 100)

    def test_sorting_classification_different(self):
        model = PredictionsModel(self.values, self.probs, self.actual)

        model.setProbInd([[2], [0]])
        model.sort(0, Qt.DescendingOrder)
        val, prob = model.data(model.index(0, 0))
        self.assertEqual(val, 2)
        np.testing.assert_equal(prob, [0, 0.1, 0.9])
        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [0.1, 0.6, 0.3])
        model.sort(2, Qt.DescendingOrder)
        val, prob = model.data(model.index(0, 0))
        self.assertEqual(val, 1)
        np.testing.assert_equal(prob, [0.3, 0.7, 0])
        val, prob = model.data(model.index(0, 2))
        self.assertEqual(val, 0)
        np.testing.assert_equal(prob, [0.9, 0.05, 0.05])

    def test_sorting_regression(self):
        model = PredictionsModel(self.values, self.no_probs, self.actual)

        self.assertEqual(model.data(model.index(0, 2))[0], 0)
        self.assertEqual(model.data(model.index(3, 2))[0], 1)

        model.setProbInd([2])
        model.sort(0, Qt.AscendingOrder)
        self.assertEqual([model.data(model.index(i, 0))[0]
                          for i in range(model.rowCount())], [0, 0, 1, 1, 2])

        model.setProbInd([])
        model.sort(0, Qt.DescendingOrder)
        self.assertEqual([model.data(model.index(i, 0))[0]
                          for i in range(model.rowCount())], [2, 1, 1, 0, 0])

        model.setProbInd(None)
        model.sort(0, Qt.AscendingOrder)
        self.assertEqual([model.data(model.index(i, 0))[0]
                          for i in range(model.rowCount())], [0, 0, 1, 1, 2])

    def test_sorting_regression_error(self):
        actual = np.array([40, 0, 12, 0, -45])
        model = PredictionsModel(values=np.array([[30, 0, 12, -5, -40]]),
                                 probs=self.no_probs[:1],
                                 actual=actual,
                                 reg_error_type=NO_ERR)

        model.setRegErrorType(DIFF_ERROR)
        model.sort(1, Qt.AscendingOrder)
        diff_error = [-10, 0, 0, -5, 5]
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            sorted(diff_error))

        model.setRegErrorType(ABSDIFF_ERROR)
        model.sort(1, Qt.AscendingOrder)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            sorted(np.abs(diff_error)))

        model.setRegErrorType(REL_ERROR)
        rel_error = [-10 / 40, 0, 0, -np.inf, 5 / 45]
        model.sort(1, Qt.AscendingOrder)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            sorted(rel_error))

        model.setRegErrorType(ABSREL_ERROR)
        model.sort(1, Qt.AscendingOrder)
        np.testing.assert_almost_equal(
            [model.data(model.index(row, 1)) for row in range(5)],
            sorted(np.abs(rel_error)))


class TestPredictionsItemDelegate(GuiTest):
    def test_displayText(self):
        delegate = PredictionsItemDelegate()
        delegate.fmt = "{value:.3f}"
        self.assertEqual(delegate.displayText((0.12345, [1, 2, 3]), Mock()), "0.123")
        delegate.fmt = "{value:.1f}"
        self.assertEqual(delegate.displayText((0.12345, [1, 2, 3]), Mock()), "0.1")
        delegate.fmt = "{value:.1f} - {dist[2]}"
        self.assertEqual(delegate.displayText((0.12345, [1, 2, 3]), Mock()), "0.1 - 3")

        self.assertEqual(delegate.displayText(None, Mock()), "")


class TestClassificationItemDelegate(GuiTest):
    @patch.object(QToolTip, "showText")
    def test_format(self, showText):
        delegate = ClassificationItemDelegate(["foo", "bar", "baz"],
                                              [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                                              (1, None, 0), ("foo", "baz")
                                              )
        delegate.helpEvent(Mock(), Mock(), Mock(), Mock())
        self.assertEqual(showText.call_args[0][1], "p(foo, baz)")
        self.assertEqual(delegate.displayText((["baz", (0.4, 0.6)]), Mock()),
                         "0.60 : - : 0.40 → baz")
        showText.reset_mock()

        delegate = ClassificationItemDelegate(["foo", "bar", "baz"],
                                              [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                                              )
        delegate.helpEvent(Mock(), Mock(), Mock(), Mock())
        self.assertEqual(showText.call_args[0][1], "")
        self.assertEqual(delegate.displayText((["baz", (0.4, 0.6)]), Mock()),
                         "baz")

    def test_drawbar(self):
        delegate = ClassificationItemDelegate(
            ["foo", "bar", "baz", "bax"],
            [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)],
            (1, None, 0, 2), ("baz", "foo", "bax"))
        painter = Mock()
        rr = painter.drawRoundedRect
        index = Mock()
        index.data = lambda *_: 2
        rect = QRect(0, 0, 256, 16)

        delegate.cachedData = lambda *_: None
        delegate.drawBar(painter, Mock(), index, rect)
        rr.assert_not_called()

        delegate.cachedData = lambda *_: (1, (0.25, 0, 0.75, 0))
        delegate.drawBar(painter, Mock(), index, rect)
        self.assertEqual(rr.call_count, 2)
        rect = rr.call_args_list[0][0][0]
        self.assertEqual(rect.width(), 64)
        self.assertEqual(rect.height(), 8)
        rect = rr.call_args_list[1][0][0]
        self.assertEqual(rect.width(), 192)
        self.assertEqual(rect.height(), 16)


class TestNoopItemDelegate(GuiTest):
    def test_donothing(self):
        delegate = NoopItemDelegate()
        delegate.paint(Mock(), Mock(), Mock(), Mock())
        delegate.sizeHint()


class TestRegressionItemDelegate(GuiTest):
    def test_format(self):
        delegate = RegressionItemDelegate("%6.3f")
        self.assertEqual(delegate.displayText((5.13, None), Mock()), " 5.130")
        self.assertEqual(delegate.offset, 0)
        self.assertEqual(delegate.span, 1)

        delegate = RegressionItemDelegate("%6.3f", 2, 5)
        self.assertEqual(delegate.displayText((5.13, None), Mock()), " 5.130")
        self.assertEqual(delegate.offset, 2)
        self.assertEqual(delegate.span, 3)

        delegate = RegressionItemDelegate(None, 2, 5)
        self.assertEqual(delegate.displayText((5.1, None), Mock()), "5.10")
        self.assertEqual(delegate.offset, 2)
        self.assertEqual(delegate.span, 3)

    def test_drawBar(self):
        delegate = RegressionItemDelegate("%6.3f", 2, 10)
        painter = Mock()
        dr = painter.drawRect
        el = painter.drawEllipse
        index = Mock()
        rect = QRect(0, 0, 256, 16)

        ### Actual is missing
        index.data = lambda *_: np.nan

        # Prediction is missing
        delegate.cachedData = lambda *_: None
        delegate.drawBar(painter, Mock(), index, rect)

        dr.assert_not_called()
        el.assert_not_called()

        # Prediction is known
        delegate.cachedData = lambda *_: (8.0, None)
        delegate.drawBar(painter, Mock(), index, rect)

        dr.assert_called_once()
        rrect = dr.call_args[0][0]
        self.assertEqual(rrect.width(), 192)
        el.assert_not_called()
        dr.reset_mock()

        ### Actual is known
        index.data = lambda *_: 8.0

        # Prediction is correct
        delegate.cachedData = lambda *_: (8.0, None)
        delegate.drawBar(painter, Mock(), index, rect)

        dr.assert_called_once()
        rrect = dr.call_args[0][0]
        self.assertEqual(rrect.width(), 192)
        el.assert_called_once()
        center = el.call_args[0][0]
        self.assertEqual(center.x(), 192)
        dr.reset_mock()
        el.reset_mock()

        # Prediction is below
        delegate.cachedData = lambda *_: (6.0, None)
        delegate.drawBar(painter, Mock(), index, rect)

        dr.assert_called_once()
        rrect = dr.call_args[0][0]
        self.assertEqual(rrect.width(), 128)
        el.assert_called_once()
        center = el.call_args[0][0]
        self.assertEqual(center.x(), 192)
        dr.reset_mock()
        el.reset_mock()

        # Prediction is above
        delegate.cachedData = lambda *_: (9.0, None)
        delegate.drawBar(painter, Mock(), index, rect)

        dr.assert_called_once()
        rrect = dr.call_args[0][0]
        self.assertEqual(rrect.width(), 224)
        el.assert_called_once()
        center = el.call_args[0][0]
        self.assertEqual(center.x(), 192)
        dr.reset_mock()
        el.reset_mock()


class TestClassificationErrorDelegate(GuiTest):
    def test_displayText(self):
        delegate = ClassificationErrorDelegate()
        self.assertEqual(delegate.displayText(0.12345, Mock()), "0.123")
        self.assertEqual(delegate.displayText(np.nan, Mock()), "?")

    def test_drawBar(self):
        delegate = ClassificationErrorDelegate()
        painter = Mock()
        dr = painter.drawRect
        index = Mock()
        rect = QRect(0, 0, 256, 16)

        delegate.cachedData = lambda *_: np.nan
        delegate.drawBar(painter, Mock(), index, rect)
        dr.assert_not_called()

        delegate.cachedData = lambda *_: None
        delegate.drawBar(painter, Mock(), index, rect)
        dr.assert_not_called()

        delegate.cachedData = lambda *_: 1 / 4
        delegate.drawBar(painter, Mock(), index, rect)
        dr.assert_called_once()
        r = dr.call_args[0][0]
        self.assertEqual(r.x(), 0)
        self.assertEqual(r.y(), 0)
        self.assertEqual(r.width(), 64)
        self.assertEqual(r.height(), 16)


class TestRegressionErrorDelegate(GuiTest):
    def test_displayText(self):
        delegate = RegressionErrorDelegate("", True, 4)
        self.assertEqual(delegate.displayText(0.1234567, Mock()), "")

        delegate = RegressionErrorDelegate("%.5f", True, 4)
        self.assertEqual(delegate.displayText(0.1234567, Mock()), "0.12346")
        self.assertEqual(delegate.displayText(np.nan, Mock()), "?")
        self.assertEqual(delegate.displayText(np.inf, Mock()), "∞")
        self.assertEqual(delegate.displayText(-np.inf, Mock()), "-∞")

    def test_drawBar(self):
        painter = Mock()
        dr = painter.drawRect
        index = Mock()
        rect = QRect(0, 0, 256, 16)

        delegate = RegressionErrorDelegate("%.5f", True, 0)
        delegate.drawBar(painter, Mock(), index, rect)
        dr.assert_not_called()

        delegate = RegressionErrorDelegate("%.5f", True, 12)

        delegate.cachedData = lambda *_: np.nan
        delegate.drawBar(painter, Mock(), index, rect)
        dr.assert_not_called()

        delegate.cachedData = lambda *_: None
        delegate.drawBar(painter, Mock(), index, rect)
        dr.assert_not_called()

        delegate.cachedData = lambda *_: 3
        delegate.drawBar(painter, Mock(), index, rect)
        r = dr.call_args[0][0]
        self.assertEqual(r.x(), 128)
        self.assertEqual(r.y(), 0)
        self.assertEqual(r.width(), 32)
        self.assertEqual(r.height(), 16)

        delegate.cachedData = lambda *_: -3
        delegate.drawBar(painter, Mock(), index, rect)
        r = dr.call_args[0][0]
        self.assertEqual(r.x(), 128)
        self.assertEqual(r.y(), 0)
        self.assertEqual(r.width(), -32)
        self.assertEqual(r.height(), 16)

        delegate = RegressionErrorDelegate("%.5f", False, 12)
        delegate.cachedData = lambda *_: 3
        delegate.drawBar(painter, Mock(), index, rect)
        r = dr.call_args[0][0]
        self.assertEqual(r.x(), 0)
        self.assertEqual(r.y(), 0)
        self.assertEqual(r.width(), 64)
        self.assertEqual(r.height(), 16)


if __name__ == "__main__":
    unittest.main()

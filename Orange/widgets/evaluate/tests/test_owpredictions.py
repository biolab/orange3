"""Tests for OWPredictions"""
import io
from unittest.mock import Mock

import numpy as np

from Orange.base import Model
from Orange.classification import LogisticRegressionLearner
from Orange.data.io import TabReader
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owpredictions import OWPredictions
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis

from Orange.data import Table, Domain, DiscreteVariable
from Orange.modelling import ConstantLearner, TreeLearner
from Orange.evaluation import Results
from Orange.widgets.tests.utils import excepthook_catch
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator


class TestOWPredictions(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPredictions)  # type: OWPredictions
        self.iris = Table("iris")

    def test_rowCount_from_model(self):
        """Don't crash if the bottom row is visible"""
        self.send_signal(self.widget.Inputs.data, self.iris[:5])
        self.widget.dataview.sizeHintForColumn(0)

    def test_nan_target_input(self):
        data = self.iris[::10].copy()
        data.Y[1] = np.nan
        yvec, _ = data.get_column_view(data.domain.class_var)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.predictors, ConstantLearner()(data), 1)
        pred = self.get_output(self.widget.Outputs.predictions)
        self.assertIsInstance(pred, Table)
        np.testing.assert_array_equal(
            yvec, pred.get_column_view(data.domain.class_var)[0])

        evres = self.get_output(self.widget.Outputs.evaluation_results)
        self.assertIsInstance(evres, Results)
        self.assertIsInstance(evres.data, Table)
        ev_yvec, _ = evres.data.get_column_view(data.domain.class_var)

        self.assertTrue(np.all(~np.isnan(ev_yvec)))
        self.assertTrue(np.all(~np.isnan(evres.actual)))

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
        np.testing.assert_allclose(out.get_column_view("constant")[0], 0)

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
            n = pred_model.rowCount()
            return [
                pred_model.mapToSource(model.index(i, 0)).row()
                for i in range(n)
            ]

        w = self.widget

        titanic = Table("titanic")
        majority_titanic = LogisticRegressionLearner()(titanic)
        self.send_signal(self.widget.Inputs.predictors, majority_titanic)
        self.send_signal(self.widget.Inputs.data, titanic)

        pred_model = w.predictionsview.model()
        data_model = w.predictionsview.model()
        n = pred_model.rowCount()

        # no sort
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        self.assertListEqual(pred_order, list(range(n)))
        self.assertListEqual(data_order, list(range(n)))

        # sort by first column in prediction table
        pred_model.sort(0)
        w.predictionsview.horizontalHeader().sectionClicked.emit(0)
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        self.assertListEqual(pred_order, data_order)

        # sort by second column in data table
        data_model.sort(1)
        w.dataview.horizontalHeader().sectionClicked.emit(0)
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        self.assertListEqual(pred_order, data_order)

        # restore order
        w.reset_button.click()
        pred_order = get_items_order(pred_model)
        data_order = get_items_order(data_model)
        self.assertListEqual(pred_order, list(range(n)))
        self.assertListEqual(data_order, list(range(n)))

    def test_sort_predictions(self):
        """
        Test whether sorting of probabilities by FilterSortProxy is correct.
        """
        def get_items_order(model):
            n = pred_model.rowCount()
            return [
                pred_model.mapToSource(model.index(i, 0)).row()
                for i in range(n)
            ]

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
        np.testing.assert_array_equal(colors, ColorPaletteGenerator.palette(3))

        # case 4: two domains different values order, matching colors
        idom = self.iris.domain
        # this way we know that default colors are not used
        colors = ColorPaletteGenerator.palette(5)[2:]
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


if __name__ == "__main__":
    import unittest
    unittest.main()

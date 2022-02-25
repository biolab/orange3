import copy
import warnings
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import QItemSelection
from pyqtgraph import InfiniteLine

from sklearn.exceptions import ConvergenceWarning

from Orange.data import Table, DiscreteVariable, Domain, ContinuousVariable
import Orange.evaluation
import Orange.classification
from Orange.evaluation import Results
from Orange.evaluation.performance_curves import Curves

from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.tests.base import WidgetTest
from Orange.tests import test_filename


class TestOWCalibrationPlot(WidgetTest, EvaluateTest):
    def setUp(self):
        super().setUp()

        n, p = (0, 1)
        actual, probs = np.array([
            (p, .8), (n, .7), (p, .6), (p, .55), (p, .54), (n, .53), (n, .52),
            (p, .51), (n, .505), (p, .4), (n, .39), (p, .38), (n, .37),
            (n, .36), (n, .35), (p, .34), (n, .33), (p, .30), (n, .1)]).T
        self.curves = Curves(actual, probs)
        probs2 = (probs + 0.5) / 2 + 1
        self.curves2 = Curves(actual, probs2)
        pred = probs > 0.5
        pred2 = probs2 > 0.5
        probs = np.vstack((1 - probs, probs)).T
        probs2 = np.vstack((1 - probs2, probs2)).T
        domain = Domain([], DiscreteVariable("y", values=("a", "b")))
        self.results = Results(
            domain=domain,
            actual=actual,
            folds=np.array([Ellipsis]),
            models=np.array([[Mock(), Mock()]]),
            row_indices=np.arange(19),
            predicted=np.array((pred, pred2)),
            probabilities=np.array([probs, probs2]))

        self.lenses = data = Table(test_filename("datasets/lenses.tab"))
        majority = Orange.classification.MajorityLearner()
        majority.name = "majority"
        knn3 = Orange.classification.KNNLearner(n_neighbors=3)
        knn3.name = "knn-3"
        knn1 = Orange.classification.KNNLearner(n_neighbors=1)
        knn1.name = "knn-1"
        self.lenses_results = Orange.evaluation.TestOnTestData(
            store_data=True, store_models=True)(
                data=data[::2], test_data=data[1::2],
                learners=[majority, knn3, knn1])
        self.lenses_results.learner_names = ["majority", "knn-3", "knn-1"]

        self.widget = self.create_widget(OWCalibrationPlot)  # type: OWCalibrationPlot
        warnings.filterwarnings("ignore", ".*", ConvergenceWarning)

    def test_initialization(self):
        """Test initialization of lists and combos"""
        def check_clsfr_names(names):
            self.assertEqual(widget.classifier_names, names)
            clsf_list = widget.controls.selected_classifiers
            self.assertEqual(
                [clsf_list.item(i).text() for i in range(clsf_list.count())],
                names)

        widget = self.widget
        tcomb = widget.controls.target_index

        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        check_clsfr_names(["majority", "knn-3", "knn-1"])
        self.assertEqual(widget.selected_classifiers, [0, 1, 2])
        self.assertEqual(
            tuple(tcomb.itemText(i) for i in range(tcomb.count())),
            self.lenses.domain.class_var.values)
        self.assertEqual(widget.target_index, 0)

        self.send_signal(widget.Inputs.evaluation_results, self.results)
        check_clsfr_names(["#1", "#2"])
        self.assertEqual(widget.selected_classifiers, [0, 1])
        self.assertEqual(
            [tcomb.itemText(i) for i in range(tcomb.count())], ["a", "b"])
        self.assertEqual(widget.target_index, 1)

        self.send_signal(widget.Inputs.evaluation_results, None)
        check_clsfr_names([])
        self.assertEqual(widget.selected_classifiers, [])
        self.assertEqual(widget.controls.target_index.count(), 0)

    def test_empty_input_error(self):
        """Show an error when data is present but empty"""
        widget = self.widget

        res = copy.copy(self.results)
        res.row_indices = res.row_indices[:0]
        res.actual = res.actual[:0]
        res.probabilities = res.probabilities[:, :0, :]
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self.assertFalse(widget.Error.empty_input.is_shown())
        self.assertTrue(bool(widget.plot.items))

        self.send_signal(widget.Inputs.evaluation_results, res)
        self.assertTrue(widget.Error.empty_input.is_shown())
        self.assertIsNone(widget.results)
        self.assertFalse(bool(widget.plot.items))

        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self.assertFalse(widget.Error.empty_input.is_shown())
        self.assertTrue(bool(widget.plot.items))

    def test_regression_input_error(self):
        """Show an error for regression data"""
        widget = self.widget

        res = copy.copy(self.results)
        res.domain = Domain([], ContinuousVariable("y"))
        res.row_indices = res.row_indices[:0]
        res.actual = res.actual[:0]
        res.probabilities = res.probabilities[:, :0, :]
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self.assertFalse(widget.Error.non_discrete_target.is_shown())
        self.assertTrue(bool(widget.plot.items))

        self.send_signal(widget.Inputs.evaluation_results, res)
        self.assertTrue(widget.Error.non_discrete_target.is_shown())
        self.assertIsNone(widget.results)
        self.assertFalse(bool(widget.plot.items))

        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self.assertFalse(widget.Error.non_discrete_target.is_shown())
        self.assertTrue(bool(widget.plot.items))

    @staticmethod
    def _set_combo(combo, val):
        combo.setCurrentIndex(val)
        combo.activated[int].emit(val)
        combo.activated[str].emit(combo.currentText())

    @staticmethod
    def _set_radio_buttons(radios, val):
        radios.buttons[val].click()

    @staticmethod
    def _set_list_selection(listview, selection):
        model = listview.model()
        selectionmodel = listview.selectionModel()
        itemselection = QItemSelection()
        for item in selection:
            itemselection.select(model.index(item, 0), model.index(item, 0))
        selectionmodel.select(itemselection, selectionmodel.ClearAndSelect)

    def _set_threshold(self, pos, done):
        _, line = self._get_curves()
        line.setPos(pos)
        if done:
            line.sigPositionChangeFinished.emit(line)
        else:
            line.sigPositionChanged.emit(line)

    def _get_curves(self):
        plot_items = self.widget.plot.items[:]
        for i, item in enumerate(plot_items):
            if isinstance(item, InfiniteLine):
                del plot_items[i]
                return plot_items, item
        return plot_items, None

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_plotting_curves(self, *_):
        """Curve coordinates match those computed by `Curves`"""
        widget = self.widget
        widget.display_rug = False
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        widget.selected_classifiers = [0]
        combo = widget.controls.score

        c = self.curves
        combinations = ([c.ca()],
                        [c.f1()],
                        [c.sensitivity(), c.specificity()],
                        [c.precision(), c.recall()],
                        [c.ppv(), c.npv()],
                        [c.tpr(), c.fpr()])
        for idx, curves_data in enumerate(combinations, start=1):
            self._set_combo(combo, idx)
            curves, line = self._get_curves()
            self.assertEqual(len(curves), len(curves_data))
            self.assertIsNotNone(line)
            for curve in curves:
                x, y = curve.getData()
                np.testing.assert_almost_equal(x, self.curves.probs)
                for i, curve_data in enumerate(curves_data):
                    if np.max(curve_data - y) < 1e-6:
                        del curves_data[i]
                        break
                else:
                    self.fail(f"invalid curve for {combo.currentText()}")

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_multiple_fold_curves(self, *_):
        widget = self.widget
        widget.display_rug = False
        widget.fold_curves = False
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_list_selection(widget.controls.selected_classifiers, [0])
        self._set_combo(widget.controls.score, 1)  # CA

        self.results.folds = [slice(1, 5), slice(5, 19)]
        self.results.models = np.array([[Mock(), Mock()]] * 2)
        curves, _ = self._get_curves()
        self.assertEqual(len(curves), 1)

        widget.controls.fold_curves.click()
        curves, _ = self._get_curves()
        self.assertEqual(len(curves), 3)

        widget.controls.fold_curves.click()
        curves, _ = self._get_curves()
        self.assertEqual(len(curves), 1)

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_change_target_class(self, *_):
        """Changing target combo changes the curves"""
        widget = self.widget
        widget.display_rug = False
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        widget.selected_classifiers = [0]
        score_combo = widget.controls.score
        target_combo = widget.controls.target_index

        self._set_combo(score_combo, 1)  # ca
        self._set_combo(target_combo, 1)
        (ca, ), _ = self._get_curves()
        np.testing.assert_almost_equal(ca.getData()[1], self.curves.ca())

        self._set_combo(target_combo, 0)
        (ca, ), _ = self._get_curves()
        curves = Curves(1 - self.curves.ytrue, 1 - self.curves.probs[:-1])
        np.testing.assert_almost_equal(ca.getData()[1], curves.ca())

    def test_changing_score_explanation(self):
        """Changing score hides/shows explanation and options for calibration"""
        widget = self.widget
        score_combo = widget.controls.score
        explanation = widget.explanation
        calibrations = widget.controls.output_calibration

        self._set_combo(score_combo, 1)  # ca
        self.assertTrue(explanation.isHidden())
        self.assertTrue(calibrations.isHidden())

        self._set_combo(score_combo, 0)  # calibration
        self.assertTrue(explanation.isHidden())
        self.assertFalse(calibrations.isHidden())

        self._set_combo(score_combo, 3)  # sens/spec
        self.assertFalse(explanation.isHidden())
        self.assertTrue(calibrations.isHidden())

    def test_rug(self):
        """Test rug appearance and positions"""
        def get_rugs():
            rugs = [None, None]
            for item in widget.plot.items:
                if item.curve.opts.get("connect", "") == "pairs":
                    x, y = item.getData()
                    np.testing.assert_almost_equal(x[::2], x[1::2])
                    rugs[int(y[0] == 1)] = x[::2]
            return rugs

        widget = self.widget
        widget.display_rug = True
        model_list = widget.controls.selected_classifiers
        self.send_signal(widget.Inputs.evaluation_results, self.results)

        self._set_list_selection(model_list, [0])
        probs = self.curves.probs[:-1]
        truex = probs[self.curves.ytrue == 1]
        falsex = probs[self.curves.ytrue == 0]
        bottom, top = get_rugs()
        np.testing.assert_almost_equal(bottom, falsex)
        np.testing.assert_almost_equal(top, truex)

        # Switching targets should switch rugs and takes other probabilities
        self._set_combo(widget.controls.target_index, 0)
        bottom, top = get_rugs()
        np.testing.assert_almost_equal(bottom, (1 - truex)[::-1])
        np.testing.assert_almost_equal(top, (1 - falsex)[::-1])
        self._set_combo(widget.controls.target_index, 1)

        # Changing models gives a different rug
        self._set_list_selection(model_list, [1])
        probs2 = self.curves2.probs[:-1]
        truex2 = probs2[self.curves2.ytrue == 1]
        falsex2 = probs2[self.curves2.ytrue == 0]
        bottom, top = get_rugs()
        np.testing.assert_almost_equal(bottom, falsex2)
        np.testing.assert_almost_equal(top, truex2)

        # Two models - two rugs - four rug items
        self._set_list_selection(model_list, [0, 1])
        self.assertEqual(sum(item.curve.opts.get("connect", "") == "pairs"
                             for item in widget.plot.items), 4)

        # No models - no rugs
        self._set_list_selection(model_list, [])
        self.assertEqual(get_rugs(), [None, None])

        # Bring the rug back
        self._set_list_selection(model_list, [1])
        self.assertIsNotNone(get_rugs()[0])

        # Disable it with checkbox
        widget.controls.display_rug.click()
        self.assertEqual(get_rugs(), [None, None])

    def test_calibration_curve(self):
        """Test the correct number of calibration curves"""
        widget = self.widget
        model_list = widget.controls.selected_classifiers
        widget.display_rug = False

        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self.assertEqual(len(widget.plot.items), 3)  # 2 + diagonal

        self._set_list_selection(model_list, [1])
        self.assertEqual(len(widget.plot.items), 2)

        self._set_list_selection(model_list, [])
        self.assertEqual(len(widget.plot.items), 1)

    def test_threshold_change_updates_info(self):
        """Changing the threshold updates info label"""
        widget = self.widget
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_combo(widget.controls.score, 1)

        original_text = widget.info_label.text()
        self._set_threshold(0.3, False)
        self.assertNotEqual(widget.info_label.text(), original_text)

    def test_threshold_rounding(self):
        """Threshold is rounded to two decimals"""
        widget = self.widget
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_combo(widget.controls.score, 1)
        self._set_threshold(0.367, False)
        self.assertAlmostEqual(widget.threshold, 0.37)

    def test_threshold_flips_on_two_classes(self):
        """Threshold changes to 1 - threshold if *binary* class is switched"""
        widget = self.widget
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_combo(widget.controls.target_index, 0)
        self._set_combo(widget.controls.score, 1) # CA
        self._set_threshold(0.25, False)
        self.assertEqual(widget.threshold, 0.25)
        self._set_combo(widget.controls.target_index, 1)
        self.assertEqual(widget.threshold, 0.75)

        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        self._set_combo(widget.controls.target_index, 0)
        self._set_combo(widget.controls.score, 1) # CA
        self._set_threshold(0.25, False)
        self.assertEqual(widget.threshold, 0.25)
        self._set_combo(widget.controls.target_index, 1)
        self.assertEqual(widget.threshold, 0.25)


    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_apply_no_output(self, *_):
        """Test no output warnings"""
        widget = self.widget
        model_list = widget.controls.selected_classifiers

        multiple_folds, multiple_selected, no_models, non_binary_class = "abcd"
        messages = {
            multiple_folds:
                "each training data sample produces a different model",
            no_models:
                "test results do not contain stored models - try testing on "
                "separate data or on training data",
            multiple_selected:
                "select a single model - the widget can output only one",
            non_binary_class:
                "cannot calibrate non-binary classes"}

        def test_shown(shown):
            widget_msg = widget.Information.no_output
            output = self.get_output(widget.Outputs.calibrated_model)
            if not shown:
                self.assertFalse(widget_msg.is_shown())
                self.assertIsNotNone(output)
            else:
                self.assertTrue(widget_msg.is_shown())
                self.assertIsNone(output)
                for msg_id in shown:
                    msg = messages[msg_id]
                    self.assertIn(msg, widget_msg.formatted,
                                  f"{msg} not included in the message")

        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_combo(widget.controls.score, 1)  # CA
        test_shown({multiple_selected})

        self._set_list_selection(model_list, [0])
        test_shown(())
        self._set_list_selection(model_list, [0, 1])

        self.results.models = None
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        test_shown({multiple_selected, no_models})

        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        test_shown({multiple_selected, non_binary_class})

        self._set_list_selection(model_list, [0])
        test_shown({non_binary_class})

        self.results.folds = [slice(0, 5), slice(5, 10), slice(10, 19)]
        self.results.models = np.array([[Mock(), Mock()]] * 3)

        self.send_signal(widget.Inputs.evaluation_results, self.results)
        test_shown({multiple_selected, multiple_folds})

        self._set_list_selection(model_list, [0])
        test_shown({multiple_folds})

        self._set_combo(widget.controls.score, 0)  # calibration
        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        self._set_list_selection(model_list, [0, 1])
        test_shown({multiple_selected})
        self._set_list_selection(model_list, [0])
        test_shown(())

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    def test_output_threshold_classifier(self, threshold_classifier):
        """Test threshold classifier on output"""
        widget = self.widget
        model_list = widget.controls.selected_classifiers
        models = self.results.models.ravel()
        target_combo = widget.controls.target_index
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_list_selection(model_list, [0])
        widget.target_index = 1

        widget.threshold = 0.3
        self._set_combo(widget.controls.score, 1)  # CA
        model = self.get_output(widget.Outputs.calibrated_model)
        threshold_classifier.assert_called_with(models[0], 0.3)
        self.assertIs(model, threshold_classifier.return_value)
        threshold_classifier.reset_mock()

        widget.auto_commit = True
        self._set_threshold(0.4, False)
        threshold_classifier.assert_not_called()

        widget.auto_commit = False
        self._set_threshold(0.35, True)
        threshold_classifier.assert_not_called()

        widget.auto_commit = True
        self._set_threshold(0.4, True)
        threshold_classifier.assert_called_with(models[0], 0.4)
        self.assertIs(model, threshold_classifier.return_value)
        threshold_classifier.reset_mock()

        self._set_combo(target_combo, 0)
        threshold_classifier.assert_called_with(models[0], 0.4)
        self.assertIs(model, threshold_classifier.return_value)
        threshold_classifier.reset_mock()

        self._set_combo(target_combo, 1)
        threshold_classifier.assert_called_with(models[0], 0.4)
        self.assertIs(model, threshold_classifier.return_value)
        threshold_classifier.reset_mock()

        self._set_list_selection(model_list, [1])
        threshold_classifier.assert_called_with(models[1], 0.4)
        self.assertIs(model, threshold_classifier.return_value)
        threshold_classifier.reset_mock()

    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_output_calibrated_classifier(self, calibrated_learner):
        """Test calibrated classifier on output"""
        calibrated_instance = calibrated_learner.return_value
        get_model = calibrated_instance.get_model

        widget = self.widget
        model_list = widget.controls.selected_classifiers
        models = self.lenses_results.models.ravel()
        results = self.lenses_results
        self.send_signal(widget.Inputs.evaluation_results, results)
        self._set_combo(widget.controls.score, 0)

        self._set_list_selection(model_list, [1])

        self._set_radio_buttons(widget.controls.output_calibration, 0)
        calibrated_learner.assert_called_with(None, 0)
        model, actual, probabilities = get_model.call_args[0]
        self.assertIs(model, models[1])
        np.testing.assert_equal(actual, results.actual)
        np.testing.assert_equal(probabilities, results.probabilities[1])
        self.assertIs(self.get_output(widget.Outputs.calibrated_model),
                      get_model.return_value)
        calibrated_learner.reset_mock()
        get_model.reset_mock()

        self._set_radio_buttons(widget.controls.output_calibration, 1)
        calibrated_learner.assert_called_with(None, 1)
        model, actual, probabilities = get_model.call_args[0]
        self.assertIs(model, models[1])
        np.testing.assert_equal(actual, results.actual)
        np.testing.assert_equal(probabilities, results.probabilities[1])
        self.assertIs(self.get_output(widget.Outputs.calibrated_model),
                      get_model.return_value)
        calibrated_learner.reset_mock()
        get_model.reset_mock()

        self._set_list_selection(model_list, [0])
        self._set_radio_buttons(widget.controls.output_calibration, 1)
        calibrated_learner.assert_called_with(None, 1)
        model, actual, probabilities = get_model.call_args[0]
        self.assertIs(model, models[0])
        np.testing.assert_equal(actual, results.actual)
        np.testing.assert_equal(probabilities, results.probabilities[0])
        self.assertIs(self.get_output(widget.Outputs.calibrated_model),
                      get_model.return_value)
        calibrated_learner.reset_mock()
        get_model.reset_mock()

    def test_contexts(self):
        """Test storing and retrieving context settings"""
        widget = self.widget
        model_list = widget.controls.selected_classifiers
        target_combo = widget.controls.target_index
        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        self._set_list_selection(model_list, [0, 2])
        self._set_combo(target_combo, 2)
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_list_selection(model_list, [0])
        self._set_combo(target_combo, 0)
        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        self.assertEqual(widget.selected_classifiers, [0, 2])
        self.assertEqual(widget.target_index, 2)

    def test_report(self):
        """Test that report does not crash"""
        widget = self.widget
        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        widget.send_report()

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_single_class(self, *_):
        """Curves are not plotted if all data belongs to (non)-target"""
        def check_error(shown):
            for error in (errors.no_target_class, errors.all_target_class,
                          errors.nan_classes):
                self.assertEqual(error.is_shown(), error is shown,
                                 f"{error} is unexpectedly"
                                 f"{'' if error.is_shown() else ' not'} shown")
            if shown is not None:
                self.assertEqual(len(widget.plot.items), 0)
            else:
                self.assertGreater(len(widget.plot.items), 0)

        widget = self.widget
        errors = widget.Error
        widget.display_rug = True
        combo = widget.controls.score

        original_actual = self.results.actual.copy()
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        widget.selected_classifiers = [0]
        for idx in range(combo.count()):
            self._set_combo(combo, idx)
            self.results.actual[:] = 0
            self.send_signal(widget.Inputs.evaluation_results, self.results)
            check_error(errors.no_target_class)

            self.results.actual[:] = 1
            self.send_signal(widget.Inputs.evaluation_results, self.results)
            check_error(errors.all_target_class)

            self.results.actual[:] = original_actual
            self.results.actual[3] = np.nan
            self.send_signal(widget.Inputs.evaluation_results, self.results)
            check_error(errors.nan_classes)

            self.results.actual[:] = original_actual
            self.send_signal(widget.Inputs.evaluation_results, self.results)
            check_error(None)

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_single_class_folds(self, *_):
        """Curves for single-class folds are not plotted"""
        widget = self.widget
        widget.display_rug = False
        widget.fold_curves = False

        results = self.lenses_results
        results.folds = [slice(0, 5), slice(5, 19)]
        results.models = results.models.repeat(2, axis=0)
        results.actual = results.actual.copy()
        results.actual[:3] = 0
        results.probabilities[1, 3:5] = np.nan
        # after this, model 1 has just negative instances in fold 0
        self.send_signal(widget.Inputs.evaluation_results, results)
        self._set_combo(widget.controls.score, 1)  # CA
        self.assertFalse(widget.Warning.omitted_folds.is_shown())
        widget.controls.fold_curves.click()
        self.assertTrue(widget.Warning.omitted_folds.is_shown())

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_warn_nan_probabilities(self, *_):
        """Warn about omitted points with nan probabiities"""
        widget = self.widget
        widget.display_rug = False
        widget.fold_curves = False

        self.results.probabilities[1, 3] = np.nan
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self.assertTrue(widget.Warning.omitted_nan_prob_points.is_shown())
        self._set_list_selection(widget.controls.selected_classifiers, [0, 2])
        self.assertFalse(widget.Warning.omitted_folds.is_shown())

    @patch("Orange.widgets.evaluate.owcalibrationplot.ThresholdClassifier")
    @patch("Orange.widgets.evaluate.owcalibrationplot.CalibratedLearner")
    def test_no_folds(self, *_):
        """Don't crash on malformed Results with folds=None"""
        widget = self.widget

        self.results.folds = None
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        widget.selected_classifiers = [0]
        widget.commit.now()
        self.assertIsNotNone(self.get_output(widget.Outputs.calibrated_model))


if __name__ == "__main__":
    unittest.main()

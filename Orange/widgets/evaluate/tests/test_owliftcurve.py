# pylint: disable=protected-access
import copy
import unittest
from unittest.mock import Mock

import numpy as np

from AnyQt.QtGui import QFont, QPen

from Orange.classification import ThresholdClassifier
from Orange.data import Table
import Orange.evaluation
import Orange.classification

from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve, cumulative_gains, \
    cumulative_gains_from_results, CurveTypes, precision_recall_from_results, \
    points_from_results
from Orange.tests import test_filename


class TestOWLiftCurve(WidgetTest, EvaluateTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lenses = data = Table(test_filename("datasets/lenses.tab"))
        test_on_test = Orange.evaluation.TestOnTestData(store_data=True)
        cls.res = test_on_test(
            data=data[::2], test_data=data[1::2],
            learners=[Orange.classification.MajorityLearner(),
                      Orange.classification.KNNLearner()]
        )

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(
            OWLiftCurve,
            stored_settings={
                "display_convex_hull": True
            }
        )  # type: OWLiftCurve

    def test_basic(self):
        self.send_signal(self.widget.Inputs.evaluation_results, self.res)
        simulate.combobox_run_through_all(self.widget.target_cb)

    def test_empty_input(self):
        res = copy.copy(self.res)
        res.actual = res.actual[:0]
        res.row_indices = res.row_indices[:0]
        res.predicted = res.predicted[:, :0]
        res.probabilities = res.probabilities[:, :0, :]
        self.send_signal(self.widget.Inputs.evaluation_results, res)

    def test_nan_input(self):
        res = copy.copy(self.res)
        res.actual = res.actual.copy()
        res.actual[0] = np.nan
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.assertTrue(self.widget.Error.invalid_results.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.assertFalse(self.widget.Error.invalid_results.is_shown())

    def test_precision_recall(self):
        self.send_signal(self.widget.Inputs.evaluation_results, self.res)
        radio_buttons = self.widget.controls.curve_type.buttons
        radio_buttons[CurveTypes.PrecisionRecall].click()
        self.assertEqual(self.widget.curve_type, CurveTypes.PrecisionRecall)

    def test_disable_convex_hull(self):
        self.send_signal(self.widget.Inputs.evaluation_results, self.res)
        self.assertTrue(self.widget.controls.display_convex_hull.isEnabled())
        radio_buttons = self.widget.controls.curve_type.buttons
        radio_buttons[CurveTypes.PrecisionRecall].click()
        self.assertFalse(self.widget.controls.display_convex_hull.isEnabled())
        radio_buttons = self.widget.controls.curve_type.buttons
        radio_buttons[CurveTypes.LiftCurve].click()
        self.assertTrue(self.widget.controls.display_convex_hull.isEnabled())

    def test_get_threshold(self):
        recall = np.array([1, 2 / 3, 2 / 3, 1 / 3, 0])
        thresholds = np.array([0.4, 0.5, 0.6, 0.9, 1])

        self.widget.rate = 1
        threshold = self.widget._get_threshold(recall, thresholds)
        self.assertEqual(threshold, 0.4)

        self.widget.rate = 0.7
        threshold = self.widget._get_threshold(recall, thresholds)
        self.assertEqual(threshold, 0.4)

        self.widget.rate = 0.5
        threshold = self.widget._get_threshold(recall, thresholds)
        self.assertEqual(threshold, 0.6)

        self.widget.rate = 0.3
        threshold = self.widget._get_threshold(recall, thresholds)
        self.assertEqual(threshold, 0.9)

        self.widget.rate = 0
        threshold = self.widget._get_threshold(recall, thresholds)
        self.assertEqual(threshold, 1)

    def test_threshold_tooltip(self):
        data = Table("heart_disease")
        test_on_test = Orange.evaluation.TestOnTestData(
            store_data=True, store_models=True)
        res = test_on_test(data=data[::2], test_data=data[1::2],
                           learners=[Orange.classification.MajorityLearner(),
                                     Orange.classification.KNNLearner()])
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.assertEqual(self.widget.tooltip.toPlainText(),
                         "Thresholds:\n— 0.526\n— 0.4")

    def test_output(self):
        data = Table("heart_disease")
        test_on_test = Orange.evaluation.TestOnTestData(
            store_data=True, store_models=True)
        res = test_on_test(data=data[::2], test_data=data[1::2],
                           learners=[Orange.classification.MajorityLearner(),
                                     Orange.classification.KNNLearner()])
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        model = self.get_output(self.widget.Outputs.calibrated_model)
        self.assertIsNone(model)
        self.assertTrue(self.widget.Information.no_output.is_shown())

        self.widget.selected_classifiers = [1]
        self.widget._on_classifiers_changed()
        model = self.get_output(self.widget.Outputs.calibrated_model)
        self.assertIsInstance(model, ThresholdClassifier)
        self.assertEqual(model.threshold, 0.6)
        self.assertFalse(self.widget.Information.no_output.is_shown())

    def test_visual_settings(self):
        graph = self.widget.plot

        def test_settings():
            font = QFont("Helvetica", italic=True, pointSize=20)
            self.assertFontEqual(
                graph.parameter_setter.title_item.item.font(), font
            )

            font.setPointSize(16)
            for item in graph.parameter_setter.axis_items:
                self.assertFontEqual(item.label.font(), font)

            font.setPointSize(15)
            for item in graph.parameter_setter.axis_items:
                self.assertFontEqual(item.style["tickFont"], font)

            self.assertEqual(
                graph.parameter_setter.title_item.item.toPlainText(), "Foo"
            )
            self.assertEqual(graph.parameter_setter.title_item.text, "Foo")

            for line in graph.curve_items:
                pen: QPen = line.opts["pen"]
                self.assertEqual(pen.width(), 3)

            for line in graph.hull_items:
                pen: QPen = line.opts["pen"]
                self.assertEqual(pen.width(), 10)

            pen: QPen = graph.default_line_item.opts["pen"]
            self.assertEqual(pen.width(), 4)

        test_on_test = Orange.evaluation.TestOnTestData(store_data=True)
        res = test_on_test(
            data=self.lenses[::2], test_data=self.lenses[1::2],
            learners=[Orange.classification.MajorityLearner(),
                      Orange.classification.KNNLearner()]
        )
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis title", "Font size"), 16
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis ticks", "Font size"), 15
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis ticks", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Wide line", "Width"), 10
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Thin Line", "Width"), 3
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Default Line", "Width"), 4
        self.widget.set_visual_settings(key, value)

        self.send_signal(self.widget.Inputs.evaluation_results, res)
        test_settings()

        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        test_settings()

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())


class UtilsTest(unittest.TestCase):
    @staticmethod
    def test_cumulative_gains():
        shuffle = [1, 2, 0, 3, 5, 4]
        y_true = np.array([1, 1, 0, 0, 1, 0])[shuffle]
        y_scores = np.array([0.9, 0.6, 0.5, 0.4, 0.4, 0.2])[shuffle]

        assert_almost_equal = np.testing.assert_almost_equal

        contacted, respondents, thresholds = cumulative_gains(y_true, y_scores)
        assert_almost_equal(contacted, np.array([1, 2, 3, 5, 6]) / 6)
        assert_almost_equal(thresholds, [0.9, 0.6, 0.5, 0.4, 0.2])
        assert_almost_equal(respondents, np.array([1, 2, 2, 3, 3]) / 3)

        contacted, respondents, thresholds = cumulative_gains(y_true, 1 - y_scores, target=0)
        assert_almost_equal(contacted, np.array([1, 3, 4, 5, 6]) / 6)
        assert_almost_equal(thresholds, [0.8, 0.6, 0.5, 0.4, 0.1])
        assert_almost_equal(respondents, np.array([1, 2, 3, 3, 3]) / 3)

        contacted, respondents, thresholds = \
            cumulative_gains(np.array([], dtype=int), np.array([]))
        assert_almost_equal(contacted, [])
        assert_almost_equal(respondents, [])
        assert_almost_equal(thresholds, [])

    @staticmethod
    def test_cumulative_gains_from_results():
        shuffle = [1, 2, 0, 3, 5, 4]
        y_true = np.array([1, 1, 0, 0, 1, 0])[shuffle]
        y_scores = np.array([0.9, 0.6, 0.5, 0.4, 0.4, 0.2])[shuffle]

        results = Mock()
        results.actual = y_true
        results.probabilities = \
            [Mock(), Mock(), np.vstack((1 - y_scores, y_scores)).T]

        assert_almost_equal = np.testing.assert_almost_equal

        contacted, respondents, thresholds = \
            cumulative_gains_from_results(results, 1, 2)
        assert_almost_equal(thresholds, [0.9, 0.6, 0.5, 0.4, 0.2])
        assert_almost_equal(contacted, np.array([1, 2, 3, 5, 6]) / 6)
        assert_almost_equal(respondents, np.array([1, 2, 2, 3, 3]) / 3)

        contacted, respondents, thresholds = \
            cumulative_gains_from_results(results, 0, 2)
        assert_almost_equal(contacted, np.array([1, 3, 4, 5, 6]) / 6)
        assert_almost_equal(thresholds, [0.8, 0.6, 0.5, 0.4, 0.1])
        assert_almost_equal(respondents, np.array([1, 2, 3, 3, 3]) / 3)

        results.actual = np.array([], dtype=int)
        results.probabilities = np.empty((3, 0, 2))
        contacted, respondents, thresholds = \
            cumulative_gains(np.array([], dtype=int), np.array([]))
        assert_almost_equal(contacted, [])
        assert_almost_equal(respondents, [])
        assert_almost_equal(thresholds, [])

    @staticmethod
    def test_precision_recall_from_results():
        y_true = np.array([1, 0, 1, 0, 0, 1])
        y_scores = np.array([0.6, 0.5, 0.9, 0.4, 0.2, 0.4])

        results = Mock()
        results.actual = y_true
        results.probabilities = \
            [Mock(), Mock(), np.vstack((1 - y_scores, y_scores)).T]

        recall, precision, thresholds = \
            precision_recall_from_results(results, 1, 2)
        np.testing.assert_equal(precision, np.array([3 / 5, 2 / 3, 1, 1, 1]))
        np.testing.assert_equal(recall, np.array([1, 2 / 3, 2 / 3, 1 / 3, 0]))
        np.testing.assert_equal(thresholds, np.array([0.4, 0.5, 0.6, 0.9, 1]))

    @staticmethod
    def test_precision_recall_from_results_multiclass():
        y_true = np.array([1, 0, 1, 0, 2, 2])
        y_scores = np.array([[0.3, 0.3, 0.4],
                             [0.3, 0.4, 0.4],
                             [0.1, 0.9, 0.1],
                             [0.4, 0.2, 0.4],
                             [0.1, 0.2, 0.7],
                             [0.1, 0.1, 0.8]])

        results = Mock()
        results.actual = y_true
        results.probabilities = [Mock(), Mock(), y_scores]

        recall, precision, thresholds = \
            precision_recall_from_results(results, 1, 2)
        np.testing.assert_equal(precision, np.array([2 / 3, 1 / 2, 1, 1]))
        np.testing.assert_equal(recall, np.array([1, 1 / 2, 1 / 2, 0]))
        np.testing.assert_equal(thresholds, np.array([0.3, 0.4, 0.9, 1]))

    @staticmethod
    def test_points_from_results_cumulative_gain():
        y_scores = np.array([0.6, 0.5, 0.9, 0.4, 0.2, 0.4])
        results = Mock()
        results.actual = np.array([1, 0, 1, 0, 0, 1])
        results.probabilities = \
            [Mock(), Mock(), np.vstack((1 - y_scores, y_scores)).T]

        contacted, respondents, thresholds = \
            cumulative_gains_from_results(results, 1, 2)
        res = points_from_results(results, 1, 2, CurveTypes.CumulativeGains)
        np.testing.assert_almost_equal(res.points.contacted, contacted)
        np.testing.assert_almost_equal(res.points.respondents, respondents)
        np.testing.assert_almost_equal(res.points.thresholds, thresholds)

    @staticmethod
    def test_points_from_results_precision_recall():
        y_scores = np.array([0.6, 0.5, 0.9, 0.4, 0.2, 0.4])
        results = Mock()
        results.actual = np.array([1, 0, 1, 0, 0, 1])
        results.probabilities = \
            [Mock(), Mock(), np.vstack((1 - y_scores, y_scores)).T]

        contacted, respondents, thresholds = \
            precision_recall_from_results(results, 1, 2)
        res = points_from_results(results, 1, 2, CurveTypes.PrecisionRecall)
        np.testing.assert_almost_equal(res.points.contacted, contacted)
        np.testing.assert_almost_equal(res.points.respondents, respondents)
        np.testing.assert_almost_equal(res.points.thresholds, thresholds)


if __name__ == "__main__":
    unittest.main()

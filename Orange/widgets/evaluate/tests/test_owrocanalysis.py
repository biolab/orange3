# pylint: disable=protected-access

import unittest
import copy
import numpy as np

from AnyQt.QtWidgets import QToolTip

import Orange.data
import Orange.evaluation
import Orange.classification

from Orange.widgets.evaluate import owrocanalysis
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import mouseMove


class TestROC(unittest.TestCase):
    def test_ROCData_from_results(self):
        data = Orange.data.Table("iris")
        learners = [
            Orange.classification.MajorityLearner(),
            Orange.classification.LogisticRegressionLearner(),
            Orange.classification.TreeLearner()
        ]
        res = Orange.evaluation.CrossValidation(data, learners, k=10)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.ROCData_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 10)
                self.assertTrue(all(c.is_valid for c in rocdata.folds))
                self.assertTrue(rocdata.avg_vertical.is_valid)
                self.assertTrue(rocdata.avg_threshold.is_valid)

        # fixed random seed because otherwise it could happen that data sample
        # contained only instances of two classes (and the test then fails)
        data = data[np.random.RandomState(0).choice(len(data), size=20)]
        res = Orange.evaluation.LeaveOneOut(data, learners)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.ROCData_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                # all individual fold curves and averaged curve data
                # should be invalid
                self.assertTrue(all(not c.is_valid for c in rocdata.folds))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)

        # equivalent test to the LeaveOneOut but from a slightly different
        # constructed Orange.evaluation.Results
        res = Orange.evaluation.CrossValidation(data, learners, k=20)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.ROCData_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                # all individual fold curves and averaged curve data
                # should be invalid
                self.assertTrue(all(not c.is_valid for c in rocdata.folds))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)


class TestOWROCAnalysis(WidgetTest, EvaluateTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lenses = data = Orange.data.Table("lenses")
        cls.res = Orange.evaluation.TestOnTestData(
            train_data=data[::2], test_data=data[1::2],
            learners=[Orange.classification.MajorityLearner(),
                      Orange.classification.KNNLearner()],
            store_data=True,
        )

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(
            OWROCAnalysis,
            stored_settings={
                "display_perf_line": True,
                "display_def_threshold": True,
                "display_convex_hull": True,
                "display_convex_curve": True
            }
        )  # type: OWROCAnalysis

    def tearDown(self):
        super().tearDown()
        self.widget.onDeleteWidget()
        self.widgets.remove(self.widget)
        self.widget = None

    def test_basic(self):
        res = self.res
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()
        self.send_signal(self.widget.Inputs.evaluation_results, None)

    def test_empty_input(self):
        res = Orange.evaluation.Results(
            data=self.lenses[:0], nmethods=2, store_data=True)
        res.row_indices = np.array([], dtype=int)
        res.actual = np.array([])
        res.predicted = np.zeros((2, 0))
        res.probabilities = np.zeros((2, 0, 3))

        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()

        res.row_indices = np.array([1], dtype=int)
        res.actual = np.array([0.0])
        res.predicted = np.zeros((2, 1))
        res.probabilities = np.zeros((2, 1, 3))

        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()

    def test_nan_input(self):
        res = copy.copy(self.res)
        res.actual = res.actual.copy()
        res.predicted = res.predicted.copy()
        res.probabilities = res.probabilities.copy()

        res.actual[0] = np.nan
        res.predicted[:, 1] = np.nan
        res.probabilities[0, 1, :] = np.nan

        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.assertTrue(self.widget.Error.invalid_results.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.assertFalse(self.widget.Error.invalid_results.is_shown())

    def test_tooltips(self):
        data_in = Orange.data.Table("titanic")
        res = Orange.evaluation.TestOnTrainingData(
            data=data_in,
            learners=[Orange.classification.KNNLearner(),
                      Orange.classification.LogisticRegressionLearner()],
            store_data=True
        )

        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget.target_index = 0
        self.widget.selected_classifiers = [0, 1]
        vb = self.widget.plot.getViewBox()
        vb.childTransform()  # Force pyqtgraph to update transforms

        curve = self.widget.plot_curves(self.widget.target_index, 0)
        curve_merge = curve.merge()
        view = self.widget.plotview
        item = curve_merge.curve_item  # type: pg.PlotCurveItem

        # no tooltips to be shown
        pos = item.mapToScene(0.0, 1.0)
        pos = view.mapFromScene(pos)
        mouseMove(view.viewport(), pos)
        self.assertIs(self.widget._tooltip_cache, None)

        # test single point
        pos = item.mapToScene(0.22504, 0.45400)
        pos = view.mapFromScene(pos)
        mouseMove(view.viewport(), pos)
        shown_thresh = self.widget._tooltip_cache[1]
        self.assertTrue(QToolTip.isVisible())
        np.testing.assert_almost_equal(shown_thresh, [0.40000], decimal=5)

        pos = item.mapToScene(0.0, 0.0)
        pos = view.mapFromScene(pos)
        # test overlapping points
        mouseMove(view.viewport(), pos)
        shown_thresh = self.widget._tooltip_cache[1]
        self.assertTrue(QToolTip.isVisible())
        np.testing.assert_almost_equal(shown_thresh, [1.8, 1.89336], decimal=5)

        # test that cache is invalidated when changing averaging mode
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        mouseMove(view.viewport(), pos)
        shown_thresh = self.widget._tooltip_cache[1]
        self.assertTrue(QToolTip.isVisible())
        np.testing.assert_almost_equal(shown_thresh, [1, 1])

        # test nan thresholds
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        mouseMove(view.viewport(), pos)
        self.assertIs(self.widget._tooltip_cache, None)

# pylint: disable=protected-access

import unittest
from unittest.mock import patch
import copy
import numpy as np
import pyqtgraph as pg
from AnyQt.QtWidgets import QToolTip

from Orange.data import Table
import Orange.evaluation
import Orange.classification
from Orange.evaluation import Results

from Orange.widgets.evaluate import owrocanalysis
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import mouseMove, simulate
from Orange.tests import test_filename


class TestROC(unittest.TestCase):
    def test_ROCData_from_results(self):
        data = Orange.data.Table("iris")
        learners = [
            Orange.classification.MajorityLearner(),
            Orange.classification.LogisticRegressionLearner(),
            Orange.classification.TreeLearner()
        ]
        cv = Orange.evaluation.CrossValidation(k=10)
        res = cv(data, learners)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.roc_data_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 10)
                self.assertTrue(all(c.is_valid for c in rocdata.folds))
                self.assertTrue(rocdata.avg_vertical.is_valid)
                self.assertTrue(rocdata.avg_threshold.is_valid)

        # fixed random seed because otherwise it could happen that data sample
        # contained only instances of two classes (and the test then fails)
        # Pylint complains about RandomState; pylint: disable=no-member
        data = data[np.random.RandomState(0).choice(len(data), size=20)]
        loo = Orange.evaluation.LeaveOneOut()
        res = loo(data, learners)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.roc_data_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                # all individual fold curves and averaged curve data
                # should be invalid
                self.assertTrue(all(not c.is_valid for c in rocdata.folds))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)

        # equivalent test to the LeaveOneOut but from a slightly different
        # constructed Orange.evaluation.Results
        cv = Orange.evaluation.CrossValidation(k=20)
        res = cv(data, learners)

        for i, _ in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.roc_data_from_results(res, i, target=c)
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
        cls.lenses = data = Table(test_filename("datasets/lenses.tab"))
        totd = Orange.evaluation.TestOnTestData(
            store_data=True,
        )
        cls.res = totd(
            data=data[::2], test_data=data[1::2],
            learners=[Orange.classification.MajorityLearner(),
                      Orange.classification.KNNLearner()]
        )
        try:
            # 'mouseRateLimit' interferes with mouse move tests
            pg.setConfigOption("mouseRateLimit", -1)
        except KeyError:
            pass

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
        # See https://people.inf.elte.hu/kiss/11dwhdm/roc.pdf for the curve
        # representing this data
        actual = np.array([float(c == "n") for c in "ppnpppnnpnpnpnnnpnpn"])
        p = np.array([.9, .8, .7, .6, .55, .54, .53, .52, .51, .505,
                      .4, .39, .38, .37, .36, .35, .34, .33, .30, .1])
        n = 1 - p
        predicted = (p > .5).astype(float)

        # The second curve is like the first except for the first three points:
        # it goes to the right and then up
        p2 = p.copy()
        p2[:4] = [0.7, 0.8, 0.9, 0.59]
        n2 = 1 - p2
        predicted2 = (p2 < .5).astype(float)

        data = Orange.data.Table(
            Orange.data.Domain(
                [],
                [Orange.data.DiscreteVariable("y", values=tuple("pn"))]),
            np.empty((len(p), 0), dtype=float),
            actual
        )
        res = Results(
            data=data,
            actual=actual,
            predicted=np.array([list(predicted), list(predicted2)]),
            probabilities=np.array([list(zip(p, n)), list(zip(p2, n2))])
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

        with patch.object(QToolTip, "showText") as show_text:
            # no tooltips to be shown
            pos = item.mapToScene(0.0, 1.0)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            show_text.assert_not_called()

            # test single point
            pos = item.mapToScene(0, 0.1)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            (_, text), _ = show_text.call_args
            self.assertIn("(#1) 0.900", text)
            self.assertNotIn("#2", text)

            # test overlapping points
            pos = item.mapToScene(0.0, 0.0)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            (_, text), _ = show_text.call_args
            self.assertIn("(#1) 1.000\n(#2) 1.000", text)

            pos = item.mapToScene(0.1, 0.3)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            (_, text), _ = show_text.call_args
            self.assertIn("(#1) 0.600\n(#2) 0.590", text)
            show_text.reset_mock()

            # test that cache is invalidated when changing averaging mode
            self.widget.roc_averaging = OWROCAnalysis.Threshold
            self.widget._replot()
            mouseMove(view.viewport(), pos)
            (_, text), _ = show_text.call_args
            self.assertIn("(#1) 0.600\n(#2) 0.590", text)
            show_text.reset_mock()

            # test nan thresholds
            self.widget.roc_averaging = OWROCAnalysis.Vertical
            self.widget._replot()
            mouseMove(view.viewport(), pos)
            show_text.assert_not_called()

    def test_target_prior(self):
        w = self.widget
        self.send_signal(w.Inputs.evaluation_results, self.res)
        # hard selected
        self.assertEqual(np.round(4/12 * 100), w.target_prior)

        simulate.combobox_activate_item(w.controls.target_index, "none")
        self.assertEqual(np.round(3/12 * 100), w.target_prior)

        simulate.combobox_activate_item(w.controls.target_index, "soft")
        self.assertEqual(np.round(5/12 * 100), w.target_prior)


if __name__ == "__main__":
    unittest.main()

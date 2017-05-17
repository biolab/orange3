import unittest
import copy
import numpy

import Orange.data
import Orange.evaluation
import Orange.classification

from Orange.widgets.evaluate import owrocanalysis
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.tests.base import WidgetTest


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
        data = data[numpy.random.RandomState(0).choice(len(data), size=20)]
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


class TestOWROCAnalysis(WidgetTest):

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
        self.send_signal("Evaluation Results", res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()
        self.send_signal("Evaluation Results", None)

    def test_empty_input(self):
        res = Orange.evaluation.Results(
            data=self.lenses[:0], nmethods=2, store_data=True)
        res.row_indices = numpy.array([], dtype=int)
        res.actual = numpy.array([])
        res.predicted = numpy.zeros((2, 0))
        res.probabilities = numpy.zeros((2, 0, 3))

        self.send_signal("Evaluation Results", res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()

        res.row_indices = numpy.array([1], dtype=int)
        res.actual = numpy.array([0.0])
        res.predicted = numpy.zeros((2, 1))
        res.probabilities = numpy.zeros((2, 1, 3))

        self.send_signal("Evaluation Results", res)
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

        res.actual[0] = numpy.nan
        res.predicted[:, 1] = numpy.nan
        res.probabilities[0, 1, :] = numpy.nan

        self.send_signal("Evaluation Results", res)
        self.assertTrue(self.widget.Error.invalid_results.is_shown())
        self.send_signal("Evaluation Results", None)
        self.assertFalse(self.widget.Error.invalid_results.is_shown())
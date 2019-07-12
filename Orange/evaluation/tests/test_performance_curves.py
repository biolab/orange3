import unittest
from unittest.mock import patch

import numpy as np

from Orange.evaluation.testing import Results
from Orange.evaluation.performance_curves import Curves


# Test data and sensitivity/specificity are taken from
# Tom Fawcett: An introduction to ROC analysis, with one true positive instance
# removed, so that the number of positive and negative does not match

class TestCurves(unittest.TestCase):
    def setUp(self):
        n, p = (0, 1)
        self.data = np.array([
            (p, .8), (n, .7), (p, .6), (p, .55), (p, .54), (n, .53),
            (n, .52), (p, .51), (n, .505), (p, .4), (n, .39), (p, .38),
            (n, .37), (n, .36), (n, .35), (p, .34), (n, .33), (p, .30), (n, .1)
        ])

    def test_curves(self):
        np.random.shuffle(self.data)
        ytrue, probs = self.data.T
        curves = Curves(ytrue, probs)

        tn = np.array(
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 9, 9, 10, 10])
        np.testing.assert_equal(curves.tn, tn)
        np.testing.assert_equal(curves.fp, 10 - tn)
        np.testing.assert_almost_equal(curves.specificity(), tn / 10)

        tp = np.array(
            [9, 9, 8, 8, 7, 7, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 2, 1, 1, 0])
        np.testing.assert_equal(curves.tp, tp)
        np.testing.assert_equal(curves.fn, 9 - tp)
        np.testing.assert_almost_equal(curves.sensitivity(), tp / 9)

        np.testing.assert_almost_equal(
            curves.ca(),
            np.array([9, 10, 9, 10, 9, 10, 11, 12, 11, 12, 11, 12, 11, 12,
                      13, 12, 11, 10, 11, 10]) / 19)

        precision = np.array(
            [9 / 19, 9 / 18, 8 / 17, 8 / 16, 7 / 15, 7 / 14, 7 / 13,
             7 / 12, 6 / 11, 6 / 10, 5 / 9, 5 / 8, 4 / 7, 4 / 6,
             4 / 5, 3 / 4, 2 / 3, 1 / 2, 1 / 1, 1])
        np.testing.assert_almost_equal(curves.precision(), precision)
        np.testing.assert_almost_equal(curves.recall(), tp / 9)

        np.testing.assert_almost_equal(curves.ppv(), precision)
        np.testing.assert_almost_equal(
            curves.npv(),
            np.array([1, 1 / 1, 1 / 2, 2 / 3, 2 / 4, 3 / 5, 4 / 6, 5 / 7,
                      5 / 8, 6 / 9, 6 / 10, 7 / 11, 7 / 12, 8 / 13, 9 / 14,
                      9 / 15, 9 / 16, 9 / 17, 10 / 18, 10 / 19]))

        np.testing.assert_almost_equal(curves.tpr(), tp / 9)
        np.testing.assert_almost_equal(curves.fpr(), (10 - tn) / 10)

    @patch("Orange.evaluation.performance_curves.Curves.__init__",
           return_value=None)
    def test_curves_from_results(self, init):
        res = Results()
        ytrue, probs = self.data.T
        res.actual = ytrue.astype(float)
        res.probabilities = np.vstack((1 - probs, probs)).T.reshape(1, -1, 2)
        Curves.from_results(res)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, ytrue)
        np.testing.assert_equal(cprobs, probs)

        Curves.from_results(res, target_class=0)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, 1 - ytrue)
        np.testing.assert_equal(cprobs, 1 - probs)

        res.actual = ytrue.astype(float)
        res.probabilities = np.random.random((2, 19, 2))
        res.probabilities[1] = np.vstack((1 - probs, probs)).T

        Curves.from_results(res, model_index=1)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, ytrue)
        np.testing.assert_equal(cprobs, probs)

        self.assertRaises(ValueError, Curves.from_results, res)

        ytrue[ytrue == 0] = 2 * (np.arange(10) % 2)
        res.actual = ytrue.astype(float)
        res.probabilities = np.random.random((2, 19, 3))
        res.probabilities[1] = np.vstack(
            ((1 - probs) / 3, probs, (1 - probs) * 2 / 3)).T

        Curves.from_results(res, model_index=1, target_class=1)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, ytrue == 1)
        np.testing.assert_equal(cprobs, probs)

        Curves.from_results(res, model_index=1, target_class=0)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, ytrue == 0)
        np.testing.assert_equal(cprobs, (1 - probs) / 3)

        Curves.from_results(res, model_index=1, target_class=2)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, ytrue == 2)
        np.testing.assert_equal(cprobs, (1 - probs) * 2 / 3)

        self.assertRaises(ValueError, Curves.from_results, res, model_index=1)

    @patch("Orange.evaluation.performance_curves.Curves.__init__",
           return_value=None)
    def test_curves_from_results_nans(self, init):
        res = Results()
        ytrue, probs = self.data.T
        ytrue[0] = np.nan
        probs[-1] = np.nan
        res.actual = ytrue.astype(float)
        res.probabilities = np.vstack((1 - probs, probs)).T.reshape(1, -1, 2)
        Curves.from_results(res)
        cytrue, cprobs = init.call_args[0]
        np.testing.assert_equal(cytrue, ytrue[1:-1])
        np.testing.assert_equal(cprobs, probs[1:-1])

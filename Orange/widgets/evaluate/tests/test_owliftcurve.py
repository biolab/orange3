import copy
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import Table
import Orange.evaluation
import Orange.classification

from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.evaluate.owliftcurve import OWLiftCurve, cumulative_gains, \
    cumulative_gains_from_results
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
        res.actual[0] = np.nan
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.assertTrue(self.widget.Error.invalid_results.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.assertFalse(self.widget.Error.invalid_results.is_shown())


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


if __name__ == "__main__":
    unittest.main()

from unittest.mock import Mock, patch

import numpy as np
import unittest

from Orange.regression.linear import OLSRegressionLearner


class TestLinear(unittest.TestCase):
    @patch("Orange.regression.linear.OLSRegressionLearner._compute_pvalues")
    def test_fit_computes_p_values(self, _compute_pvalues):
        p_values = np.array([0.1, 0.2, 0.3])
        _compute_pvalues.return_value = p_values
        learner = OLSRegressionLearner()
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        classifier = learner.fit(x, y)
        _compute_pvalues.assert_called_once()
        args = _compute_pvalues.call_args[0]
        np.testing.assert_array_equal(args[1], x)
        np.testing.assert_array_equal(args[2], y)
        np.testing.assert_almost_equal(classifier.p_values, p_values)

        _compute_pvalues.side_effect = ValueError("test")
        classifier = learner.fit(x, y)
        _compute_pvalues.assert_called()
        self.assertTrue(np.all(np.isnan(classifier.p_values)))
        self.assertEqual(classifier.p_values.shape[0], x.shape[1] + 1)

        learner = OLSRegressionLearner(fit_intercept=False)
        classifier = learner.fit(x, y)
        _compute_pvalues.assert_called()
        self.assertTrue(np.all(np.isnan(classifier.p_values)))
        self.assertEqual(classifier.p_values.shape[0], x.shape[1])

    def test_compute_p_values(self):
        # The data for this test was generated using statsmodels.api.OLS:
        #
        # from statsmodels.api as sm
        # x = np.random.randint(-10, 10, (10, 5))
        # y = np.sum(x * [2, -1, 3, 5, 6], axis=1) + np.random.uniform(-2, 2, 10)
        # model = sm.OLS(y, sm.add_constant(x)).fit()
        # interept, *coefficients = model.params
        # predictions = model.fittedvalues
        # p_values = model.pvalues
        #
        x = np.array([
            [8, 9, 9, 9, 3],
            [8, -6, 0, -6, 3],
            [-10, -1, 2, -4, -4],
            [-4, 1, 6, 0, -8],
            [-4, -10, -6, 6, 4],
            [7, 6, -5, -1, -1],
            [-3, -10, -4, -3, -8],
            [8, -4, -6, -2, 3],
            [8, 9, 4, 2, 0],
            [-7, 5, -9, 7, -6]])
        y = np.array(
            [95.09214324, 8.95689637, -57.17925557, -38.71883822,
             38.33283747, -17.89237331, -71.98402682, 8.92551392,
             27.29567782, -47.20940499])
        model = Mock()

        model.fit_intercept = True
        model.predict = lambda _: np.array(
            [95.39443023, 8.99369606, -56.80606079, -39.67108666,
             37.82843204, -18.76834255, -71.40699824,   9.16945086,
             27.74232094, -46.85667099])
        model.coef_ = np.array(
            [1.91871327, -0.983126, 2.95603488, 4.96955402, 6.02835226])
        model.intercept_ = -0.52249876
        model.positive = False
        p_values = OLSRegressionLearner._compute_pvalues(model, x, y)
        np.testing.assert_almost_equal(
            p_values,
            np.array([2.13567899e-01, 7.22409281e-06, 7.03986732e-05,
                      6.22288559e-07, 2.91292141e-07, 3.35988040e-07]))

        model = Mock()
        model.fit_intercept = False
        model.predict = lambda _: np.array([
            95.65013398, 9.4403285, -55.90440701, -39.51794241,
            38.26756636, -18.51585771, -71.57093931, 9.44024785,
            28.04171045, -46.70613575])
        model.coef_ = np.array(
            [1.86525612, -0.9533842 ,  2.96382506,  4.92240953,  6.11081051])
        model.positive = False
        p_values = OLSRegressionLearner._compute_pvalues(model, x, y)
        np.testing.assert_almost_equal(
            p_values,
            np.array(
                [5.58893923e-07, 1.84684967e-05, 5.77583632e-08,
                 1.45358756e-08, 8.45328074e-09]))

    def test_invalid_rank(self):
        model = Mock()
        model.fit_intercept = True
        model.predict = lambda _: np.array([1, 2, 3])
        model.coef_ = np.array([1, 2, 3])
        model.positive = False

        x = np.array([[1, 2], [1, 2], [1, 2]])
        y = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            OLSRegressionLearner._compute_pvalues(model, x, y)

        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            OLSRegressionLearner._compute_pvalues(model, x, y)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        with self.assertRaises(ValueError):
            OLSRegressionLearner._compute_pvalues(model, x, y)


if __name__ == "__main__":
    unittest.main()
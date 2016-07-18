# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest

from Orange.base import SklLearner
from Orange.regression import LinearRegressionLearner
from Orange.classification import LogisticRegressionLearner
from Orange.data import Table

from sklearn.linear_model import LogisticRegression


class TestSklLearner(unittest.TestCase):
    def test_supports_weights(self):
        class DummySklLearner:
            def fit(self, X, y, sample_weight=None):
                pass

        class DummyLearner(SklLearner):
            __wraps__ = DummySklLearner

        self.assertTrue(DummyLearner().supports_weights)

        class DummySklLearner:
            def fit(self, X, y):
                pass

        class DummyLearner(SklLearner):
            __wraps__ = DummySklLearner

        self.assertFalse(DummyLearner().supports_weights)

    def test_linreg(self):
        self.assertTrue(LinearRegressionLearner().supports_weights,
                        "Either LinearRegression no longer supports weighted tables "
                        "or SklLearner.supports_weights is out-of-date.")

    def test_logreg(self):
        self.assertFalse(LogisticRegressionLearner().supports_weights,
                         "Logistic regression has its supports_weights overridden because "
                         "liblinear doesn't support them (even though the parameter exists)")

    def test_assert_liblinear_doesnt_accept_weights(self):
        data = Table('iris')
        data.set_weights(1.2)
        with self.assertRaises(ValueError):
            skl = LogisticRegression(solver='liblinear')
            skl.fit(data.X, data.Y, data.W)

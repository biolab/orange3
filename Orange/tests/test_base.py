# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest

from Orange.base import SklLearner
from Orange.regression import LinearRegressionLearner


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

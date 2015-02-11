import unittest

from sklearn.preprocessing import Imputer

from Orange.data import ContinuousVariable, DiscreteVariable, Table
from Orange.preprocess import (ANOVA, Gini, UnivariateLinearRegression,
                               SelectBestFeatures)


class TestFSS(unittest.TestCase):
    def test_select_1(self):
        data = Table('titanic')
        gini = Gini()
        s = SelectBestFeatures(method=gini, k=1)
        data2 = s(data)
        best = max((gini(f, data), f) for f in data.domain.attributes)[1]
        self.assertEqual(data2.domain.attributes[0], best)

    def test_select_threshold(self):
        data = Table('wine')
        anova = ANOVA()
        t = 30
        data2 = SelectBestFeatures(method=anova, threshold=t)(data)
        self.assertTrue(all(anova(f, data) >= t for f in data2.domain.attributes))

    def test_error(self):
        data = Table('wine')
        s = SelectBestFeatures(method=UnivariateLinearRegression(), k=3)
        with self.assertRaises(ValueError):
            s(data)

    def test_mixed_features(self):
        data = Table('auto-mpg')
        data.X = Imputer().fit_transform(data.X)
        s = SelectBestFeatures(method=UnivariateLinearRegression(), k=2)
        data2 = s(data)
        self.assertEqual(sum(1 for f in data2.domain.attributes
                             if isinstance(f, ContinuousVariable)), 2)
        self.assertEqual(sum(1 for f in data2.domain.attributes
                             if isinstance(f, DiscreteVariable)),
                         sum(1 for f in data.domain.attributes
                             if isinstance(f, DiscreteVariable)))

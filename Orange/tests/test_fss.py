import unittest

from sklearn.preprocessing import Imputer

import Orange
from Orange.data import ContinuousVariable, DiscreteVariable


class TestFSS(unittest.TestCase):
    def test_select_1(self):
        data = Orange.data.Table('titanic')
        gini = Orange.preprocess.Gini()
        s = Orange.preprocess.SelectKBest(k=1, method=gini)
        data2 = s(data)
        best = max((gini(f, data), f) for f in data.domain.attributes)[1]
        self.assertEqual(data2.domain.attributes[0], best)

    def test_select_t(self):
        data = Orange.data.Table('wine')
        anova = Orange.preprocess.ANOVA()
        t = 30
        s = Orange.preprocess.SelectThreshold(threshold=t, method=anova)
        data2 = s(data)
        self.assertTrue(all(anova(f, data) >= t for f in data2.domain.attributes))

    def test_error(self):
        data = Orange.data.Table('wine')
        ulr = Orange.preprocess.UnivariateLinearRegression()
        s = Orange.preprocess.SelectKBest(k=3, method=ulr)
        with self.assertRaises(ValueError):
            s(data)

    def test_mixed_features(self):
        data = Orange.data.Table('auto-mpg')
        data.X = Imputer().fit_transform(data.X)
        ulr = Orange.preprocess.UnivariateLinearRegression()
        s = Orange.preprocess.SelectKBest(k=2, method=ulr)
        data2 = s(data)
        self.assertEqual(sum(1 for f in data2.domain.attributes
                             if isinstance(f, ContinuousVariable)), 2)
        self.assertEqual(sum(1 for f in data2.domain.attributes
                             if isinstance(f, DiscreteVariable)),
                         sum(1 for f in data.domain.attributes
                             if isinstance(f, DiscreteVariable)))

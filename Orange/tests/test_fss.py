import unittest

import numpy as np

import Orange
from Orange.data import ContinuousVariable, DiscreteVariable, Table, Variable
from Orange.preprocess.score import ANOVA, Gini, UnivariateLinearRegression, \
    Chi2, GainRatio
from Orange.preprocess import SelectBestFeatures, Impute


class TestFSS(unittest.TestCase):
    def setUp(self):
        Variable._clear_all_caches()

    def test_select_1(self):
        data = Table('titanic')
        gini = Gini()
        s = SelectBestFeatures(method=gini, k=1)
        data2 = s(data)
        best = max((gini(data, f), f) for f in data.domain.attributes)[1]
        self.assertEqual(data2.domain.attributes[0], best)

    def test_select_threshold(self):
        data = Table('wine')
        anova = ANOVA()
        t = 30
        data2 = SelectBestFeatures(method=anova, threshold=t)(data)
        self.assertTrue(all(anova(data, f) >= t for f in data2.domain.attributes))

    def test_error_when_using_regression_score_on_classification_data(self):
        data = Table('wine')
        s = SelectBestFeatures(method=UnivariateLinearRegression(), k=3)
        with self.assertRaises(ValueError):
            s(data)

    def test_discrete_scores_on_continuous_features(self):
        data = Table('iris')
        c = data.columns
        for method in (Gini, Chi2):
            d1 = SelectBestFeatures(method=method)(data)
            expected = \
                (c.petal_length, c.petal_width, c.sepal_length, c.sepal_width)
            self.assertSequenceEqual(d1.domain.attributes, expected)

            scores = method(d1)
            self.assertEqual(len(scores), 4)

            score = method(d1, c.petal_length)
            self.assertIsInstance(score, float)

    def test_continuous_scores_on_discrete_features(self):
        data = Impute(Table('auto-mpg'))
        with self.assertRaises(ValueError):
            UnivariateLinearRegression(data)

        d1 = SelectBestFeatures(method=UnivariateLinearRegression)(data)
        self.assertEqual(len(d1.domain), len(data.domain))

    def test_defaults(self):
        fs = SelectBestFeatures(k=3)
        data2 = fs(Impute(Table('auto-mpg')))
        self.assertTrue(all(isinstance(a, ContinuousVariable) for a in data2.domain.attributes))
        data2 = fs(Table('wine'))
        self.assertTrue(all(isinstance(a, ContinuousVariable) for a in data2.domain.attributes))
        data2 = fs(Table('titanic'))
        self.assertTrue(all(isinstance(a, DiscreteVariable) for a in data2.domain.attributes))


class TestRemoveNaNColumns(unittest.TestCase):
    def test_column_filtering(self):
        data = Orange.data.Table("iris")
        data.X[:, (1, 3)] = np.NaN

        new_data = Orange.preprocess.RemoveNaNColumns(data)
        self.assertEqual(len(new_data.domain.attributes),
                         len(data.domain.attributes) - 2)

        data = Orange.data.Table("iris")
        data.X[0, 0] = np.NaN
        new_data = Orange.preprocess.RemoveNaNColumns(data)
        self.assertEqual(len(new_data.domain.attributes),
                         len(data.domain.attributes))


class TestSelectRandomFeatures(unittest.TestCase):
    def test_select_random_features(self):
        data = Orange.data.Table("voting")
        srf = Orange.preprocess.SelectRandomFeatures(k=3)
        new_data = srf(data)
        self.assertEqual(len(new_data.domain.attributes), 3)

        srf = Orange.preprocess.SelectRandomFeatures(k=0.25)
        new_data = srf(data)
        self.assertEqual(len(new_data.domain.attributes), 4)

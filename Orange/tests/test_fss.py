# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table, Variable
from Orange.preprocess.score import ANOVA, Gini, UnivariateLinearRegression, \
    Chi2
from Orange.preprocess import SelectBestFeatures, Impute, SelectRandomFeatures
from Orange.tests import test_filename


class TestFSS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.titanic = Table('titanic')
        cls.heart_disease = Table('heart_disease')
        cls.iris = Table('iris')
        cls.imports = Table(test_filename('datasets/imports-85.tab'))

    def test_select_1(self):
        gini = Gini()
        s = SelectBestFeatures(method=gini, k=1)
        data2 = s(self.titanic)
        best = max((gini(self.titanic, f), f) for f in self.titanic.domain.attributes)[1]
        self.assertEqual(data2.domain.attributes[0], best)

    def test_select_2(self):
        gini = Gini()
        # 100th percentile = selection of top1 attribute
        sel1 = SelectBestFeatures(method=gini, k=1.0)
        data2 = sel1(self.titanic)
        best = max((gini(self.titanic, f), f) for f in self.titanic.domain.attributes)[1]
        self.assertEqual(data2.domain.attributes[0], best)

        # no k and no threshold, select all attributes
        sel2 = SelectBestFeatures(method=gini, k=0)
        data2 = sel2(self.titanic)
        self.assertEqual(len(data2.domain.attributes), len(self.titanic.domain.attributes))

        # 31% = selection of top  (out of 3) attributes
        sel3 = SelectBestFeatures(method=gini, k=0.31)
        data2 = sel3(self.titanic)
        self.assertEqual(len(data2.domain.attributes), 1)

        # 35% = selection of top  (out of 3) attributes
        sel3 = SelectBestFeatures(method=gini, k=0.35)
        data2 = sel3(self.titanic)
        self.assertEqual(len(data2.domain.attributes), 1)

        # 1% = select one (out of 3) attributes
        sel3 = SelectBestFeatures(method=gini, k=0.01)
        data2 = sel3(self.titanic)
        self.assertEqual(len(data2.domain.attributes), 1)

        # number of selected attrs should be relative to number of current input attrs
        sel3 = SelectBestFeatures(method=gini, k=1.0)
        data2 = sel3(self.heart_disease)
        self.assertEqual(len(data2.domain.attributes), 13)

    def test_select_threshold(self):
        anova = ANOVA()
        t = 30
        data2 = SelectBestFeatures(method=anova, threshold=t)(self.heart_disease)
        self.assertTrue(all(anova(self.heart_disease, f) >= t
                            for f in data2.domain.attributes))

    def test_error_when_using_regression_score_on_classification_data(self):
        s = SelectBestFeatures(method=UnivariateLinearRegression(), k=3)
        with self.assertRaises(ValueError):
            s(self.heart_disease)

    def test_discrete_scores_on_continuous_features(self):
        c = self.iris.columns
        for method in (Gini(), Chi2()):
            d1 = SelectBestFeatures(method=method)(self.iris)
            expected = \
                (c.petal_length, c.petal_width, c.sepal_length, c.sepal_width)
            self.assertSequenceEqual(d1.domain.attributes, expected)

            scores = method(d1)
            self.assertEqual(len(scores), 4)

            score = method(d1, c.petal_length)
            self.assertIsInstance(score, float)

    def test_continuous_scores_on_discrete_features(self):
        data = Impute()(self.imports)
        with self.assertRaises(ValueError):
            UnivariateLinearRegression()(data)

        d1 = SelectBestFeatures(method=UnivariateLinearRegression())(data)
        self.assertEqual(len(d1.domain.variables), len(data.domain.variables))

    def test_defaults(self):
        fs = SelectBestFeatures(k=3)
        data2 = fs(Impute()(self.imports))
        self.assertTrue(all(a.is_continuous for a in data2.domain.attributes))
        data2 = fs(self.iris)
        self.assertTrue(all(a.is_continuous for a in data2.domain.attributes))
        data2 = fs(self.titanic)
        self.assertTrue(all(a.is_discrete for a in data2.domain.attributes))


class TestSelectRandomFeatures(unittest.TestCase):
    def test_select_random_features(self):
        data = Table("heart_disease")
        for k_features, n_attributes in ((3, 3), (0.35, 4)):
            srf = SelectRandomFeatures(k=k_features)
            new_data = srf(data)
            self.assertEqual(len(new_data.domain.attributes), n_attributes)

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA, RMSE
from Orange.classification import RandomForestLearner
from Orange.regression import RandomForestRegressionLearner


class RandomForestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.house = Table('housing')
        cls.forestClassification = RandomForestLearner()
        cls.regressionForest = RandomForestRegressionLearner()
        cls.modelClassification = cls.forestClassification(cls.iris)
        cls.modelRegression = cls.regressionForest(cls.house)

    def test_RandomForest(self):
        results = CrossValidation(self.iris, [self.forestClassification], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        for ins in self.iris:
            self.modelClassification(ins)
            _, _ = self.modelClassification(ins, self.modelClassification.ValueProbs)

    def test_predict(self):
        for data in (self.iris,  # table
                     self.iris.X):  # numpy
            self.modelClassification(data)
            _, _ = self.modelClassification(data, self.modelClassification.ValueProbs)

    def test_classification_scorer(self):
        scores = self.forestClassification.score_data(self.iris)
        self.assertEqual(len(scores), len(self.iris.domain.attributes))
        self.assertNotEqual(sum(scores), 0)
        self.assertEqual(['petal length', 'petal width'],
                         sorted([self.iris.domain.attributes[i].name
                                 for i in np.argsort(scores)[-2:]]))

    def test_scorer_feature(self):
        np.random.seed(42)
        data = Table('test4.tab')
        scores = self.forestClassification.score_data(data)
        for i, attr in enumerate(data.domain.attributes):
            np.random.seed(42)
            score = self.forestClassification.score_data(data, attr)
            self.assertEqual(score, scores[i])

    def test_RandomForestRegression(self):
        results = CrossValidation(self.house, [self.regressionForest], k=10)
        _ = RMSE(results)

    def test_predict_single_instance_reg(self):
        for ins in self.house:
            pred = self.modelRegression(ins)
            self.assertGreater(pred, 0)

    def test_predict_reg(self):
        for data in (self.house,  # table
                     self.house.X):  # numpy
            pred = self.modelRegression(data)
            self.assertEqual(len(data), len(pred))
            self.assertGreater(all(pred), 0)

    def test_regression_scorer(self):
        scores = self.regressionForest.score_data(self.house)
        self.assertEqual(['LSTAT', 'RM'],
                         sorted([self.house.domain.attributes[i].name
                                 for i in np.argsort(scores)[-2:]]))

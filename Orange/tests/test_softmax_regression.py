# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.classification import Model, SoftmaxRegressionLearner
from Orange.evaluation import CrossValidation, CA


class TestSoftmaxRegressionLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')

    def test_SoftmaxRegression(self):
        learner = SoftmaxRegressionLearner()
        cv = CrossValidation(k=3)
        results = cv(self.iris, [learner])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 1.0)

    def test_SoftmaxRegressionPreprocessors(self):
        table = self.iris.copy()
        with table.unlocked():
            table.X[:, 2] = table.X[:, 2] * 0.001
            table.X[:, 3] = table.X[:, 3] * 0.001
        learners = [SoftmaxRegressionLearner(preprocessors=[]),
                    SoftmaxRegressionLearner()]
        cv = CrossValidation(k=10)
        results = cv(table, learners)
        ca = CA(results)

        self.assertLess(ca[0], ca[1])

    def test_probability(self):
        learn = SoftmaxRegressionLearner()
        clf = learn(self.iris)
        p = clf(self.iris, ret=Model.Probs)
        self.assertLess(abs(p.sum(axis=1) - 1).all(), 1e-6)

    def test_predict_table(self):
        learner = SoftmaxRegressionLearner()
        c = learner(self.iris)
        c(self.iris)
        vals, probs = c(self.iris, c.ValueProbs)

    def test_predict_numpy(self):
        learner = SoftmaxRegressionLearner()
        c = learner(self.iris)
        c(self.iris.X)
        vals, probs = c(self.iris.X, c.ValueProbs)

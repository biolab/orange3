# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.classification import NaiveBayesLearner
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import CrossValidation, CA


class TestNaiveBayesLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = Table('titanic')
        cls.learner = NaiveBayesLearner()
        cls.model = cls.learner(data)
        cls.table = data[::20]

    def test_NaiveBayes(self):
        results = CrossValidation(self.table, [self.learner], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

    def test_predict_single_instance(self):
        for ins in self.table:
            self.model(ins)
            val, prob = self.model(ins, self.model.ValueProbs)

    def test_predict_table(self):
        self.model(self.table)
        vals, probs = self.model(self.table, self.model.ValueProbs)

    def test_predict_numpy(self):
        X = self.table.X[::20]
        self.model(X)
        vals, probs = self.model(X, self.model.ValueProbs)

    def test_degenerate(self):
        d = Domain((ContinuousVariable(name="A"),
                    ContinuousVariable(name="B"),
                    ContinuousVariable(name="C")),
                    DiscreteVariable(name="CLASS", values=["M", "F"]))
        t = Table(d, [[0,1,0,0], [0,1,0,1], [0,1,0,1]])
        nb = NaiveBayesLearner()
        model = nb(t)
        self.assertEqual(model.domain.attributes, ())
        self.assertEqual(model(t[0]), 1)
        self.assertTrue(all(model(t) == 1))

    def test_allnan_cv(self):
        # GH 2740
        data = Table('voting')
        results = CrossValidation(data, [self.learner])
        self.assertFalse(any(results.failed))

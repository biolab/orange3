import unittest

import Orange
from Orange.classification import NaiveBayesLearner
from  Orange.data import Table

class NaiveBayesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.titanic = Table('titanic')
    def test_NaiveBayes(self):
        bayes = NaiveBayesLearner()
        results = Orange.evaluation.CrossValidation(self.titanic[::20], [bayes], k=10)
        ca = Orange.evaluation.CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

    def test_predict_single_instance(self):
        bayes = NaiveBayesLearner()
        c = bayes(self.titanic)
        for ins in self.titanic[::20]:
            c(ins)
            val, prob = c(ins, c.ValueProbs)

    def test_predict_table(self):
        bayes = NaiveBayesLearner()
        c = bayes(self.titanic)
        table = self.titanic[::20]
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        bayes = NaiveBayesLearner()
        c = bayes(self.titanic)
        X = self.titanic.X[::20]
        c(X)
        vals, probs = c(X, c.ValueProbs)

import unittest

import Orange
from Orange.classification import NaiveBayesLearner


class NaiveBayesTest(unittest.TestCase):
    def test_NaiveBayes(self):
        table = Orange.data.Table('titanic')
        bayes = NaiveBayesLearner()
        results = Orange.evaluation.CrossValidation(table[::20], [bayes], k=10)
        ca = Orange.evaluation.CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

    def test_predict_single_instance(self):
        table = Orange.data.Table('titanic')
        bayes = NaiveBayesLearner()
        c = bayes(table)
        for ins in table[::20]:
            c(ins)
            val, prob = c(ins, c.ValueProbs)

    def test_predict_table(self):
        table = Orange.data.Table('titanic')
        bayes = NaiveBayesLearner()
        c = bayes(table)
        table = table[::20]
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Orange.data.Table('titanic')
        bayes = NaiveBayesLearner()
        c = bayes(table)
        X = table.X[::20]
        c(X)
        vals, probs = c(X, c.ValueProbs)

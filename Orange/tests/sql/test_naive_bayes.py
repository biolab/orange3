import unittest

from numpy import array

import Orange.classification.naive_bayes as nb
from Orange.data.discretization import DiscretizeTable
from Orange.data.sql.table import SqlTable
from Orange.data.variable import DiscreteVariable


class NaiveBayesTest(unittest.TestCase):
    def test_NaiveBayes(self):
        table = SqlTable(host='localhost', database='test', table='iris',
                         type_hints=dict(iris=DiscreteVariable(
                             values=['Iris-setosa', 'Iris-versicolor',
                                     'Iris-virginica']),
                                         __class_vars__=['iris']))
        table = DiscretizeTable(table)
        bayes = nb.BayesLearner()
        clf = bayes(table)
        # Single instance prediction
        self.assertEqual(clf(table[0]), table[0].get_class())
        # Table prediction
        pred = clf(table)
        actual = array([ins.get_class() for ins in table])
        ca = pred == actual
        ca = ca.sum() / len(ca)
        self.assertGreater(ca, 0.95)
        self.assertLess(ca, 1.)

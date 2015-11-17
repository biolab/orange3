import unittest

from numpy import array

import Orange.classification.naive_bayes as nb
from Orange import preprocess
from Orange.data.sql.table import SqlTable
from Orange.data import Domain
from Orange.data.variable import DiscreteVariable
from Orange.tests.sql.base import sql_test, connection_params


@unittest.skip("Fails on travis.")
@sql_test
class NaiveBayesTest(unittest.TestCase):
    def test_NaiveBayes(self):
        table = SqlTable(connection_params(), 'iris',
                         type_hints=Domain([], DiscreteVariable("iris",
                                values=['Iris-setosa', 'Iris-virginica',
                                        'Iris-versicolor'])))
        table = preprocess.Discretize(table)
        bayes = nb.NaiveBayesLearner()
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

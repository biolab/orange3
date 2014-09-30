import unittest

from numpy import array

import Orange.classification.naive_bayes as nb
from Orange.data.discretization import DiscretizeTable
from Orange.data.sql.table import SqlTable
from Orange.data import Domain
from Orange.data.variable import DiscreteVariable
from Orange.tests.sql.base import has_psycopg2


@unittest.skipIf(not has_psycopg2, "Psycopg2 is required for sql tests.")
class NaiveBayesTest(unittest.TestCase):
    def test_NaiveBayes(self):
        table = SqlTable(host='localhost', database='test', table='iris',
                         type_hints=Domain([], DiscreteVariable("iris", 
                                values=['Iris-setosa', 'Iris-virginica',
                                        'Iris-versicolor'])))
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

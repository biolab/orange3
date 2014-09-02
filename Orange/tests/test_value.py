import pickle
import unittest
from Orange.data import Table


class ValueTests(unittest.TestCase):
    def test_pickling_discrete_values(self):
        iris = Table('iris')

        a = iris[0]['iris']
        b = pickle.loads(pickle.dumps(a))

        self.assertEqual(float(a), float(b))
        self.assertEqual(a.value, b.value)

    def test_pickling_string_values(self):
        zoo = Table('zoo')

        a = zoo[0]['name']
        b = pickle.loads(pickle.dumps(a))

        self.assertEqual(float(a), float(b))
        self.assertEqual(a.value, b.value)

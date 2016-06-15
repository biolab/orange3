# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import pickle
import unittest
import numpy as np

from Orange.data import Table, Domain, DiscreteVariable


class TestValue(unittest.TestCase):
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

    def test_compare_continuous(self):
        auto = Table('auto-mpg')
        acc1 = auto[0]['acceleration']  # 12
        acc2 = auto[1]['acceleration']  # 11.5
        self.assertTrue(acc1 > acc2)
        self.assertTrue(acc1 >= 12.0)
        self.assertFalse(acc1 != acc1)

    def test_compare_discrete(self):
        data = Table(Domain([DiscreteVariable(name="G",values=["M", "F"])]),
                     np.array([[0],[1]]))
        self.assertTrue(data[0]['G'] < data[1]['G'])
        self.assertTrue(data[0]['G'] >= data[0]['G'])
        self.assertTrue(data[0]['G'] < 1)
        self.assertTrue(data[0]['G'] < "F")
        self.assertFalse(data[1]['G'] < "F")

    def test_compare_string(self):
        zoo = Table('zoo')
        zoo1 = zoo[0]['name']  # aardvark
        zoo2 = zoo[1]['name']  # antelope
        self.assertTrue(zoo1 < zoo2)
        self.assertTrue(zoo1 >= "aardvark")

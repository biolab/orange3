import random
from unittest import TestCase

import numpy as np

import Orange.preprocess.discretization


class TestEqualFreq(TestCase):
    def test_equifreq_with_too_few_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        data = Orange.data.Table(X)
        disc = Orange.preprocess.discretization.EqualFreq(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])

    def test_equifreq_100_to_4(self):
        X = np.arange(100).reshape((100, 1))
        data = Orange.data.Table(X)
        disc = Orange.preprocess.discretization.EqualFreq(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [24.5, 49.5, 74.5])

    def test_equifreq_with_k_instances(self):
        X = np.array([[1], [2], [3], [4]])
        data = Orange.data.Table(X)
        disc = Orange.preprocess.discretization.EqualFreq(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [1.5, 2.5, 3.5])


class TestEqualWidth(TestCase):
    def test_equalwidth_on_two_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        data = Orange.data.Table(X)
        disc = Orange.preprocess.discretization.EqualWidth(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [0.25, 0.5, 0.75])

    def test_equalwidth_100_to_4(self):
        X = np.arange(101).reshape((101, 1))
        data = Orange.data.Table(X)
        disc = Orange.preprocess.discretization.EqualWidth(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [25, 50, 75])

    def test_equalwidth_const_value(self):
        X = np.ones((100, 1))
        data = Orange.data.Table(X)
        disc = Orange.preprocess.discretization.EqualFreq(n=4)
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])


class TestEntropyMDL(TestCase):
    def test_entropy_with_two_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        data = Orange.data.Table(X, X)
        disc = Orange.preprocess.discretization.EntropyMDL()
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])

    def test_entropy_with_two_values_useless(self):
        X = np.array([0] * 50 + [1] * 50).reshape((100, 1))
        Y = np.array([0] * 25 + [1] * 50 + [0] * 25)
        data = Orange.data.Table(X, Y)
        disc = Orange.preprocess.discretization.EntropyMDL()
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])

    def test_entropy_constant(self):
        X = np.ones((100, 1))
        data = Orange.data.Table(X, X)
        disc = Orange.preprocess.discretization.EntropyMDL()
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])

    def test_entropy(self):
        X = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
                     ).reshape((100, 1))
        Y = np.array([0] * 25 + [1] * 75)
        data = Orange.data.Table(X, Y)
        disc = Orange.preprocess.discretization.EntropyMDL()
        dvar = disc(data, data.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])


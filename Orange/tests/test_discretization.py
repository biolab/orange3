import random
from unittest import TestCase
from mock import Mock

import numpy as np

from Orange import preprocess
from Orange import data


# noinspection PyPep8Naming
class TestEqualFreq(TestCase):
    def test_equifreq_with_too_few_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        table = data.Table(X)
        disc = preprocess.EqualFreq(n=4)
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])

    def test_equifreq_100_to_4(self):
        X = np.arange(100).reshape((100, 1))
        table = data.Table(X)
        disc = preprocess.EqualFreq(n=4)
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [24.5, 49.5, 74.5])

    def test_equifreq_with_k_instances(self):
        X = np.array([[1], [2], [3], [4]])
        table = data.Table(X)
        disc = preprocess.EqualFreq(n=4)
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [1.5, 2.5, 3.5])


# noinspection PyPep8Naming
class TestEqualWidth(TestCase):
    def test_equalwidth_on_two_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        table = data.Table(X)
        disc = preprocess.EqualWidth(n=4)
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [0.25, 0.5, 0.75])

    def test_equalwidth_100_to_4(self):
        X = np.arange(101).reshape((101, 1))
        table = data.Table(X)
        disc = preprocess.EqualWidth(n=4)
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [25, 50, 75])

    def test_equalwidth_const_value(self):
        X = np.ones((100, 1))
        table = data.Table(X)
        disc = preprocess.EqualFreq(n=4)
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])


# noinspection PyPep8Naming
class TestEntropyMDL(TestCase):
    def test_entropy_with_two_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        table = data.Table(X, X)
        disc = preprocess.EntropyMDL()
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])

    def test_entropy_with_two_values_useless(self):
        X = np.array([0] * 50 + [1] * 50).reshape((100, 1))
        Y = np.array([0] * 25 + [1] * 50 + [0] * 25)
        table = data.Table(X, Y)
        disc = preprocess.EntropyMDL()
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])

    def test_entropy_constant(self):
        X = np.ones((100, 1))
        table = data.Table(X, X)
        disc = preprocess.EntropyMDL()
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])

    def test_entropy(self):
        X = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
                     ).reshape((100, 1))
        Y = np.array([0] * 25 + [1] * 75)
        table = data.Table(X, Y)
        disc = preprocess.EntropyMDL()
        dvar = disc(table, table.domain.variables[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])


# noinspection PyPep8Naming
class TestDiscretizer(TestCase):
    def setUp(self):
        self.var = Mock(data.ContinuousVariable, number_of_decimals=1)
        self.var.name = "x"

    def test_create_discretized_var(self):
        dvar = preprocess.Discretizer.create_discretized_var(
            self.var, [1, 2, 3])
        self.assertEqual(dvar.values, ["<1", "[1, 2)", "[2, 3)", ">=3"])
        self.assertIsInstance(dvar.compute_value, preprocess.Discretizer)
        self.assertEqual(dvar.compute_value.points, [1, 2, 3])

    def test_discretizer_computation(self):
        dvar = preprocess.Discretizer.create_discretized_var(
            self.var, [1, 2, 3])
        X = np.array([0, 0.9, 1, 1.1, 1.9, 2, 2.5, 3, 3.5])
        np.testing.assert_equal(dvar.compute_value._transform(X), np.floor(X))


# noinspection PyPep8Naming
class TestDiscretizeTable(TestCase):
    def setUp(self):
        s = [0] * 50 + [1] * 50
        X1 = np.array(s).reshape((100, 1))
        X2 = np.arange(100).reshape((100, 1))
        X3 = np.ones((100, 1))
        X = np.hstack([X1, X2, X3])
        self.table_no_class = data.Table(X)
        self.table_class = data.Table(X, X1)

    def test_discretize_exclude_constant(self):
        dt = preprocess.DiscretizeTable(self.table_no_class)
        dom = dt.domain
        self.assertEqual(len(dom.attributes), 2)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])

        dt = preprocess.DiscretizeTable(self.table_no_class, clean=False)
        dom = dt.domain
        self.assertEqual(len(dom.attributes), 3)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])
        self.assertEqual(dom[2].compute_value.points, [])

        dt = preprocess.DiscretizeTable(self.table_class)
        dom = dt.domain
        self.assertEqual(len(dom.attributes), 2)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])

    def test_discretize_class(self):
        dt = preprocess.DiscretizeTable(self.table_class)
        self.assertIs(dt.domain.class_var, self.table_class.domain.class_var)

        dt = preprocess.DiscretizeTable(self.table_class, discretize_class=True)
        self.assertIs(dt.domain.class_var, self.table_class.domain.class_var)

    def test_method(self):
        dt = preprocess.DiscretizeTable(self.table_class)
        self.assertEqual(len(dt.domain[1].values), 4)

        dt = preprocess.DiscretizeTable(self.table_class,
                                        method=preprocess.EqualWidth(n=2))
        self.assertEqual(len(dt.domain[1].values), 2)

    def test_fixed(self):
        dt = preprocess.DiscretizeTable(self.table_no_class,
                                        method=preprocess.EqualWidth(n=2),
                                        fixed={"Feature 2": [1, 11]})
        dom = dt.domain
        self.assertEqual(len(dom.attributes), 2)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [6])

    def test_leave_discrete(self):
        s = [0] * 50 + [1] * 50
        X1 = np.array(s).reshape((100, 1))
        X2 = np.arange(100).reshape((100, 1))
        X3 = np.ones((100, 1))
        X = np.hstack([X1, X2, X3])
        domain = data.Domain([data.DiscreteVariable("a", values="MF"),
                              data.ContinuousVariable("b"),
                              data.DiscreteVariable("c", values="AB")],
                             data.ContinuousVariable("d"))
        table = data.Table(domain, X, X1)
        dt = preprocess.DiscretizeTable(table)
        dom = dt.domain
        self.assertIs(dom[0], table.domain[0])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])
        self.assertIs(dom[2], table.domain[2])
        self.assertIs(dom.class_var, table.domain.class_var)

        domain = data.Domain([data.DiscreteVariable("a", values="MF"),
                              data.ContinuousVariable("b"),
                              data.DiscreteVariable("c", values="AB")],
                             data.DiscreteVariable("d"))
        table = data.Table(domain, X, X1)
        dt = preprocess.DiscretizeTable(table)
        dom = dt.domain
        self.assertIs(dom[0], table.domain[0])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])
        self.assertIs(dom[2], table.domain[2])
        self.assertIs(dom.class_var, table.domain.class_var)


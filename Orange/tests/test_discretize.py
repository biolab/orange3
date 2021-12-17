# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import sys
import random
from unittest import TestCase

import numpy as np
import scipy.sparse as sp

from Orange.preprocess import discretize, Discretize, decimal_binnings
from Orange import data
from Orange.data import Table, Instance, Domain, ContinuousVariable, DiscreteVariable


# noinspection PyPep8Naming
from Orange.widgets.tests.utils import table_dense_sparse


class TestEqualFreq(TestCase):
    def test_equifreq_with_too_few_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        table = data.Table.from_numpy(None, X)
        disc = discretize.EqualFreq(n=4)
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])

    def test_equifreq_100_to_4(self):
        X = np.arange(100).reshape((100, 1))
        table = data.Table.from_numpy(None, X)
        disc = discretize.EqualFreq(n=4)
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [24.5, 49.5, 74.5])

    def test_equifreq_with_k_instances(self):
        X = np.array([[1], [2], [3], [4]])
        table = data.Table.from_numpy(None, X)
        disc = discretize.EqualFreq(n=4)
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [1.5, 2.5, 3.5])

    def test_below_precision(self):
        eps = sys.float_info.epsilon

        # Test with n >= number of distinct values
        X = np.array([[1], [1 + eps], [1 + 2 * eps], [1 + 3 * eps]])
        # Test the test: check that these are indeed distinct values
        assert len(np.unique(X).flatten()) == 4
        table = data.Table.from_numpy(None, X)
        var = discretize.EqualFreq(n=4)(table, table.domain[0])
        points = var.compute_value.points
        self.assertEqual(len(np.unique(points)), len(points))

        # Test with n < number of distinct values
        X = np.array([[1 + i * eps] for i in range(10)])
        # Test the test: check that these are indeed distinct values
        assert len(np.unique(X).flatten()) == 10
        table = data.Table.from_numpy(None, X)
        var = discretize.EqualFreq(n=8)(table, table.domain[0])
        points = var.compute_value.points
        self.assertEqual(len(np.unique(points)), len(points))

# noinspection PyPep8Naming
class TestEqualWidth(TestCase):
    def test_equalwidth_on_two_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        table = data.Table.from_numpy(None, X)
        disc = discretize.EqualWidth(n=4)
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [0.25, 0.5, 0.75])

    @table_dense_sparse
    def test_equalwidth_100_to_4(self, prepare_table):
        X = np.arange(101).reshape((101, 1))
        table = prepare_table(data.Table.from_numpy(None, X))
        disc = discretize.EqualWidth(n=4)
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 4)
        self.assertEqual(dvar.compute_value.points, [25, 50, 75])

    def test_equalwidth_const_value(self):
        X = np.ones((100, 1))
        table = data.Table.from_numpy(None, X)
        disc = discretize.EqualFreq(n=4)
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])


class TestBinning(TestCase):
    def test_decimal_binnings(self):
        values = np.array([
            -0.2, -0.2, -0.6, 1.0, 0.2, -0.6, 0.6, 1.0, 0.4, -0.5, -0.4, -0.4,
            -0.6, 0.6, 0.75, 0.4, -0.2, 0.2, 0.0, 0.0, -1.0, -0.6, -0.2, -0.6,
        ])
        binning = decimal_binnings(values, factors=[0.2, 0.25, 0.5])
        self.assertEqual(len(binning), 3)

        np.testing.assert_array_equal(
            binning[0].thresholds,
            [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.assertEqual(binning[0].width, 0.2)

        np.testing.assert_array_equal(
            binning[1].thresholds,
            [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        )
        self.assertEqual(binning[1].width, 0.25)

        np.testing.assert_array_equal(
            binning[2].thresholds,
            [-1, -0.5, 0, 0.5, 1]
        )
        self.assertEqual(binning[2].width, 0.5)


# noinspection PyPep8Naming
class TestEntropyMDL(TestCase):
    def test_entropy_with_two_values(self):
        s = [0] * 50 + [1] * 50
        random.shuffle(s)
        X = np.array(s).reshape((100, 1))
        table = data.Table.from_numpy(None, X, X)
        disc = discretize.EntropyMDL()
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])

    def test_entropy_with_two_values_useless(self):
        X = np.array([0] * 50 + [1] * 50).reshape((100, 1))
        Y = np.array([0] * 25 + [1] * 50 + [0] * 25)
        table = data.Table.from_numpy(None, X, Y)
        disc = discretize.EntropyMDL()
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])

    def test_entropy_constant(self):
        X = np.zeros((100, 1))
        domain = Domain([ContinuousVariable('v1')],
                        [DiscreteVariable('c1', values=("1",))])
        table = data.Table(domain, X, X)
        disc = discretize.EntropyMDL()
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 1)
        self.assertEqual(dvar.compute_value.points, [])

    def test_entropy(self):
        X = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
                    ).reshape((100, 1))
        Y = np.array([0] * 25 + [1] * 75)
        table = data.Table.from_numpy(None, X, Y)
        disc = discretize.EntropyMDL()
        dvar = disc(table, table.domain[0])
        self.assertEqual(len(dvar.values), 2)
        self.assertEqual(dvar.compute_value.points, [0.5])


# noinspection PyPep8Naming
class TestDiscretizer(TestCase):
    def setUp(self):
        self.var = data.ContinuousVariable("x", number_of_decimals=1)

    def test_create_discretized_var(self):
        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [1, 2, 3])
        self.assertIsInstance(dvar.compute_value,
                              discretize.Discretizer)
        self.assertEqual(dvar.compute_value.points, [1, 2, 3])


    def test_create_discretized_var_formatting(self):
        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [1, 2, 3])
        self.assertEqual(dvar.values, ("< 1", "1 - 2", "2 - 3", "≥ 3"))

        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [10])
        self.assertEqual(dvar.values, ("< 10", "≥ 10"))

        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [10.1234])
        self.assertEqual(dvar.values, ("< 10.123", "≥ 10.123"))

        self.var.number_of_decimals = 3

        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [5, 10.25])
        self.assertEqual(dvar.values, ("< 5", "5 - 10.25", "≥ 10.25"))

        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [5, 10.1234])
        self.assertEqual(dvar.values, ("< 5", "5 - 10.1234", "≥ 10.1234"))

    def test_discretizer_computation(self):
        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [1, 2, 3])
        X = np.array([0, 0.9, 1, 1.1, 1.9, 2, 2.5, 3, 3.5])
        np.testing.assert_equal(dvar.compute_value.transform(X), np.floor(X))

    def test_discretizer_computation_sparse(self):
        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [1, 2, 3])
        X = sp.csr_matrix(np.array([0, 0.9, 1, 1.1, 1.9, 2, 2.5, 3, 3.5]))
        self.assertEqual((dvar.compute_value.transform(X) != X.floor()).nnz, 0)

    def test_discretizer_computation_sparse_no_points(self):
        dvar = discretize.Discretizer.create_discretized_var(
            self.var, [])
        X = sp.csr_matrix(np.array([0, 0.9, 1, 1.1, 1.9, 2, 2.5, 3, 3.5]))
        empty = sp.csr_matrix(X.shape)
        self.assertEqual((dvar.compute_value.transform(X) != empty).nnz, 0)

    def test_transform(self):
        table = data.Table('iris')
        table2 = Discretize()(table)
        ins = data.Instance(table2.domain, table[0])
        table3 = table[:10].transform(table2.domain)
        self.assertEqual(ins, table3[0])

    def test_remove_constant(self):
        table = data.Table('iris')
        with table.unlocked():
            table[:, 0] = 1
        discretize = Discretize(remove_const=True)
        new_table = discretize(table)
        self.assertNotEqual(len(table.domain.attributes),
                            len(new_table.domain.attributes))

    def test_keep_constant(self):
        table = data.Table('iris')
        with table.unlocked():
            table[:, 0] = 1
        discretize = Discretize(remove_const=False)
        new_table = discretize(table)
        self.assertEqual(len(table.domain.attributes),
                         len(new_table.domain.attributes))

    def test_discretize_class(self):
        table = data.Table('iris')
        domain = table.domain
        regr_domain = data.Domain(domain.attributes[:3],
                                  [domain.attributes[3], domain.class_var])
        table = data.Table.from_table(regr_domain, table)

        discretize = Discretize(remove_const=False)
        new_table = discretize(table)
        self.assertIs(new_table.domain.class_vars[0],
                      new_table.domain.class_vars[0])
        self.assertIs(new_table.domain.class_vars[1],
                      new_table.domain.class_vars[1])

        discretize = Discretize(remove_const=False, discretize_classes=True)
        new_table = discretize(table)
        self.assertIsInstance(new_table.domain.class_vars[0], DiscreteVariable)
        self.assertIs(new_table.domain.class_vars[1],
                      new_table.domain.class_vars[1])

    def test_discretize_metas(self):
        table = data.Table('iris')
        domain = table.domain
        regr_domain = data.Domain(domain.attributes[:3],
                                  [],
                                  [domain.attributes[3], domain.class_var])
        table = data.Table.from_table(regr_domain, table)

        discretize = Discretize(remove_const=False)
        new_table = discretize(table)
        self.assertIs(new_table.domain.metas[0],
                      new_table.domain.metas[0])
        self.assertIs(new_table.domain.metas[1],
                      new_table.domain.metas[1])

        discretize = Discretize(remove_const=False, discretize_metas=True)
        new_table = discretize(table)
        self.assertIsInstance(new_table.domain.metas[0], DiscreteVariable)
        self.assertIs(new_table.domain.metas[1],
                      new_table.domain.metas[1])


# noinspection PyPep8Naming
class TestDiscretizeTable(TestCase):
    @classmethod
    def setUpClass(cls):
        s = [0] * 50 + [1] * 50
        X1 = np.array(s).reshape((100, 1))
        X2 = np.arange(100).reshape((100, 1))
        X3 = np.ones((100, 1))
        X = np.hstack([X1, X2, X3])
        cls.table_no_class = data.Table.from_numpy(None, X)
        cls.table_class = data.Table.from_numpy(None, X, X1)

    def test_discretize_exclude_constant(self):
        dom = discretize.DomainDiscretizer()(self.table_no_class)
        self.assertEqual(len(dom.attributes), 2)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])

        dom = discretize.DomainDiscretizer(clean=False)(self.table_no_class)
        self.assertEqual(len(dom.attributes), 3)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])
        self.assertEqual(dom[2].compute_value.points, [])

        dom = discretize.DomainDiscretizer()(self.table_class)
        self.assertEqual(len(dom.attributes), 2)
        self.assertEqual(dom[0].compute_value.points, [0.5])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])

    def test_discretize_class(self):
        dom = discretize.DomainDiscretizer()(self.table_class)
        self.assertIs(dom.class_var, self.table_class.domain.class_var)

        dom = discretize.DomainDiscretizer(discretize_class=True)
        dom = dom(self.table_class)
        self.assertIs(dom.class_var, self.table_class.domain.class_var)

    def test_method(self):
        dom = discretize.DomainDiscretizer()(self.table_class)
        self.assertEqual(len(dom[1].values), 4)

        dom = discretize.DomainDiscretizer(method=discretize.EqualWidth(n=2))
        dom = dom(self.table_class)
        self.assertEqual(len(dom[1].values), 2)

    def test_fixed(self):
        dom = discretize.DomainDiscretizer(method=discretize.EqualWidth(n=2),
                                           fixed={"Feature 2": [1, 11]})
        dom = dom(self.table_no_class)
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
        dom = discretize.DomainDiscretizer()(table)
        self.assertIs(dom[0], table.domain[0])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])
        self.assertIs(dom[2], table.domain[2])
        self.assertIs(dom.class_var, table.domain.class_var)

        domain = data.Domain([data.DiscreteVariable("a", values="MF"),
                              data.ContinuousVariable("b"),
                              data.DiscreteVariable("c", values="AB")],
                             data.DiscreteVariable("d"))
        table = data.Table(domain, X, X1)
        dom = discretize.DomainDiscretizer()(table)
        self.assertIs(dom[0], table.domain[0])
        self.assertEqual(dom[1].compute_value.points, [24.5, 49.5, 74.5])
        self.assertIs(dom[2], table.domain[2])
        self.assertIs(dom.class_var, table.domain.class_var)


class TestInstanceConversion(TestCase):
    def test_single_instance(self):
        iris = Table("iris")
        inst = Instance(iris.domain, [5.2, 3.8, 1.4, 0.5, "Iris-virginica"])
        d_iris = Discretize()(iris)
        Instance(d_iris.domain, inst)

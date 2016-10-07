# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from functools import reduce
import numpy as np
import scipy.sparse as sp

from Orange import preprocess
from Orange.preprocess import impute
from Orange import data
from Orange.data import Unknown, Table

from Orange.classification import MajorityLearner, SimpleTreeLearner
from Orange.regression import MeanLearner

class TestReplaceUnknowns(unittest.TestCase):
    def test_replacement(self):
        a = np.arange(10, dtype=float)
        a[1] = a[5] = Unknown
        ia = preprocess.ReplaceUnknowns(None).transform(a)
        np.testing.assert_equal(ia, [0, 0, 2, 3, 4, 0, 6, 7, 8, 9])

        a[1] = a[5] = Unknown
        ia = preprocess.ReplaceUnknowns(None, value=42).transform(a)
        np.testing.assert_equal(ia, [0, 42, 2, 3, 4, 42, 6, 7, 8, 9])

    def test_sparse(self):
        m = sp.csr_matrix(np.eye(10))
        rm = preprocess.ReplaceUnknowns(None, value=42).transform(m)
        self.assertEqual((m!=rm).nnz, 0)


class TestDropInstances(unittest.TestCase):
    def test_drop(self):
        X = np.random.rand(5, 3)
        X[3:5, 1] = np.nan
        table = data.Table.from_numpy(None, X)
        drop = impute.DropInstances()
        ind = drop(table, table.domain[1])
        self.assertEqual(list(ind), [False, False, False, True, True])


class TestDoNotImpute(unittest.TestCase):
    def test_str(self):
        table = data.Table.from_file("iris")
        imputer = impute.BaseImputeMethod()
        var = table.domain[0]
        self.assertIn(var.name, imputer.format_variable(var))
        self.assertIn(imputer.short_name, imputer.format_variable(var))
        self.assertIn(imputer.name, str(imputer))

    def test_copy(self):
        imputer = impute.BaseImputeMethod()
        self.assertIs(imputer, imputer.copy())

    def test_call(self):
        X = np.random.rand(5, 3)
        X[3:5, 1] = np.nan
        table = data.Table.from_numpy(None, X)
        imputer = impute.DoNotImpute()
        var = imputer(table, table.domain[1])
        self.assertEqual(var, table.domain[1])

    def test_support(self):
        table = data.Table.from_file("iris")
        continuous = table.domain.variables[0]
        discrete = table.domain.variables[-1]

        imputer = impute.DoNotImpute()
        self.assertTrue(imputer.supports_variable(discrete))
        self.assertTrue(imputer.supports_variable(continuous))


class TestAverage(unittest.TestCase):
    def test_replacement(self):
        s = [0] * 50 + [1] * 50
        c1 = np.array(s).reshape((100, 1))
        s = [0] * 5 + [1] * 5 + [2] * 90
        c2 = np.array(s).reshape((100, 1))
        x = np.hstack([c1, c2])
        domain = data.Domain([data.ContinuousVariable("a"),
                              data.DiscreteVariable("b", values="ABC")],
                             data.ContinuousVariable("c"),)
        table = Table(domain, x, c1)
        for col, computed_value in ((0, 0.5), (1, 2)):
            var1 = preprocess.Average()(table, col)
            self.assertIsInstance(var1.compute_value, preprocess.ReplaceUnknowns)
            self.assertEqual(var1.compute_value.value, computed_value)


class TestDefault(unittest.TestCase):
    def test_replacement(self):
        nan = np.nan
        X = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]

        table = Table.from_numpy(None, np.array(X))
        var1 = impute.Default(0.0)(table, 0)
        self.assertTrue(np.all(np.isfinite(var1.compute_value(table))))
        self.assertTrue(all(var1.compute_value(table) == [1.0, 2.0, 0.0]))

        imputer = preprocess.Impute(method=impute.Default(42))
        idata = imputer(table)
        np.testing.assert_allclose(
            idata.X,
            [[1.0, 42., 0.0],
             [2.0, 1.0, 3.0],
             [42., 42., 42.]])

    def test_default(self):
        nan = np.nan
        X = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        domain = data.Domain(
            (data.DiscreteVariable("A", values=["0", "1", "2"],
                                   base_value=2),
             data.DiscreteVariable("B", values=["a", "b", "c"]),
             data.ContinuousVariable("C"))
        )
        table = data.Table.from_numpy(domain, np.array(X))
        v1 = impute.Default(1)(table, domain["A"])
        self.assertEqual(v1.compute_value.value, 1)

        v2 = impute.Default(42)(table, domain["C"])
        self.assertEqual(v2.compute_value.value, 42)

        v3 = impute.Default()(table, domain["C"], default=42)
        self.assertEqual(v3.compute_value.value, 42)

    def test_copy(self):
        imputer = impute.Default(1)
        copied = imputer.copy()
        imputer.default = 2
        self.assertEqual(copied.default, 1)

    def test_str(self):
        imputer = impute.Default(1)
        self.assertIn('1', imputer.format_variable(data.Variable()))


class TestAsValue(unittest.TestCase):
    def test_replacement(self):
        nan = np.nan
        X = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        domain = data.Domain(
            (data.DiscreteVariable("A", values=["0", "1", "2"]),
             data.ContinuousVariable("B"),
             data.ContinuousVariable("C"))
        )
        table = data.Table.from_numpy(domain, np.array(X))

        v1 = impute.AsValue()(table, domain[0])
        self.assertTrue(np.all(np.isfinite(v1.compute_value(table))))
        self.assertTrue(np.all(v1.compute_value(table) == [1., 2., 3.]))
        self.assertEqual([v1.str_val(v) for v in v1.compute_value(table)],
                         ["1", "2", "N/A"])

        v1, v2 = impute.AsValue()(table, domain[1])
        self.assertTrue(np.all(np.isfinite(v1.compute_value(table))))
        self.assertTrue(np.all(np.isfinite(v2.compute_value(table))))
        self.assertTrue(np.all(v2.compute_value(table) == [0., 1., 0.]))
        self.assertEqual([v2.str_val(v) for v in v2.compute_value(table)],
                         ["undef", "def", "undef"])

        vars = reduce(lambda acc, v:
                      acc + (list(v) if isinstance(v, (tuple, list))
                             else [v]),
                      [impute.AsValue()(table, var) for var in table.domain],
                      [])
        domain = data.Domain(vars)
        idata = table.from_table(domain, table)

        np.testing.assert_allclose(
            idata.X,
            [[1, 1.0, 0, 0.0, 1],
             [2, 1.0, 1, 3.0, 1],
             [3, 1.0, 0, 1.5, 0]]
        )


class TestModel(unittest.TestCase):
    def test_replacement(self):
        nan = np.nan
        X = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        unknowns = np.isnan(X)

        domain = data.Domain(
            (data.DiscreteVariable("A", values=["0", "1", "2"]),
             data.ContinuousVariable("B"),
             data.ContinuousVariable("C"))
        )
        table = data.Table.from_numpy(domain, np.array(X))

        v = impute.Model(MajorityLearner())(table, domain[0])
        self.assertTrue(np.all(np.isfinite(v.compute_value(table))))
        self.assertTrue(np.all(v.compute_value(table) == [1., 2., 1.]) or
                        np.all(v.compute_value(table) == [1., 2., 2.]))
        v = impute.Model(MeanLearner())(table, domain[1])
        self.assertTrue(np.all(np.isfinite(v.compute_value(table))))
        self.assertTrue(np.all(v.compute_value(table) == [1., 1., 1.]))

        imputer = preprocess.Impute(impute.Model(SimpleTreeLearner()))
        itable = imputer(table)

        # Original data should keep unknowns
        self.assertTrue(np.all(np.isnan(table.X) == unknowns))
        self.assertTrue(np.all(itable.X[~unknowns] == table.X[~unknowns]))

        Aimp = itable.domain["A"].compute_value
        self.assertIsInstance(Aimp, impute.ReplaceUnknownsModel)

        col = Aimp(table)
        self.assertEqual(col.shape, (len(table),))
        self.assertTrue(np.all(np.isfinite(col)))

        v = Aimp(table[-1])
        self.assertEqual(v.shape, (1,))
        self.assertTrue(np.all(np.isfinite(v)))

    def test_copy(self):
        imputer = impute.Model(MajorityLearner())
        copied = imputer.copy()
        imputer.learner = MajorityLearner()
        self.assertIsNot(copied.learner, imputer.learner)

    def test_support(self):
        table = data.Table.from_file("iris")
        continuous = table.domain.variables[0]
        discrete = table.domain.variables[-1]

        imputer = impute.Model(MajorityLearner())
        self.assertTrue(imputer.supports_variable(discrete))
        self.assertFalse(imputer.supports_variable(continuous))

        imputer = impute.Model(MeanLearner())
        self.assertFalse(imputer.supports_variable(discrete))
        self.assertTrue(imputer.supports_variable(continuous))

    def test_str(self):
        imputer = impute.Model(MajorityLearner())
        self.assertIn(MajorityLearner.name, imputer.format_variable(data.Variable()))

    def test_bad_domain(self):
        table = data.Table.from_file('iris')
        imputer = impute.Model(MajorityLearner())
        self.assertRaises(ValueError, imputer, data=table,
                          variable=table.domain[0])


class TestRandom(unittest.TestCase):
    def test_replacement(self):
        nan = np.nan
        X = [
            [1.0, nan, 0.0],
            [2.0, 1.0, 3.0],
            [nan, nan, nan]
        ]
        unknowns = np.isnan(X)

        domain = data.Domain(
            (data.DiscreteVariable("A", values=["0", "1", "2"]),
             data.ContinuousVariable("B"),
             data.ContinuousVariable("C"))
        )
        table = data.Table.from_numpy(domain, np.array(X))

        for i in range(0, 3):
            v = impute.Random()(table, domain[i])
            self.assertTrue(np.all(np.isfinite(v.compute_value(table))))

        imputer = preprocess.Impute(method=impute.Random())
        itable = imputer(table)
        self.assertTrue(np.all(np.isfinite(itable.X)))

        # Original data should keep unknowns
        self.assertTrue(np.all(unknowns == np.isnan(table.X)))
        self.assertTrue(np.all(itable.X[~unknowns] == table.X[~unknowns]))


class TestImputer(unittest.TestCase):
    def test_imputer(self):
        auto = data.Table('auto-mpg')
        auto2 = preprocess.Impute(auto)
        self.assertFalse(np.isnan(auto2.X).any())

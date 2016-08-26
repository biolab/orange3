# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import os
import unittest
from itertools import chain
from math import isnan
import random
import warnings

from unittest.mock import Mock, MagicMock, patch
from scipy.sparse import csr_matrix, issparse
import numpy as np
import pandas as pd

from Orange.data import Variable, DiscreteVariable, Table, SparseTable, StringVariable, \
    get_sample_datasets_dir, dataset_dirs, TableSeries, Domain, ContinuousVariable
from Orange.data.io import FileFormat
from Orange.tests import test_dirname


@np.vectorize
def naneq(a, b):
    try:
        return (isnan(a) and isnan(b)) or a == b
    except TypeError:
        return a == b


def assert_array_nanequal(*args, **kwargs):
    # similar as np.testing.assert_array_equal but with better handling of
    # object arrays
    return np.testing.utils.assert_array_compare(naneq, *args, **kwargs)


def cols_wo_weights(table):
    # semantic beautification for this filter
    return [c for c in table.columns if c != Table._WEIGHTS_COLUMN]


class TableTestCase(unittest.TestCase):
    def setUp(self):
        Variable._clear_all_caches()
        dataset_dirs.append(test_dirname())

    def tearDown(self):
        dataset_dirs.remove(test_dirname())

    def test_filename(self):
        dir = get_sample_datasets_dir()
        d = Table("iris")
        self.assertEqual(d.__file__, os.path.join(dir, "iris.tab"))

        d = Table("test2.tab")
        self.assertTrue(d.__file__.endswith("test2.tab"))  # platform dependent

    def test_can_read_iris(self):
        t = Table('iris')
        self.assertEqual(4, len(t.domain.attributes))
        self.assertEqual(1, len(t.domain.class_vars))
        self.assertEqual(0, len(t.domain.metas))
        self.assertTrue(all(v.is_continuous for v in t.domain.attributes))
        self.assertTrue(all(v.is_discrete for v in t.domain.class_vars))
        self.assertEqual({'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'},
                         set(t["iris"].unique()))
        self.assertEqual(150, len(t))
        self.assertEqual(6, len(t.columns))  # 5 + weights
        self.assertEqual({'sepal length', 'sepal width', 'petal length', 'petal width', 'iris'},
                         set(c for c in t.columns if c != Table._WEIGHTS_COLUMN))
        self.assertEqual(3.2, t['sepal width'].iloc[2])
        self.assertEqual(0.2, t['petal width'].iloc[22])
        self.assertTrue((t.weights == 1).all())

    def test_hash(self):
        d = Table("zoo")
        d.iloc[42, 3] = 0
        crc1 = hash(d)
        d.iloc[42, 3] = 1
        crc2 = hash(d)
        self.assertNotEqual(crc1, crc2)
        d.iloc[42, 3] = 0
        crc3 = hash(d)
        self.assertEqual(crc1, crc3)
        _ = d.iloc[42].loc["name"]
        d.iloc[42].loc["name"] = "non-animal"
        crc4 = hash(d)
        self.assertEqual(crc1, crc4)
        crc5 = hash(d)
        self.assertEqual(crc1, crc5)

    def test_total_weight(self):
        d = Table("zoo")
        self.assertEqual(d.weights.sum(), len(d))

        d.set_weights(0)
        d.loc[0, Table._WEIGHTS_COLUMN] = 0.1
        d.loc[10, Table._WEIGHTS_COLUMN] = 0.2
        d.loc[-1, Table._WEIGHTS_COLUMN] = 0.2
        self.assertAlmostEqual(d.weights.sum(), 0.5)
        d.drop(10, axis=0, inplace=True)
        self.assertAlmostEqual(d.weights.sum(), 0.3)
        d.drop(d.index, inplace=True)
        self.assertAlmostEqual(d.weights.sum(), 0)

    def test_set_weights_list(self):
        t = Table('iris')
        w = list(np.random.random(len(t)))
        t.set_weights(w)
        np.testing.assert_almost_equal(t.weights, w)

    def test_set_weights_list_wrong_length(self):
        t = Table('iris')
        w = list(np.random.random(len(t) - 1))
        with self.assertRaises(ValueError):
            t.set_weights(w)

    def test_set_weights_array(self):
        t = Table('iris')
        w = np.random.random(len(t))
        t.set_weights(w)
        np.testing.assert_almost_equal(t.weights, w)

    def test_set_weights_nan(self):
        t = Table('iris')
        with self.assertRaises(ValueError):
            t.set_weights(np.nan)

    def test_set_weights_existing_column(self):
        t = Table('iris')
        t.set_weights("sepal width")
        np.testing.assert_almost_equal(t.weights, t["sepal width"])

    def test_set_weights_nonexisting_column(self):
        t = Table('iris')
        with self.assertRaises(ValueError):
            t.set_weights("this column doesnt exist in iris")

    def test_set_weights_existing_column_with_nan(self):
        t = Table('iris')
        t.loc[t.index[0], "sepal width"] = np.nan
        with self.assertRaises(ValueError):
            t.set_weights("sepal width")

    def test_set_weights_series(self):
        t = Table('iris')
        w = pd.Series(np.random.random(len(t)))
        t.set_weights(w)
        np.testing.assert_almost_equal(t.weights, w.values)

    def test_set_weights_series_with_nan(self):
        t = Table('iris')
        w = pd.Series(np.random.random(len(t)))
        w.iloc[2] = np.nan
        with self.assertRaises(ValueError):
            t.set_weights(w)

    def test_set_weights_invalid(self):
        t = Table('iris')
        with self.assertRaises(TypeError):
            t.set_weights({1: 2, "orange": "cool"})

    def test_has_missing(self):
        d = Table("zoo")
        self.assertFalse(d.has_missing())
        self.assertFalse(d.has_missing_class())

        d.iloc[10, 3] = np.nan
        self.assertTrue(d.has_missing())
        self.assertFalse(d.has_missing_class())

        d.ix[10, d.domain.class_var] = np.nan
        self.assertTrue(d.has_missing())
        self.assertTrue(d.has_missing_class())

        d = Table("test3")
        self.assertFalse(d.has_missing())
        self.assertFalse(d.has_missing_class())

    @staticmethod
    def not_less_ex(ex1, ex2):
        for v1, v2 in zip(ex1, ex2):
            if v1 != v2:
                return v1 < v2
        return True

    @staticmethod
    def sorted(d):
        for i in range(1, len(d)):
            if not TableTestCase.not_less_ex(d[i - 1], d[i]):
                return False
        return True

    @staticmethod
    def not_less_ex_ord(ex1, ex2, ord):
        for a in ord:
            if ex1[a] != ex2[a]:
                return ex1[a] < ex2[a]
        return True

    @staticmethod
    def sorted_ord(d, ord):
        for i in range(1, len(d)):
            if not TableTestCase.not_less_ex_ord(d[i - 1], d[i], ord):
                return False
        return True

    def test_copy_sparse(self):
        data = Table("iris")
        t = Table(
            Domain(data.domain.attributes, [ContinuousVariable("y")]),
            csr_matrix(data.X),
            csr_matrix(np.array([data.Y]).T)
        )
        copy = t.copy()

        self.assertEqual((t.X != copy.X).nnz, 0)      # sparse matrices match by content
        np.testing.assert_equal(t.X.todense(), copy.X.todense())
        np.testing.assert_equal(t.Y.todense(), copy.Y.todense())
        np.testing.assert_equal(t.metas.todense(), copy.metas.todense())

    def test_concatenate_no_tables(self):
        with self.assertRaises(ValueError):
            Table.concatenate([])

    def test_concatenate_single_copies(self):
        t1 = Table('iris')
        t2 = Table.concatenate([t1])
        self.assertTrue(t1.equals(t2))
        self.assertEqual(t1.domain, t2.domain)
        self.assertNotEqual(id(t1), id(t2))

    def test_concatenate_stacking(self):
        t1 = Table(Domain([ContinuousVariable("c1"), ContinuousVariable("c2")]),
                   [[1, 2],
                    [3, 4]])
        t2 = Table(Domain([ContinuousVariable("c3"), ContinuousVariable("c4")]),
                   [[5, 6],
                    [7, 8]])
        rowstack = Table.concatenate([t1, t2], axis=0, rowstack=True)
        self.assertEqual(len(rowstack.columns), 3)  # 2 + weights
        self.assertEqual(id(t1.domain), id(rowstack.domain))
        np.testing.assert_almost_equal(rowstack.X, [[1, 2],
                                                    [3, 4],
                                                    [5, 6],
                                                    [7, 8]])
        colstack = Table.concatenate([t1, t2], axis=1, colstack=True)
        self.assertEqual(len(colstack.columns), 5)  # 4 + weights
        self.assertTrue(all(v in colstack.domain
                            for v in t1.domain.attributes + t2.domain.attributes))
        np.testing.assert_almost_equal(colstack.X, [[1, 2, 5, 6],
                                                    [3, 4, 7, 8]])

    def test_concatenate_stacking_mismatch(self):
        t1 = Table(Domain([ContinuousVariable("c1"), ContinuousVariable("c2")]),
                   [[1, 2],
                    [3, 4]])
        t2 = Table(Domain([ContinuousVariable("c3"), ContinuousVariable("c4"), ContinuousVariable("c5")]),
                   [[5, 6, 0],
                    [7, 8, 0],
                    [0, 0, 0]])
        with self.assertRaises(ValueError):
            Table.concatenate([t1, t2], axis=0, rowstack=True)
        with self.assertRaises(ValueError):
            Table.concatenate([t1, t2], axis=1, colstack=True)

    def test_concatenate_nostacking(self):
        t1 = Table(Domain([ContinuousVariable("c1"), ContinuousVariable("c2")]),
                   [[1, 2],
                    [3, 4]])
        t1.index = [0, 1]
        t2 = Table(Domain([ContinuousVariable("c2"), ContinuousVariable("c3")]),
                   [[5, 6],
                    [7, 8]])
        t2.index = [1, 2]
        rows = Table.concatenate([t1, t2], axis=0, rowstack=False)
        self.assertEqual(len(rows.columns), 4)  # 3 + weights
        self.assertTrue([v in rows.domain for v in ("c1", "c2", "c3")])
        self.assertEqual(len(rows.domain), 3)
        np.testing.assert_almost_equal(rows.X, [[1, 2, np.nan],
                                                [3, 4, np.nan],
                                                [np.nan, 5, 6],
                                                [np.nan, 7, 8]])
        # we can't have duplicate columns now
        t2.domain = Domain([ContinuousVariable("c3"), ContinuousVariable("c4")])
        t2.columns = ["c3", "c4", Table._WEIGHTS_COLUMN]
        cols = Table.concatenate([t1, t2], axis=1, colstack=False)
        self.assertEqual(len(cols.columns), 5)  # 4 + weights
        self.assertTrue(all(v in cols.domain
                            for v in t1.domain.attributes + t2.domain.attributes))
        np.testing.assert_almost_equal(cols.X, [[1, 2, np.nan, np.nan],
                                                [3, 4, 5, 6],
                                                [np.nan, np.nan, 7, 8]])

    def test_concatenate(self):
        d1 = Domain([ContinuousVariable('a1')])
        t1 = Table.from_list(d1, [[1],
                                  [2]])
        d2 = Domain([ContinuousVariable('a2')], metas=[StringVariable('s')])
        t2 = Table.from_list(d2, [[3, 'foo'],
                                  [4, 'fuu']])
        self.assertRaises(ValueError, lambda: Table.concatenate((t1, t2), axis=5))

        t3 = Table.concatenate((t1, t2))
        self.assertEqual(t3.domain.attributes, t1.domain.attributes + t2.domain.attributes)
        self.assertEqual(len(t3.domain.metas), 1)
        self.assertEqual(t3.X.shape, (2, 2))
        self.assertRaises(ValueError, lambda: Table.concatenate((t3, t1)))

        t4 = Table.concatenate((t3, t3), axis=0)
        np.testing.assert_equal(t4.X, [[1, 3],
                                       [2, 4],
                                       [1, 3],
                                       [2, 4]])
        t4 = Table.concatenate((t3, t1), axis=0)
        np.testing.assert_equal(t4.X, [[1, 3],
                                       [2, 4],
                                       [1, np.nan],
                                       [2, np.nan]])

    def test_pickle(self):
        import pickle

        d = Table("zoo")
        s = pickle.dumps(d)
        d2 = pickle.loads(s)
        self.assertEqual(d.attributes, d2.attributes)
        self.assertTrue(d.iloc[0].equals(d2.iloc[0]))

        d = Table("iris")
        s = pickle.dumps(d)
        d2 = pickle.loads(s)
        self.assertEqual(d.name, d2.name)
        self.assertTrue(d.iloc[0].equals(d2.iloc[0]))

    def test_saveTab(self):
        d = Table("iris").iloc[:3]
        d.save("test-save.tab")
        try:
            d2 = Table("test-save.tab")
            for (_, e1), (_, e2) in zip(d.iterrows(), d2.iterrows()):
                self.assertEqual(list(e1), list(e2))
        finally:
            os.remove("test-save.tab")

        dom = Domain([ContinuousVariable("a")])
        d = Table(dom)
        d["a"] = [0, 1, 2]
        d.save("test-save.tab")
        try:
            d2 = Table("test-save.tab")
            self.assertEqual(len(d.domain.attributes), 1)
            self.assertEqual(d.domain.class_var, None)
            for i in range(3):
                self.assertEqual(list(d2.iloc[i]), [i, 1])  # + weights
        finally:
            os.remove("test-save.tab")

        dom = Domain([ContinuousVariable("a")], None)
        d = Table(dom)
        d["a"] = [0, 1, 2]
        d.save("test-save.tab")
        try:
            d2 = Table("test-save.tab")
            self.assertEqual(len(d.domain.attributes), 1)
            for i in range(3):
                self.assertEqual(list(d2.iloc[i]), [i, 1])  # + weights
        finally:
            os.remove("test-save.tab")

        d = Table("zoo")
        d.save("test-zoo.tab")
        dd = Table("test-zoo")
        try:
            self.assertTupleEqual(d.domain.metas, dd.domain.metas, msg="Meta attributes don't match.")
            self.assertTupleEqual(d.domain.variables, dd.domain.variables, msg="Attributes don't match.")

            np.testing.assert_almost_equal(d.weights, dd.weights, err_msg="Weights don't match.")
            for i in range(10):
                for j in d.domain.variables:
                    self.assertEqual(d.iloc[i].loc[j], dd.iloc[i].loc[j])
        finally:
            os.remove("test-zoo.tab")

        d = Table("zoo")
        d.set_weights(range(len(d)))
        d.save("test-zoo-weights.tab")
        dd = Table("test-zoo-weights")
        try:
            self.assertTupleEqual(d.domain.metas, dd.domain.metas, msg="Meta attributes don't match.")
            self.assertTupleEqual(d.domain.variables, dd.domain.variables, msg="Attributes don't match.")

            np.testing.assert_almost_equal(d.weights, dd.weights, err_msg="Weights don't match.")
            for i in range(10):
                for j in d.domain.variables:
                    self.assertEqual(d.iloc[i].loc[j], dd.iloc[i].loc[j])
        finally:
            os.remove("test-zoo-weights.tab")

    def test_save_unknown_extension(self):
        t = Table('iris')
        with self.assertRaises(IOError):
            t.save("iris.extensionfromtheyear3000")

    def test_save_pickle(self):
        table = Table("iris")
        try:
            table.save("iris.pickle")
            table2 = Table.from_file("iris.pickle")
            np.testing.assert_almost_equal(table.X, table2.X)
            np.testing.assert_almost_equal(table.Y, table2.Y)
            self.assertIs(table.domain[0], table2.domain[0])
        finally:
            os.remove("iris.pickle")

    def test_from_numpy(self):
        a = np.arange(16, dtype="d").reshape((4, 4))
        b = np.array(["no", "no", "no", "yes"])
        dom = Domain([ContinuousVariable(x) for x in "abcd"],
                          DiscreteVariable("e", values=["no", "yes"]))
        table = Table(dom, a, b)
        for i in range(4):
            self.assertEqual(table.iloc[i]["e"], "no" if i < 3 else "yes")
            for j in range(5):
                if j < a.shape[1]:
                    self.assertEqual(a[i, j], table.iloc[i, j + 1])
                else:
                    self.assertEqual(b[i], table.iloc[i, j + 1])

    def test_table_dtypes(self):
        table = Table("iris")
        new_domain = Domain(table.domain.attributes, table.domain.class_vars, table.domain.class_vars)
        new_table = Table(new_domain, table)

        self.assertEqual(new_table.X.dtype, np.float64)
        self.assertEqual(new_table.Y.dtype, np.float64)
        self.assertEqual(new_table.metas.dtype, object)

    def test_attributes(self):
        table = Table("iris")
        self.assertEqual(table.attributes, {})
        table.attributes[1] = "test"
        table2 = table[:4]
        self.assertEqual(table2.attributes[1], "test")
        table2.attributes[1] = "modified"
        self.assertEqual(table.attributes[1], "modified")

    def test_get_column_view(self):
        # get_column_view is deprecated
        t = Table('iris')
        with warnings.catch_warnings(record=True) as w:
            cv = t.get_column_view(0)
            self.assertTrue(any(w))
            np.testing.assert_almost_equal(cv[0], t[t.domain[0]].values)
            self.assertFalse(cv[1])

        with warnings.catch_warnings(record=True) as w:
            cv = t.get_column_view("petal length")
            self.assertTrue(any(w))
            np.testing.assert_almost_equal(cv[0], t["petal length"].values)
            self.assertFalse(cv[1])

    def test_density(self):
        t = Table("iris")
        self.assertTrue(t.is_dense)
        self.assertFalse(t.is_sparse)
        self.assertEqual(t.density, 1)
        t.iloc[0, 0] = np.nan
        self.assertLess(t.density, 1)

    # TODO Test conjunctions and disjunctions of conditions


def column_sizes(table):
    return (len(table.domain.attributes),
            len(table.domain.class_vars),
            len(table.domain.metas))


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attributes = ["Feature %i" % i for i in range(10)]
        cls.class_vars = ["Class %i" % i for i in range(1)]
        cls.metas = ["Meta %i" % i for i in range(5)]
        cls.nrows = 10
        cls.row_indices = (1, 5, 7, 9)

        cls.data = np.random.random((cls.nrows, len(cls.attributes)))
        cls.class_data = np.random.random((cls.nrows, len(cls.class_vars)))
        if len(cls.class_vars) == 1:
            cls.class_data = cls.class_data.flatten()
        cls.meta_data = np.random.randint(0, 5, (cls.nrows, len(cls.metas)))
        cls.weight_data = np.random.random((cls.nrows, 1))

    def mock_domain(self, with_classes=False, with_metas=False):
        attributes = [ContinuousVariable(name=a) for a in self.attributes]
        class_vars = [ContinuousVariable(name=c) for c in self.class_vars] if with_classes else []
        metas = [DiscreteVariable(name=m, values=list(map(int, range(5)))) for m in self.metas] if with_metas else []
        variables = attributes + class_vars
        return MagicMock(Domain,
                         attributes=attributes,
                         class_vars=class_vars,
                         metas=metas,
                         variables=variables)

    def create_domain(self, attributes=(), classes=(), metas=()):
        attr_vars = [ContinuousVariable(name=a) if isinstance(a, str)
                     else a for a in attributes]
        class_vars = [ContinuousVariable(name=c) if isinstance(c, str)
                      else c for c in classes]
        meta_vars = [DiscreteVariable(name=m, values=list(map(int, range(5))))
                     if isinstance(m, str) else m for m in metas]
        return Domain(attr_vars, class_vars, meta_vars)

    def assert_discretes_are_categoricals(self, table):
        for v in chain(table.domain.variables, table.domain.metas):
            if isinstance(v, DiscreteVariable):
                self.assertIsInstance(table[v.name].dtype, pd.types.dtypes.CategoricalDtype)


class CreateEmptyTable(TableTests):
    def test_calling_new_with_no_parameters_constructs_a_new_instance(self):
        table = Table()
        self.assertIsInstance(table, Table)

    def test_table_has_file(self):
        table = Table()
        self.assertIsNone(table.__file__)


class CreateTableWithFilename(TableTests):
    filename = "data.tab"

    @patch("os.path.exists", Mock(return_value=True))
    def test_read_data_calls_reader(self):
        table_mock = Mock(Table)
        reader_instance = Mock(read=Mock(return_value=table_mock))
        reader_mock = Mock(return_value=reader_instance)

        with patch.dict(FileFormat.readers,
                        {'.xlsx': reader_mock}):
            table = Table.from_file("test.xlsx")

        reader_mock.assert_called_with("test.xlsx")
        reader_instance.read.assert_called_with()
        self.assertEqual(table, table_mock)

    @patch("os.path.exists", Mock(return_value=False))
    def test_raises_error_if_file_does_not_exist(self):
        with self.assertRaises(IOError):
            Table.from_file(self.filename)

    @patch("os.path.exists", Mock(return_value=True))
    def test_raises_error_if_file_has_unknown_extension(self):
        with self.assertRaises(IOError):
            Table.from_file("file.invalid_extension")

    @patch("Orange.data.Table.from_file")
    def test_calling_new_with_string_argument_calls_read_data(self, read_data):
        Table(self.filename)

        read_data.assert_called_with(self.filename)

    @patch("Orange.data.Table.from_file")
    def test_calling_new_with_keyword_argument_filename_calls_read_data(
            self, read_data):
        Table(self.filename)

        read_data.assert_called_with(self.filename)


class CreateTableWithUrl(TableTests):
    def test_load_from_url(self):
        d1 = Table('iris')
        d2 = Table('https://raw.githubusercontent.com/biolab/orange3/master/Orange/datasets/iris.tab')
        np.testing.assert_array_equal(d1.X, d2.X)
        np.testing.assert_array_equal(d1.Y, d2.Y)

    class _MockUrlOpen(MagicMock):
        headers = {'content-disposition': 'attachment; filename="Something-FormResponses.tsv"; '
                                          'filename*=UTF-8''Something%20%28Responses%29.tsv'}
        url = 'https://docs.google.com/spreadsheets/d/ABCD/edit'

        def __enter__(self): return self

        def __exit__(self, *args, **kwargs): pass

        def read(self): return b'''\
a\tb\tc
1\t2\t3
2\t3\t4'''

    urlopen = _MockUrlOpen()

    @patch('Orange.data.io.urlopen', urlopen)
    def test_google_sheets(self):
        d = Table(self.urlopen.url)
        self.urlopen.assert_called_with('https://docs.google.com/spreadsheets/d/ABCD/export?format=tsv',
                                        timeout=10)
        self.assertEqual(len(d), 2)
        self.assertEqual(d.name, 'Something-FormResponses')


class CreateTableWithDomain(TableTests):
    def test_creates_an_empty_table_with_given_domain(self):
        domain = self.mock_domain()
        table = Table.from_domain(domain)
        self.assertEqual(table.domain, domain)
        self.assert_discretes_are_categoricals(table)

    @patch("Orange.data.Table.from_domain")
    def test_calling_new_with_domain_calls_new_from_domain(
            self, new_from_domain):
        domain = self.mock_domain()
        Table(domain)

        new_from_domain.assert_called_with(domain)


class CreateTableWithData(TableTests):
    def test_creates_a_table_with_given_X(self):
        # from numpy
        table = Table(np.array(self.data))
        self.assertIsInstance(table.domain, Domain)
        np.testing.assert_almost_equal(table.X, self.data)
        self.assert_discretes_are_categoricals(table)

        # from list
        table = Table(list(self.data))
        self.assertIsInstance(table.domain, Domain)
        np.testing.assert_almost_equal(table.X, self.data)
        self.assert_discretes_are_categoricals(table)

    def test_creates_a_table_from_domain_and_list(self):
        domain = Domain([DiscreteVariable(name="a", values="mf"),
                              ContinuousVariable(name="b")],
                             DiscreteVariable(name="y", values="abc"))
        table = Table(domain, [["m", 1, "c"],
                               ["f", 2, "?"],
                               ["m", 3, "a"],
                               ["?", "?", "c"]])
        self.assertIs(table.domain, domain)
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(
            table.X, np.array([[0, 1], [1, 2], [0, 3], [np.nan, np.nan]]))
        np.testing.assert_almost_equal(table.Y, np.array([2, np.nan, 0, 2]))

    def test_creates_a_table_from_domain_and_list_non_rectangular(self):
        domain = Domain([DiscreteVariable(name="a", values="mf"),
                         ContinuousVariable(name="b")],
                        DiscreteVariable(name="y", values="abc"))
        with self.assertRaises(ValueError):
            table = Table(domain, [["m", 1, "c"],
                                   ["f", 2],
                                   ["m", 3, "a", "f"],
                                   ["?", "?", "c"]])

    def test_creates_a_table_from_domain_and_list_column_mismatch(self):
        domain = Domain([DiscreteVariable(name="a", values="mf")],
                        DiscreteVariable(name="y", values="abc"))
        with self.assertRaises(ValueError):
            table = Table(domain, [["m", 1, "c"],
                                   ["f", 2, "a"],
                                   ["m", 3, "a"],
                                   ["?", "?", "c"]])

    def test_creates_a_table_from_domain_and_list_and_weights(self):
        domain = Domain([DiscreteVariable(name="a", values="mf"),
                              ContinuousVariable(name="b")],
                             DiscreteVariable(name="y", values="abc"))
        table = Table(domain, [["m", 1, "c"],
                               ["f", 2, "?"],
                               ["m", 3, "a"],
                               ["?", "?", "c"]], [1, 2, 3, 4])
        self.assertIs(table.domain, domain)
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(
            table.X, np.array([[0, 1], [1, 2], [0, 3], [np.nan, np.nan]]))
        np.testing.assert_almost_equal(table.Y, np.array([2, np.nan, 0, 2]))
        np.testing.assert_almost_equal(table.weights, np.array([1, 2, 3, 4]))

    def test_creates_a_table_from_domain_and_list_and_weights_mismatched(self):
        domain = Domain([DiscreteVariable(name="a", values="mf"),
                         ContinuousVariable(name="b")],
                        DiscreteVariable(name="y", values="abc"))
        with self.assertRaises(ValueError):
            table = Table(domain, [["m", 1, "c"],
                                   ["f", 2, "?"],
                                   ["m", 3, "a"],
                                   ["?", "?", "c"]], [1, 2, 3])

    def test_creates_a_table_from_domain_and_list_and_metas(self):
        metas = [DiscreteVariable("Meta 1", values="XYZ"),
                 ContinuousVariable("Meta 2"),
                 StringVariable("Meta 3")]
        domain = Domain([DiscreteVariable(name="a", values="mf"),
                              ContinuousVariable(name="b")],
                             DiscreteVariable(name="y", values="abc"),
                             metas=metas)
        table = Table(domain, [["m", 1, "c", "X", 2, "bb"],
                                    ["f", 2, "?", "Y", 1, "aa"],
                                    ["m", 3, "a", "Z", 3, "bb"],
                                    ["?", "?", "c", "X", 1, "aa"]])
        self.assertIs(table.domain, domain)
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(
            table.X, np.array([[0, 1], [1, 2], [0, 3], [np.nan, np.nan]]))
        np.testing.assert_almost_equal(table.Y, np.array([2, np.nan, 0, 2]))
        np.testing.assert_array_equal(table.metas,
                                      np.array([[0, 2., "bb"],
                                                [1, 1., "aa"],
                                                [2, 3., "bb"],
                                                [0, 1., "aa"]],
                                               dtype=object))

    def test_creates_a_table_from_list_of_instances(self):
        table = Table('iris')
        # skip weights
        new_table = Table(table.domain, [[v for i, v in enumerate(list(row)) if i != len(table.columns) - 1]
                                              for _, row in table.iterrows()])
        self.assertIs(table.domain, new_table.domain)
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(table.X, new_table.X)
        np.testing.assert_almost_equal(table.Y, new_table.Y)
        np.testing.assert_almost_equal(table.weights, new_table.weights)
        self.assertEqual(table.domain, new_table.domain)
        np.testing.assert_array_equal(table.metas, new_table.metas)

    def test_creates_a_table_from_list_of_instances_metas(self):
        table = Table('zoo')
        # skip weights
        new_table = Table(Domain(attributes=[], metas=table.domain.metas), table.metas.tolist())
        self.assertEqual(table.domain.metas, new_table.domain.metas)
        self.assert_discretes_are_categoricals(table)
        self.assertEqual(new_table.X.size, 0)
        self.assertEqual(new_table.Y.size, 0)
        np.testing.assert_array_equal(table.metas, new_table.metas)

    def test_creates_a_table_with_domain_and_given_X(self):
        domain = self.mock_domain()

        table = Table(domain, self.data)
        self.assertIsInstance(table.domain, Domain)
        self.assert_discretes_are_categoricals(table)
        self.assertEqual(table.domain, domain)
        np.testing.assert_almost_equal(table.X, self.data)

    def test_creates_a_table_with_given_X_and_Y(self):
        table = Table(self.data, self.class_data)

        self.assertIsInstance(table.domain, Domain)
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)

    def test_creates_a_table_with_given_X_Y_and_metas(self):
        table = Table(self.data, self.class_data, self.meta_data)

        self.assertIsInstance(table.domain, Domain)
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)
        np.testing.assert_almost_equal(table.metas, self.meta_data)

    def test_creates_a_discrete_class_if_Y_has_few_distinct_values(self):
        Y = np.array([float(np.random.randint(0, 2)) for i in self.data])
        table = Table(self.data, Y, self.meta_data)

        np.testing.assert_almost_equal(table.Y, Y)
        self.assertIsInstance(table.domain.class_vars[0], DiscreteVariable)

    def test_creates_a_table_with_given_domain(self):
        domain = self.mock_domain()
        table = Table.from_numpy(domain, self.data)

        self.assertEqual(table.domain, domain)
        self.assert_discretes_are_categoricals(table)

    def test_sets_Y_if_given(self):
        domain = self.mock_domain(with_classes=True)
        table = Table.from_numpy(domain, self.data, self.class_data)

        np.testing.assert_almost_equal(table.Y, self.class_data)
        self.assert_discretes_are_categoricals(table)

    def test_sets_metas_if_given(self):
        domain = self.mock_domain(with_metas=True)
        table = Table.from_numpy(domain, self.data, metas=self.meta_data)

        np.testing.assert_almost_equal(table.metas, self.meta_data)
        self.assert_discretes_are_categoricals(table)

    def test_sets_weights_if_given(self):
        domain = self.mock_domain()
        table = Table.from_numpy(domain, self.data, weights=self.weight_data)

        np.testing.assert_equal(table.weights.shape, (len(self.data), ))
        self.assert_discretes_are_categoricals(table)
        np.testing.assert_almost_equal(table.weights.flatten(), self.weight_data.flatten())

    def test_initializes_Y_metas_and_W_if_not_given(self):
        domain = self.mock_domain()
        table = Table.from_numpy(domain, self.data)

        self.assertEqual(table.Y.shape, (self.nrows, len(domain.class_vars)))
        self.assertEqual(table.metas.shape, (self.nrows, len(domain.metas)))
        self.assertEqual(table.weights.shape, (self.nrows,))
        self.assert_discretes_are_categoricals(table)

    def test_raises_error_if_columns_in_domain_and_data_do_not_match(self):
        domain = self.mock_domain(with_classes=True, with_metas=True)
        ones = np.zeros((self.nrows, 1))

        with self.assertRaises(ValueError):
            data_ = np.hstack((self.data, ones))
            Table.from_numpy(domain, data_, self.class_data,
                                  self.meta_data)

        with self.assertRaises(ValueError):
            classes_ = np.hstack((self.class_data, ones))
            Table.from_numpy(domain, self.data, classes_,
                                  self.meta_data)

        with self.assertRaises(ValueError):
            metas_ = np.hstack((self.meta_data, ones))
            Table.from_numpy(domain, self.data, self.class_data,
                                  metas_)

    def test_raises_error_if_lengths_of_data_do_not_match(self):
        domain = self.mock_domain(with_classes=True, with_metas=True)

        with self.assertRaises(ValueError):
            data_ = np.vstack((self.data, np.zeros((1, len(self.attributes)))))
            Table(domain, data_, self.class_data, self.meta_data)

        with self.assertRaises(ValueError):
            class_data_ = np.vstack((self.class_data,
                                     np.zeros((1, len(self.class_vars)))))
            Table(domain, self.data, class_data_, self.meta_data)

        with self.assertRaises(ValueError):
            meta_data_ = np.vstack((self.meta_data,
                                    np.zeros((1, len(self.metas)))))
            Table(domain, self.data, self.class_data, meta_data_)

    @patch("Orange.data.Table.from_numpy")
    def test_calling_new_with_domain_and_numpy_arrays_calls_new_from_numpy(
            self, new_from_numpy):
        domain = self.mock_domain()
        Table(domain, self.data)
        new_from_numpy.assert_called_with(domain, self.data)

        domain = self.mock_domain(with_classes=True)
        Table(domain, self.data, self.class_data)
        new_from_numpy.assert_called_with(domain, self.data, self.class_data)

        domain = self.mock_domain(with_classes=True, with_metas=True)
        Table(domain, self.data, self.class_data, self.meta_data)
        new_from_numpy.assert_called_with(
            domain, self.data, self.class_data, self.meta_data)

        Table(domain, self.data, self.class_data,
                   self.meta_data, self.weight_data)
        new_from_numpy.assert_called_with(domain, self.data, self.class_data,
                                          self.meta_data, self.weight_data)

    def test_from_numpy_reconstructable(self):
        def assert_equal(T1, T2):
            np.testing.assert_array_equal(T1.X, T2.X)
            np.testing.assert_array_equal(T1.Y, T2.Y)
            np.testing.assert_array_equal(T1.metas, T2.metas)
            np.testing.assert_array_equal(T1.weights, T2.weights)

        nullcol = np.empty((self.nrows, 0))
        domain = self.create_domain(self.attributes)
        table = Table(domain, self.data)
        self.assert_discretes_are_categoricals(table)

        table_1 = Table.from_numpy(
            domain, table.X, table.Y, table.metas, table.weights)
        assert_equal(table, table_1)
        self.assert_discretes_are_categoricals(table_1)

        domain = self.create_domain(classes=self.class_vars)
        table = Table(domain, nullcol, self.class_data)
        self.assert_discretes_are_categoricals(table)

        table_1 = Table.from_numpy(
            domain, table.X, table.Y, table.metas, table.weights)
        self.assert_discretes_are_categoricals(table_1)
        assert_equal(table, table_1)

        domain = self.create_domain(metas=self.metas)
        table = Table(domain, nullcol, nullcol, self.meta_data)
        self.assert_discretes_are_categoricals(table)

        table_1 = Table.from_numpy(
            domain, table.X, table.Y, table.metas, table.weights)
        self.assert_discretes_are_categoricals(table_1)
        assert_equal(table, table_1)

    def test_from_numpy_mismatched_columns(self):
        domain = self.create_domain(self.attributes, self.class_vars)
        with self.assertRaises(ValueError):
            t = Table(domain, self.data, None)


class CreateTableWithDomainAndTable(TableTests):
    interesting_slices = [
        slice(0, 0),  # [0:0] - empty slice
        slice(1),  # [:1]  - only first element
        slice(1, None),  # [1:]  - all but first
        slice(-1, None),  # [-1:] - only last element
        slice(-1),  # [:-1] - all but last
        slice(None),  # [:]   - all elements
        slice(None, None, 2),  # [::2] - even elements
        slice(None, None, -1),  # [::-1]- all elements reversed
    ]

    row_indices = [1, 5, 6, 7]

    def setUp(self):
        super(CreateTableWithDomainAndTable, self).setUp()
        self.domain = self.create_domain(
            self.attributes, self.class_vars, self.metas)
        self.table = Table(
            self.domain, self.data, self.class_data, self.meta_data)

    def test_creates_table_with_given_domain(self):
        new_table = Table.from_table(self.table.domain, self.table)

        self.assertIsInstance(new_table, Table)
        self.assert_discretes_are_categoricals(new_table)
        self.assertIsNot(self.table, new_table)
        self.assertEqual(new_table.domain, self.domain)

    def test_creates_table_with_empty_domain(self):
        new_table = Table.from_table(Domain(attributes=[]), self.table)
        self.assertEqual(len(new_table), 0)

    def test_can_copy_table(self):
        new_table = Table.from_table(self.domain, self.table)
        self.assertTrue(new_table.equals(self.table))

    def test_can_change_attribute_roles(self):
        x = self.table.domain.attributes[:3]
        y = self.table.domain.attributes[3:5]
        meta = self.table.domain.attributes[5:]
        new_domain = Domain(x, y, meta)
        new_table = Table(new_domain, self.table)
        self.assert_discretes_are_categoricals(new_table)

        self.assertEqual(new_table.X.shape[1], len(x))
        self.assertEqual(new_table.Y.shape[1], len(y))
        self.assertEqual(new_table.metas.shape[1], len(meta))
        self.assertAlmostEqual(new_table.X.sum(), self.table[x].values.sum() - self.table.weights.sum())
        self.assertAlmostEqual(new_table.Y.sum(), self.table[y].values.sum() - self.table.weights.sum())
        self.assertAlmostEqual(new_table.metas.sum(), self.table[meta].values.sum() - self.table.weights.sum())

    def test_creates_table_with_given_domain_and_row_filter(self):
        new_domain = Domain(self.table.domain.attributes[:2], [], [])
        new_table = Table.from_table(new_domain, self.table, [2, 3, 8])
        np.testing.assert_array_almost_equal(self.table.X[[2, 3, 8], :2], new_table.X)

    def test_from_table_on_sparse_data(self):
        iris = Table("iris")
        iris = Table(
            Domain(iris.domain.attributes),
            csr_matrix(iris.X)
        )

        new_domain = Domain(iris.domain.attributes[:2], iris.domain.class_vars,
                                        iris.domain.metas, source=iris.domain)
        new_iris = SparseTable.from_table(new_domain, iris)
        self.assertTrue(issparse(new_iris.X))
        self.assertEqual(new_iris.X.shape[1], 2)
        self.assertEqual(len(new_iris.domain.attributes), 2)

        all_vars = chain(iris.domain.variables, iris.domain.metas)
        n_all = len(iris.domain) + len(iris.domain.metas)
        new_domain = Domain([], [], all_vars, source=iris.domain)
        new_iris = Table.from_table(new_domain, iris)
        self.assertEqual(len(new_iris.domain.metas), n_all)
        self.assertEqual(new_iris.metas.shape[1], n_all)


def isspecial(s):
    return isinstance(s, slice) or s is Ellipsis


def split_columns(indices, t):
    a, c, m = column_sizes(t)
    if indices is ...:
        return slice(a), slice(c), slice(m)
    elif isinstance(indices, slice):
        return indices, slice(0, 0), slice(0, 0)
    elif not isinstance(indices, list) and not isinstance(indices, tuple):
        indices = [indices]
    return (
        [t.domain.index(x)
         for x in indices if 0 <= t.domain.index(x) < a] or slice(0, 0),
        [t.domain.index(x) - a
         for x in indices if t.domain.index(x) >= a] or slice(0, 0),
        [-t.domain.index(x) - 1
         for x in indices if t.domain.index(x) < 0] or slice(0, 0))


def getname(variable):
    return variable.name


class TableIndexingTests(TableTests):
    def setUp(self):
        super().setUp()
        d = self.domain = \
            self.create_domain(self.attributes, self.class_vars, self.metas)
        t = self.table = \
            Table(self.domain, self.data, self.class_data, self.meta_data)
        self.magic_table = \
            np.column_stack((self.table.X, self.table.Y,
                             self.table.metas[:, ::-1]))

        self.rows = [0, -1]
        self.multiple_rows = [slice(0, 0), ..., slice(1, -1, -1)]
        a, c, m = column_sizes(t)
        columns = [0, a - 1, a, a + c - 1, -1, -m]
        self.columns = chain(columns,
                             map(lambda x: d[x], columns),
                             map(lambda x: d[x].name, columns))
        self.multiple_columns = chain(
            self.multiple_rows,
            [d.attributes, d.class_vars, d.metas, [0, a, -1]],
            [self.attributes, self.class_vars, self.metas],
            [self.attributes + self.class_vars + self.metas])

    def test_can_select_a_single_value(self):
        for i, attr in enumerate(self.table.domain.attributes):
            for j in range(len(self.table)):
                self.assertAlmostEqual(self.table[attr].iloc[j], self.data[j, i])

    def test_can_select_a_single_row(self):
        for r in self.rows:
            row = self.table.iloc[r][cols_wo_weights(self.table)]
            new_row = np.hstack(
                (self.data[r, :],
                 self.class_data[r, None],
                 self.meta_data[r, :]))
            np.testing.assert_almost_equal(
                np.array(list(row)), new_row)

    def test_can_select_a_subset_of_rows_and_columns(self):
        subset = self.table[self.table.domain.attributes[2:4]].iloc[3:7].values[:, :-1]  # last column are weights
        np.testing.assert_almost_equal(subset, self.data[3:7, 2:4])


class TableElementAssignmentTest(TableTests):
    def setUp(self):
        super(TableElementAssignmentTest, self).setUp()
        self.domain = self.create_domain(self.attributes, self.class_vars, self.metas)
        self.table = Table(self.domain, self.data, self.class_data, self.meta_data)

    def test_can_assign_values(self):
        self.table.loc[self.table.index[0], self.domain.attributes[0]] = 123.
        self.assertAlmostEqual(self.table.X[0, 0], 123.)

    def test_can_assign_values_to_classes(self):
        a, c, m = column_sizes(self.table)
        self.table.loc[self.table.index[0], self.domain[a]] = 42.
        self.assertAlmostEqual(self.table.Y[0], 42.)

    def test_cannot_assign_values_to_categorical(self):
        with self.assertRaises(ValueError):
            self.table.iloc[0, -1] = 42

    def test_can_assign_rows_to_rows(self):
        self.table.iloc[0] = self.table.iloc[1]
        np.testing.assert_almost_equal(
            self.table.X[0], self.table.X[1])
        np.testing.assert_almost_equal(
            self.table.Y[0], self.table.Y[1])
        np.testing.assert_almost_equal(
            self.table.metas[0], self.table.metas[1])

    def test_can_assign_lists(self):
        a, c, m = column_sizes(self.table)
        new_example = [0 for _ in range(len(self.attributes + self.class_vars + self.metas) + 1)]
        self.table.iloc[0] = new_example
        np.testing.assert_almost_equal(
            self.table.X[0], np.array(new_example[:a]))
        np.testing.assert_almost_equal(
            self.table.Y[0], np.array(new_example[a:]))

    def test_can_assign_np_array(self):
        a, c, m = column_sizes(self.table)
        new_example = np.array([0 for _ in range(len(self.attributes + self.class_vars + self.metas) + 1)])
        self.table.iloc[0] = new_example
        np.testing.assert_almost_equal(self.table.X[0], new_example[:a])
        np.testing.assert_almost_equal(self.table.Y[0], new_example[a:])


class InterfaceTest(unittest.TestCase):
    """
    Basic tests each implementation of Table should pass.
    Does not test functionality completely handled by pandas.
    """

    features = (
        ContinuousVariable(name="Continuous Feature 1"),
        ContinuousVariable(name="Continuous Feature 2"),
        DiscreteVariable(name="Discrete Feature 1", values=[0, 1]),
        DiscreteVariable(name="Discrete Feature 2", values=["value1", "value2"]),
    )

    class_vars = (
        ContinuousVariable(name="Continuous Class"),
        DiscreteVariable(name="Discrete Class", values=[0, 1])
    )

    feature_data = [
        [1, 0, 0, "value1"],
        [0, 1, 0, "value1"],
        [0, 0, 1, "value1"],
        [0, 0, 0, "value2"],
    ]

    class_data = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]

    data = list(list(a + c) for a, c in zip(feature_data, class_data))

    nrows = 4

    def setUp(self):
        self.domain = Domain(attributes=self.features, class_vars=self.class_vars)
        self.table = Table.from_list(self.domain, self.data)

    def test_iteration(self):
        for (idx, row), expected_data in zip(self.table.iterrows(), self.data):
            self.assertEqual(list(row[cols_wo_weights(self.table)]), expected_data)

    def test_row_indexing(self):
        for i in range(self.nrows):
            self.assertEqual(list(self.table.iloc[i][cols_wo_weights(self.table)]), self.data[i])

    def test_value_indexing(self):
        for i in range(self.nrows):
            for j, c in enumerate(cols_wo_weights(self.table)):
                self.assertEqual(self.table.iloc[i][c], self.data[i][j])

    def test_row_assignment(self):
        for i in range(self.nrows):
            new_row = [2, 3, 1, "value2", 1, 1] + [1]  # additional item: weight
            self.table.iloc[i] = new_row
            self.assertEqual(list(self.table.iloc[i]), new_row)

    def test_delete_rows(self):
        for i in range(self.nrows):
            self.table = self.table.iloc[1:]
            for j in range(len(self.table)):
                self.assertEqual(list(self.table.iloc[j][cols_wo_weights(self.table)]), self.data[i + j + 1])

    def test_subclasses(self):
        from pathlib import Path

        # also TablePanel, but we don't need that here
        # see pandas' subclassing docs and our impl for more info

        class _ExtendedSeries(TableSeries):
            @property
            def _constructor(self):
                return _ExtendedSeries

            @property
            def _constructor_expanddim(self):
                return _ExtendedTable

        class _ExtendedTable(Table):
            @property
            def _constructor(self):
                return _ExtendedTable

            @property
            def _constructor_sliced(self):
                return _ExtendedSeries

        data_file = _ExtendedTable('iris')
        data_url = _ExtendedTable.from_url(
            Path(os.path.dirname(__file__), 'test1.tab').as_uri())

        self.assertIsInstance(data_file, _ExtendedTable)
        self.assertIsInstance(data_url, _ExtendedTable)


class TestPandasInteraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # test assertions: a must bot have weight set here
        cls.a = Table([[1, 2, 3], [3, 4, 5]])
        cls.b = Table([[1, 2], [3, 4], [5, 6]])
        cls.b.set_weights(np.random.random(len(cls.b)))

    def test_unique_index_across_tables(self):
        self.assertEqual(len(self.a) + len(self.b), len(set(self.a.index).union(set(self.b.index))))

    def test_unique_index_from_empty_table(self):
        c = Table()
        c["foo"] = [1, 2, 3, 4, 5]
        self.assertEqual(len(self.a) + len(self.b) + len(c),
                         len(set(self.a.index).union(set(self.b.index)).union(set(c.index))))

    def test_default_weights_from_constructed_table(self):
        np.testing.assert_array_equal(np.repeat(1, len(self.a)), self.a.weights)

    def test_default_weights_from_empty_table(self):
        c = Table()
        c["foo"] = [1, 2]
        np.testing.assert_array_equal(np.repeat(1, len(c)), c.weights)

    def test_weights_transfer_on_multi_column_subset(self):
        subset = self.b[self.b.columns[1:]]
        np.testing.assert_array_equal(self.b.weights, subset.weights)

    def test_weights_transfer_on_multi_column_subset_single(self):
        subset = self.b[[self.b.columns[1]]]
        np.testing.assert_array_equal(self.b.weights, subset.weights)

    def test_weights_transfer_on_empty_column_subset(self):
        subset = self.b[[]]
        np.testing.assert_array_equal(self.b.weights, subset.weights)

    def test_series_have_weights(self):
        s = self.b.iloc[0]
        np.testing.assert_almost_equal(s.weights, self.b.weights[0])

    def test_merge(self):
        merged = self.a.merge(self.b, left_on=self.a.domain[0], right_on=self.b.domain[0])
        self.assertEqual(len(merged.columns), len(self.a.columns) + len(self.b.columns) - 1)  # -1 internal dup
        self.assertEqual(len(self.a.domain) + len(self.b.domain), len(merged.columns) - 1)

    def test_iterrows_wrapper_needed(self):
        # see TableBase.iterrows, remove the wrapper and this test when this fails
        class DFS(pd.DataFrame):
            @property
            def _constructor(self):
                return DFS

            @property
            def _constructor_sliced(self):
                return SS

        class SS(pd.Series):
            @property
            def _constructor(self):
                return SS

            @property
            def _constructor_expanddim(self):
                return DFS

        dfs = DFS([[1, 2, 3], [4, 5, 6]])
        _, row = next(dfs.iterrows())
        self.assertEqual(row.__class__, pd.Series)  # proper behaviour is SS

if __name__ == "__main__":
    unittest.main()


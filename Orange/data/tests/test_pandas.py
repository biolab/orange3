# pylint: disable=import-outside-toplevel
import unittest
from datetime import date, datetime, timezone

import numpy as np
import pytz
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from Orange.data import ContinuousVariable, DiscreteVariable, TimeVariable, Table, Domain, \
    StringVariable
from Orange.data.pandas_compat import OrangeDataFrame
from Orange.data.tests.test_variable import TestTimeVariable

try:
    import pandas as pd
except ImportError:
    pd = None


@unittest.skipIf(pd is None, "Missing package 'pandas'")
class TestPandasCompat(unittest.TestCase):
    def test_table_from_frame(self):
        from Orange.data.pandas_compat import table_from_frame

        nan = np.nan
        df = pd.DataFrame([['a', 1, pd.Timestamp('2017-12-19')],
                           ['b', 0, pd.Timestamp('1724-12-20')],
                           ['c', 0, pd.Timestamp('1724-12-20')],
                           [nan, nan, nan]])
        table = table_from_frame(df)
        np.testing.assert_equal(table.X,
                                [[1, pd.Timestamp('2017-12-19').timestamp()],
                                 [0, pd.Timestamp('1724-12-20').timestamp()],
                                 [0, pd.Timestamp('1724-12-20').timestamp()],
                                 [nan, nan]])
        np.testing.assert_equal(table.metas.tolist(), [['a'],
                                                       ['b'],
                                                       ['c'],
                                                       [nan]])
        names = [var.name for var in table.domain.attributes]
        types = [type(var) for var in table.domain.attributes]
        self.assertEqual(names, ['1', '2'])
        self.assertEqual(types, [ContinuousVariable, TimeVariable])

        # Force strings nominal
        table = table_from_frame(df, force_nominal=True)
        np.testing.assert_equal(table.X, [[0, 1, pd.Timestamp('2017-12-19').timestamp()],
                                          [1, 0, pd.Timestamp('1724-12-20').timestamp()],
                                          [2, 0, pd.Timestamp('1724-12-20').timestamp()],
                                          [nan, nan, nan]])
        np.testing.assert_equal(table.metas.tolist(), [[], [], [], []])
        names = [var.name for var in table.domain.attributes]
        types = [type(var) for var in table.domain.attributes]
        self.assertEqual(names, ['0', '1', '2'])
        self.assertEqual(types, [DiscreteVariable, ContinuousVariable, TimeVariable])

        # Include index
        df.index = list('abaa')
        table = table_from_frame(df)
        np.testing.assert_equal(table.X,
                                [[0, 1, pd.Timestamp('2017-12-19').timestamp()],
                                 [1, 0, pd.Timestamp('1724-12-20').timestamp()],
                                 [0, 0, pd.Timestamp('1724-12-20').timestamp()],
                                 [0, nan, nan]])
        np.testing.assert_equal(table.metas.tolist(), [['a'],
                                                       ['b'],
                                                       ['c'],
                                                       [nan]])
        names = [var.name for var in table.domain.attributes]
        types = [type(var) for var in table.domain.attributes]
        self.assertEqual(names, ['index', '1', '2'])
        self.assertEqual(types, [DiscreteVariable, ContinuousVariable, TimeVariable])

    def test_table_from_frame_keep_ids(self):
        """ Test if indices are correctly transferred to Table"""
        from Orange.data.pandas_compat import table_from_frame
        df = OrangeDataFrame(Table('iris')[:6])
        df.index = [1, "_oa", "_o", "1", "_o20", "_o30"]
        table = table_from_frame(df)
        self.assertEqual(table.ids[-2:].tolist(), [20, 30])
        self.assertTrue(np.issubdtype(table.ids.dtype, np.number))

    def test_table_to_frame(self):
        from Orange.data.pandas_compat import table_to_frame
        table = Table("iris")
        df = table_to_frame(table)
        table_column_names = [var.name for var in table.domain.variables]
        frame_column_names = df.columns

        self.assertEqual(sorted(table_column_names), sorted(frame_column_names))
        self.assertEqual(type(df['iris'].dtype), pd.api.types.CategoricalDtype)
        self.assertEqual(list(df['sepal length'])[0:4], [5.1, 4.9, 4.7, 4.6])
        self.assertEqual(list(df['iris'])[0:2], ['Iris-setosa', 'Iris-setosa'])

    def test_table_to_frame_object_dtype(self):
        from Orange.data.pandas_compat import table_to_frame

        domain = Domain([], metas=[ContinuousVariable("a", number_of_decimals=0)])
        table = Table.from_numpy(
            domain, np.empty((10, 0)), metas=np.ones((10, 1), dtype=object)
        )

        df = table_to_frame(table, include_metas=True)
        self.assertEqual(["a"], df.columns)
        np.testing.assert_array_equal(df["a"].values, np.ones((10,)))

    def test_table_to_frame_nans(self):
        from Orange.data.pandas_compat import table_to_frame
        domain = Domain(
            [ContinuousVariable("a", number_of_decimals=0), ContinuousVariable("b")]
        )
        table = Table(
            domain, np.column_stack((np.ones(10), np.hstack((np.ones(9), [np.nan]))))
        )

        df = table_to_frame(table)
        table_column_names = [var.name for var in table.domain.variables]
        frame_column_names = df.columns

        self.assertEqual(sorted(table_column_names), sorted(frame_column_names))
        self.assertEqual(df["a"].dtype, int)
        self.assertEqual(df["b"].dtype, float)
        self.assertEqual([1, 1, 1], list(df["a"].iloc[-3:]))
        self.assertTrue(np.isnan(df["b"].iloc[-1]))

    def test_table_to_frame_metas(self):
        from Orange.data.pandas_compat import table_to_frame

        table = Table("zoo")
        domain = table.domain

        df = table_to_frame(table)
        cols = pd.Index([var.name for var in domain.variables])
        pd.testing.assert_index_equal(df.columns, cols)

        df = table_to_frame(table, include_metas=True)
        cols = pd.Index([var.name for var in domain.variables + domain.metas])
        pd.testing.assert_index_equal(df.columns, cols)

    def test_not_orangedf(self):
        table = Table("iris")
        xdf, ydf, mdf = table.to_pandas_dfs()
        xtable = xdf.to_orange_table()
        ytable = ydf.to_orange_table()
        mtable = mdf.to_orange_table()

        np.testing.assert_array_equal(table.X, xtable.X)
        np.testing.assert_array_equal(table.Y, ytable.Y)
        np.testing.assert_array_equal(table.metas, mtable.metas)
        np.testing.assert_array_equal(table.W, xtable.W)
        np.testing.assert_array_equal(table.W, ytable.W)
        np.testing.assert_array_equal(table.W, mtable.W)
        self.assertEqual(table.attributes, xtable.attributes)
        self.assertEqual(table.attributes, ytable.attributes)
        self.assertEqual(table.attributes, mtable.attributes)
        np.testing.assert_array_equal(table.ids, xtable.ids)
        np.testing.assert_array_equal(table.ids, ytable.ids)
        np.testing.assert_array_equal(table.ids, mtable.ids)

        d1 = table.domain
        d2 = xtable.domain

        vars1 = d1.variables + d1.metas
        vars2 = d2.variables + d2.metas

        for v1, v2 in zip(vars1, vars2):
            self.assertEqual(type(v1), type(v2))

    def test_table_from_frame_date(self):
        from Orange.data.pandas_compat import table_from_frame

        df = pd.DataFrame(
            [[pd.Timestamp("2017-12-19")], [pd.Timestamp("1724-12-20")], [np.nan]]
        )
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19").timestamp()],
                [pd.Timestamp("1724-12-20").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 0)
        self.assertEqual(table.domain.variables[0].have_date, 1)

        df = pd.DataFrame([["2017-12-19"], ["1724-12-20"], [np.nan]])
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19").timestamp()],
                [pd.Timestamp("1724-12-20").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 0)
        self.assertEqual(table.domain.variables[0].have_date, 1)

        df = pd.DataFrame([[date(2017, 12, 19)], [date(1724, 12, 20)], [np.nan]])
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19").timestamp()],
                [pd.Timestamp("1724-12-20").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 0)
        self.assertEqual(table.domain.variables[0].have_date, 1)

    def test_table_from_frame_time(self):
        from Orange.data.pandas_compat import table_from_frame

        df = pd.DataFrame(
            [[pd.Timestamp("00:00:00.25")], [pd.Timestamp("20:20:20.30")], [np.nan]]
        )
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("1970-01-01 00:00:00.25").timestamp()],
                [pd.Timestamp("1970-01-01 20:20:20.30").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 1)
        self.assertEqual(table.domain.variables[0].have_date, 0)

        df = pd.DataFrame([["00:00:00.25"], ["20:20:20.30"], [np.nan]])
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("1970-01-01 00:00:00.25").timestamp()],
                [pd.Timestamp("1970-01-01 20:20:20.30").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 1)
        self.assertEqual(table.domain.variables[0].have_date, 0)

    def test_table_from_frame_datetime(self):
        from Orange.data.pandas_compat import table_from_frame

        df = pd.DataFrame(
            [
                [pd.Timestamp("2017-12-19 00:00:00.50")],
                [pd.Timestamp("1724-12-20 20:20:20.30")],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00.50").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20.30").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 1)
        self.assertEqual(table.domain.variables[0].have_date, 1)

        df = pd.DataFrame(
            [["2017-12-19 00:00:00.50"], ["1724-12-20 20:20:20.30"], [np.nan]]
        )
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00.50").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20.30").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 1)
        self.assertEqual(table.domain.variables[0].have_date, 1)

        df = pd.DataFrame(
            [
                [datetime(2017, 12, 19, 0, 0, 0, 500000)],
                [datetime(1724, 12, 20, 20, 20, 20, 300000)],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00.50").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20.30").timestamp()],
                [np.nan],
            ],
        )
        self.assertEqual(table.domain.variables[0].have_time, 1)
        self.assertEqual(table.domain.variables[0].have_date, 1)

    def test_table_from_frame_timezones(self):
        from Orange.data.pandas_compat import table_from_frame

        df = pd.DataFrame(
            [
                [pd.Timestamp("2017-12-19 00:00:00")],
                [pd.Timestamp("1724-12-20 20:20:20")],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        self.assertEqual(table.domain.variables[0].timezone, timezone.utc)

        df = pd.DataFrame(
            [
                [pd.Timestamp("2017-12-19 00:00:00Z")],
                [pd.Timestamp("1724-12-20 20:20:20Z")],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        self.assertEqual(pytz.utc, table.domain.variables[0].timezone)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20").timestamp()],
                [np.nan],
            ],
        )

        df = pd.DataFrame(
            [
                [pd.Timestamp("2017-12-19 00:00:00+1")],
                [pd.Timestamp("1724-12-20 20:20:20+1")],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        self.assertEqual(pytz.FixedOffset(60), table.domain.variables[0].timezone)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00+1").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20+1").timestamp()],
                [np.nan],
            ],
        )

        df = pd.DataFrame(
            [
                [pd.Timestamp("2017-12-19 00:00:00", tz="CET")],
                [pd.Timestamp("1724-12-20 20:20:20", tz="CET")],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        self.assertEqual(pytz.timezone("CET"), table.domain.variables[0].timezone)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00+1").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20+1").timestamp()],
                [np.nan],
            ],
        )

        df = pd.DataFrame(
            [
                [pd.Timestamp("2017-12-19 00:00:00", tz="CET")],
                [pd.Timestamp("1724-12-20 20:20:20")],
                [np.nan],
            ]
        )
        table = table_from_frame(df)
        self.assertEqual(pytz.utc, table.domain.variables[0].timezone)
        np.testing.assert_equal(
            table.X,
            [
                [pd.Timestamp("2017-12-19 00:00:00+1").timestamp()],
                [pd.Timestamp("1724-12-20 20:20:20").timestamp()],
                [np.nan],
            ],
        )

    def test_table_from_frame_no_datetim(self):
        """
        In case when dtype of column is object and column contains numbers only,
        column could be recognized as a TimeVarialbe since pd.to_datetime can parse
        numbers as datetime. That column must be result either in StringVariable
        or DiscreteVariable since it's dtype is object.
        """
        from Orange.data.pandas_compat import table_from_frame

        df = pd.DataFrame([[1], [2], [3]], dtype="object")
        table = table_from_frame(df)
        # check if exactly ContinuousVariable and not subtype TimeVariable
        self.assertIsInstance(table.domain.metas[0], StringVariable)

        df = pd.DataFrame([[1], [2], [2]], dtype="object")
        table = table_from_frame(df)
        # check if exactly ContinuousVariable and not subtype TimeVariable
        self.assertIsInstance(table.domain.attributes[0], DiscreteVariable)

    def test_time_variable_compatible(self):
        from Orange.data.pandas_compat import table_from_frame

        def to_df(val):
            return pd.DataFrame([[pd.Timestamp(val)]])

        for datestr, timestamp, outstr in TestTimeVariable.TESTS:
            var = TimeVariable("time")
            var_parse = var.to_val(datestr)
            try:
                pandas_parse = table_from_frame(to_df(datestr)).X[0, 0]
            except ValueError:
                # pandas cannot parse some formats in the list skip them
                continue
            if not (np.isnan(var_parse) and np.isnan(pandas_parse)):
                # nan == nan => False
                self.assertEqual(var_parse, pandas_parse)
                self.assertEqual(pandas_parse, timestamp)

            self.assertEqual(var.repr_val(var_parse), var.repr_val(var_parse))
            self.assertEqual(outstr, var.repr_val(var_parse))

    @unittest.skip("Convert all Orange demo dataset. It takes about 5s which is way to slow")
    def test_table_to_frame_on_all_orange_dataset(self):
        from os import listdir
        from Orange.data.pandas_compat import table_to_frame

        dataset_directory = "Orange/datasets/"

        def _filename_to_dataset_name(f):
            return f.split('.')[0]

        def _get_orange_demo_datasets():
            x = [_filename_to_dataset_name(f) for f in listdir(dataset_directory) if '.tab' in f]
            return x

        for name in _get_orange_demo_datasets():
            table = Table(name)
            df = table_to_frame(table)
            assert_message = "Failed to process Table('{}')".format(name)

            self.assertEqual(type(df), pd.DataFrame, assert_message)
            self.assertEqual(len(df), len(table), assert_message)
            self.assertEqual(len(df.columns), len(table.domain.variables), assert_message)

    def test_table_from_frames(self):
        table = Table("brown-selected")  # dataset with all X, Y and metas
        table.ids = np.arange(100, len(table) + 100, 1, dtype=int)

        x, y, m = table.to_pandas_dfs()
        new_table = Table.from_pandas_dfs(x, y, m)

        np.testing.assert_array_equal(table.X, new_table.X)
        np.testing.assert_array_equal(table.Y, new_table.Y)
        np.testing.assert_array_equal(table.metas, new_table.metas)
        np.testing.assert_array_equal(table.ids, new_table.ids)
        self.assertTupleEqual(table.domain.attributes, new_table.domain.attributes)
        self.assertTupleEqual(table.domain.metas, new_table.domain.metas)
        self.assertEqual(table.domain.class_var, new_table.domain.class_var)

    def test_table_from_frames_not_orange_dataframe(self):
        x = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["x1", "x2", "x3"])
        y = pd.DataFrame([[5], [6]], columns=["y"])
        m = pd.DataFrame([[1, 2], [4, 5]], columns=["m1", "m2"])
        new_table = Table.from_pandas_dfs(x, y, m)

        np.testing.assert_array_equal(x, new_table.X)
        np.testing.assert_array_equal(y.values.flatten(), new_table.Y)
        np.testing.assert_array_equal(m, new_table.metas)
        d = new_table.domain
        self.assertListEqual(x.columns.tolist(), [a.name for a in d.attributes])
        self.assertEqual(y.columns[0], d.class_var.name)
        self.assertListEqual(m.columns.tolist(), [a.name for a in d.metas])

    def test_table_from_frames_same_index(self):
        """
        Test that index column is placed in metas. Function should fail
        with ValueError when indexes are different
        """
        index = np.array(["a", "b"])
        x = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=["x1", "x2", "x3"], index=index
        )
        y = pd.DataFrame([[5], [6]], columns=["y"], index=index)
        m = pd.DataFrame([[1, 2], [4, 5]], columns=["m1", "m2"], index=index)
        new_table = Table.from_pandas_dfs(x, y, m)

        # index should be placed in metas
        np.testing.assert_array_equal(x, new_table.X)
        np.testing.assert_array_equal(y.values.flatten(), new_table.Y)
        np.testing.assert_array_equal(
            np.hstack((index[:, None], m.values.astype("object"))), new_table.metas
        )
        d = new_table.domain
        self.assertListEqual(x.columns.tolist(), [a.name for a in d.attributes])
        self.assertEqual(y.columns[0], d.class_var.name)
        self.assertListEqual(["index"] + m.columns.tolist(), [a.name for a in d.metas])

        index2 = np.array(["a", "c"])
        x = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=["x1", "x2", "x3"], index=index
        )
        y = pd.DataFrame([[5], [6]], columns=["y"], index=index2)
        m = pd.DataFrame([[1, 2], [4, 5]], columns=["m1", "m2"], index=index)
        with self.assertRaises(ValueError):
            Table.from_pandas_dfs(x, y, m)


class TestTablePandas(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, 'table'):
            self.skipTest('Base class')

    def test_basic(self):
        xdf, ydf, mdf = self.table.to_pandas_dfs()
        xtable = xdf.to_orange_table()
        ytable = ydf.to_orange_table()
        mtable = mdf.to_orange_table()

        self.__arreq__(self.table.X, xtable.X)
        self.__arreq__(self.table.Y, ytable.Y)
        self.__arreq__(self.table.metas, mtable.metas)
        self.__arreq__(self.table.W, xtable.W)
        self.__arreq__(self.table.W, ytable.W)
        self.__arreq__(self.table.W, mtable.W)
        self.assertEqual(self.table.attributes, xtable.attributes)
        self.assertEqual(self.table.attributes, ytable.attributes)
        self.assertEqual(self.table.attributes, mtable.attributes)
        self.__arreq__(self.table.ids, xtable.ids)
        self.__arreq__(self.table.ids, ytable.ids)
        self.__arreq__(self.table.ids, mtable.ids)

        d1 = self.table.domain
        d2 = xtable.domain

        vars1 = d1.variables + d1.metas
        vars2 = d2.variables + d2.metas

        for v1, v2 in zip(vars1, vars2):
            self.assertEqual(type(v1), type(v2))

    def test_slice(self):
        df = self.table.X_df
        table2 = df[['c2', 'd1']].to_orange_table()

        self.__arreq__(self.table.ids, table2.ids)
        self.__arreq__(self.table.W, table2.W)
        self.__arreq__(self.table.attributes, table2.attributes)

        domain = table2.domain
        target_domain = Domain([ContinuousVariable("c2"),
                                DiscreteVariable("d1")])

        self.assertEqual(domain, target_domain)

    def test_copy(self):
        df = self.table.X_df
        df2 = df.copy()

        np.testing.assert_array_equal(df.index, df2.index)
        self.assertEqual(df.orange_variables, df2.orange_variables)
        self.assertEqual(df.orange_role, df2.orange_role)
        self.assertEqual(df.orange_attributes, df2.orange_attributes)
        self.assertEqual(df.orange_weights, df2.orange_weights)

    def test_concat_table(self):
        domain2 = Domain(
            [self.table.domain['c2'],
             self.table.domain['d1']]
        )
        table2 = Table.from_numpy(
            domain2,
            np.array(
                [[0, 0],
                 [1, 1]]),
            W=np.array([1, 1])
        )

        df = self.table.X_df
        df2 = table2.X_df
        df3 = pd.concat([df, df2])
        table3 = df3.to_orange_table()

        np.testing.assert_array_equal(table3.ids,
                                      np.concatenate([self.table.ids,
                                                      table2.ids]))
        np.testing.assert_array_equal(table3.W[:-2],
                                      self.table.W)
        np.testing.assert_array_equal(table3.W[-2:],
                                      table2.W)

        attrs = {}
        attrs.update(self.table.attributes)
        attrs.update(table2.attributes)

        self.assertEqual(table3.attributes, attrs)

        vars1 = self.table.domain.variables
        vars2 = table3.domain.variables

        for v1, v2 in zip(vars1, vars2):
            self.assertEqual(type(v1), type(v2))

    def test_concat_df(self):
        df2 = pd.DataFrame(np.array([[0, 0],
                                     [1, 1]]))

        df = self.table.X_df
        df3 = pd.concat([df, df2])
        table3 = df3.to_orange_table()

        np.testing.assert_array_equal(table3.ids[:-2],
                                      self.table.ids)

        d1 = self.table.domain
        d2 = table3.domain

        vars1 = d1.attributes
        vars2 = d2.variables + d2.metas

        for v1 in vars1:
            self.assertIn(v1, vars2)

    def test_merge(self):
        domain2 = Domain(
            [ContinuousVariable("c2"),
             ContinuousVariable("d15")]
        )
        table2 = Table.from_numpy(
            domain2,
            np.array(
                [[0, 4],
                 [-1, 15],
                 [1, 23]])
        )

        df = self.table.X_df
        df2 = table2.X_df
        df3 = pd.merge(df, df2, on='c2')
        table2 = df.to_orange_table()
        table3 = df3.to_orange_table()

        self.assertEqual(len(table2), len(table3))
        self.assertEqual(0, table3.W.size)
        self.assertEqual(self.table.attributes, table3.attributes)

        d1 = table2.domain
        d2 = table3.domain

        vars1 = d1.variables + d1.metas
        vars2 = d2.variables + d2.metas

        self.assertEqual(len(vars2), len(vars1) + 1)

        new_var = next(v for v in vars2
                       if v.name == 'd15')
        self.assertIsInstance(new_var, ContinuousVariable)

        new_var_excluded = [v for v in vars2
                            if v != new_var]
        for v1, v2 in zip(vars1, new_var_excluded):
            self.assertEqual(type(v1), type(v2))

    def test_new_column(self):
        df = self.table.X_df
        table2 = df.to_orange_table()
        df['new'] = np.array([0, 1, 0, 1, 6, 1, 1])
        table3 = df.to_orange_table()

        d1 = table2.domain
        d2 = table3.domain

        vars1 = d1.variables + d1.metas
        vars2 = d2.variables + d2.metas

        self.assertEqual(len(vars2), len(vars1) + 1)

        new_var = next(v for v in vars2
                       if v.name == 'new')
        self.assertIsInstance(new_var, ContinuousVariable)

        new_var_excluded = [v for v in vars2
                            if v != new_var]
        for v1, v2 in zip(vars1, new_var_excluded):
            self.assertEqual(type(v1), type(v2))

    def test_selection(self):
        df = self.table.X_df

        tsel = self.table[1:3]
        dfsel = df[1:3].to_orange_table()

        self.__arreq__(tsel.X, dfsel.X)
        self.__arreq__(tsel.ids, dfsel.ids)
        self.__arreq__(tsel.W, dfsel.W)
        self.__arreq__(tsel.attributes, dfsel.attributes)


class TestDenseTablePandas(TestTablePandas):
    def setUp(self):
        self.domain = Domain(
            [ContinuousVariable("c1"),
             ContinuousVariable("c2"),
             DiscreteVariable("d1", values=("a", "b"))],
            ContinuousVariable("y"),
            [ContinuousVariable("c3"),
             DiscreteVariable("d2", values=("c", "d")),
             StringVariable("s1"),
             StringVariable("s2")]
        )
        metas = np.array(
            [0, 1, 0, 1, 1, 2, 1] +
            [0, 0, 0, 0, 4, 1, 1] +
            "a  b  c  d  e     f    g".split() +
            list("ABCDEF") + [""], dtype=object).reshape(-1, 7).T.copy()
        self.table = Table.from_numpy(
            self.domain,
            np.array(
                [[0, 0, 0],
                 [0, -1, 0],
                 [7, 1, 0],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]),
            np.array(
                [0, 1, 0, 1, 1, 1, 1]),
            metas,
            attributes={'haha': 'hoho'},
            W=np.array([0, 1, 1, 0, 1, 0, 1])
        )

        self.__arreq__ = np.testing.assert_equal

    def test_contiguous_x(self):
        table = self.table
        df = table.X_df
        table2 = df.to_orange_table()
        self.assertTrue(np.shares_memory(df.values, table.X))
        self.assertTrue(np.shares_memory(df.values, table2.X))

    def test_contiguous_y(self):
        table = self.table
        df = table.Y_df
        table2 = df.to_orange_table()
        self.assertTrue(np.shares_memory(df.values, table.Y))
        self.assertTrue(np.shares_memory(df.values, table2.Y))

    @unittest.skipUnless(pd.__version__ >= '1.4.0',
                         'pandas-dev/pandas#39263')
    def test_contiguous_metas(self):
        table = self.table
        df = table.metas_df
        table2 = df.to_orange_table()
        self.assertTrue(np.shares_memory(df.values, table.metas))
        self.assertTrue(np.shares_memory(df.values, table2.metas))

    def test_to_dfs(self):
        table = self.table
        dfs = table.to_pandas_dfs()

        tables = [df.to_orange_table() for df in dfs]
        for t in tables:
            arrs = (t.X, t.Y, t.metas)
            self.assertEqual(sum(
                arr.size > 0
                for arr in arrs
            ), 1)

    def test_amend(self):
        with self.table.unlocked():
            df = self.table.X_df
            df.iloc[0][0] = 0
        X = self.table.X
        with self.table.unlocked():
            self.table.X_df = df
        self.assertTrue(np.shares_memory(df.values, X))

    def test_amend_dimension_mismatch(self):
        df = self.table.X_df
        df = df.append([0, 1])
        try:
            self.table.X_df = df
        except ValueError as e:
            self.assertEqual(str(e),
                             'Leading dimension mismatch (not 7 == 9)')
        else:
            self.fail()


class TestSparseTablePandas(TestTablePandas):
    features = (
        ContinuousVariable(name="c2"),
        ContinuousVariable(name="Continuous Feature 2"),
        DiscreteVariable(name="d1", values=("0", "1")),
        DiscreteVariable(name="Discrete Feature 2",
                         values=("value1", "value2")),
    )

    class_vars = (
        ContinuousVariable(name="Continuous Class"),
        DiscreteVariable(name="Discrete Class", values=("m", "f"))
    )

    feature_data = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 0, 0),
        (0, 1, 1, 0),
    )

    class_data = (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
    )

    def setUp(self):
        self.domain = Domain(attributes=self.features,
                             class_vars=self.class_vars)
        table = Table.from_numpy(
            self.domain,
            np.array(self.feature_data),
            np.array(self.class_data),
        )
        self.table = Table.from_numpy(
            self.domain,
            csr_matrix(table.X),
            csr_matrix(table.Y),
            W=np.array([1, 0, 1, 0, 1, 1, 1])
        )

        def arreq(t1, t2):
            if all(sp.issparse(t) for t in (t1, t2)):
                return self.assertEqual((t1 != t2).nnz, 0)
            else:
                return np.array_equal(t1, t2)

        self.__arreq__ = arreq

    def test_to_dense(self):
        df = self.table.X_df

        self.assertIsInstance(df, OrangeDataFrame)

        ddf = df.sparse.to_dense()
        np.testing.assert_array_equal(df.index, ddf.index)
        np.testing.assert_array_equal(df.orange_variables, ddf.orange_variables)
        np.testing.assert_array_equal(df.orange_attributes, ddf.orange_attributes)
        np.testing.assert_array_equal(df.orange_role, ddf.orange_role)
        np.testing.assert_array_equal(df.orange_weights, ddf.orange_weights)

        table = self.table.to_dense()
        table2 = ddf.to_orange_table()

        np.testing.assert_array_equal(table2.X, table.X)
        np.testing.assert_array_equal(table2.ids, table.ids)
        np.testing.assert_array_equal(table2.W, table.W)
        np.testing.assert_array_equal(table2.attributes, table.attributes)


if __name__ == "__main__":
    unittest.main()

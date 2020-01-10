# pylint: disable=import-outside-toplevel
import unittest
import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, TimeVariable, Table

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

    @unittest.skip("Convert all Orange demo dataset. It takes about 5s which is way to slow")
    def test_table_to_frame_on_all_orange_dataset(self):
        from os import listdir
        from Orange.data.pandas_compat import table_to_frame
        import pandas as pd

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
            self.assertEqual(len(df.columns), len(table.domain), assert_message)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, TimeVariable

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

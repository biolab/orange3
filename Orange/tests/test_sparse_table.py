# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import scipy.sparse as sp
import numpy as np
import pandas as pd

from Orange.data import Table, Domain, ContinuousVariable, SparseTable, StringVariable
from Orange.tests import test_table as tabletests


class InterfaceTest(tabletests.InterfaceTest):
    def setUp(self):
        super().setUp()

        # data_sparseformat: 0 -> np.nan, string -> variable.to_val(string)
        feature_data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        class_data = [[1, 0], [0, 1], [1, 0], [0, 1]]
        data = list(list(a + c) for a, c in zip(feature_data, class_data))
        self.data_sparseformat = np.array(data, dtype=float)
        self.data_sparseformat[self.data_sparseformat == 0] = np.nan

        self.table = Table.from_numpy(
            Domain([ContinuousVariable("x" + str(i)) for i in range(len(self.domain.attributes))],
                   [ContinuousVariable("y" + str(i)) for i in range(len(self.domain.class_vars))]),
            sp.csr_matrix(np.array(feature_data)),
            sp.csr_matrix(np.array(class_data)),
        )

    def test_can_select_single_item_from_series(self):
        # a counterpart to test_cant_select_multiple_items_from_series
        # should produce results without fail
        values = self.table.iloc[0]['x0']

    def test_cant_select_multiple_items_from_series(self):
        # getting a series should be okay
        s = self.table.iloc[0]
        with self.assertRaises(ValueError):
            # this doesn't work for sparse series, but works for normal series
            # likely because of overridden __getitem__
            # _modified versions of tests below exist because of this
            values = s[['x0', 'x1']]

    def test_delete_rows(self):
        with self.assertRaises(ValueError):
            super().test_delete_rows()

    def test_delete_rows_modified(self):
        for i in range(self.nrows):
            self.table = self.table.iloc[1:]
            for j in range(len(self.table)):
                np.testing.assert_array_equal(list(self.table.iloc[j][:-1]), self.data_sparseformat[i + j + 1])

    def test_iteration(self):
        with self.assertRaises(ValueError):
            super().test_iteration()

    def test_iteration_modified(self):
        for (idx, row), expected_data in zip(self.table.iterrows(), self.data_sparseformat):
            np.testing.assert_array_equal(list(row[:-1]), expected_data)

    def test_row_assignment(self):
        with self.assertRaises(NotImplementedError):
            super().test_row_assignment()

    def test_row_indexing(self):
        with self.assertRaises(ValueError):
            super().test_row_indexing()

    def test_row_indexing_modified(self):
        for i in range(self.nrows):
            np.testing.assert_array_equal(list(self.table.iloc[i][:-1]), self.data_sparseformat[i])

    def test_value_assignment(self):
        with self.assertRaises(NotImplementedError):
            self.table.iloc[0, 0] = 3
        # creates a copy, pandas (actually general sparse) limitation
        newt = self.table.set_value(0, 0, 42)
        assert newt is not self.table

    def test_value_indexing(self):
        for i in range(self.nrows):
            for j, c in enumerate(tabletests.cols_wo_weights(self.table)):
                np.testing.assert_equal(self.table.iloc[i][c], self.data_sparseformat[i][j])

    def test_pandas_subclass_slicing_bug(self):
        # see https://github.com/pydata/pandas/pull/13787
        self.assertIsInstance(self.table, SparseTable)
        self.assertIsInstance(self.table.iloc[:2], pd.SparseDataFrame)  # should be SparseTable
        self.assertIsInstance(self.table[:2], pd.SparseDataFrame)  # should be SparseTable

    def test_string_variables_unsupported(self):
        mat = sp.coo_matrix([[1, 2],
                             [3, 4]])
        with self.assertRaises(ValueError):
            st = SparseTable(Domain([], [], [StringVariable("s")]), mat)

    def test_density(self):
        self.assertTrue(self.table.is_sparse)
        self.assertFalse(self.table.is_dense)
        self.assertLess(self.table.density, 0.5)
        self.assertGreater(self.table.density, 0.25)

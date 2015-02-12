import unittest

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from Orange import data

from Orange.tests import test_table as tabletests


class InterfaceTest(tabletests.InterfaceTest):
    def setUp(self):
        super().setUp()
        self.table = data.Table.from_numpy(
            self.domain,
            csr_matrix(self.table.X),
            csr_matrix(self.table.Y),
        )

    def test_append_rows(self):
        self.assertRaises(ValueError,
                          self.table.append, [2] * len(self.data[0]))

    def test_insert_rows(self):
        self.assertRaises(ValueError,
                          self.table.insert, 0, [2] * len(self.data[0]))

    def test_delete_rows(self):
        with self.assertRaises(ValueError):
            del self.table[0]

    def test_clear(self):
        self.assertRaises(ValueError, self.table.clear)

    def test_row_assignment(self):
        new_value = 2.
        for i in range(self.nrows):
            new_row = [new_value] * len(self.data[i])
            self.table[i] = np.array(new_row)
            self.assertEqual(list(self.table[i]), new_row)

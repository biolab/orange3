from scipy.sparse import csr_matrix

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
        with self.assertRaises(ValueError):
            super().test_append_rows()

    def test_insert_rows(self):
        with self.assertRaises(ValueError):
            super().test_insert_rows()

    def test_delete_rows(self):
        with self.assertRaises(ValueError):
            super().test_delete_rows()

    def test_clear(self):
        with self.assertRaises(ValueError):
            super().test_clear()

    def test_row_assignment(self):
        with self.assertRaises(ValueError):
            super().test_row_assignment()

    def test_value_assignment(self):
        with self.assertRaises(ValueError):
            super().test_value_assignment()

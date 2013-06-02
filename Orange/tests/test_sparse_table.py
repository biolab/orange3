import unittest

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

    @unittest.skip("CSR sparse matrices do not support resize.")
    def test_append_rows(self):
        pass

    @unittest.skip("CSR sparse matrices do not support resize.")
    def test_insert_rows(self):
        pass

    @unittest.skip("CSR sparse matrices do not support resize.")
    def test_delete_rows(self):
        pass

    @unittest.skip("CSR sparse matrices do not support resize.")
    def test_clear(self):
        pass

    @unittest.skip("CSR sparse matrices do not support row assignment.")
    def test_row_assignment(self):
        pass

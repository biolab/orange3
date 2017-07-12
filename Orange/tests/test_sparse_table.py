# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import numpy as np
from scipy.sparse import csr_matrix

from Orange import data
from Orange.data import Table
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
        with self.assertRaises(Exception):
            super().test_append_rows()

    def test_insert_rows(self):
        with self.assertRaises(Exception):
            super().test_insert_rows()

    def test_insert_view(self):
        with self.assertRaises(Exception):
            super().test_insert_view()

    def test_delete_rows(self):
        with self.assertRaises(ValueError):
            super().test_delete_rows()

    def test_clear(self):
        with self.assertRaises(ValueError):
            super().test_clear()

    def test_row_assignment(self):
        super().test_row_assignment()

    def test_value_assignment(self):
        super().test_value_assignment()

    def test_str(self):
        iris = Table('iris')
        iris.X, iris.Y = csr_matrix(iris.X), csr_matrix(iris.Y)
        str(iris)

    def test_Y_setter_1d(self):
        iris = Table('iris')
        assert iris.Y.shape == (150,)
        iris.Y = csr_matrix(iris.Y)
        # We expect the Y shape to match the X shape, which is (150, 4) in iris
        self.assertEqual(iris.Y.shape, (150, 1))

    def test_Y_setter_2d(self):
        iris = Table('iris')
        assert iris.Y.shape == (150,)
        # Convert iris.Y to (150, 1) shape
        new_y = iris.Y[:, np.newaxis]
        iris.Y = np.hstack((new_y, new_y))
        iris.Y = csr_matrix(iris.Y)
        # We expect the Y shape to match the X shape, which is (150, 4) in iris
        self.assertEqual(iris.Y.shape, (150, 2))

    def test_Y_setter_2d_single_instance(self):
        iris = Table('iris')[:1]
        # Convert iris.Y to (1, 1) shape
        new_y = iris.Y[:, np.newaxis]
        iris.Y = np.hstack((new_y, new_y))
        iris.Y = csr_matrix(iris.Y)
        # We expect the Y shape to match the X shape, which is (1, 4) in iris
        self.assertEqual(iris.Y.shape, (1, 2))

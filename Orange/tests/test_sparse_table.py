# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import warnings

import numpy as np
from scipy.sparse import csr_matrix, SparseEfficiencyWarning

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

    def test_row_assignment(self):
        # This warning will appear in actual code that assigns rows of
        # sparse matrix, which is OK, but We don't need it in test outputs
        warnings.filterwarnings("ignore", ".*", SparseEfficiencyWarning)
        super().test_row_assignment()

    def test_value_assignment(self):
        # This warning will appear in actual code that assigns rows of
        # sparse matrix, which is OK, but We don't need it in test outputs
        warnings.filterwarnings("ignore", ".*", SparseEfficiencyWarning)
        super().test_value_assignment()

    def test_str(self):
        iris = Table('iris')
        with iris.unlocked():
            iris.X, iris.Y = csr_matrix(iris.X), csr_matrix(iris.Y)
        str(iris)

    def test_Y_setter_1d(self):
        iris = Table('iris')
        assert iris.Y.shape == (150,)
        with iris.unlocked():
            iris.Y = csr_matrix(iris.Y)
        # We expect the Y shape to match the X shape, which is (150, 4) in iris
        self.assertEqual(iris.Y.shape, (150,))

    def test_Y_setter_2d(self):
        iris = Table('iris')
        assert iris.Y.shape == (150,)
        # Convert iris.Y to (150, 1) shape
        new_y = iris.Y[:, np.newaxis]
        with iris.unlocked():
            iris.Y = np.hstack((new_y, new_y))
            iris.Y = csr_matrix(iris.Y)
        # We expect the Y shape to match the X shape, which is (150, 4) in iris
        self.assertEqual(iris.Y.shape, (150, 2))

    def test_Y_setter_2d_single_instance(self):
        iris = Table('iris')[:1]
        # Convert iris.Y to (1, 1) shape
        new_y = iris.Y[:, np.newaxis]
        with iris.unlocked_reference():
            iris.Y = np.hstack((new_y, new_y))
            iris.Y = csr_matrix(iris.Y)
        # We expect the Y shape to match the X shape, which is (1, 4) in iris
        self.assertEqual(iris.Y.shape, (1, 2))

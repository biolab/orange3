# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np
import scipy.sparse.linalg as sla

import Orange
from Orange.projection import CUR
from Orange.tests import test_filename


class TestCUR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ionosphere = Orange.data.Table(test_filename('datasets/ionosphere.tab'))

    def test_cur_projection(self):
        self.__projection_test_helper(self.ionosphere, rank=10, max_error=10)
        self.__projection_test_helper(self.ionosphere, rank=20, max_error=10)
        self.__projection_test_helper(self.ionosphere, rank=5, max_error=1)

    def __projection_test_helper(self, data, rank, max_error):
        cur = CUR(rank=rank, max_error=max_error)
        cur_model = cur(data)
        self.assertEqual(data.X.shape[0], cur_model.C_.shape[0])
        self.assertEqual(data.X.shape[1], cur_model.R_.shape[1])
        np.testing.assert_array_equal(cur_model(data).X, cur_model.C_)

    def test_cur_reconstruction(self):
        self.__reconstruction_test_helper(self.ionosphere, rank=20, max_error=5)
        self.__reconstruction_test_helper(self.ionosphere, rank=25, max_error=1)
        self.__reconstruction_test_helper(self.ionosphere, rank=30, max_error=0.1)

    def __reconstruction_test_helper(self, data, rank, max_error):
        U, s, V = sla.svds(data.X, rank)
        S = np.diag(s)
        X_k = np.dot(U, np.dot(S, V))
        err_svd = np.linalg.norm(data.X - X_k, 'fro')
        cur = CUR(rank=rank, max_error=max_error, compute_U=True, random_state=0)
        cur_model = cur(data)
        X_hat = np.dot(cur_model.C_, np.dot(cur_model.U_, cur_model.R_))
        err_cur = np.linalg.norm(data.X - X_hat, 'fro')
        self.assertLess(err_cur, (3 + cur_model.max_error) * err_svd)

    def test_cur_axis(self):
        data1 = self.ionosphere[:100]
        data2 = self.ionosphere[100:]
        cur = CUR(rank=5, max_error=1)
        cur_model = cur(data1)

        data2_trans1 = cur_model(data2)
        data2_trans2 = data2.X[:, cur_model.features_]
        np.testing.assert_array_equal(data2_trans1.X, data2_trans2)

        data1_trans1 = cur_model(data1, axis=1)
        data1_trans2 = data1.X[cur_model.samples_, :]
        np.testing.assert_array_equal(data1_trans1.X, data1_trans2)

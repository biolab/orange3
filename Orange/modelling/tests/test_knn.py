import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import sklearn

from Orange.data import Table
from Orange.modelling import KNNLearner

knn_init = sklearn.neighbors.KNeighborsClassifier.__init__


class Test(unittest.TestCase):
    def setUp(self):
        x = np.array([[3, 0, 4],
                      [12, 5, 0],
                      [1, 2, 2]])
        y = np.array([1, 0, 1])
        self.data = Table.from_numpy(None, x, y)

    @patch("Orange.base.SklLearner.fit")
    def test_cosine_normalizes(self, fit):
        learner = KNNLearner(metric="cosine")
        learner(self.data)
        X = fit.call_args[0][0]
        np.testing.assert_allclose(np.sum(X ** 2, axis=1), 1)
        np.testing.assert_allclose(X, [
            [3 / 5, 0, 4 / 5],
            [12 / 13, 5 / 13, 0],
            [1 / 3, 2 / 3, 2 / 3]])
        fit.reset_mock()

        with self.data.unlocked():
            self.data.X = sparse.csr_matrix(self.data.X)
            learner(self.data)
            X = fit.call_args[0][0]
            np.testing.assert_allclose(np.sum(X ** 2, axis=1), 1)
            row0 = X[0]
            np.testing.assert_allclose(row0, row0 / np.linalg.norm(row0))

    def test_cosine_does_not_modify_input_data(self):
        original = self.data.X.copy()
        learner = KNNLearner(metric="cosine")
        learner(self.data)
        np.testing.assert_array_equal(self.data.X, original)

    @patch("sklearn.neighbors.KNeighborsClassifier.__init__",
           side_effect=knn_init, autospec=True)
    def test_cosine_computes_euclidean(self, mock_init):
        learner = KNNLearner(metric="cosine")
        learner(self.data)
        self.assertEqual(mock_init.call_args[1]["metric"], "euclidean")

        learner = KNNLearner(metric="manhattan")
        learner(self.data)
        self.assertEqual(mock_init.call_args[1]["metric"], "manhattan")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np
from Orange.misc import DistMatrix


class DistMatrixTest(unittest.TestCase):
    def test_reader_selection(self):
        with patch("Orange.misc._distmatrix_xlsx.read_matrix") as read_matrix, \
                patch.object(DistMatrix, "_from_dst") as _from_dst:
            read_matrix.return_value = (np.zeros((3, 4)), None, None, 1)
            _from_dst.return_value = (np.zeros((2, 2)), None, None, 1)

            matrix = DistMatrix.from_file("test.dst")
            self.assertEqual(matrix.shape, (2, 2))

            matrix = DistMatrix.from_file("test.xlsx")
            self.assertEqual(matrix.shape, (3, 4))

    def test_auto_symmetricized_result(self):
        data = np.array([[np.nan, np.nan, np.nan],
                         [1528.13, np.nan, np.nan],
                         [1497.61, 999.25, np.nan],
                         [1062.89, 1372.59, 651.62]])

        exp_sym = np.array([[1528.13, 1497.61, 1062.89],
                            [1497.61, 999.25, 1372.59],
                            [1062.89, 1372.59, 651.62]])

        labels = list("ABC")
        for ri, li in ((labels, labels),
                       (labels, None),
                       (None, labels),
                       (None, None)):
            matrix = DistMatrix(data[1:], ri, li)
            sym = matrix.auto_symmetricized()
            np.testing.assert_almost_equal(sym, exp_sym)

        matrix = DistMatrix(data[1:].T)
        sym = matrix.auto_symmetricized()
        np.testing.assert_almost_equal(sym, exp_sym)

        labels = list("ABCD")
        data[1, 1] =  2
        exp_sym = np.array(
            [[   0.  , 1528.13, 1497.61, 1062.89],
             [1528.13,    2.  ,  999.25, 1372.59],
             [1497.61,  999.25,    0.  ,  651.62],
             [1062.89, 1372.59,  651.62,    0.  ]])
        matrix = DistMatrix(data, labels)
        sym = matrix.auto_symmetricized()
        np.testing.assert_almost_equal(sym, exp_sym)

        matrix = DistMatrix(data.T, None, labels)
        sym = matrix.auto_symmetricized()
        np.testing.assert_almost_equal(sym, exp_sym)

    def test_auto_symmetricized_dont_apply(self):
        data = DistMatrix(np.array([[np.nan, np.nan]] * 3 + [[1, np.nan]]))
        self.assertIs(data.auto_symmetricized(), data)

        data = np.array([[np.nan, np.nan, 1],
                         [1528.13, np.nan, np.nan],
                         [1497.61, 999.25, np.nan],
                         [1062.89, 1372.59, 651.62]])
        matrix = DistMatrix(data)
        self.assertIs(matrix.auto_symmetricized(), matrix)

        data = np.array([[np.nan, np.nan, 1],
                         [1528.13, np.nan, np.nan],
                         [1497.61, 999.25, np.nan],
                         [1062.89, 1372.59, 651.62]])
        matrix = DistMatrix(data)
        sym = matrix.auto_symmetricized(copy=True)
        np.testing.assert_equal(matrix, sym)
        self.assertIsNot(sym, matrix)

        matrix = DistMatrix(data.T)
        self.assertIs(matrix.auto_symmetricized(), matrix)

        data = np.array([[np.nan, np.nan, np.nan],
                         [1528.13, np.nan, np.nan],
                         [1497.61, 999.25, np.nan],
                         [1062.89, 1372.59, 651.62]])
        matrix = DistMatrix(data, None, list("abc"))
        self.assertIs(matrix.auto_symmetricized(), matrix)

        matrix = DistMatrix(data.T, list("abc"))
        self.assertIs(matrix.auto_symmetricized(), matrix)

        data = np.array([[1528.13, np.nan, np.nan],
                         [1497.61, 999.25, np.nan],
                         [1062.89, 1372.59, 651.62]])
        matrix = DistMatrix(data, list("def"), list("abc"))
        self.assertIs(matrix.auto_symmetricized(), matrix)


if __name__ == "__main__":
    unittest.main()

# pylint: disable=protected-access

import unittest
from unittest.mock import patch

import numpy as np
from Orange.data import ContinuousVariable, StringVariable, Table, Domain
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
            self.assertEqual(sym.row_items, sym.col_items)
            self.assertIs((ri or li), sym.row_items)

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

    def test_trivial_labels(self):
        matrix = DistMatrix(np.array([[1, 2, 3], [4, 5, 6]]))

        self.assertFalse(matrix._trivial_labels(matrix.row_items))
        self.assertIsNone(matrix.get_labels(matrix.row_items))

        matrix.row_items = list("abc")
        self.assertTrue(matrix._trivial_labels(matrix.row_items))
        self.assertEqual(matrix.get_labels(matrix.row_items), list("abc"))

        matrix.row_items = ["a", 1, "c"]
        self.assertFalse(matrix._trivial_labels(matrix.row_items))
        self.assertIsNone(matrix.get_labels(matrix.row_items))

        c1, c2 = (ContinuousVariable(c) for c in "xy")
        s1, s2 = (StringVariable(c) for c in "st")
        data = Table.from_list(Domain([c1], None, [c2, s1]),
                               [[1, 0, "a"], [2, 2, "b"], [3, 1, "c"]])
        matrix.row_items = data

        matrix.axis = 1
        self.assertTrue(matrix._trivial_labels(matrix.row_items))
        self.assertEqual(list(matrix.get_labels(matrix.row_items)), list("abc"))

        matrix.axis = 0
        self.assertTrue(matrix._trivial_labels(matrix.row_items))
        self.assertEqual(list(matrix.get_labels(matrix.row_items)), list("x"))


        data = Table.from_list(Domain([c1], None, [c2, s1, s2]),
                               [[1, 2, "a", "2"],
                                [2, 4, "b", "5"],
                                [3, 0, "c", "g"]])
        matrix.row_items = data

        matrix.axis = 1
        self.assertFalse(matrix._trivial_labels(matrix.row_items))
        self.assertIsNone(matrix.get_labels(matrix.row_items))

        matrix.axis = 0
        self.assertTrue(matrix._trivial_labels(matrix.row_items))
        self.assertEqual(list(matrix.get_labels(matrix.row_items)), list("x"))

        data = Table.from_list(Domain([c1], None, [c2]),
                               [[1, 2],
                                [2, 4],
                                [3, 0]])
        matrix.row_items = data
        matrix.axis = 1
        self.assertFalse(matrix._trivial_labels(matrix.row_items))
        self.assertIsNone(matrix.get_labels(matrix.row_items))

        matrix.axis = 0
        self.assertTrue(matrix._trivial_labels(matrix.row_items))
        self.assertEqual(matrix.get_labels(matrix.row_items), list("x"))


if __name__ == "__main__":
    unittest.main()

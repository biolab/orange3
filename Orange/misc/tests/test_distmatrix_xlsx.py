# pylint: disable=protected-access

import os
import unittest
from unittest.mock import patch, Mock

import numpy as np
import openpyxl

from Orange.misc import DistMatrix
from Orange.misc._distmatrix_xlsx import read_matrix, _get_sheet, \
    _non_empty_cells, _get_labels, _matrix_from_cells, write_matrix

import Orange.tests
from Orange.tests import named_file

files_dir = os.path.join(os.path.split(Orange.tests.__file__)[0], "xlsx_files")


class ReadMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file = os.path.join(files_dir, "distances.xlsx")

    def test_layouts(self):
        def test(sheet, exp_matrix, exp_row_labels, exp_col_labels):
            matrix, row_labels, col_labels, _ = read_matrix(self.file, sheet)
            np.testing.assert_almost_equal(matrix, exp_matrix)
            self.assertEqual(row_labels, exp_row_labels)
            self.assertEqual(col_labels, exp_col_labels)

        labels = "Barcelona Belgrade Berlin Brussels".split()
        data = np.array([[np.nan, np.nan, np.nan],
                         [1528.13, np.nan, np.nan],
                         [1497.61, 999.25, np.nan],
                         [1062.89, 1372.59, 651.62]])

        test("lower_row_labels", data, labels, None)
        test("upper_col_labels", data.T, None, labels)

        data = np.array([[1528.13, np.nan, np.nan, np.nan],
                         [1497.61, 999.25, np.nan, np.nan],
                         [1062.89, 1372.59, 651.62, np.nan]])
        test("lower_col_labels", data, None, labels)

        data = np.array([[np.nan, 1528.13, 1497.61, 1062.89],
                         [np.nan, np.nan, 999.25, 1372.59],
                         [np.nan, np.nan, np.nan, 651.62],
                         [np.nan, np.nan, np.nan, np.nan]])
        test("upper_row_labels", data, labels, None)
        test("upper_both_labels", data, list("AERU"), labels)
        test("lower_both_labels", data.T, labels, list("AERU"))
        test("upper_no_labels", data[:-1, 1:], None, None)
        test("lower_no_labels", data[:-1, 1:].T, None, None)

        data[np.diag_indices(4)] = [1, 2, 3, 4]
        test("upper_with_diag", data, labels, None)
        test("lower_with_diag", data.T, labels, None)
        test("with_nans",
             np.array([[1, np.nan, 1, 2],
                       [np.nan, 2, np.nan, 4],
                       [2, np.nan, 3, 5],
                       [np.nan, 4, np.nan, 4]]), labels, None)

        data = np.array([[5, 5, np.nan, 47, 7, 4],
                         [7, 5, np.nan, 2, np.nan, np.nan],
                         [2, 7, np.nan, np.nan, 27, 5],
                         [np.nan, 2, 2, np.nan, 2, np.nan]])
        test("non_square_both", data, labels, list("abcdef"))
        test("non_square_row_labels", data, labels, None)
        test("non_square_col_labels", data, None, list("abcdef"))
        test("non_square_no_labels", data, None, None)

        test("non_square_off",
             np.array([[np.nan] * 8,
                       [np.nan] * 8,
                       [5, 5, np.nan, 47, 7, 4, np.nan, np.nan],
                       [7, 5, np.nan, 2, np.nan, np.nan, np.nan, np.nan],
                       [2, 7, np.nan, np.nan, 27, 5, np.nan, np.nan],
                       [np.nan, 2, 2, np.nan, 2, np.nan, np.nan, np.nan]]),
             list("abcd??"), list("???ABCDE"), )

        test("just_numbers", [[1, 2, 3], [4, 5, 6]], None, None)

    def test_fast_floats(self):
        with patch("numpy.cumsum", Mock(wraps=np.cumsum)) as cumsum:
            read_matrix(self.file, "non_square_off")
            cumsum.assert_called()  # sanity check
            cumsum.reset_mock()

            data, row_labels, col_labels, _ = \
                read_matrix(self.file, "numbers_upper_left")
            cumsum.assert_not_called()
            np.testing.assert_almost_equal(data, [[1.2, 4.6, 1.8],
                                                  [2.6, 6.4, 1.7]])
            self.assertIsNone(row_labels)
            self.assertIsNone(col_labels)

    def test_errors(self):
        self.assertRaisesRegex(
            ValueError, "sheet", read_matrix, self.file, "koala")

        self.assertRaisesRegex(
            ValueError, "E15", read_matrix, self.file, "non_square_off_err")

        self.assertRaisesRegex(
            ValueError, "empty", read_matrix, self.file, "no data")

    def test_active_worksheet(self):
        # Either succeed or report error, just load something :)
        try:
            matrix, *_ = read_matrix(self.file)
            self.assertIsNotNone(matrix)
        except ValueError as exc:
            self.assertTrue({"E15", "sheet", "empty"} & set(str(exc).split()))


class FunctionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file = os.path.join(files_dir, "distances.xlsx")

    def test_get_sheet(self):
        # Just return something...
        self.assertIsInstance(_get_sheet(self.file, None),
                              openpyxl.worksheet.worksheet.Worksheet)
        self.assertIsInstance(_get_sheet(self.file, "lower_row_labels"),
                              openpyxl.worksheet.worksheet.Worksheet)

    def test_non_empty_cells(self):
        sheet = _get_sheet(self.file, "upper_row_labels")
        cells, row_off, col_off = _non_empty_cells(sheet)
        self.assertEqual(cells.shape, (4, 5))
        self.assertEqual(row_off, 0)
        self.assertEqual(col_off, 0)

        sheet = _get_sheet(self.file, "non_square_both")
        cells, row_off, col_off = _non_empty_cells(sheet)
        self.assertEqual(cells.shape, (5, 7))
        self.assertEqual(row_off, 5)
        self.assertEqual(col_off, 2)

        sheet = _get_sheet(self.file, "non_square_off")
        cells, row_off, col_off = _non_empty_cells(sheet)
        self.assertEqual(cells.shape, (7, 9))
        self.assertEqual(row_off, 10)
        self.assertEqual(col_off, 1)
        self.assertEqual(cells[1, 0], "a")
        self.assertEqual(cells[0, 8], "E")
        self.assertIsNone(cells[6, 8])

        sheet = _get_sheet(self.file, "no data")
        self.assertRaisesRegex(ValueError, ".*empty.*", _non_empty_cells, sheet)

        with patch("numpy.cumsum", Mock(wraps=np.cumsum)) as cumsum:
            sheet = _get_sheet(self.file, "non_square_off")
            _non_empty_cells(sheet)
            cumsum.assert_called()  # sanity check
            cumsum.reset_mock()

            sheet = _get_sheet(self.file, "numbers_upper_left")
            _non_empty_cells(sheet)
            cumsum.assert_not_called()

    def test_get_labels(self):
        self.assertEqual(_get_labels(["a", "b", "c"]), ["a", "b", "c"])
        self.assertEqual(_get_labels(["a", "bb", 1, 2]), ["a", "bb", "1", "2"])
        self.assertEqual(_get_labels([None, "b", None]), ["?", "b", "?"])
        self.assertIsNone(_get_labels([None, "1.5", 2]), None)
        self.assertIsNone(_get_labels([]), None)

    def test_matrix_from_cells(self):
        np.testing.assert_almost_equal(
            _matrix_from_cells(
                np.array([[1, 2, None], ["3.15", None, ""]]),
                1, 2),
            np.array([[1, 2, np.nan], [3.15, np.nan, np.nan]])
        )

        self.assertRaisesRegex(
            ValueError, ".*D3.*", _matrix_from_cells,
            np.array([[1, 2, None], ["3.15", "foo", ""]]),
            1, 2)
        self.assertRaisesRegex(
            ValueError, ".*D3.*", _matrix_from_cells,
            np.array([[1, 2, None], ["3.15", object(), ""]]),
            1, 2)

    def test_write(self):
        with named_file("", suffix=".xlsx") as fname:
            matrix = DistMatrix([[1, 2, 3], [4, 5, 6]])
            write_matrix(matrix, fname)
            matrix2, *_ = read_matrix(fname)
            np.testing.assert_equal(matrix, matrix2)

            matrix.row_items = mrow_items = ["aa", "bb"]
            matrix.col_items = mcol_items = ["cc", "dd", "ee"]

            matrix.row_items = mrow_items
            matrix.col_items = mcol_items
            write_matrix(matrix, fname)
            matrix2, row_labels, col_labels, _ = read_matrix(fname)
            np.testing.assert_equal(matrix, matrix2)
            self.assertEqual(row_labels, mrow_items)
            self.assertEqual(col_labels, mcol_items)

            matrix.row_items = None
            matrix.col_items = mcol_items
            write_matrix(matrix, fname)
            matrix2, row_labels, col_labels, _ = read_matrix(fname)
            np.testing.assert_equal(matrix, matrix2)
            self.assertIsNone(row_labels)
            self.assertEqual(col_labels, mcol_items)

            matrix.row_items = mrow_items
            matrix.col_items = None
            write_matrix(matrix, fname)
            matrix2, row_labels, col_labels, _ = read_matrix(fname)
            np.testing.assert_equal(matrix, matrix2)
            self.assertEqual(row_labels, mrow_items)
            self.assertEqual(col_labels, None)

            matrix.row_items = matrix._labels_to_tables(mrow_items)
            matrix.col_items = matrix._labels_to_tables(mcol_items)
            write_matrix(matrix, fname)
            matrix2, row_labels, col_labels, _ = read_matrix(fname)
            np.testing.assert_equal(matrix, matrix2)
            self.assertEqual(row_labels, mrow_items)
            self.assertEqual(col_labels, mcol_items)

            matrix = DistMatrix([[1, 2, 3], [2, 0, 4], [3, 4, 0]])
            write_matrix(matrix, fname)
            matrix2, *_ = read_matrix(fname)
            np.testing.assert_equal(matrix2, [[1, np.nan, np.nan],
                                              [2, 0, np.nan],
                                              [3, 4, 0]])

            matrix = DistMatrix([[0, 2, 3], [2, 0, 4], [3, 4, 0]])
            matrix.col_items = mcol_items
            write_matrix(matrix, fname)
            matrix2, row_items, col_items, *_ = read_matrix(fname)
            np.testing.assert_equal(matrix2, [[np.nan, np.nan, np.nan],
                                              [2, np.nan, np.nan],
                                              [3, 4, np.nan]])
            self.assertIsNone(row_items)
            self.assertEqual(col_items, mcol_items)


if __name__ == "__main__":
    unittest.main()

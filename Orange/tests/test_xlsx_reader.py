# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import os

import numpy as np

from Orange.data import io, ContinuousVariable, DiscreteVariable, Table


def get_dataset(name):
    return os.path.join(os.path.dirname(__file__), "xlsx_files", name)


def read_file(name):
    return io.ExcelReader(get_dataset(name)).read()


class TestExcelHeader0(unittest.TestCase):
    def test_read(self):
        table = read_file("header_0.xlsx")
        domain = table.domain
        self.assertIsNone(domain.class_var)
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 4)
        for i, var in enumerate(domain.attributes):
            self.assertIsInstance(var, ContinuousVariable)
            self.assertEqual(var.name, "Feature {}".format(i + 1))
        np.testing.assert_almost_equal(table.X,
                                       np.array([[0.1, 0.5, 0.1, 21],
                                                 [0.2, 0.1, 2.5, 123],
                                                 [0, 0, 0, 0]]))
        self.assertEqual(table.name, 'header_0')


class TextExcelSheets(unittest.TestCase):
    def setUp(self):
        self.reader = io.ExcelReader(get_dataset("header_0_sheet.xlsx"))

    def test_sheets(self):
        self.assertSequenceEqual(self.reader.sheets,
                                 ["Sheet1", "my_sheet", "Sheet3"])

    def test_named_sheet(self):
        self.reader.select_sheet("my_sheet")
        table = self.reader.read()
        self.assertEqual(len(table.domain.attributes), 4)
        self.assertEqual(table.name, 'header_0_sheet-my_sheet')

    def test_named_sheet_table(self):
        table = Table.from_file(get_dataset("header_0_sheet.xlsx"),
                                sheet="my_sheet")
        self.assertEqual(len(table.domain.attributes), 4)
        self.assertEqual(table.name, 'header_0_sheet-my_sheet')


class TestExcelHeader1(unittest.TestCase):
    def test_no_flags(self):
        table = read_file("header_1_no_flags.xlsx")
        domain = table.domain
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 4)
        self.assertIsInstance(domain[0], DiscreteVariable)
        self.assertIsInstance(domain[1], ContinuousVariable)
        self.assertIsInstance(domain[2], DiscreteVariable)
        self.assertIsInstance(domain[3], ContinuousVariable)
        for i, var in enumerate(domain.variables):
            self.assertEqual(var.name, chr(97 + i))
        self.assertEqual(domain[0].values, ["green", "red"])
        np.testing.assert_almost_equal(table.X,
                                       np.array([[1, 0.5, 0, 21],
                                                 [1, 0.1, 0, 123],
                                                 [0, 0, np.nan, 0]]))
        np.testing.assert_equal(table.Y, np.array([]).reshape(3, 0))

    def test_flags(self):
        table = read_file("header_1_flags.xlsx")
        domain = table.domain

        self.assertEqual(len(domain.attributes), 1)
        attr = domain.attributes[0]
        self.assertEqual(attr.name, "d")
        self.assertIsInstance(attr, ContinuousVariable)
        np.testing.assert_almost_equal(table.X, np.arange(23).reshape(23, 1))

        self.assertEqual(len(domain.class_vars), 1)
        class_ = domain.class_var
        self.assertEqual(class_.name, "b")
        self.assertIsInstance(class_, ContinuousVariable)
        np.testing.assert_almost_equal(
            table.Y, np.array([.5, .1, 0, 0] * 5 + [.5, .1, 0]))

        self.assertEqual(len(domain.metas), 3)
        for n, var in zip("acf", domain.metas):
            self.assertEqual(var.name, n)
        self.assertIsInstance(domain.metas[0], DiscreteVariable)
        self.assertEqual(domain.metas[0].values, ["green", "red"])
        self.assertIsInstance(domain.metas[1], ContinuousVariable)
        np.testing.assert_almost_equal(
            table.metas[:, 0], np.array([1, 1, 0] * 7 + [1, 1]))
        np.testing.assert_almost_equal(
            table.metas[:, 1], np.array([0, 1, 2, 3] * 5 + [0, 1, 2]))


class TestExcelHeader3(unittest.TestCase):
    def test_read(self):
        table = read_file("header_3.xlsx")
        domain = table.domain

        self.assertEqual(len(domain.attributes), 2)
        attr = domain.attributes[0]
        self.assertEqual(attr.name, "d")
        self.assertIsInstance(attr, ContinuousVariable)
        np.testing.assert_almost_equal(table.X[:, 0], np.arange(23))
        attr = domain.attributes[1]
        self.assertEqual(attr.name, "g")
        self.assertIsInstance(attr, DiscreteVariable)
        np.testing.assert_almost_equal(table.X[:, 1],
                                       np.array([1, 0] + [float("nan")] * 21))

        self.assertEqual(len(domain.class_vars), 1)
        class_ = domain.class_var
        self.assertEqual(class_.name, "b")
        self.assertIsInstance(class_, ContinuousVariable)
        np.testing.assert_almost_equal(
            table.Y, np.array([.5, .1, 0, 0] * 5 + [.5, .1, 0]))

        self.assertEqual(len(domain.metas), 3)
        for n, var in zip("acf", domain.metas):
            self.assertEqual(var.name, n)
        self.assertIsInstance(domain.metas[0], DiscreteVariable)
        self.assertEqual(domain.metas[0].values, ["green", "red"])
        self.assertIsInstance(domain.metas[1], ContinuousVariable)
        np.testing.assert_almost_equal(
            table.metas[:, 0], np.array([1, 1, 0] * 7 + [1, 1]))
        np.testing.assert_almost_equal(
            table.metas[:, 1], np.array([0, 1, 2, 3] * 5 + [0, 1, 2]))
        np.testing.assert_equal(
            table.metas[:, 2], np.array(list("abcdefghijklmnopqrstuvw")))


if __name__ == "__main__":
    unittest.main()

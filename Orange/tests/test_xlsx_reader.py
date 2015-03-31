import unittest
import os

import numpy as np

from Orange.data import io, ContinuousVariable, DiscreteVariable, StringVariable


def read_file(name):
    return io.ExcelFormat().read_file(
        os.path.join(os.path.dirname(__file__), "xlsx_files", name))


class TestExcelHeader0(unittest.TestCase):
    def test_read(self):
        table = read_file("header_0.xlsx")
        domain = table.domain
        self.assertEqual(len(domain.class_vars), 0)
        self.assertIsNone(domain.class_var)
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 4)
        for i, var in enumerate(domain.attributes):
            self.assertIsInstance(var, ContinuousVariable)
            self.assertEqual(var.name, "Var{:04}".format(i + 1))
        np.testing.assert_almost_equal(table.X,
                                       np.array([[0.1, 0.5, 0.1, 21],
                                                 [0.2, 0.1, 2.5, 123],
                                                 [0, 0, 0, 0]]))


class TextExcelSheets(unittest.TestCase):
    def test_named_sheet(self):
        table = read_file("header_0_sheet.xlsx:my_sheet")
        self.assertEqual(len(table.domain.attributes), 4)


class TestExcelHeader1(unittest.TestCase):
    def test_no_flags(self):
        table = read_file("header_1_no_flags.xlsx")
        domain = table.domain
        self.assertEqual(len(domain.class_vars), 1)
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 3)
        for i, var in enumerate(domain.variables):
            self.assertIsInstance(var,
                                  [DiscreteVariable, ContinuousVariable][i > 0])
            self.assertEqual(var.name, chr(97 + i))
        self.assertEqual(domain[0].values, ["green", "red"])
        np.testing.assert_almost_equal(table.X,
                                       np.array([[1, 0.5, 0],
                                                 [1, 0.1, 0],
                                                 [0, 0, float("nan")]]))
        np.testing.assert_almost_equal(table.Y, np.array([21, 123, 0]))

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
        self.assertIsInstance(class_, DiscreteVariable)
        np.testing.assert_almost_equal(
            table.Y, np.array([2, 1, 0, 0] * 5 + [2, 1, 0]))

        self.assertEqual(len(domain.metas), 3)
        for n, var in zip("acf", domain.metas):
            self.assertEqual(var.name, n)
        self.assertIsInstance(domain.metas[0], DiscreteVariable)
        self.assertEqual(domain.metas[0].values, ["green", "red"])
        self.assertIsInstance(domain.metas[1], ContinuousVariable)
        self.assertIsInstance(domain.metas[2], StringVariable)
        np.testing.assert_almost_equal(
            table.metas[:, 0], np.array([1, 1, 0] * 7 + [1, 1]))
        np.testing.assert_almost_equal(
            table.metas[:, 1], np.array([0, 1, 2, 3] * 5 + [0, 1, 2]))
        np.testing.assert_equal(
            table.metas[:, 2], np.array(list("abcdefghijklmnopqrstuvw")))

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
        self.assertIsInstance(class_, DiscreteVariable)
        np.testing.assert_almost_equal(
            table.Y, np.array([2, 1, 0, 0] * 5 + [2, 1, 0]))

        self.assertEqual(len(domain.metas), 3)
        for n, var in zip("acf", domain.metas):
            self.assertEqual(var.name, n)
        self.assertIsInstance(domain.metas[0], DiscreteVariable)
        self.assertEqual(domain.metas[0].values, ["green", "red"])
        self.assertIsInstance(domain.metas[1], ContinuousVariable)
        self.assertIsInstance(domain.metas[2], StringVariable)
        np.testing.assert_almost_equal(
            table.metas[:, 0], np.array([1, 1, 0] * 7 + [1, 1]))
        np.testing.assert_almost_equal(
            table.metas[:, 1], np.array([0, 1, 2, 3] * 5 + [0, 1, 2]))
        np.testing.assert_equal(
            table.metas[:, 2], np.array(list("abcdefghijklmnopqrstuvw")))

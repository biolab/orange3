import unittest

import numpy as np

from Orange.data import io, ContinuousVariable, DiscreteVariable, StringVariable

class TestExcelHeader0(unittest.TestCase):
    def test_read(self):
        table = io.ExcelReader().read_file("xlsx_files/header_0.xlsx")
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

class TestExcelHeader1(unittest.TestCase):
    def test_just_floats(self):
        table = io.ExcelReader().read_file("xlsx_files/header_1_floats.xlsx")
        domain = table.domain
        self.assertEqual(len(domain.class_vars), 1)
        self.assertEqual(len(domain.metas), 0)
        self.assertEqual(len(domain.attributes), 3)
        for i, var in enumerate(domain.variables):
            self.assertIsInstance(var, ContinuousVariable)
            self.assertEqual(var.name, chr(97 + i))
        np.testing.assert_almost_equal(table.X,
                                       np.array([[0.1, 0.5, 0.1],
                                                 [0.2, 0.1, 2.5],
                                                 [0, 0, 0]]))
        np.testing.assert_almost_equal(table.Y, np.array([21, 123, 0]))

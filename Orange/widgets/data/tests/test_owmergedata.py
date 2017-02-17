import unittest
from collections import defaultdict

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.tests import test_filename
from Orange.widgets.data.owmergedata import group_table_indices, OWMergeData
from Orange.widgets.tests.base import WidgetTest


class MergeDataTest(unittest.TestCase):
    def test_group_table_indices(self):
        table = Table(test_filename("test9.tab"))
        dd = defaultdict(list)
        dd[("1",)] = [0, 1]
        dd[("huh",)] = [2]
        dd[("hoy",)] = [3]
        dd[("?",)] = [4]
        dd[("2",)] = [5]
        dd[("oh yeah",)] = [6]
        dd[("3",)] = [7]
        self.assertEqual(dd, group_table_indices(table, ["g"]))


class TestOWMergeData(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWMergeData)
        xA = np.array([[3], [1], [2], [4]])
        yA = np.array([1, 2, 3, 4])
        domainA = Domain([TimeVariable("tA")], ContinuousVariable("clsA"))
        self.tableA = Table(domainA, xA, yA)
        xB = np.array([[2, 11], [4, 22], [6, 33], [4, 23], [np.nan, 24]])
        yB = np.array([10, 20, 30, 21, 22])
        domainB = Domain([TimeVariable("tB"), ContinuousVariable("cB")],
                         ContinuousVariable("clsB"))
        self.tableB = Table(domainB, xB, yB)

    def test_merge_timestamp(self):
        self.send_signal("Data A", self.tableA)
        self.send_signal("Data B", self.tableB)
        self.widget.attr_a = "tA"
        self.widget.attr_b = "tB"
        self.widget.commit()
        tableAB = self.get_output("Merged Data A+B")
        tableBA = self.get_output("Merged Data B+A")
        self.assertEqual(
            self.tableA.domain.class_vars + self.tableB.domain.class_vars,
            tableAB.domain.class_vars)
        self.assertEqual(
            self.tableA.domain.attributes + self.tableB.domain.attributes,
            tableAB.domain.attributes)

        resultAB_X = np.array([[3, 2, 11],
                               [1, np.nan, np.nan],
                               [2, 2, 11],
                               [4, 4, 22]])
        resultAB_Y = np.array([[1, 10],
                               [2, np.nan],
                               [3, 10],
                               [4, 20]])
        resultBA_X = np.array([[2, 11, 2],
                               [4, 22, 4],
                               [6, 33, 4],
                               [4, 23, 4],
                               [np.nan, 24, np.nan]])
        resultBA_Y = np.array([[10, 3],
                               [20, 4],
                               [30, 4],
                               [21, 4],
                               [22, np.nan]])
        np.testing.assert_array_equal(tableAB.X, resultAB_X)
        np.testing.assert_array_equal(tableAB.Y, resultAB_Y)
        np.testing.assert_array_equal(tableBA.X, resultBA_X)
        np.testing.assert_array_equal(tableBA.Y, resultBA_Y)

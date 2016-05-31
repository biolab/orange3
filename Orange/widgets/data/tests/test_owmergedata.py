import unittest
from collections import defaultdict

from Orange.data import Table
from Orange.widgets.data.owmergedata import group_table_indices
from Orange.tests import test_filename


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

import unittest
from collections import defaultdict

from os.path import dirname

from Orange.data import Table, dataset_dirs
from Orange.widgets.data.owmergedata import group_table_indices

class MergeDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from Orange import tests
        dataset_dirs.append(dirname(tests.__file__))

    def test_group_table_indices(self):
        table = Table("test9.tab")
        dd = defaultdict(list)
        dd[("1",)] = [0, 1]
        dd[("huh",)] = [2]
        dd[("hoy",)] = [3]
        dd[("?",)] = [4]
        dd[("2",)] = [5]
        dd[("oh yeah",)] = [6]
        dd[("3",)] = [7]
        self.assertEqual(dd, group_table_indices(table, ["g"]))

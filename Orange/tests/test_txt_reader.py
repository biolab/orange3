import unittest
from tempfile import NamedTemporaryFile
import os

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.data.io import TxtReader

tab_file = """\
Feature 1\tFeature 2\tFeature 3
1.0      \t1.3        \t5
2.0      \t42        \t7
"""

spc_file = """\
Feature_1 Feature_2 Feature_3
1.0      1.3        5
2.0      42         7
"""

csv_file = """\
Feature 1,   Feature 2,Feature 3
1.0,      1.3,       5
2.0,      42,        7
"""

tab_file_nh = """\
1.0      \t1.3        \t5
2.0      \t42        \t7
"""

spc_file_nh = """\
1.0      1.3        5
2.0      42         7
"""

csv_file_nh = """\
1.0,      1.3,       5
2.0,      42,        7
"""


class TestTabReader(unittest.TestCase):
    def read_easy(self, s, name):
        file = NamedTemporaryFile("wt", delete=False)
        filename = file.name
        try:
            file.write(s)
            file.close()
            table = TxtReader().read_file(filename)

            f1, f2, f3 = table.domain.variables
            self.assertIsInstance(f1, ContinuousVariable)
            self.assertEqual(f1.name, name + "1")
            self.assertIsInstance(f2, ContinuousVariable)
            self.assertEqual(f2.name, name + "2")
            self.assertIsInstance(f3, ContinuousVariable)
            self.assertEqual(f3.name, name + "3")

            self.assertEqual(len(table.domain.class_vars), 0)
        finally:
            os.remove(filename)

    def test_read_tab(self):
        self.read_easy(tab_file, "Feature ")
        self.read_easy(tab_file_nh, "Var000")

    def test_read_spc(self):
        self.read_easy(spc_file, "Feature_")
        self.read_easy(spc_file_nh, "Var000")

    def test_read_csv(self):
        self.read_easy(csv_file, "Feature ")
        self.read_easy(csv_file_nh, "Var000")


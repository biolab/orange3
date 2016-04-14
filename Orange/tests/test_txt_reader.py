# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from tempfile import NamedTemporaryFile
import os
import warnings
from io import StringIO

import numpy as np

from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.data.io import CSVFormat

tab_file = """\
Feature 1\tFeature 2\tFeature 3
1.0      \t1.3        \t5
2.0      \t42        \t7
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

csv_file_nh = """\
1.0,      1.3,       5
2.0,      42,        7
"""

noncont_marked_cont = '''\
a,b
d,c
,
e,1
f,g
'''


class TestTabReader(unittest.TestCase):
    def read_easy(self, s, name):
        file = NamedTemporaryFile("wt", delete=False)
        filename = file.name
        try:
            file.write(s)
            file.close()
            table = CSVFormat().read_file(filename)

            f1, f2, f3 = table.domain
            self.assertIsInstance(f1, DiscreteVariable)
            self.assertEqual(f1.name, name + "1")
            self.assertIsInstance(f2, ContinuousVariable)
            self.assertEqual(f2.name, name + "2")
            self.assertIsInstance(f3, ContinuousVariable)
            self.assertEqual(f3.name, name + "3")
        finally:
            os.remove(filename)

    def test_read_tab(self):
        self.read_easy(tab_file, "Feature ")
        self.read_easy(tab_file_nh, "Feature ")

    def test_read_csv(self):
        self.read_easy(csv_file, "Feature ")
        self.read_easy(csv_file_nh, "Feature ")

    def test_read_nonutf8_encoding(self):
        with self.assertRaises(ValueError) as cm:
            data = Table('binary-blob.tab')
        self.assertIn('NULL byte', cm.exception.args[0])

        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                data = Table('invalid_characters.tab')

    def test_noncontinous_marked_continuous(self):
        file = NamedTemporaryFile("wt", delete=False)
        file.write(noncont_marked_cont)
        file.close()
        with self.assertRaises(ValueError) as cm:
            table = CSVFormat().read_file(file.name)
        self.assertIn('line 5, column 2', cm.exception.args[0])

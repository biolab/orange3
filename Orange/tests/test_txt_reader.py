# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from tempfile import NamedTemporaryFile
import os
import io
import warnings

from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.data.io import CSVReader
from Orange.tests import test_filename

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


csv_file_missing = """\
A,B
1,A
2,B
3,A
?,B
5,?
"""


class TestTabReader(unittest.TestCase):
    def read_easy(self, s, name):
        file = NamedTemporaryFile("wt", delete=False)
        filename = file.name
        try:
            file.write(s)
            file.close()
            table = CSVReader(filename).read()

            f1, f2, f3 = table.domain.variables
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

    def test_read_csv_with_na(self):
        c = io.StringIO(csv_file_missing)
        table = CSVReader(c).read()
        f1, f2 = table.domain.variables
        self.assertIsInstance(f1, ContinuousVariable)
        self.assertIsInstance(f2, DiscreteVariable)

    def test_read_nonutf8_encoding(self):
        with self.assertRaises(ValueError) as cm:
            data = Table(test_filename('datasets/binary-blob.tab'))
        self.assertIn('NUL', cm.exception.args[0])

        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                data = Table(test_filename('datasets/invalid_characters.tab'))

    def test_noncontinous_marked_continuous(self):
        file = NamedTemporaryFile("wt", delete=False)
        file.write(noncont_marked_cont)
        file.close()
        with self.assertRaises(ValueError) as cm:
            table = CSVReader(file.name).read()
        self.assertIn('line 5, column 2', cm.exception.args[0])

    def test_pr1734(self):
        ContinuousVariable('foo')
        file = NamedTemporaryFile("wt", delete=False)
        filename = file.name
        try:
            file.write('''\
foo
time

123123123
''')
            file.close()
            CSVReader(filename).read()
        finally:
            os.remove(filename)

    def test_csv_sniffer(self):
        # GH-2785
        reader = CSVReader(test_filename('datasets/test_asn_data_working.csv'))
        data = reader.read()
        self.assertEqual(len(data), 8)
        self.assertEqual(len(data.domain.variables) + len(data.domain.metas), 15)

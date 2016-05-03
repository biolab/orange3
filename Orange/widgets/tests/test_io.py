import unittest
import os
from Orange.data import Table
from Orange.data.io import CSVReader


class IOFileFormatTest(unittest.TestCase):
    CSV_FILE = "test_format.csv"

    def tearDown(self):
        os.remove(self.CSV_FILE)

    def test_csv_format_missing_values(self):
        f = CSVReader()
        data = Table("../../tests/test9.tab")
        self.assertTrue(any([x is "" for instance in
                             data.metas for x in instance]))
        f.write_file(self.CSV_FILE, data)
        new_data = f.read_file(self.CSV_FILE)
        for new_instance, old_instance in zip(new_data, data):
            self.assertEqual(new_instance, old_instance)

import unittest

from Orange.data.io import UrlReader


class TestUrlReader(unittest.TestCase):
    def test_basic_file(self):
        data = UrlReader("https://datasets.biolab.si/core/titanic.tab").read()
        self.assertEqual(2201, len(data))

        data = UrlReader("https://datasets.biolab.si/core/grades.xlsx").read()
        self.assertEqual(16, len(data))

    def test_zipped(self):
        """ Test zipped files with two extensions"""
        data = UrlReader(
            "http://datasets.biolab.si/core/philadelphia-crime.csv.xz"
        ).read()
        self.assertEqual(9666, len(data))

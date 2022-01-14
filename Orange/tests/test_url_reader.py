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

    def test_special_characters(self):
        # TO-DO - replace this file with a more appropriate one (e.g. .csv)
        #  and change the assertion accordingly
        path = "http://file.biolab.si/text-semantics/data/elektrotehniski-" \
               "vestnik-clanki/detektiranje-utrdb-v-šahu-.txt"
        self.assertRaises(OSError, UrlReader(path).read)

    def test_base_url_with_parameters(self):
        data = UrlReader("https://datasets.biolab.si/core/grades.xlsx?a=1&b=2").read()
        self.assertEqual(16, len(data))

    def test_special_characters_with_parameters(self):
        path = "http://file.biolab.si/text-semantics/data/elektrotehniski-" \
               "vestnik-clanki/detektiranje-utrdb-v-šahu-.txt?a=1&b=2"
        self.assertRaises(OSError, UrlReader(path).read)


if __name__ == "__main__":
    unittest.main()

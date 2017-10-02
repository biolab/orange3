# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import os
import tempfile
import shutil

from Orange.data.io import FileFormat, TabReader, CSVReader, PickleReader
from Orange.data.table import get_sample_datasets_dir


class WildcardReader(FileFormat):
    EXTENSIONS = ('.wild', '.wild[0-9]')
    DESCRIPTION = "Dummy reader for testing extensions"

    def read(self):
        pass


class TestChooseReader(unittest.TestCase):

    def test_usual_extensions(self):
        self.assertIsInstance(FileFormat.get_reader("t.tab"), TabReader)
        self.assertIsInstance(FileFormat.get_reader("t.csv"), CSVReader)
        self.assertIsInstance(FileFormat.get_reader("t.pkl"), PickleReader)
        with self.assertRaises(OSError):
            FileFormat.get_reader("test.undefined_extension")

    def test_wildcard_extension(self):
        self.assertIsInstance(FileFormat.get_reader("t.wild"),
                              WildcardReader)
        self.assertIsInstance(FileFormat.get_reader("t.wild2"),
                              WildcardReader)
        with self.assertRaises(OSError):
            FileFormat.get_reader("t.wild2a")


class SameExtension(FileFormat):
    PRIORITY = 100
    EXTENSIONS = ('.same_extension',)
    DESCRIPTION = "Same extension, different priority"

    def read(self):
        pass


class SameExtensionPreferred(SameExtension):
    PRIORITY = 90


class SameExtensionL(SameExtension):
    PRIORITY = 110


class TestMultipleSameExtension(unittest.TestCase):

    def test_find_reader(self):
        reader = FileFormat.get_reader("some.same_extension")
        self.assertIsInstance(reader, SameExtensionPreferred)


class TestLocate(unittest.TestCase):

    def test_locate_sample_datasets(self):
        with self.assertRaises(OSError):
            FileFormat.locate("iris.tab",
                              search_dirs=[os.path.dirname(__file__)])
        iris = FileFormat.locate("iris.tab",
                                 search_dirs=[get_sample_datasets_dir()])
        self.assertEqual(os.path.basename(iris), "iris.tab")
        # test extension adding
        iris = FileFormat.locate("iris",
                                 search_dirs=[get_sample_datasets_dir()])
        self.assertEqual(os.path.basename(iris), "iris.tab")

    def test_locate_wildcard_extension(self):
        tempdir = tempfile.mkdtemp()
        with self.assertRaises(OSError):
            FileFormat.locate("t.wild9", search_dirs=[tempdir])
        fn = os.path.join(tempdir, "t.wild8")
        with open(fn, "wt") as f:
            f.write("\n")
        l = FileFormat.locate("t.wild8", search_dirs=[tempdir])
        self.assertEqual(l, fn)
        # test extension adding
        l = FileFormat.locate("t", search_dirs=[tempdir])
        self.assertEqual(l, fn)
        shutil.rmtree(tempdir)


class TestReader(unittest.TestCase):

    def test_open_bad_pickle(self):
        """
        Raise TypeError when PickleReader reads a pickle
        file without a table (and it suppose to be there).
        GH-2232
        """
        reader = PickleReader("")
        with unittest.mock.patch("pickle.load", return_value=None):
            self.assertRaises(TypeError, reader.read, "foo")

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import io
import os
from os import path, remove
import unittest
import shutil
import time
from collections import OrderedDict
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np

from Orange.data import Table, DiscreteVariable
from Orange.data.io import TabReader


def read_tab_file(filename):
    return TabReader(filename).read()


class TestTabReader(unittest.TestCase):

    def setUp(self):
        self.data = Table.from_numpy(None, [[1, 2, 3]])

    def test_read_easy(self):
        simplefile = """\
        Feature 1\tFeature 2\tClass 1\tClass 42
        d        \tM F      \td      \td
                 \t         \tclass  \tclass
        1.0      \tM        \t5      \trich
                 \tF        \t7      \tpoor
        2.0      \tM        \t4      \t
        """

        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(simplefile)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f1, DiscreteVariable)
        self.assertEqual(f1.name, "Feature 1")
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertIsInstance(c1, DiscreteVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertIsInstance(c2, DiscreteVariable)
        self.assertEqual(c2.name, "Class 42")

        np.testing.assert_almost_equal(table.X, np.array([[0, 0], [np.nan, 1], [1, 0]]))
        np.testing.assert_almost_equal(table.Y, np.array([[1, 1], [2, 0], [0, np.nan]]))

    def test_read_save_quoted(self):
        quoted = '''\
        S\tA
        s\td
        m\t
        """a"""\ti
        """b"""\tj
        """c\td"""\tk
        '''
        expected = ['"a"', '"b"', '"c\td"']
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(quoted)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)
        self.assertSequenceEqual(table.metas[:, 0].tolist(), expected)

        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            pass
        TabReader.write_file(tmp.name, table)
        with open(tmp.name, encoding="utf-8") as f:
            saved = f.read()
        os.unlink(tmp.name)

        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(saved)
        table1 = read_tab_file(tmp.name)
        os.unlink(tmp.name)

        self.assertSequenceEqual(table1.metas[:, 0].tolist(), expected)

    def test_read_and_save_attributes(self):
        samplefile = """\
        Feature 1\tFeature 2\tClass 1\tClass 42
        d        \tM F      \td      \td
                 \ta=1 b=2 \tclass x=a\\ longer\\ string \tclass
        1.0      \tM        \t5      \trich
        """
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(samplefile)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)

        _, f2, c1, _ = table.domain.variables
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertEqual(f2.attributes, {'a': 1, 'b': 2})
        self.assertIn(c1, table.domain.class_vars)
        self.assertIsInstance(c1, DiscreteVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertEqual(c1.attributes, {'x': 'a longer string'})

        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            pass
        TabReader.write_file(tmp.name, table)
        with open(tmp.name, encoding="utf-8") as f:
            saved = f.read()
        os.unlink(tmp.name)

        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(saved)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)

        _, f2, c1, _ = table.domain.variables
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertEqual(f2.attributes, {'a': 1, 'b': 2})
        self.assertIn(c1, table.domain.class_vars)
        self.assertIsInstance(c1, DiscreteVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertEqual(c1.attributes, {'x': 'a longer string'})

        spath = "/path/to/somewhere"
        c1.attributes["path"] = spath
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            pass
        TabReader.write_file(tmp.name, table)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)
        _, _, c1, _ = table.domain.variables
        self.assertEqual(c1.attributes["path"], spath)

    def test_read_data_oneline_header(self):
        samplefile = """\
        data1\tdata2\tdata3
        0.1\t0.2\t0.3
        1.1\t1.2\t1.5
        """
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(samplefile)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(len(table), 2)
        self.assertEqual(len(table.domain.variables), 3)
        self.assertEqual(table.domain[0].name, 'data1')

    def test_read_data_no_header(self):
        samplefile = """\
        0.1\t0.2\t0.3
        1.1\t1.2\t1.5
        """
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(samplefile)
        table = read_tab_file(tmp.name)
        os.unlink(tmp.name)

        self.assertEqual(len(table), 2)
        self.assertEqual(len(table.domain.variables), 3)
        self.assertTrue(table.domain[0].is_continuous)
        self.assertEqual(table.domain[0].name, 'Feature 1')

    def test_read_data_no_header_feature_reuse(self):
        samplefile = """\
        0.1\t0.2\t0.3
        1.1\t1.2\t1.5
        """
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(samplefile)
        t1 = read_tab_file(tmp.name)
        os.unlink(tmp.name)
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(samplefile)
        t2 = read_tab_file(tmp.name)
        os.unlink(tmp.name)
        self.assertEqual(t1.domain[0], t2.domain[0])

    def test_renaming(self):
        simplefile = """\
            a\t  b\t  a\t  a\t  b\t     a\t     c\t  a\t b
            c\t  c\t  c\t  c\t  c\t     c\t     c\t  c\t c
             \t   \t  \t   \t   class\t class\t  \t  \t  meta
            0\t  0\t  0\t  0\t  0\t     0\t     0\t  0 """
        file = NamedTemporaryFile("wt", delete=False, suffix=".tab")
        filename = file.name
        try:
            file.write(simplefile)
            file.close()
            table = read_tab_file(filename)
            domain = table.domain
            self.assertEqual([x.name for x in domain.attributes],
                             ["a (1)", "b (1)", "a (2)", "a (3)", "c", "a (5)"])
            self.assertEqual([x.name for x in domain.class_vars], ["b (2)", "a (4)"])
            self.assertEqual([x.name for x in domain.metas], ["b (3)"])
        finally:
            remove(filename)


    def test_dataset_with_weird_names_and_column_attributes(self):
        data = Table(path.join(path.dirname(__file__), 'datasets/weird.tab'))
        self.assertEqual(len(data), 6)
        self.assertEqual(len(data.domain.variables), 1)
        self.assertEqual(len(data.domain.metas), 1)
        NAME = ['5534fab7fad58d5df50061f1', '5534fab8fad58d5de20061f8']
        self.assertEqual(data.domain[0].name, str(NAME))
        ATTRIBUTES = dict(
            Timepoint=20,
            id=NAME,
            Name=['Gene expressions (dd_AX4_on_Ka_20Hr_bio1_mapped.bam)',
                  'Gene expressions (dd_AX4_on_Ka_20Hr_bio2_mapped.bam)'],
            Replicate=['1', '2'],
        )
        self.assertEqual(data.domain[0].attributes, ATTRIBUTES)

    def test_sheets(self):
        file1 = io.StringIO("\n".join("xd dbac"))
        reader = TabReader(file1)

        self.assertEqual(reader.sheets, [])

    def test_attributes_saving(self):
        tempdir = mkdtemp()
        try:
            self.assertEqual(self.data.attributes, {})
            self.data.attributes[1] = "test"
            self.data.save(path.join(tempdir, "out.tab"))
            table = Table(path.join(tempdir, "out.tab"))
            self.assertEqual(table.attributes[1], "test")
        finally:
            shutil.rmtree(tempdir)

    def test_attributes_saving_as_txt(self):
        tempdir = mkdtemp()
        try:
            self.data.attributes = OrderedDict()
            self.data.attributes["a"] = "aa"
            self.data.attributes["b"] = "bb"
            self.data.save(path.join(tempdir, "out.tab"))
            table = Table(path.join(tempdir, "out.tab"))
            self.assertIsInstance(table.attributes, OrderedDict)
            self.assertEqual(table.attributes["a"], "aa")
            self.assertEqual(table.attributes["b"], "bb")
        finally:
            shutil.rmtree(tempdir)

    def test_data_name(self):
        table1 = Table('iris')
        table2 = TabReader(table1.__file__).read()
        self.assertEqual(table1.name, 'iris')
        self.assertEqual(table2.name, 'iris')

    def test_metadata(self):
        tempdir = mkdtemp()
        try:
            self.data.attributes = OrderedDict()
            self.data.attributes["a"] = "aa"
            self.data.attributes["b"] = "bb"
            fname = path.join(tempdir, "out.tab")
            TabReader.write_table_metadata(fname, self.data)
            self.assertTrue(path.isfile(fname + ".metadata"))
        finally:
            shutil.rmtree(tempdir)

    def test_no_metadata(self):
        tempdir = mkdtemp()
        try:
            self.data.attributes = OrderedDict()
            fname = path.join(tempdir, "out.tab")
            TabReader.write_table_metadata(fname, self.data)
            self.assertFalse(path.isfile(fname + ".metadata"))
        finally:
            shutil.rmtree(tempdir)

    def test_had_metadata_now_there_is_none(self):
        tempdir = mkdtemp()
        try:
            self.data.attributes["a"] = "aa"
            fname = path.join(tempdir, "out.tab")
            TabReader.write_table_metadata(fname, self.data)
            self.assertTrue(path.isfile(fname + ".metadata"))
            del self.data.attributes["a"]
            TabReader.write_table_metadata(fname, self.data)
            self.assertFalse(path.isfile(fname + ".metadata"))
        finally:
            shutil.rmtree(tempdir)

    def test_number_of_decimals(self):
        data = Table("heart_disease")
        self.assertEqual(data.domain["age"].number_of_decimals, 0)
        self.assertEqual(data.domain["ST by exercise"].number_of_decimals, 1)

        data = Table("housing")
        self.assertEqual(data.domain["CRIM"].number_of_decimals, 5)
        self.assertEqual(data.domain["INDUS"].number_of_decimals, 2)
        self.assertEqual(data.domain["AGE"].number_of_decimals, 1)

    @staticmethod
    def test_many_discrete():
        with NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("Poser\nd\n\n")
            tmp.writelines("K" + str(i) + "\n" for i in range(30000))
        start = time.time()
        _ = TabReader(tmp.name).read()
        elapsed = time.time() - start
        os.unlink(tmp.name)
        if elapsed > 2:
            raise AssertionError()


if __name__ == "__main__":
    unittest.main()

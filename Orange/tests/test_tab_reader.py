# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import io
from os import path, remove
import unittest
import tempfile
import shutil
import pickle
from collections import OrderedDict

import numpy as np

from Orange.data import Table, DiscreteVariable
from Orange.data.io import TabReader


def read_tab_file(filename):
    return TabReader(filename).read()


class TestTabReader(unittest.TestCase):

    def setUp(self):
        DiscreteVariable._clear_cache()

    def test_read_easy(self):
        simplefile = """\
        Feature 1\tFeature 2\tClass 1\tClass 42
        d        \tM F      \td      \td
                 \t         \tclass  \tclass
        1.0      \tM        \t5      \trich
                 \tF        \t7      \tpoor
        2.0      \tM        \t4      \t
        """

        file = io.StringIO(simplefile)
        table = read_tab_file(file)

        f1, f2, c1, c2 = table.domain
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

    def test_read_and_save_attributes(self):
        samplefile = """\
        Feature 1\tFeature 2\tClass 1\tClass 42
        d        \tM F      \td      \td
                 \ta=1 b=2 \tclass x=a\\ longer\\ string \tclass
        1.0      \tM        \t5      \trich
        """
        file = io.StringIO(samplefile)
        table = read_tab_file(file)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertEqual(f2.attributes, {'a': 1, 'b': 2})
        self.assertIn(c1, table.domain.class_vars)
        self.assertIsInstance(c1, DiscreteVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertEqual(c1.attributes, {'x': 'a longer string'})
        outf = io.StringIO()
        outf.close = lambda: None
        TabReader.write_file(outf, table)
        saved = outf.getvalue()

        file = io.StringIO(saved)
        table = read_tab_file(file)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertEqual(f2.attributes, {'a': 1, 'b': 2})
        self.assertIn(c1, table.domain.class_vars)
        self.assertIsInstance(c1, DiscreteVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertEqual(c1.attributes, {'x': 'a longer string'})

        path = "/path/to/somewhere"
        c1.attributes["path"] = path
        outf = io.StringIO()
        outf.close = lambda: None
        TabReader.write_file(outf, table)
        outf.seek(0)

        table = read_tab_file(outf)
        f1, f2, c1, c2 = table.domain.variables
        self.assertEqual(c1.attributes["path"], path)

    def test_read_data_oneline_header(self):
        samplefile = """\
        data1\tdata2\tdata3
        0.1\t0.2\t0.3
        1.1\t1.2\t1.5
        """
        file = io.StringIO(samplefile)
        table = read_tab_file(file)

        self.assertEqual(len(table), 2)
        self.assertEqual(len(table.domain), 3)
        self.assertEqual(table.domain[0].name, 'data1')

    def test_read_data_no_header(self):
        samplefile = """\
        0.1\t0.2\t0.3
        1.1\t1.2\t1.5
        """
        file = io.StringIO(samplefile)
        table = read_tab_file(file)

        self.assertEqual(len(table), 2)
        self.assertEqual(len(table.domain), 3)
        self.assertTrue(table.domain[0].is_continuous)
        self.assertEqual(table.domain[0].name, 'Feature 1')

    def test_reuse_variables(self):
        file1 = io.StringIO("\n".join("xd dbac"))
        t1 = read_tab_file(file1)

        self.assertSequenceEqual(t1.domain['x'].values, 'abcd')
        np.testing.assert_almost_equal(t1.X.ravel(), [3, 1, 0, 2])

        file2 = io.StringIO("\n".join("xd hgacb"))
        t2 = read_tab_file(file2)

        self.assertSequenceEqual(t2.domain['x'].values, 'abcdgh')
        np.testing.assert_almost_equal(t2.X.ravel(), [5, 4, 0, 2, 1])

    def test_renaming(self):
        simplefile = """\
            a\t  b\t  a\t  a\t  b\t     a\t     c\t  a\t b
            c\t  c\t  c\t  c\t  c\t     c\t     c\t  c\t c
             \t   \t  \t   \t   class\t class\t  \t  \t  meta
            0\t  0\t  0\t  0\t  0\t     0\t     0\t  0 """
        file = tempfile.NamedTemporaryFile("wt", delete=False, suffix=".tab")
        filename = file.name
        try:
            file.write(simplefile)
            file.close()
            table = read_tab_file(filename)
            domain = table.domain
            self.assertEqual([x.name for x in domain.attributes],
                             ["a_1", "b_1", "a_2", "a_3", "c", "a_5"])
            self.assertEqual([x.name for x in domain.class_vars], ["b_2", "a_4"])
            self.assertEqual([x.name for x in domain.metas], ["b_3"])
        finally:
            remove(filename)


    def test_dataset_with_weird_names_and_column_attributes(self):
        data = Table(path.join(path.dirname(__file__), 'weird.tab'))
        self.assertEqual(len(data), 6)
        self.assertEqual(len(data.domain), 1)
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

        self.assertEqual(reader.sheets, ())

    def test_attributes_saving(self):
        tempdir = tempfile.mkdtemp()
        table = Table("titanic")
        self.assertEqual(table.attributes, {})
        table.attributes[1] = "test"
        table.save(path.join(tempdir, "out.tab"))
        table = Table(path.join(tempdir, "out.tab"))
        self.assertEqual(table.attributes[1], "test")
        shutil.rmtree(tempdir)

    def test_attributes_saving_as_txt(self):
        tempdir = tempfile.mkdtemp()
        table = Table("titanic")
        table.attributes = OrderedDict()
        table.attributes["a"] = "aa"
        table.attributes["b"] = "bb"
        table.save(path.join(tempdir, "out.tab"))
        table = Table(path.join(tempdir, "out.tab"))
        self.assertIsInstance(table.attributes, OrderedDict)
        self.assertEqual(table.attributes["a"], "aa")
        self.assertEqual(table.attributes["b"], "bb")
        shutil.rmtree(tempdir)

    def test_data_name(self):
        table1 = Table('iris')
        table2 = TabReader(table1.__file__).read()
        self.assertEqual(table1.name, 'iris')
        self.assertEqual(table2.name, 'iris')

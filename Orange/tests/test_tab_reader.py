import io
import unittest

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.data.io import TabDelimFormat


class TestTabReader(unittest.TestCase):

    def setUp(self):
        DiscreteVariable._clear_cache()

    def test_read_easy(self):
        simplefile = """\
        Feature 1\tFeature 2\tClass 1\tClass 42
        c        \tM F      \tc      \td
                 \t         \tclass  \tclass
        1.0      \tM        \t5      \trich
                 \tF        \t7      \tpoor
        2.0      \tM        \t4      \t
        """

        file = io.StringIO(simplefile)
        table = TabDelimFormat()._read_file(file)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f1, ContinuousVariable)
        self.assertEqual(f1.name, "Feature 1")
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertIsInstance(c1, ContinuousVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertIsInstance(c2, DiscreteVariable)
        self.assertEqual(c2.name, "Class 42")

        np.testing.assert_almost_equal(table.X, np.array([[1, 0], [np.nan, 1], [2, 0]]))
        np.testing.assert_almost_equal(table.Y, np.array([[5, 1], [7, 0], [4, np.nan]]))

    def test_read_and_save_attributes(self):
        samplefile = """\
        Feature 1\tFeature 2\tClass 1\tClass 42
        c        \tM F      \tc      \td
                 \ta=1 b=2 \tclass x=a\\ longer\\ string \tclass
        1.0      \tM        \t5      \trich
        """
        file = io.StringIO(samplefile)
        table = TabDelimFormat()._read_file(file)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertEqual(f2.attributes, {'a': '1', 'b': '2'})
        self.assertIn(c1, table.domain.class_vars)
        self.assertIsInstance(c1, ContinuousVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertEqual(c1.attributes, {'x': 'a longer string'})

        outf = io.StringIO()
        outf.close = lambda: None
        TabDelimFormat.write_file(outf, table)
        saved = outf.getvalue()

        file = io.StringIO(saved)
        table = TabDelimFormat()._read_file(file)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertEqual(f2.attributes, {'a': '1', 'b': '2'})
        self.assertIn(c1, table.domain.class_vars)
        self.assertIsInstance(c1, ContinuousVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertEqual(c1.attributes, {'x': 'a longer string'})

    def test_reuse_variables(self):
        file1 = io.StringIO("\n".join("xd dbac"))
        t1 = TabDelimFormat()._read_file(file1)

        self.assertSequenceEqual(t1.domain['x'].values, 'abcd')
        np.testing.assert_almost_equal(t1.X.ravel(), [3, 1, 0, 2])

        file2 = io.StringIO("\n".join("xd hgacb"))
        t2 = TabDelimFormat()._read_file(file2)

        self.assertSequenceEqual(t2.domain['x'].values, 'abcdgh')
        np.testing.assert_almost_equal(t2.X.ravel(), [5, 4, 0, 2, 1])

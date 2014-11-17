import io
import unittest

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.data.io import TabDelimReader

simplefile = """\
Feature 1\tFeature 2\tClass 1\tClass 42
c        \tM F      \tc      \td
         \t         \tclass  \tclass
1.0      \tM        \t5      \trich
         \tF        \t7      \tpoor
2.0      \tM        \t4      \t
"""


class TestTabReader(unittest.TestCase):
    def test_read_easy(self):
        file = io.StringIO(simplefile)
        table = TabDelimReader()._read_file(file)

        f1, f2, c1, c2 = table.domain.variables
        self.assertIsInstance(f1, ContinuousVariable)
        self.assertEqual(f1.name, "Feature 1")
        self.assertIsInstance(f2, DiscreteVariable)
        self.assertEqual(f2.name, "Feature 2")
        self.assertIsInstance(c1, ContinuousVariable)
        self.assertEqual(c1.name, "Class 1")
        self.assertIsInstance(c2, DiscreteVariable)
        self.assertEqual(c2.name, "Class 42")

        print(table.domain.attributes[1].values)
        np.testing.assert_almost_equal(table.X, np.array([[1, 0], [np.nan, 1], [2, 0]]))
        np.testing.assert_almost_equal(table.Y, np.array([[5, 1], [7, 0], [4, np.nan]]))

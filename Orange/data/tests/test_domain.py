import unittest

import numpy as np

from Orange.data import Domain, Table, DiscreteVariable, ContinuousVariable
from Orange.util import OrangeDeprecationWarning


class DomainTest(unittest.TestCase):
    def test_bool_raises_warning(self):
        self.assertWarns(OrangeDeprecationWarning, bool, Domain([]))
        self.assertWarns(OrangeDeprecationWarning, bool,
                         Domain([ContinuousVariable("y")]))

    def test_empty(self):
        var = ContinuousVariable("y")
        self.assertTrue(Domain([]).empty())

        self.assertFalse(Domain([var]).empty())
        self.assertFalse(Domain([], [var]).empty())
        self.assertFalse(Domain([], [], [var]).empty())

    def test_conversion(self):
        a1 = DiscreteVariable("a", values=tuple("abc"))
        a2 = DiscreteVariable("a", values=tuple("cab"))
        b1 = DiscreteVariable("b", values=tuple("def"))
        b2 = DiscreteVariable("b", values=tuple("efg"))
        c1 = ContinuousVariable("c")
        c2 = DiscreteVariable("c", values=tuple("efg"))
        dom1 = Domain([a1, b1, c1])
        dom2 = Domain([a2, b2, c2])

        data1 = Table.from_numpy(
            dom1, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
        np.testing.assert_array_equal(
            data1.transform(dom2),
            [[1, np.nan, np.nan], [2, 0, np.nan], [0, 1, np.nan]])

        data2 = Table.from_numpy(
            dom2, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
        np.testing.assert_array_equal(
            data2.transform(dom1),
            [[2, 1, np.nan], [0, 2, np.nan], [1, np.nan, np.nan]])


if __name__ == "__main__":
    unittest.main()

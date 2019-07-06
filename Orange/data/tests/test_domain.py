import unittest

from Orange.data import Domain, ContinuousVariable
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

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from unittest import TestCase

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.widgets.data.oweditdomain import EditDomainReport

SECTION_NAME = "NAME"


class TestEditDomainReport(TestCase):
    # This tests test private methods
    # pylint: disable=protected-access

    def setUp(self):
        self.report = EditDomainReport([], [])

    def test_section_yields_nothing_for_no_changes(self):
        result = self.report._section(SECTION_NAME, [])
        self.assertEmpty(result)

    def test_section_yields_header_for_changes(self):
        result = self.report._section(SECTION_NAME, ["a"])
        self.assertTrue(any(SECTION_NAME in item for item in result))

    def test_value_changes_yields_nothing_for_no_change(self):
        a = DiscreteVariable("a", values="abc")
        self.assertEmpty(self.report._value_changes(a, a))

    def test_value_changes_yields_nothing_for_continuous_variables(self):
        v1, v2 = ContinuousVariable("a"), ContinuousVariable("b")
        self.assertEmpty(self.report._value_changes(v1, v2))

    def test_value_changes_yields_changed_values(self):
        v1, v2 = DiscreteVariable("a", "ab"), DiscreteVariable("b", "ac")
        self.assertNotEmpty(self.report._value_changes(v1, v2))

    def test_label_changes_yields_nothing_for_no_change(self):
        v1 = ContinuousVariable("a")
        v1.attributes["a"] = "b"
        self.assertEmpty(self.report._value_changes(v1, v1))

    def test_label_changes_yields_added_labels(self):
        v1 = ContinuousVariable("a")
        v2 = v1.copy(None)
        v2.attributes["a"] = "b"
        self.assertNotEmpty(self.report._label_changes(v1, v2))

    def test_label_changes_yields_removed_labels(self):
        v1 = ContinuousVariable("a")
        v1.attributes["a"] = "b"
        v2 = v1.copy(None)
        del v2.attributes["a"]
        self.assertNotEmpty(self.report._label_changes(v1, v2))

    def test_label_changes_yields_modified_labels(self):
        v1 = ContinuousVariable("a")
        v1.attributes["a"] = "b"
        v2 = v1.copy(None)
        v2.attributes["a"] = "c"
        self.assertNotEmpty(self.report._label_changes(v1, v2))

    def assertEmpty(self, iterable):
        self.assertRaises(StopIteration, lambda: next(iter(iterable)))

    def assertNotEmpty(self, iterable):
        try:
            next(iter(iterable))
        except StopIteration:
            self.fail("Iterator did not produce any lines")

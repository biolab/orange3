import unittest

from Orange.data import Domain, ContinuousVariable
from Orange.data.util import get_unique_names, get_unique_names_duplicates


class TestGetUniqueNames(unittest.TestCase):
    def test_get_unique_names(self):
        names = ["foo", "bar", "baz", "baz (3)"]
        self.assertEqual(get_unique_names(names, ["qux"]), ["qux"])
        self.assertEqual(get_unique_names(names, ["foo"]), ["foo (1)"])
        self.assertEqual(get_unique_names(names, ["baz"]), ["baz (4)"])
        self.assertEqual(get_unique_names(names, ["baz (3)"]), ["baz (3) (1)"])
        self.assertEqual(
            get_unique_names(names, ["qux", "quux"]), ["qux", "quux"])
        self.assertEqual(
            get_unique_names(names, ["bar", "baz"]), ["bar (4)", "baz (4)"])
        self.assertEqual(
            get_unique_names(names, ["qux", "baz"]), ["qux (4)", "baz (4)"])
        self.assertEqual(
            get_unique_names(names, ["qux", "bar"]), ["qux (1)", "bar (1)"])

        self.assertEqual(get_unique_names(names, "qux"), "qux")
        self.assertEqual(get_unique_names(names, "foo"), "foo (1)")
        self.assertEqual(get_unique_names(names, "baz"), "baz (4)")

        self.assertEqual(get_unique_names(tuple(names), "baz"), "baz (4)")

    def test_get_unique_names_with_domain(self):
        a, b, c, d = map(ContinuousVariable, ["foo", "bar", "baz", "baz (3)"])
        domain = Domain([a, b], c, [d])
        self.assertEqual(get_unique_names(domain, ["qux"]), ["qux"])
        self.assertEqual(get_unique_names(domain, ["foo"]), ["foo (1)"])
        self.assertEqual(get_unique_names(domain, ["baz"]), ["baz (4)"])
        self.assertEqual(get_unique_names(domain, ["baz (3)"]), ["baz (3) (1)"])
        self.assertEqual(
            get_unique_names(domain, ["qux", "quux"]), ["qux", "quux"])
        self.assertEqual(
            get_unique_names(domain, ["bar", "baz"]), ["bar (4)", "baz (4)"])
        self.assertEqual(
            get_unique_names(domain, ["qux", "baz"]), ["qux (4)", "baz (4)"])
        self.assertEqual(
            get_unique_names(domain, ["qux", "bar"]), ["qux (1)", "bar (1)"])

        self.assertEqual(get_unique_names(domain, "qux"), "qux")
        self.assertEqual(get_unique_names(domain, "foo"), "foo (1)")
        self.assertEqual(get_unique_names(domain, "baz"), "baz (4)")

    def test_get_unique_names_from_duplicates(self):
        self.assertEqual(
            get_unique_names_duplicates(["foo", "bar", "baz"]),
            ["foo", "bar", "baz"])
        self.assertEqual(
            get_unique_names_duplicates(["foo", "bar", "baz", "bar"]),
            ["foo", "bar (1)", "baz", "bar (2)"])
        self.assertEqual(
            get_unique_names_duplicates(["x", "x", "x (1)"]),
            ["x (2)", "x (3)", "x (1)"])
        self.assertEqual(
            get_unique_names_duplicates(["x (2)", "x", "x", "x (2)", "x (3)"]),
            ["x (2) (1)", "x (1)", "x (4)", "x (2) (2)", "x (3)"])
        self.assertEqual(
            get_unique_names_duplicates(["x", "", "", None, None, "x"]),
            ["x (1)", "", "", None, None, "x (2)"])


if __name__ == "__main__":
    unittest.main()

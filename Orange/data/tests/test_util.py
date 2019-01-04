import unittest

from Orange.data import Domain, ContinuousVariable
from Orange.data.util import get_unique_names


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


if __name__ == "__main__":
    unittest.main()

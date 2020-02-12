import unittest

from Orange.data import Domain, ContinuousVariable
from Orange.data.util import get_unique_names, get_unique_names_duplicates, \
    get_unique_names_domain, progress_callback


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

    def test_get_unique_names_domain(self):
        (attrs, classes, metas), renamed = \
            get_unique_names_domain(["a", "t", "c", "t"], ["t", "d"], ["d", "e"])
        self.assertEqual(attrs, ["a", "t (1)", "c", "t (2)"])
        self.assertEqual(classes, ["t (3)", "d (1)"])
        self.assertEqual(metas, ["d (2)", "e"])
        self.assertEqual(renamed, ["t", "d"])

        (attrs, classes, metas), renamed = \
            get_unique_names_domain(["a", "t", "c", "t"], ["t", "d"])
        self.assertEqual(attrs, ["a", "t (1)", "c", "t (2)"])
        self.assertEqual(classes, ["t (3)", "d"])
        self.assertEqual(metas, [])
        self.assertEqual(renamed, ["t"])

        (attrs, classes, metas), renamed = \
            get_unique_names_domain(["a", "t", "c"])
        self.assertEqual(attrs, ["a", "t", "c"])
        self.assertEqual(classes, [])
        self.assertEqual(metas, [])
        self.assertEqual(renamed, [])

        (attrs, classes, metas), renamed = \
            get_unique_names_domain(["a", "t", "d", "t"], [], ["d", "e"])
        self.assertEqual(attrs, ["a", "t (1)", "d (1)", "t (2)"])
        self.assertEqual(classes, [])
        self.assertEqual(metas, ["d (2)", "e"])
        self.assertEqual(renamed, ["t", "d"])

        (attrs, classes, metas), renamed = \
            get_unique_names_domain([], ["t", "d"], ["d", "e"])
        self.assertEqual(attrs, [])
        self.assertEqual(classes, ["t", "d (1)"])
        self.assertEqual(metas, ["d (2)", "e"])
        self.assertEqual(renamed, ["d"])

        (attrs, classes, metas), renamed = \
            get_unique_names_domain([], ["t", "t", "d"], [])
        self.assertEqual(attrs, [])
        self.assertEqual(classes, ["t (1)", "t (2)", "d"])
        self.assertEqual(metas, [])
        self.assertEqual(renamed, ["t"])

        (attrs, classes, metas), renamed = \
            get_unique_names_domain([], [], [])
        self.assertEqual(attrs, [])
        self.assertEqual(classes, [])
        self.assertEqual(metas, [])
        self.assertEqual(renamed, [])


class TestProgressCallback(unittest.TestCase):
    def test_wrap(self):
        def func(i):
            return i

        f = progress_callback(func, start=0, end=0.8)
        self.assertEqual(f(0), 0)
        self.assertEqual(round(f(0.1), 2), 0.08)
        self.assertEqual(f(1), 0.8)

        f = progress_callback(func, start=0.1, end=0.8)
        self.assertEqual(f(0), 0.1)
        self.assertEqual(f(0.1), 0.17)
        self.assertEqual(f(1), 0.8)


if __name__ == "__main__":
    unittest.main()

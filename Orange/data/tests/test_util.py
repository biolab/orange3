import unittest
import numpy as np

from Orange.data import Domain, ContinuousVariable
from Orange.data.util import get_unique_names, get_unique_names_duplicates, \
    get_unique_names_domain, one_hot, sanitized_name, redefines_eq_and_hash


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

    def test_get_unique_names_not_equal(self):
        names = ["foo", "bar", "baz", "baz (3)"]
        self.assertEqual(
            get_unique_names(names, ["qux"], equal_numbers=False), ["qux"]
        )
        self.assertEqual(
            get_unique_names(names, ["foo"], equal_numbers=False), ["foo (1)"]
        )
        self.assertEqual(
            get_unique_names(names, ["baz"], equal_numbers=False), ["baz (4)"]
        )
        self.assertEqual(
            get_unique_names(names, ["baz (3)"], equal_numbers=False),
            ["baz (3) (1)"]
        )
        self.assertEqual(
            get_unique_names(names, ["qux", "quux"], equal_numbers=False),
            ["qux", "quux"]
        )
        self.assertEqual(
            get_unique_names(names, ["bar", "baz"], equal_numbers=False),
            ["bar (1)", "baz (4)"]
        )
        self.assertEqual(
            get_unique_names(names, ["qux", "baz"], equal_numbers=False),
            ["qux", "baz (4)"]
        )
        self.assertEqual(
            get_unique_names(names, ["qux", "bar"], equal_numbers=False),
            ["qux", "bar (1)"]
        )
        self.assertEqual(
            get_unique_names(names, ["foo", "bar", "baz"], equal_numbers=False),
            ["foo (1)", "bar (1)", "baz (4)"]
        )

        a, b, c, d = map(ContinuousVariable, ["foo", "bar", "baz", "baz (3)"])
        domain = Domain([a, b], c, [d])
        self.assertEqual(
            get_unique_names(names, ["qux"], equal_numbers=False), ["qux"]
        )
        self.assertEqual(
            get_unique_names(names, ["foo"], equal_numbers=False), ["foo (1)"]
        )
        self.assertEqual(
            get_unique_names(names, ["baz"], equal_numbers=False), ["baz (4)"]
        )
        self.assertEqual(
            get_unique_names(names, ["baz (3)"], equal_numbers=False),
            ["baz (3) (1)"]
        )
        self.assertEqual(
            get_unique_names(domain, ["qux", "quux"], equal_numbers=False),
            ["qux", "quux"]
        )
        self.assertEqual(
            get_unique_names(domain, ["bar", "baz"], equal_numbers=False),
            ["bar (1)", "baz (4)"]
        )
        self.assertEqual(
            get_unique_names(domain, ["qux", "baz"], equal_numbers=False),
            ["qux", "baz (4)"]
        )
        self.assertEqual(
            get_unique_names(domain, ["qux", "bar"], equal_numbers=False),
            ["qux", "bar (1)"]
        )
        self.assertEqual(
            get_unique_names(domain, ["foo", "bar", "baz"], equal_numbers=False),
            ["foo (1)", "bar (1)", "baz (4)"]
        )

    def test_get_unique_names_duplicated_proposals(self):
        names = ["foo", "bar", "baz", "baz (3)"]

        self.assertEqual(
            get_unique_names(names, ["foo", "boo", "boo"]),
            ['foo (1)', 'boo (1)', 'boo (2)']
        )
        self.assertEqual(
            get_unique_names(names, ["foo", "boo", "boo", "baz"]),
            ['foo (4)', 'boo (4)', 'boo (5)', 'baz (4)']
        )
        self.assertEqual(
            get_unique_names([], ["foo", "boo", "boo", "baz"]),
            ['foo', 'boo (1)', 'boo (2)', 'baz']
        )
        self.assertEqual(
            get_unique_names(["foo", "bong"], ["foo", "boo", "boo", "baz"]),
            ['foo (1)', 'boo (1)', 'boo (2)', 'baz']
        )

        self.assertEqual(
            get_unique_names(names, ["foo", "boo", "boo"],
                             equal_numbers=False),
            ['foo (1)', 'boo (1)', 'boo (2)']
        )
        self.assertEqual(
            get_unique_names(names, ["foo", "boo", "boo", "baz"],
                             equal_numbers=False),
            ['foo (1)', 'boo (1)', 'boo (2)', 'baz (4)']
        )
        self.assertEqual(
            get_unique_names([], ["foo", "boo", "boo", "baz"],
                             equal_numbers=False),
            ['foo', 'boo (1)', 'boo (2)', 'baz']
        )
        self.assertEqual(
            get_unique_names(["foo", "bong"], ["foo", "boo", "boo", "baz"],
                             equal_numbers=False),
            ['foo (1)', 'boo (1)', 'boo (2)', 'baz']
        )

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
            ["x (2) (1)", "x (4)", "x (5)", "x (2) (2)", "x (3)"])
        self.assertEqual(
                        get_unique_names_duplicates(["iris", "iris", "iris (1)"]),
                        ["iris (2)", "iris (3)", "iris (1)"])

        self.assertEqual(
            get_unique_names_duplicates(["foo", "bar", "baz"], return_duplicated=True),
            (["foo", "bar", "baz"], []))
        self.assertEqual(
            get_unique_names_duplicates(["foo", "bar", "baz", "bar"], return_duplicated=True),
            (["foo", "bar (1)", "baz", "bar (2)"], ["bar"]))
        self.assertEqual(
            get_unique_names_duplicates(["x", "x", "x (1)"], return_duplicated=True),
            (["x (2)", "x (3)", "x (1)"], ["x"]))
        self.assertEqual(
            get_unique_names_duplicates(["x (2)", "x", "x", "x (2)", "x (3)"], return_duplicated=True),
            (["x (2) (1)", "x (4)", "x (5)", "x (2) (2)", "x (3)"], ["x (2)", "x"]))
        self.assertEqual(
            get_unique_names_duplicates(["x", "", "", None, None, "x"]),
            ["x (1)", "", "", None, None, "x (2)"])
        self.assertEqual(
            get_unique_names_duplicates(["iris", "iris", "iris (1)", "iris (2)"], return_duplicated=True),
            (["iris (3)", "iris (4)", "iris (1)", "iris (2)"], ["iris"]))

        self.assertEqual(
            get_unique_names_duplicates(["iris (1) (1)", "iris (1)", "iris (1)"]),
            ["iris (1) (1)", "iris (1) (2)", "iris (1) (3)"]
        )

        self.assertEqual(
            get_unique_names_duplicates(["iris (1) (1)", "iris (1)", "iris (1)", "iris", "iris"]),
            ["iris (1) (1)", "iris (1) (2)", "iris (1) (3)", "iris (2)", "iris (3)"]
        )

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


class TestOneHot(unittest.TestCase):

    def setUp(self):
        self.values = np.array([0, 1, 2, 1])

    @staticmethod
    def test_empty():
        np.testing.assert_array_equal(
            one_hot([]), np.zeros((0, 0))
        )
        np.testing.assert_array_equal(
            one_hot([], dim=2), np.zeros((0, 2))
        )

    def test_one_hot(self):
        np.testing.assert_array_equal(
            one_hot(self.values),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [0, 1, 0]]
        )

    def test_dtype(self):
        res = one_hot(self.values)
        self.assertEqual(res.dtype, float)
        res = one_hot(self.values, dtype=int)
        self.assertEqual(res.dtype, int)

    def test_dim(self):
        np.testing.assert_array_equal(
            one_hot(self.values, dim=4),
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
        )

    def test_dim_too_low(self):
        with self.assertRaises(ValueError):
            one_hot(self.values, dim=2)


class TestSanitizedName(unittest.TestCase):
    def test_sanitized_name(self):
        self.assertEqual(sanitized_name("Foo"), "Foo")
        self.assertEqual(sanitized_name("Foo Bar"), "Foo_Bar")
        self.assertEqual(sanitized_name("0Foo"), "_0Foo")
        self.assertEqual(sanitized_name("1 Foo Bar"), "_1_Foo_Bar")


class TestRedefinesEqAndHash(unittest.TestCase):

    class Valid:
        def __eq__(self, other):
            pass

        def __hash__(self):
            pass

    class Subclass(Valid):
        pass

    class OnlyEq:
        def __eq__(self, other):
            pass

    class OnlyHash:
        def __hash__(self):
            pass

    def test_valid(self):
        self.assertTrue(redefines_eq_and_hash(self.Valid))
        self.assertTrue(redefines_eq_and_hash(self.Valid()))

    def test_subclass(self):
        self.assertFalse(redefines_eq_and_hash(self.Subclass))

    def test_only_eq(self):
        self.assertFalse(redefines_eq_and_hash(self.OnlyEq))

    def test_only_hash(self):
        self.assertFalse(redefines_eq_and_hash(self.OnlyHash))


if __name__ == "__main__":
    unittest.main()

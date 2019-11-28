import unittest

from Orange.misc.collections import frozendict


class TestFrozenDict(unittest.TestCase):
    def test_removed_methods(self):
        d = frozendict({"a": 12})
        self.assertRaises(AttributeError, d.clear)
        self.assertRaises(AttributeError, d.pop, "a")
        self.assertRaises(AttributeError, d.popitem)
        self.assertRaises(AttributeError, d.update, {"b": 5})
        self.assertRaises(AttributeError, d.setdefault, "b", 5)
        with self.assertRaises(AttributeError):
            del d["a"]
        with self.assertRaises(AttributeError):
            d["a"] = 13
        self.assertEqual(d, {"a": 12})

    def test_functions_as_dict(self):
        d = frozendict({"a": 12, "b": 13})
        self.assertEqual(len(d), 2)
        self.assertEqual(d["a"], 12)
        self.assertEqual(d.get("a"), 12)
        self.assertEqual(d.get("c", 14), 14)
        self.assertEqual(set(d), {"a", "b"})
        self.assertEqual(set(d.keys()), {"a", "b"})
        self.assertEqual(set(d.values()), {12, 13})
        self.assertEqual(set(d.items()), {("a", 12), ("b", 13)})


if __name__ == "__main__":
    unittest.main()

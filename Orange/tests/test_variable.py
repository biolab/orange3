import math
import unittest

from Orange.testing import create_pickling_tests
from Orange import data


class DiscreteVariableTest(unittest.TestCase):
    def test_to_val(self):
        values = ["F", "M"]
        var = data.DiscreteVariable(name="Feature 0", values=values)

        self.assertEqual(var.to_val(0), 0)
        self.assertEqual(var.to_val("F"), 0)
        self.assertEqual(var.to_val(0.), 0)
        self.assertTrue(math.isnan(var.to_val("?")))

        # TODO: with self.assertRaises(ValueError): var.to_val(2)
        with self.assertRaises(ValueError):
            var.to_val("G")

    def test_find_compatible_unordered(self):
        gend = data.DiscreteVariable("gend", values=["F", "M"])

        find_comp = data.DiscreteVariable._find_compatible
        self.assertIs(find_comp("gend"), gend)
        self.assertIs(find_comp("gend", values=["F"]), gend)
        self.assertIs(find_comp("gend", values=["F", "M"]), gend)
        self.assertIs(find_comp("gend", values=["M", "F"]), gend)

        # Incompatible since it is ordered
        self.assertIsNone(find_comp("gend", values=["M", "F"], ordered=True))
        self.assertIsNone(find_comp("gend", values=["F", "M"], ordered=True))
        self.assertIsNone(find_comp("gend", values=["F"], ordered=True))
        self.assertIsNone(find_comp("gend", values=["M"], ordered=True))
        self.assertIsNone(find_comp("gend", values=["N"], ordered=True))

        # Incompatible due to empty intersection
        self.assertIsNone(find_comp("gend", values=["N"]))

        # Compatible, adds values
        self.assertIs(find_comp("gend", values=["F", "N", "R"]), gend)
        self.assertEqual(gend.values, ["F", "M", "N", "R"])

    def test_find_compatible_unordered(self):
        abc = data.DiscreteVariable("abc", values="abc", ordered=True)

        find_comp = data.DiscreteVariable._find_compatible

        self.assertIsNone(find_comp("abc"))
        self.assertIsNone(find_comp("abc", list("abc")))
        self.assertIs(find_comp("abc", ordered=True), abc)
        self.assertIs(find_comp("abc", ["a"], ordered=True), abc)
        self.assertIs(find_comp("abc", ["a", "b"], ordered=True), abc)
        self.assertIs(find_comp("abc", ["a", "b", "c"], ordered=True), abc)
        self.assertIs(find_comp("abc", ["a", "b", "c", "d"], ordered=True), abc)

        abd = data.DiscreteVariable.make(
            "abc", values=["a", "d", "b"], ordered=True)
        self.assertIsNot(abc, abd)

        abc_un = data.DiscreteVariable.make("abc", values=["a", "b", "c"])
        self.assertIsNot(abc_un, abc)

        self.assertIs(
            find_comp("abc", values=["a", "d", "b"], ordered=True), abd)
        self.assertIs(find_comp("abc", values=["a", "b", "c"]), abc_un)

    def test_make(self):
        var = data.DiscreteVariable.make("a", values=["F", "M"])
        self.assertIsInstance(var, data.DiscreteVariable)
        self.assertEqual(var.name, "a")
        self.assertEqual(var.values, ["F", "M"])


class ContinuousVariableTest(unittest.TestCase):
    def test_make(self):
        data.ContinuousVariable._clear_cache()
        age1 = data.ContinuousVariable.make("age")
        age2 = data.ContinuousVariable.make("age")
        age3 = data.ContinuousVariable("age")
        self.assertIs(age1, age2)
        self.assertIsNot(age1, age3)

    def test_decimals(self):
        a = data.ContinuousVariable("a", 4)
        self.assertEqual(a.str_val(4.654321), "4.6543")

    def test_adjust_decimals(self):
        a = data.ContinuousVariable("a")
        self.assertEqual(a.str_val(4.654321), "4.654")
        a.val_from_str_add("5")
        self.assertEqual(a.str_val(4.654321), "5")
        a.val_from_str_add("  5.12    ")
        self.assertEqual(a.str_val(4.654321), "4.65")
        a.val_from_str_add("5.1234")
        self.assertEqual(a.str_val(4.654321), "4.6543")




PickleContinuousVariable = create_pickling_tests(
    "PickleContinuousVariable",
    ("variable", lambda: data.ContinuousVariable()),
    ("with_name", lambda: data.ContinuousVariable(name="Feature 0")),
)

PickleDiscreteVariable = create_pickling_tests(
    "PickleDiscreteVariable",
    ("variable", lambda: data.DiscreteVariable()),
    ("with_name", lambda: data.DiscreteVariable(name="Feature 0")),
    ("with_int_values", lambda: data.DiscreteVariable(name="Feature 0",
                                                      values=[1, 2, 3])),
    ("with_str_value", lambda: data.DiscreteVariable(name="Feature 0",
                                                     values=["F", "M"])),
    ("ordered", lambda: data.DiscreteVariable(name="Feature 0",
                                              values=["F", "M"],
                                              ordered=True)),
    ("with_base_value", lambda: data.DiscreteVariable(name="Feature 0",
                                                      values=["F", "M"],
                                                      base_value=0)),
)

PickleStringVariable = create_pickling_tests(
    "PickleStringVariable",
    ("variable", lambda: data.StringVariable()),
    ("with_name", lambda: data.StringVariable(name="Feature 0")),
)

if __name__ == "__main__":
    unittest.main()

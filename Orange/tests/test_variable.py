import math
import unittest
import pickle

from Orange.testing import create_pickling_tests
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable


# noinspection PyPep8Naming,PyUnresolvedReferences
class VariableTest:
    def setUp(self):
        self.varcls._clear_all_caches()

    def test_dont_pickle_anonymous_variables(self):
        self.assertRaises(pickle.PickleError, pickle.dumps, self.varcls())

    def test_dont_store_anonymous_variables(self):
        self.varcls()
        self.assertEqual(len(self.varcls._all_vars), 0)

    def test_dont_make_anonymous_variables(self):
        self.assertRaises(ValueError, self.varcls.make, "")


def variabletest(varcls):
    def decorate(cls):
        return type(cls.__name__, (cls, unittest.TestCase), {'varcls': varcls})
    return decorate


@variabletest(DiscreteVariable)
class DiscreteVariableTest(VariableTest):
    def test_to_val(self):
        values = ["F", "M"]
        var = DiscreteVariable(name="Feature 0", values=values)

        self.assertEqual(var.to_val(0), 0)
        self.assertEqual(var.to_val("F"), 0)
        self.assertEqual(var.to_val(0.), 0)
        self.assertTrue(math.isnan(var.to_val("?")))

        # TODO: with self.assertRaises(ValueError): var.to_val(2)
        with self.assertRaises(ValueError):
            var.to_val("G")

    def test_find_compatible_unordered(self):
        gend = DiscreteVariable("gend", values=["F", "M"])

        find_comp = DiscreteVariable._find_compatible
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

    def test_find_compatible_ordered(self):
        abc = DiscreteVariable("abc", values="abc", ordered=True)

        find_comp = DiscreteVariable._find_compatible

        self.assertIsNone(find_comp("abc"))
        self.assertIsNone(find_comp("abc", list("abc")))
        self.assertIs(find_comp("abc", ordered=True), abc)
        self.assertIs(find_comp("abc", ["a"], ordered=True), abc)
        self.assertIs(find_comp("abc", ["a", "b"], ordered=True), abc)
        self.assertIs(find_comp("abc", ["a", "b", "c"], ordered=True), abc)
        self.assertIs(find_comp("abc", ["a", "b", "c", "d"], ordered=True), abc)

        abd = DiscreteVariable.make(
            "abc", values=["a", "d", "b"], ordered=True)
        self.assertIsNot(abc, abd)

        abc_un = DiscreteVariable.make("abc", values=["a", "b", "c"])
        self.assertIsNot(abc_un, abc)

        self.assertIs(
            find_comp("abc", values=["a", "d", "b"], ordered=True), abd)
        self.assertIs(find_comp("abc", values=["a", "b", "c"]), abc_un)

    def test_make(self):
        var = DiscreteVariable.make("a", values=["F", "M"])
        self.assertIsInstance(var, DiscreteVariable)
        self.assertEqual(var.name, "a")
        self.assertEqual(var.values, ["F", "M"])


@variabletest(ContinuousVariable)
class ContinuousVariableTest(VariableTest):
    def test_make(self):
        ContinuousVariable._clear_cache()
        age1 = ContinuousVariable.make("age")
        age2 = ContinuousVariable.make("age")
        age3 = ContinuousVariable("age")
        self.assertIs(age1, age2)
        self.assertIsNot(age1, age3)

    def test_decimals(self):
        a = ContinuousVariable("a", 4)
        self.assertEqual(a.str_val(4.654321), "4.6543")

    def test_adjust_decimals(self):
        a = ContinuousVariable("a")
        self.assertEqual(a.str_val(4.654321), "4.654")
        a.val_from_str_add("5")
        self.assertEqual(a.str_val(4.654321), "5")
        a.val_from_str_add("  5.12    ")
        self.assertEqual(a.str_val(4.654321), "4.65")
        a.val_from_str_add("5.1234")
        self.assertEqual(a.str_val(4.654321), "4.6543")


@variabletest(StringVariable)
class StringVariableTest(VariableTest):
    pass


PickleContinuousVariable = create_pickling_tests(
    "PickleContinuousVariable",
    ("with_name", lambda: ContinuousVariable(name="Feature 0")),
)

PickleDiscreteVariable = create_pickling_tests(
    "PickleDiscreteVariable",
    ("with_name", lambda: DiscreteVariable(name="Feature 0")),
    ("with_int_values", lambda: DiscreteVariable(name="Feature 0",
                                                 values=[1, 2, 3])),
    ("with_str_value", lambda: DiscreteVariable(name="Feature 0",
                                                values=["F", "M"])),
    ("ordered", lambda: DiscreteVariable(name="Feature 0",
                                         values=["F", "M"],
                                         ordered=True)),
    ("with_base_value", lambda: DiscreteVariable(name="Feature 0",
                                                 values=["F", "M"],
                                                 base_value=0))
)


PickleStringVariable = create_pickling_tests(
    "PickleStringVariable",
    ("with_name", lambda: StringVariable(name="Feature 0"))
)


if __name__ == "__main__":
    unittest.main()

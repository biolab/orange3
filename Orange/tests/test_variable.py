import math
import unittest
import pickle

import numpy as np

from Orange.testing import create_pickling_tests
from Orange.data import Variable, ContinuousVariable, DiscreteVariable, \
    StringVariable, Unknown, Value


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


class BaseVariableTest(unittest.TestCase):
    def setUp(self):
        self.var = Variable("x")

    def test_name(self):
        self.assertEqual(repr(self.var), "Variable('x')")

    def test_to_val(self):
        string_var = StringVariable("x")
        self.assertEqual(string_var.to_val("foo"), "foo")
        self.assertEqual(string_var.to_val(42), "42")

        cont_var = ContinuousVariable("x")
        self.assertTrue(math.isnan(cont_var.to_val("?")))
        self.assertTrue(math.isnan(Unknown))

        var = Variable("x")
        self.assertEqual(var.to_val("x"), "x")

    def test_repr_is_abstract(self):
        self.assertRaises(RuntimeError, self.var.repr_val, None)

    def test_properties(self):
        a = ContinuousVariable()
        self.assertTrue(a.is_continuous)
        self.assertFalse(a.is_discrete)
        self.assertFalse(a.is_string)
        self.assertTrue(a.is_primitive())

        a = DiscreteVariable()
        self.assertFalse(a.is_continuous)
        self.assertTrue(a.is_discrete)
        self.assertFalse(a.is_string)
        self.assertTrue(a.is_primitive())

        a = StringVariable()
        self.assertFalse(a.is_continuous)
        self.assertFalse(a.is_discrete)
        self.assertTrue(a.is_string)
        self.assertFalse(a.is_primitive())


    def test_strange_eq(self):
        a = ContinuousVariable()
        b = ContinuousVariable()
        self.assertTrue(a == a)
        self.assertFalse(a == b)
        self.assertFalse(a == "somestring")

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

    def test_val_from_str(self):
        var = DiscreteVariable.make("a", values=["F", "M"])
        self.assertTrue(math.isnan(var.to_val(None)))
        self.assertEqual(var.to_val(1), 1)

    def test_repr(self):
        var = DiscreteVariable.make("a", values=["F", "M"])
        self.assertEqual(
            repr(var),
            "DiscreteVariable('a', values=['F', 'M'])")
        var.base_value = 1
        self.assertEqual(
            repr(var),
            "DiscreteVariable('a', values=['F', 'M'], base_value=1)")
        var.ordered = True
        self.assertEqual(
            repr(var),
            "DiscreteVariable('a', values=['F', 'M'], "
            "ordered=True, base_value=1)")

        var = DiscreteVariable.make("a", values="1234567")
        self.assertEqual(
            repr(var),
            "DiscreteVariable('a', values=['1', '2', '3', '4', '5', ...])")

    def test_colors(self):
        var = DiscreteVariable.make("a", values=["F", "M"])
        self.assertIsNone(var._colors)
        self.assertEqual(var.colors.shape, (2, 3))
        self.assertIs(var._colors, var.colors)
        self.assertEqual(var.colors.shape, (2, 3))
        self.assertFalse(var.colors.flags.writeable)

        var.colors = np.arange(6).reshape((2, 3))
        np.testing.assert_almost_equal(var.colors, [[0, 1, 2], [3, 4, 5]])
        self.assertFalse(var.colors.flags.writeable)
        with self.assertRaises(ValueError):
            var.colors[0] = [42, 41, 40]
        var.set_color(0, [42, 41, 40])
        np.testing.assert_almost_equal(var.colors, [[42, 41, 40], [3, 4, 5]])

        var = DiscreteVariable.make("x", values=["A", "B"])
        var.attributes["colors"] = ['#0a0b0c', '#0d0e0f']
        np.testing.assert_almost_equal(var.colors, [[10, 11, 12], [13, 14, 15]])


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
        self.assertEqual(a.str_val(Unknown), "?")

    def test_adjust_decimals(self):
        a = ContinuousVariable("a")
        self.assertEqual(a.str_val(4.654321), "4.654")
        a.val_from_str_add("5")
        self.assertEqual(a.str_val(4.654321), "5")
        a.val_from_str_add("  5.12    ")
        self.assertEqual(a.str_val(4.654321), "4.65")
        a.val_from_str_add("5.1234")
        self.assertEqual(a.str_val(4.654321), "4.6543")

    def test_colors(self):
        a = ContinuousVariable("a")
        self.assertEqual(a.colors, ((0, 0, 255), (255, 255, 0), False))
        self.assertIs(a.colors, a._colors)

        a = ContinuousVariable("a")
        a.attributes["colors"] = ['#010203', '#040506', True]
        self.assertEqual(a.colors, ((1, 2, 3), (4, 5, 6), True))

        a.colors = ((3, 2, 1), (6, 5, 4), True)
        self.assertEqual(a.colors, ((3, 2, 1), (6, 5, 4), True))


@variabletest(StringVariable)
class StringVariableTest(VariableTest):
    def test_val(self):
        a = StringVariable("a")
        self.assertEqual(a.to_val(None), "")
        self.assertEqual(a.str_val(None), "?")
        self.assertEqual(a.str_val(Value(a, None)), "?")
        self.assertEqual(a.repr_val(Value(a, "foo")), '"foo"')


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


@variabletest(DiscreteVariable)
class VariableTestMakeProxy(unittest.TestCase):
    def test_make_proxy_disc(self):
        abc = DiscreteVariable("abc", values="abc", ordered=True)
        abc1 = abc.make_proxy()
        abc2 = abc1.make_proxy()
        self.assertIs(abc.master, abc)
        self.assertIs(abc1.master, abc)
        self.assertIs(abc2.master, abc)
        self.assertEqual(abc, abc1)
        self.assertEqual(abc, abc2)
        self.assertEqual(abc1, abc2)

        abcx = DiscreteVariable("abc", values="abc", ordered=True)
        self.assertNotEqual(abc, abcx)

        abc1p = pickle.loads(pickle.dumps(abc1))
        self.assertIs(abc1p.master, abc)
        self.assertEqual(abc1p, abc)

        abcp, abc1p, abc2p = pickle.loads(pickle.dumps((abc, abc1, abc2)))
        self.assertIs(abcp.master, abcp)
        self.assertIs(abc1p.master, abcp)
        self.assertIs(abc2p.master, abcp)
        self.assertEqual(abcp, abc1p)
        self.assertEqual(abcp, abc2p)
        self.assertEqual(abc1p, abc2p)

    def test_make_proxy_cont(self):
        abc = ContinuousVariable("abc")
        abc1 = abc.make_proxy()
        abc2 = abc1.make_proxy()
        self.assertIs(abc.master, abc)
        self.assertIs(abc1.master, abc)
        self.assertIs(abc2.master, abc)
        self.assertEqual(abc, abc1)
        self.assertEqual(abc, abc2)
        self.assertEqual(abc1, abc2)

if __name__ == "__main__":
    unittest.main()

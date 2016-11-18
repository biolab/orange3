# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import sys
import math
import unittest
import pickle
import pkgutil
from datetime import datetime, timezone

from io import StringIO

import numpy as np

from Orange.data import Variable, ContinuousVariable, DiscreteVariable, \
    StringVariable, TimeVariable, Unknown, Value
from Orange.data.io import CSVReader
from Orange.tests.base import create_pickling_tests


def is_on_path(name):
    """
    Is a top level package/module found on sys.path

    Parameters
    ----------
    name : str
        Top level module/package name

    Returns
    -------
    found : bool
    """
    for loader, name_, ispkg in pkgutil.iter_modules(sys.path):
        if name == name_:
            return True
    else:
        return False


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

    def test_copy_copies_attributes(self):
        var = self.varcls()
        var.attributes["a"] = "b"
        var2 = var.copy(compute_value=None)
        self.assertIn("a", var2.attributes)
        self.assertIsInstance(var2, type(var))

        var2.attributes["a"] = "c"
        # Attributes of original value should not change
        self.assertEqual(var.attributes["a"], "b")


class TestVariable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.var = Variable("x")

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
        self.assertEqual(a, a)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, "somestring")


def variabletest(varcls):
    def decorate(cls):
        return type(cls.__name__, (cls, unittest.TestCase), {'varcls': varcls})
    return decorate


@variabletest(DiscreteVariable)
class TestDiscreteVariable(VariableTest):
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

    @unittest.skipUnless(is_on_path("PyQt4"), "PyQt4 is not importable")
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
class TestContinuousVariable(VariableTest):
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
class TestStringVariable(VariableTest):
    def test_val(self):
        a = StringVariable("a")
        self.assertEqual(a.to_val(None), "")
        self.assertEqual(a.str_val(""), "?")
        self.assertEqual(a.str_val(Value(a, "")), "?")
        self.assertEqual(a.repr_val(Value(a, "foo")), '"foo"')


@variabletest(TimeVariable)
class TestTimeVariable(VariableTest):
    TESTS = [
        # in str, UTC timestamp, out str (in UTC)
        ('2015-10-12 14:13:11.01+0200', 1444651991.01, '2015-10-12 14:13:11.010000+0200'),
        ('2015-10-12T14:13:11.81+0200', 1444651991.81, '2015-10-12 14:13:11.810000+0200'),
        ('2015-10-12 14:13:11.81', 1444659191.81, '2015-10-12 14:13:11.810000'),
        ('2015-10-12T14:13:11.81', 1444659191.81, '2015-10-12 14:13:11.810000'),
        ('2015-10-12 14:13:11+0200', 1444651991, '2015-10-12 14:13:11+0200'),
        ('2015-10-12T14:13:11+0200', 1444651991, '2015-10-12 14:13:11+0200'),
        ('20151012T141311+0200', 1444651991, '2015-10-12 14:13:11+0200'),
        ('20151012141311+0200', 1444651991, '2015-10-12 14:13:11+0200'),
        ('2015-10-12 14:13:11', 1444659191, '2015-10-12 14:13:11'),
        ('2015-10-12T14:13:11', 1444659191, '2015-10-12 14:13:11'),
        ('2015-10-12 14:13', 1444659180, '2015-10-12 14:13:00'),
        ('20151012T141311', 1444659191, '2015-10-12 14:13:11'),
        ('20151012141311', 1444659191, '2015-10-12 14:13:11'),
        ('2015-10-12', 1444608000, '2015-10-12'),
        ('20151012', 1444608000, '2015-10-12'),
        ('2015-285', 1444608000, '2015-10-12'),
        ('2015-10', 1443657600, '2015-10-01'),
        ('2015', 1420070400, '2015-01-01'),
        ('01:01:01.01', 3661.01, '01:01:01.010000'),
        ('010101.01', 3661.01, '01:01:01.010000'),
        ('01:01:01', 3661, '01:01:01'),
        ('01:01', 3660, '01:01:00'),
        ('1970-01-01 00:00:00', 0, '1970-01-01 00:00:00'),
        ('1969-12-31 23:59:59', -1, '1969-12-31 23:59:59'),
        ('1969-12-31 23:59:58.9', -1.1, '1969-12-31 23:59:58.900000'),
        ('1900-01-01', -2208988800, '1900-01-01'),
        ('nan', np.nan, '?'),
        ('1444651991.81', 1444651991.81, '2015-10-12 12:13:11.810000'),
    ]

    def test_parse_repr(self):
        for datestr, timestamp, outstr in self.TESTS:
            var = TimeVariable('time')
            ts = var.to_val(datestr)  # calls parse for strings
            if not np.isnan(ts):
                self.assertEqual(ts, timestamp, msg=datestr)
            self.assertEqual(var.repr_val(ts), outstr, msg=datestr)

    def test_parse_utc(self):
        var = TimeVariable('time')
        datestr, offset = '2015-10-18 22:48:20', '+0200'
        ts1 = var.parse(datestr + offset)
        self.assertEqual(var.repr_val(ts1), datestr + offset)
        # Once a value is without a TZ, all the values lose it
        ts2 = var.parse(datestr)
        self.assertEqual(var.repr_val(ts2), datestr)
        self.assertEqual(var.repr_val(ts1), '2015-10-18 20:48:20')

    def test_parse_timestamp(self):
        var = TimeVariable("time")
        datestr = str(datetime(2016, 6, 14, 23, 8, tzinfo=timezone.utc).timestamp())
        ts1 = var.parse(datestr)
        self.assertEqual(var.repr_val(ts1), '2016-06-14 23:08:00')

    def test_parse_invalid(self):
        var = TimeVariable('var')
        with self.assertRaises(ValueError):
            var.parse('123')

    def test_have_date(self):
        var = TimeVariable('time')
        ts = var.parse('1937-08-02')  # parse date
        self.assertEqual(var.repr_val(ts), '1937-08-02')
        ts = var.parse('16:20')  # parse time
        # observe have datetime
        self.assertEqual(var.repr_val(ts), '1970-01-01 16:20:00')

    def test_no_date_no_time(self):
        self.assertEqual(TimeVariable('relative time').repr_val(1.6), '1.6')

    def test_readwrite_timevariable(self):
        output_csv = StringIO()
        input_csv = StringIO("""\
Date,Feature
time,continuous
,
1920-12-12,1.0
1920-12-13,3.0
1920-12-14,5.5
""")
        for stream in (output_csv, input_csv):
            stream.close = lambda: None  # HACK: Prevent closing of streams

        table = CSVReader(input_csv).read()
        self.assertIsInstance(table.domain['Date'], TimeVariable)
        self.assertEqual(table[0, 'Date'], '1920-12-12')
        # Dates before 1970 are negative
        self.assertTrue(all(inst['Date'] < 0 for inst in table))

        CSVReader.write_file(output_csv, table)
        self.assertEqual(input_csv.getvalue().splitlines(),
                         output_csv.getvalue().splitlines())

    def test_repr_value(self):
        # https://github.com/biolab/orange3/pull/1760
        var = TimeVariable('time')
        self.assertEqual(var.repr_val(Value(var, 416.3)), '416.3')


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

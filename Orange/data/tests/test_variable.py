# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import sys
import math
import unittest
import pickle
import pkgutil
from datetime import datetime, timezone
from distutils.version import LooseVersion

from io import StringIO

import numpy as np
import scipy.sparse as sp

import Orange
from Orange.data import Variable, ContinuousVariable, DiscreteVariable, \
    StringVariable, TimeVariable, Unknown, Value
from Orange.data.io import CSVReader
from Orange.tests.base import create_pickling_tests
from Orange.util import OrangeDeprecationWarning


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
    for _, name_, _ in pkgutil.iter_modules(sys.path):
        if name == name_:
            return True
    return False


# noinspection PyPep8Naming,PyUnresolvedReferences
class VariableTest:
    def test_dont_pickle_anonymous_variables(self):
        with self.assertWarns(OrangeDeprecationWarning):
            self.assertRaises(pickle.PickleError, pickle.dumps, self.varcls())

    def test_dont_make_anonymous_variables(self):
        self.assertWarns(OrangeDeprecationWarning, self.varcls.make, "")

    def test_copy_copies_attributes(self):
        var = self.varcls("x")
        var.attributes["a"] = "b"
        var2 = var.copy(compute_value=None)
        self.assertIn("a", var2.attributes)
        self.assertIsInstance(var2, type(var))

        var2.attributes["a"] = "c"
        # Attributes of original value should not change
        self.assertEqual(var.attributes["a"], "b")

    def test_rename(self):
        var = self.varcls_modified("x")
        var2 = var.copy(name="x2")
        self.assertIsInstance(var2, type(var))
        self.assertIsNot(var, var2)
        self.assertEqual(var2.name, "x2")
        var.__dict__.pop("_name")
        var2.__dict__.pop("_name")
        self.assertDictEqual(var.__dict__, var2.__dict__)

    def varcls_modified(self, name):
        var = self.varcls(name)
        var._compute_value = lambda x: x
        var.sparse = True
        var.attributes = {"a": 1}
        return var


class TestVariable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.var = Variable("x")

    def test_name(self):
        self.assertEqual(repr(self.var), "Variable(name='x')")

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
        a = ContinuousVariable("y")
        self.assertTrue(a.is_continuous)
        self.assertFalse(a.is_discrete)
        self.assertFalse(a.is_string)
        self.assertTrue(a.is_primitive())

        a = DiscreteVariable("d")
        self.assertFalse(a.is_continuous)
        self.assertTrue(a.is_discrete)
        self.assertFalse(a.is_string)
        self.assertTrue(a.is_primitive())

        a = StringVariable("s")
        self.assertFalse(a.is_continuous)
        self.assertFalse(a.is_discrete)
        self.assertTrue(a.is_string)
        self.assertFalse(a.is_primitive())

    def test_properties_as_predicates(self):
        a = ContinuousVariable("y")
        self.assertTrue(Variable.is_continuous(a))
        self.assertFalse(Variable.is_discrete(a))
        self.assertFalse(Variable.is_string(a))
        self.assertTrue(Variable.is_primitive(a))

        a = StringVariable("s")
        self.assertFalse(Variable.is_continuous(a))
        self.assertFalse(Variable.is_discrete(a))
        self.assertTrue(Variable.is_string(a))
        self.assertFalse(Variable.is_primitive(a))

    def test_strange_eq(self):
        a = ContinuousVariable("a")
        b = ContinuousVariable("a")
        self.assertEqual(a, a)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)
        self.assertNotEqual(a, "somestring")
        self.assertEqual(hash(a), hash(b))


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

    def test_make(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        self.assertIsInstance(var, DiscreteVariable)
        self.assertEqual(var.name, "a")
        self.assertEqual(var.values, ("F", "M"))

    def test_val_from_str(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        self.assertTrue(math.isnan(var.to_val(None)))
        self.assertEqual(var.to_val(1), 1)

    def test_val_from_str_add(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        self.assertTrue(math.isnan(var.val_from_str_add(None)))
        self.assertEqual(var.val_from_str_add("M"), 1)
        self.assertEqual(var.val_from_str_add("F"), 0)
        self.assertEqual(var.values, ("F", "M"))
        self.assertEqual(var.val_from_str_add("N"), 2)
        self.assertEqual(var.values, ("F", "M", "N"))
        self.assertEqual(var._value_index, {"F": 0, "M": 1, "N": 2})
        self.assertEqual(var.val_from_str_add("M"), 1)
        self.assertEqual(var.val_from_str_add("F"), 0)
        self.assertEqual(var.val_from_str_add("N"), 2)


    def test_repr(self):
        var = DiscreteVariable.make("a", values=("F", "M"))
        self.assertEqual(
            repr(var),
            "DiscreteVariable(name='a', values=('F', 'M'))")
        var.ordered = True
        self.assertEqual(
            repr(var),
            "DiscreteVariable(name='a', values=('F', 'M'), ordered=True)")

        var = DiscreteVariable.make("a", values="1234567")
        self.assertEqual(
            repr(var),
            "DiscreteVariable(name='a', values=('1', '2', '3', '4', '5', '6', '7'))")

    def test_no_nonstringvalues(self):
        self.assertRaises(TypeError, DiscreteVariable, "foo", values=("a", 42))
        a = DiscreteVariable("foo", values=("a", "b", "c"))
        self.assertRaises(TypeError, a.add_value, 42)

    def test_no_duplicated_values(self):
        a = DiscreteVariable("foo", values=["a", "b", "c"])
        a.add_value("b")
        self.assertEqual(list(a.values), ["a", "b", "c"])
        self.assertEqual(list(a._value_index), ["a", "b", "c"])

    def test_tuple_list_warning(self):
        if LooseVersion(Orange.version.short_version) >= LooseVersion("3.27"):
            self.fail("Remove class TupleList; replace it with tuple.")

        d1 = DiscreteVariable("A", values=("one", "two"))
        with self.assertWarns(DeprecationWarning):
            val = d1.values + ["three", ]
        self.assertEqual(val, ["one", "two", "three"])
        with self.assertWarns(DeprecationWarning):
            val = d1.values.copy()
        self.assertIsInstance(val, list)
        self.assertSequenceEqual(val, d1.values)

    def test_unpickle(self):
        d1 = DiscreteVariable("A", values=("two", "one"))
        s = pickle.dumps(d1)
        d2 = DiscreteVariable.make("A", values=("one", "two", "three"))
        d2_values = tuple(d2.values)
        d1c = pickle.loads(s)
        # See: gh-3238
        # The unpickle reconstruction picks an existing variable (d2), on which
        # __setstate__ or __dict__.update is called
        self.assertSequenceEqual(d2.values, d2_values)
        self.assertSequenceEqual(d1c.values, d1.values)
        s = pickle.dumps(d2)
        d1 = DiscreteVariable("A", values=("one", "two"))
        d2 = pickle.loads(s)
        self.assertSequenceEqual(d2.values, ("one", "two", "three"))
        self.assertSequenceEqual(d1.values, ("one", "two"))

    def test_mapper_dense(self):
        abc = DiscreteVariable("a", values=tuple("abc"))
        dca = DiscreteVariable("a", values=tuple("dca"))
        mapper = dca.get_mapper_from(abc)

        self.assertEqual(mapper(0), 2)
        self.assertTrue(np.isnan(mapper(1)))
        self.assertEqual(mapper(2), 1)

        self.assertEqual(mapper(0.), 2)
        self.assertTrue(np.isnan(mapper(1.)))
        self.assertEqual(mapper(2.), 1)
        self.assertTrue(np.isnan(mapper(np.nan)))

        self.assertEqual(mapper("a"), 2)
        self.assertTrue(np.isnan(mapper("b")))
        self.assertEqual(mapper("c"), 1)

        arr = np.array([0, 0, 2, 1, 0, 1, np.nan])
        self.assertIsNot(mapper(arr), arr)
        np.testing.assert_array_equal(
            mapper(arr), np.array([2, 2, 1, np.nan, 2, np.nan, np.nan]))
        # dtype=int can have no nans; isnan shouldn't crash the mapper

        arr_int = arr[:-1].astype(int)
        self.assertIsNot(mapper(arr_int), arr_int)
        np.testing.assert_array_equal(
            mapper(arr_int), np.array([2, 2, 1, np.nan, 2, np.nan]))

        arr_obj = arr.astype(object)
        self.assertIsNot(mapper(arr_obj), arr_obj)
        np.testing.assert_array_equal(
            mapper(arr_obj), np.array([2, 2, 1, np.nan, 2, np.nan, np.nan]))

        arr_list = list(arr)
        self.assertIsNot(mapper(arr_list), arr_list)
        self.assertTrue(
            all(x == y or (np.isnan(x) and np.isnan(y))
                for x, y in zip(mapper(arr_list),
                                [2, 2, 1, np.nan, 2, np.nan, np.nan])))

        self.assertTrue(
            x == y or (np.isnan(x) and np.isnan(y))
            for x, y in zip(mapper(tuple(arr)),
                            (2, 2, 1, np.nan, 2, np.nan, np.nan)))

        self.assertRaises(ValueError, mapper, object())

    def test_mapper_sparse(self):
        abc = DiscreteVariable("a", values=tuple("abc"))
        dca = DiscreteVariable("a", values=tuple("dca"))
        mapper = dca.get_mapper_from(abc)

        arr = np.array([0, 0, 2, 1, 0, 1, np.nan])

        # 0 does map to 0 -> convert to dense
        marr = mapper(sp.csr_matrix(arr))
        self.assertIsInstance(marr, np.ndarray)
        np.testing.assert_array_equal(
            marr,
            np.array([2, 2, 1, np.nan, 2, np.nan, np.nan]))

        marr = mapper(sp.csr_matrix(arr))
        self.assertIsInstance(marr, np.ndarray)
        np.testing.assert_array_equal(
            marr,
            np.array([2, 2, 1, np.nan, 2, np.nan, np.nan]))

        # 0 maps to 0 -> keep sparse
        acd = DiscreteVariable("a", values=tuple("acd"))
        mapper = acd.get_mapper_from(abc)

        arr_csr = sp.csr_matrix(arr)
        marr = mapper(arr_csr)
        self.assertIsNot(arr_csr, marr)
        self.assertTrue(sp.isspmatrix_csr(marr))
        np.testing.assert_array_equal(
            marr.todense(),
            np.array([[0, 0, 1, np.nan, 0, np.nan, np.nan]]))

        arr_csc = sp.csc_matrix(arr)
        marr = mapper(arr_csc)
        self.assertIsNot(arr_csc, marr)
        self.assertTrue(sp.isspmatrix_csc(marr))
        np.testing.assert_array_equal(
            marr.todense(),
            np.array([[0, 0, 1, np.nan, 0, np.nan, np.nan]]))

    def test_mapper_inplace(self):
        s = list(range(7))
        abc = DiscreteVariable("a", values=tuple("abc"))
        dca = DiscreteVariable("a", values=tuple("dca"))
        mapper = dca.get_mapper_from(abc)

        arr = np.array([[0, 0, 2, 1, 0, 1, np.nan], s]).T
        mapper(arr, 0)
        np.testing.assert_array_equal(
            arr, np.array([[2, 2, 1, np.nan, 2, np.nan, np.nan], s]).T)

        self.assertRaises(ValueError, mapper, sp.csr_matrix(arr), 0)
        self.assertRaises(ValueError, mapper, [1, 2, 3], 0)
        self.assertRaises(ValueError, mapper, 1, 0)

        acd = DiscreteVariable("a", values=tuple("acd"))
        mapper = acd.get_mapper_from(abc)

        arr = np.array([[0, 0, 2, 1, 0, 1, np.nan], s]).T
        mapper(arr, 0)
        np.testing.assert_array_equal(
            arr, np.array([[0, 0, 1, np.nan, 0, np.nan, np.nan], s]).T)

        arr = sp.csr_matrix(np.array([[0, 0, 2, 1, 0, 1, np.nan], s]).T)
        mapper(arr, 0)
        np.testing.assert_array_equal(
            arr.todense(),
            np.array([[0, 0, 1, np.nan, 0, np.nan, np.nan], s]).T)

        arr = sp.csc_matrix(np.array([[0, 0, 2, 1, 0, 1, np.nan], s]).T)
        mapper(arr, 0)
        np.testing.assert_array_equal(
            arr.todense(),
            np.array([[0, 0, 1, np.nan, 0, np.nan, np.nan], s]).T)

    def test_mapper_dim_check(self):
        abc = DiscreteVariable("a", values=tuple("abc"))
        dca = DiscreteVariable("a", values=tuple("dca"))
        mapper = dca.get_mapper_from(abc)

        self.assertRaises(ValueError, mapper, np.zeros((7, 2)))
        self.assertRaises(ValueError, mapper, sp.csr_matrix(np.zeros((7, 2))))
        self.assertRaises(ValueError, mapper, sp.csc_matrix(np.zeros((7, 2))))

    def test_mapper_from_no_values(self):
        abc = DiscreteVariable("a", values=())
        dca = DiscreteVariable("a", values=tuple("dca"))
        mapper = dca.get_mapper_from(abc)

        arr = np.full(7, np.nan)
        self.assertIsNot(mapper(arr), arr)
        np.testing.assert_array_equal(mapper(arr), arr)

        arr_csr = sp.csr_matrix(arr)
        self.assertIsNot(arr_csr, mapper(arr_csr))
        np.testing.assert_array_equal(
            mapper(arr_csr).todense(), np.atleast_2d(arr))

        arr_csc = sp.csc_matrix(arr)
        self.assertIsNot(arr_csc, mapper(arr_csc))
        np.testing.assert_array_equal(
            mapper(arr_csc).todense(), np.atleast_2d(arr))

        self.assertRaises(ValueError, mapper, sp.csr_matrix(arr), 0)
        self.assertRaises(ValueError, mapper, sp.csc_matrix(arr), 0)

    def varcls_modified(self, name):
        var = super().varcls_modified(name)
        var.add_value("A")
        var.add_value("B")
        var.ordered = True
        return var

    def test_copy_checks_len_values(self):
        var = DiscreteVariable("gender", values=("F", "M"))
        self.assertEqual(var.values, ("F", "M"))

        self.assertRaises(ValueError, var.copy, values=("F", "M", "N"))
        self.assertRaises(ValueError, var.copy, values=("F",))
        self.assertRaises(ValueError, var.copy, values=())

        var2 = var.copy()
        self.assertEqual(var2.values, ("F", "M"))

        var2 = var.copy(values=None)
        self.assertEqual(var2.values, ("F", "M"))

        var2 = var.copy(values=("W", "M"))
        self.assertEqual(var2.values, ("W", "M"))


@variabletest(ContinuousVariable)
class TestContinuousVariable(VariableTest):
    def test_make(self):
        age1 = ContinuousVariable.make("age")
        age2 = ContinuousVariable.make("age")
        self.assertEqual(age1, age2)
        self.assertIsNot(age1, age2)

    def test_decimals(self):
        a = ContinuousVariable("a", 4)
        self.assertEqual(a.str_val(4.654321), "4.6543")
        self.assertEqual(a.str_val(4.654321654321), "4.6543")
        self.assertEqual(a.str_val(Unknown), "?")
        a = ContinuousVariable("a", 5)
        self.assertEqual(a.str_val(0.000000000001), "0.00000")
        a = ContinuousVariable("a", 10)
        self.assertEqual(a.str_val(0.000000000001), "1e-12")

    def test_adjust_decimals(self):
        a = ContinuousVariable("a")
        self.assertEqual(a.str_val(4.65432), "4.65432")
        a.val_from_str_add("5")
        self.assertEqual(a.str_val(4.65432), "5")
        a.val_from_str_add("  5.12    ")
        self.assertEqual(a.str_val(4.65432), "4.65")
        a.val_from_str_add("5.1234")
        self.assertEqual(a.str_val(4.65432), "4.6543")

    def varcls_modified(self, name):
        var = super().varcls_modified(name)
        var.number_of_decimals = 5
        return var


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

    def test_have_date_have_time_in_construct(self):
        """Test if have_time and have_date is correctly set"""
        var = TimeVariable('time', have_date=1)
        self.assertTrue(var.have_date)
        self.assertFalse(var.have_time)

    def varcls_modified(self, name):
        var = super().varcls_modified(name)
        var.number_of_decimals = 5
        var.have_date = 1
        var.have_time = 1
        return var


PickleContinuousVariable = create_pickling_tests(
    "PickleContinuousVariable",
    ("with_name", lambda: ContinuousVariable(name="Feature 0")),
)

PickleDiscreteVariable = create_pickling_tests(
    "PickleDiscreteVariable",
    ("with_name", lambda: DiscreteVariable(name="Feature 0")),
    ("with_str_value", lambda: DiscreteVariable(name="Feature 0",
                                                values=("F", "M"))),
    ("ordered", lambda: DiscreteVariable(name="Feature 0",
                                         values=("F", "M"),
                                         ordered=True)),
)


PickleStringVariable = create_pickling_tests(
    "PickleStringVariable",
    ("with_name", lambda: StringVariable(name="Feature 0"))
)


class VariableTestMakeProxy(unittest.TestCase):
    def test_make_proxy_disc(self):
        abc = DiscreteVariable("abc", values="abc", ordered=True)
        abc1 = abc.make_proxy()
        abc2 = abc1.make_proxy()
        self.assertEqual(abc, abc1)
        self.assertEqual(abc, abc2)
        self.assertEqual(abc1, abc2)
        self.assertEqual(hash(abc), hash(abc1))
        self.assertEqual(hash(abc1), hash(abc2))

        abcx = DiscreteVariable("abc", values="abc", ordered=True)
        self.assertEqual(abc, abcx)
        self.assertIsNot(abc, abcx)

        abc1p = pickle.loads(pickle.dumps(abc1))
        self.assertEqual(abc1p, abc)

        abcp, abc1p, abc2p = pickle.loads(pickle.dumps((abc, abc1, abc2)))
        self.assertEqual(abcp, abc1p)
        self.assertEqual(abcp, abc2p)
        self.assertEqual(abc1p, abc2p)

    def test_make_proxy_cont(self):
        abc = ContinuousVariable("abc")
        abc1 = abc.make_proxy()
        abc2 = abc1.make_proxy()
        self.assertEqual(abc, abc1)
        self.assertEqual(abc, abc2)
        self.assertEqual(abc1, abc2)
        self.assertEqual(hash(abc), hash(abc1))
        self.assertEqual(hash(abc1), hash(abc2))

    def test_proxy_has_separate_attributes(self):
        image = StringVariable("image")
        image1 = image.make_proxy()
        image2 = image1.make_proxy()

        image.attributes["origin"] = "a"
        image1.attributes["origin"] = "b"
        image2.attributes["origin"] = "c"

        self.assertEqual(image.attributes["origin"], "a")
        self.assertEqual(image1.attributes["origin"], "b")
        self.assertEqual(image2.attributes["origin"], "c")


if __name__ == "__main__":
    unittest.main()

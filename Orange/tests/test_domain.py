# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from time import time
from numbers import Real
from itertools import starmap
import unittest
import pickle

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import (ContinuousVariable, DiscreteVariable, Domain, Table,
                         StringVariable, Variable, DomainConversion)
from Orange.data.domain import filter_visible
from Orange.preprocess import Continuize, Impute
from Orange.tests.base import create_pickling_tests


def create_domain(*ss):
    Variable._clear_all_caches()
    vars = dict(
        age=ContinuousVariable(name="AGE"),
        gender=DiscreteVariable(name="Gender", values=["M", "F"]),
        incomeA=ContinuousVariable(name="incomeA"),
        income=ContinuousVariable(name="income"),
        education=DiscreteVariable(name="education", values=["GS", "HS", "C"]),
        ssn=StringVariable(name="SSN"),
        race=DiscreteVariable(name="race",
                              values=["White", "Hypsanic", "African", "Other"]))

    def map_vars(s):
        return [vars[x] for x in s]
    return Domain(*[map_vars(s) for s in ss])


PickleDomain = create_pickling_tests(
    "PickleDomain",
    ("empty_domain", lambda: create_domain([])),
    ("with_continuous_variable", lambda: create_domain(["age"])),
    ("with_discrete_variable", lambda: create_domain(["gender"])),
    ("with_mixed_variables", lambda: create_domain(["age", "gender"])),
    ("with_continuous_class", lambda: create_domain(["age", "gender"], ["incomeA"])),
    ("with_discrete_class", lambda: create_domain(["age", "gender"], ["education"])),
    ("with_multiple_classes", lambda: create_domain(["age", "gender"],
                                                    ["incomeA", "education"])),
    ("with_metas", lambda: create_domain(["age", "gender"], [], ["ssn"])),
    ("with_class_and_metas", lambda: create_domain(["age", "gender"],
                                                   ["incomeA", "education"],
                                                   ["ssn"])),
)


age, gender, incomeA, income, education, ssn, race = \
    create_domain([], [],
                  ["age", "gender", "incomeA", "income", "education", "ssn",
                   "race"]).metas


class TestDomainInit(unittest.TestCase):
    def test_init_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes, race)
        self.assertEqual(d.variables, attributes + (race,))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race,))
        self.assertEqual(d.metas, ())

    def test_init_class_list(self):
        attributes = (age, gender, income)
        d = Domain(attributes, [race])
        self.assertEqual(d.variables, attributes + (race,))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race,))
        self.assertEqual(d.metas, ())

    def test_init_no_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes)
        self.assertEqual(d.variables, attributes)
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, None)
        self.assertEqual(d.class_vars, ())
        self.assertEqual(d.metas, ())

    def test_init_no_class_false(self):
        attributes = (age, gender, income)
        d = Domain(attributes, None)
        self.assertEqual(d.variables, attributes)
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, None)
        self.assertEqual(d.class_vars, ())
        self.assertEqual(d.metas, ())

    def test_init_multi_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes, (education, race))
        self.assertEqual(d.variables, attributes + (education, race))
        self.assertEqual(d.attributes, attributes)
        self.assertIsNone(d.class_var)
        self.assertEqual(d.class_vars, (education, race))
        self.assertEqual(d.metas, ())

    def test_init_source(self):
        attributes = (age, gender, income)
        d = Domain(attributes, (education, race))
        d2 = Domain(["Gender", 0, income], source=d)
        self.assertEqual(d2.variables, (gender, age, income))

    def test_init_source_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes, (education, race))
        d2 = Domain(["Gender", 0], "income", source=d)
        self.assertEqual(d2.variables, (gender, age, income))

    def test_init_metas(self):
        attributes = (age, gender, income)
        metas = (ssn, race)
        d = Domain(attributes, race, metas=metas)
        self.assertEqual(d.variables, attributes + (race, ))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race, ))
        self.assertEqual(d.metas, metas)

    def test_from_numpy_names(self):
        for n_cols, name in [(5, "Feature {}"),
                             (99, "Feature {:02}"),
                             (100, "Feature {:03}")]:
            d = Domain.from_numpy(np.zeros((1, n_cols)))
            self.assertTrue(d.anonymous)
            self.assertEqual([var.name for var in d.attributes],
                             [name.format(i) for i in range(1, n_cols+1)])

        d = Domain.from_numpy(np.zeros((1, 1)))
        self.assertTrue(d.anonymous)
        self.assertEqual(d.attributes[0].name, "Feature")

        d = Domain.from_numpy(np.zeros((1, 3)), np.zeros((1, 1)),
                              np.zeros((1, 100)))
        self.assertTrue(d.anonymous)
        self.assertEqual([var.name for var in d.attributes],
                         ["Feature {}".format(i) for i in range(1, 4)])
        self.assertEqual(d.class_var.name, "Target")
        self.assertEqual([var.name for var in d.metas],
                         ["Meta {:03}".format(i) for i in range(1, 101)])

    def test_from_numpy_dimensions(self):
        for dimension in [[5], [5, 1]]:
            d = Domain.from_numpy(np.zeros((1, 1)), np.zeros(dimension))
            self.assertTrue(d.anonymous)
            self.assertEqual(len(d.class_vars), 1)

        self.assertRaises(ValueError, Domain.from_numpy, np.zeros(2))
        self.assertRaises(ValueError, Domain.from_numpy, np.zeros((2, 2, 2)))
        self.assertRaises(ValueError, Domain.from_numpy, np.zeros((2, 2)), np.zeros((2, 2, 2)))

    def test_from_numpy_values(self):
        for aran_min, aran_max, vartype in [(1, 3, ContinuousVariable),
                                            (0, 2, DiscreteVariable),
                                            (18, 23, ContinuousVariable)]:
            n_rows, n_cols, = aran_max - aran_min, 1
            d = Domain.from_numpy(np.zeros((1, 1)), np.arange(aran_min, aran_max).reshape(n_rows, n_cols))
            self.assertTrue(d.anonymous)
            self.assertIsInstance(d.class_var, vartype)
            if isinstance(vartype, DiscreteVariable):
                self.assertEqual(d.class_var.values, ["v{}".format(i) for i in range(1, 3)])

    def test_wrong_vartypes(self):
        attributes = (age, gender, income)
        for args in ((attributes, ssn),
                     (attributes + (ssn,)),
                     ((ssn, ) + attributes)):
            with self.assertRaises(TypeError):
                Domain(*args)

    def test_wrong_vartypes_w_source(self):
        d = Domain((age, gender), metas=(ssn,))
        with self.assertRaises(TypeError):
            Domain(-1, source=d)

    def test_wrong_types(self):
        with self.assertRaises(TypeError):
            Domain((age, []))
        with self.assertRaises(TypeError):
            Domain((age, "income"))
        with self.assertRaises(TypeError):
            Domain(([], age))
        with self.assertRaises(TypeError):
            Domain(("income", age))
        with self.assertRaises(TypeError):
            Domain((age,), self)
        with self.assertRaises(TypeError):
            Domain((age,), metas=("income",))

    def test_get_item(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        for idx, var in [(age, age),
                         ("AGE", age),
                         (0, age),
                         (income, income),
                         ("income", income),
                         (2, income),
                         (ssn, ssn),
                         ("SSN", ssn),
                         (-1, ssn),
                         (-2, race)]:
            self.assertEqual(d[idx], var)

    def test_index(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        for idx, var in [(age, 0),
                         ("AGE", 0),
                         (0, 0),
                         (np.int_(0), 0),
                         (income, 2),
                         ("income", 2),
                         (2, 2),
                         (np.int_(2), 2),
                         (ssn, -1),
                         ("SSN", -1),
                         (-1, -1),
                         (np.int_(-1), -1),
                         (-2, -2), (np.int_(-2), -2)]:
            self.assertEqual(d.index(idx), var)

    def test_get_item_slices(self):
        d = Domain((age, gender, income, race), metas=(ssn, race))
        self.assertEqual(d[:2], (age, gender))
        self.assertEqual(d[1:3], (gender, income))
        self.assertEqual(d[2:], (income, race))

    def test_get_item_error(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        for idx in (3, -3, incomeA, "no_such_thing"):
            with self.assertRaises(KeyError):
                _ = d[idx]

        with self.assertRaises(TypeError):
            _ = d[[2]]

    def test_index_error(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        for idx in (3, np.int(3), -3, np.int(-3), incomeA, "no_such_thing"):
            with self.assertRaises(ValueError):
                d.index(idx)

        with self.assertRaises(TypeError):
            d.index([2])

    def test_contains(self):
        d = Domain((age, gender, income), metas=(ssn,))
        for var in ["AGE", age, 0, np.int_(0), "income", income, 2, np.int_(2), "SSN", ssn, -1, np.int_(-1)]:
            self.assertIn(var, d)

        for var in ["no_such_thing", race, 3, np.int_(3), -2, np.int_(-2)]:
            self.assertNotIn(var, d)

        with self.assertRaises(TypeError):
            {} in d
        with self.assertRaises(TypeError):
            [] in d

    def test_iter(self):
        d = Domain((age, gender, income), metas=(ssn,))
        self.assertEqual([var for var in d], [age, gender, income])

        d = Domain((age, ), metas=(ssn,))
        self.assertEqual([var for var in d], [age])

        d = Domain((), metas=(ssn,))
        self.assertEqual([var for var in d], [])

    def test_str(self):
        cases = (
            (((),), "[]"),
            (((age,),), "[AGE]"),
            (((), age), "[ | AGE]"),
            (((gender,), age), "[Gender | AGE]"),
            (((gender, income), None), "[Gender, income]"),
            (((gender, income), age), "[Gender, income | AGE]"),
            (((gender,), (age, income)), "[Gender | AGE, income]"),
            (((gender,), (age, income), (ssn,)),
             "[Gender | AGE, income] {SSN}"),
            (((gender,), (age, income), (ssn, race)),
             "[Gender | AGE, income] {SSN, race}"),
            (((), (), (ssn, race)), "[] {SSN, race}"),
        )

        for args, printout in cases:
            self.assertEqual(str(Domain(*args)), printout)

    def test_has_discrete(self):
        self.assertFalse(Domain([]).has_discrete_attributes())
        self.assertFalse(Domain([], [age]).has_discrete_attributes())
        self.assertFalse(Domain([], race).has_discrete_attributes())

        self.assertFalse(Domain([age], None).has_discrete_attributes())
        self.assertTrue(Domain([race], None).has_discrete_attributes())
        self.assertTrue(Domain([age, race], None).has_discrete_attributes())
        self.assertTrue(Domain([race, age], None).has_discrete_attributes())

        self.assertFalse(Domain([], [age]).has_discrete_attributes(True))
        self.assertTrue(Domain([], [race]).has_discrete_attributes(True))
        self.assertFalse(Domain([age], None).has_discrete_attributes(True))
        self.assertTrue(Domain([race], None).has_discrete_attributes(True))
        self.assertTrue(Domain([age], race).has_discrete_attributes(True))
        self.assertTrue(Domain([race], age).has_discrete_attributes(True))
        self.assertTrue(Domain([], [race, age]).has_discrete_attributes(True))

        d = Domain([], None, [gender])
        self.assertTrue(d.has_discrete_attributes(False, True))
        d = Domain([], None, [age])
        self.assertFalse(d.has_discrete_attributes(False, True))
        d = Domain([], [age], [gender])
        self.assertTrue(d.has_discrete_attributes(True, True))
        d = Domain([], [incomeA], [age])
        self.assertFalse(d.has_discrete_attributes(True, True))

    def test_has_continuous(self):
        self.assertFalse(Domain([]).has_continuous_attributes())
        self.assertFalse(Domain([], [age]).has_continuous_attributes())
        self.assertFalse(Domain([], [race]).has_continuous_attributes())

        self.assertTrue(Domain([age], None).has_continuous_attributes())
        self.assertFalse(Domain([race], None).has_continuous_attributes())
        self.assertTrue(Domain([age, race], None).has_continuous_attributes())
        self.assertTrue(Domain([race, age], None).has_continuous_attributes())

        self.assertTrue(Domain([], [age]).has_continuous_attributes(True))
        self.assertFalse(Domain([], [race]).has_continuous_attributes(True))
        self.assertTrue(Domain([age], None).has_continuous_attributes(True))
        self.assertFalse(Domain([race], None).has_continuous_attributes(True))
        self.assertTrue(Domain([age], race).has_continuous_attributes(True))
        self.assertTrue(Domain([race], age).has_continuous_attributes(True))
        self.assertTrue(Domain([], [race, age]).has_continuous_attributes(True))

        d = Domain([], None, [age])
        self.assertTrue(d.has_continuous_attributes(False, True))
        d = Domain([], None, [gender])
        self.assertFalse(d.has_continuous_attributes(False, True))
        d = Domain([], [gender], [age])
        self.assertTrue(d.has_continuous_attributes(True, True))
        d = Domain([], [race], [gender])
        self.assertFalse(d.has_continuous_attributes(True, True))

    def test_get_conversion(self):
        compute_value = lambda: 42
        new_income = income.copy(compute_value=compute_value)

        d = Domain((age, gender, income), metas=(ssn, race))
        e = Domain((gender, race), None, metas=(age, gender, ssn))
        f = Domain((gender,), (race, income), metas=(age, income, ssn))
        g = Domain((), metas=(age, gender, ssn))
        h = Domain((gender,), (race, new_income), metas=(age, new_income, ssn))

        for conver, domain, attr, class_vars, metas in ((d, e, [1, -2], [], [0, 1, -1]),
                                                        (d, f, [1], [-2, 2], [0, 2, -1]),
                                                        (f, g, [], [], [-1, 0, -3]),
                                                        (g, h, [-2], [None, compute_value], [-1, compute_value, -3])):
            to_domain = domain.get_conversion(conver)
            self.assertIs(to_domain.source, conver)
            self.assertEqual(to_domain.attributes, attr)
            self.assertEqual(to_domain.class_vars, class_vars)
            self.assertEqual(to_domain.metas, metas)

    def test_conversion(self):
        domain = Domain([age, income], [race],
                        [gender, education, ssn])

        x, y, metas = domain.convert([42, 13, "White"])
        assert_array_equal(x, np.array([42, 13]))
        assert_array_equal(y, np.array([0]))
        metas_exp = [gender.Unknown, education.Unknown, ssn.Unknown]

        def eq(a, b):
            if isinstance(a, Real) and isinstance(b, Real) and \
                    np.isnan(a) and np.isnan(b):
                return True
            else:
                return a == b

        self.assertTrue(all(starmap(eq, zip(metas, metas_exp))))

        x, y, metas = domain.convert([42, 13, "White", "M", "HS", "1234567"])
        assert_array_equal(x, np.array([42, 13]))
        assert_array_equal(y, np.array([0]))
        assert_array_equal(metas, np.array([0, 1, "1234567"], dtype=object))

    def test_conversion_size(self):
        domain = Domain([age, gender, income], [race])
        self.assertRaises(ValueError, domain.convert, [0] * 3)
        self.assertRaises(ValueError, domain.convert, [0] * 5)

        domain = Domain([age, income], [race],
                        [gender, education, ssn])
        self.assertRaises(ValueError, domain.convert, [0] * 2)
        self.assertRaises(ValueError, domain.convert, [0] * 4)
        self.assertRaises(ValueError, domain.convert, [0] * 7)
        domain.convert([0] * 3)
        domain.convert([0] * 6)

    def test_preprocessor_chaining(self):
        domain = Domain([DiscreteVariable("a", values="01"),
                         DiscreteVariable("b", values="01")],
                        DiscreteVariable("y", values="01"))
        table = Table(domain, [[0, 1], [1, np.NaN]], [0, 1])
        pre1 = Continuize(Impute(table))
        pre2 = Table(pre1.domain, table)
        np.testing.assert_almost_equal(pre1.X, pre2.X)

    def test_unpickling_recreates_known_domains(self):
        Variable._clear_all_caches()
        domain = Domain([])
        unpickled_domain = pickle.loads(pickle.dumps(domain))
        self.assertTrue(hasattr(unpickled_domain, '_known_domains'))

    def test_different_domains_with_same_attributes_are_equal(self):
        domain1 = Domain([])
        domain2 = Domain([])
        self.assertEqual(domain1, domain2)

        var1 = ContinuousVariable('var1')
        domain1.attributes = (var1,)
        self.assertNotEqual(domain1, domain2)

        domain2.attributes = (var1,)
        self.assertEqual(domain1, domain2)

        domain1.class_vars = (var1,)
        self.assertNotEqual(domain1, domain2)

        domain2.class_vars = (var1,)
        self.assertEqual(domain1, domain2)

        domain1._metas = (var1,)
        self.assertNotEqual(domain1, domain2)

        domain2._metas = (var1,)
        self.assertEqual(domain1, domain2)

    def test_domain_conversion_is_fast_enough(self):
        attrs = [ContinuousVariable("f%i" % i) for i in range(10000)]
        class_vars = [ContinuousVariable("c%i" % i) for i in range(10)]
        metas = [ContinuousVariable("m%i" % i) for i in range(10)]
        source = Domain(attrs, class_vars, metas)

        start = time()
        cases = (((attrs[:1000], class_vars, metas),
                  list(range(1000)), list(range(10000, 10010)), list(range(-1, -11, -1))),
                 ((metas, attrs[:1000], class_vars),
                  list(range(-1, -11, -1)), list(range(1000)), list(range(10000, 10010))),
                 ((class_vars, metas, attrs[:1000]),
                  list(range(10000, 10010)), list(range(-1, -11, -1)), list(range(1000))))

        for domain_args, attributes, class_vars, metas in cases:
            c1 = DomainConversion(source, Domain(*domain_args))
            self.assertEqual(c1.attributes, attributes)
            self.assertEqual(c1.class_vars, class_vars)
            self.assertEqual(c1.metas, metas)

        self.assertLessEqual(time() - start, 1)


class TestDomainFilter(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')

    def test_filter_visible(self):
        n_feats = len(self.iris.domain.attributes)

        self.iris.domain.attributes[0].attributes.update({'hidden': True})
        filtered = list(filter_visible(self.iris.domain.attributes))
        self.assertNotIn(self.iris.domain.attributes[0], filtered)
        self.assertEqual(len(filtered), n_feats - 1)


if __name__ == "__main__":
    unittest.main()

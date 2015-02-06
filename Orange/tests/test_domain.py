import unittest
import pickle

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import (ContinuousVariable, DiscreteVariable, Domain,
                         StringVariable, Unknown, Variable)
from Orange.testing import create_pickling_tests


age = ContinuousVariable(name="AGE")
gender = DiscreteVariable(name="Gender", values=["M", "F"])
incomeA = ContinuousVariable(name="AGE")
income = ContinuousVariable(name="income")
education = DiscreteVariable(name="education", values=["GS", "HS", "C"])
ssn = StringVariable(name="SSN")
race = DiscreteVariable(name="race",
                        values=["White", "Hypsanic", "African", "Other"])

PickleDomain = create_pickling_tests(
    "PickleDomain",
    ("empty_domain", lambda: Domain([])),
    ("with_continuous_variable", lambda: Domain([age])),
    ("with_discrete_variable", lambda: Domain([gender])),
    ("with_mixed_variables", lambda: Domain([age, gender])),
    ("with_continuous_class", lambda: Domain([age, gender], [incomeA])),
    ("with_discrete_class", lambda: Domain([age, gender], [education])),
    ("with_multiple_classes", lambda: Domain([age, gender],
                                             [incomeA, education])),
    ("with_metas", lambda: Domain([age, gender], metas=[ssn])),
    ("with_class_and_metas", lambda: Domain([age, gender],
                                            [incomeA, education],
                                            [ssn])),
)


class TestDomainInit(unittest.TestCase):
    def test_init_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes, race)
        self.assertEqual(d.variables, attributes + (race,))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race,))
        self.assertEqual(d.metas, ())
        self.assertEqual(d._indices,
                         {"AGE": 0, "Gender": 1, "income": 2, "race": 3})

    def test_init_class_list(self):
        attributes = (age, gender, income)
        d = Domain(attributes, [race])
        self.assertEqual(d.variables, attributes + (race,))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race,))
        self.assertEqual(d.metas, ())
        self.assertEqual(d._indices,
                         {"AGE": 0, "Gender": 1, "income": 2, "race": 3})

    def test_init_no_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes)
        self.assertEqual(d.variables, attributes)
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, None)
        self.assertEqual(d.class_vars, ())
        self.assertEqual(d.metas, ())
        self.assertEqual(d._indices,
                         {"AGE": 0, "Gender": 1, "income": 2})

    def test_init_no_class_false(self):
        attributes = (age, gender, income)
        d = Domain(attributes, None)
        self.assertEqual(d.variables, attributes)
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, None)
        self.assertEqual(d.class_vars, ())
        self.assertEqual(d.metas, ())
        self.assertEqual(d._indices,
                         {"AGE": 0, "Gender": 1, "income": 2})

    def test_init_multi_class(self):
        attributes = (age, gender, income)
        d = Domain(attributes, (education, race))
        self.assertEqual(d.variables, attributes + (education, race))
        self.assertEqual(d.attributes, attributes)
        self.assertIsNone(d.class_var)
        self.assertEqual(d.class_vars, (education, race))
        self.assertEqual(d.metas, ())
        self.assertEqual(d._indices,
                         {"AGE": 0, "Gender": 1, "income": 2,
                          "education": 3, "race": 4})

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
        self.assertEqual(d._indices, {"AGE": 0, "Gender": 1, "income": 2,
                                     "SSN": -1, "race": -2})

    def test_wrong_vartypes(self):
        attributes = (age, gender, income)
        with self.assertRaises(TypeError):
            Domain(attributes, ssn)
        with self.assertRaises(TypeError):
            Domain(attributes + (ssn,))
        with self.assertRaises(TypeError):
            Domain((ssn, ) + attributes)

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
        self.assertEqual(d[age], age)
        self.assertEqual(d["AGE"], age)
        self.assertEqual(d[0], age)

        self.assertEqual(d[income], income)
        self.assertEqual(d["income"], income)
        self.assertEqual(d[2], income)

        self.assertEqual(d[ssn], ssn)
        self.assertEqual(d["SSN"], ssn)
        self.assertEqual(d[-1], ssn)

        self.assertEqual(d[-2], race)

    def test_index(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        self.assertEqual(d.index(age), 0)
        self.assertEqual(d.index("AGE"), 0)
        self.assertEqual(d.index(0), 0)

        self.assertEqual(d.index(income), 2)
        self.assertEqual(d.index("income"), 2)
        self.assertEqual(d.index(2), 2)

        self.assertEqual(d.index(ssn), -1)
        self.assertEqual(d.index("SSN"), -1)
        self.assertEqual(d.index(-1), -1)

        self.assertEqual(d.index(-2), -2)

    def test_get_item_slices(self):
        d = Domain((age, gender, income, race), metas=(ssn, race))
        self.assertEqual(d[:2], (age, gender))
        self.assertEqual(d[1:3], (gender, income))
        self.assertEqual(d[2:], (income, race))

    def test_get_item_error(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        with self.assertRaises(IndexError):
            _ = d[3]
        with self.assertRaises(IndexError):
            _ = d[-3]
        with self.assertRaises(IndexError):
            _ = d[incomeA]
        with self.assertRaises(IndexError):
            _ = d["no_such_thing"]
        with self.assertRaises(TypeError):
            _ = d[[2]]

    def test_index_error(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        with self.assertRaises(ValueError):
            d.index(3)
        with self.assertRaises(ValueError):
            d.index(-3)
        with self.assertRaises(ValueError):
            d.index(incomeA)
        with self.assertRaises(ValueError):
            d.index("no_such_thing")
        with self.assertRaises(TypeError):
            d.index([2])

    def test_var_from_domain(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        self.assertEqual(d.var_from_domain(incomeA), incomeA)
        self.assertEqual(d.var_from_domain(incomeA, False), incomeA)
        with self.assertRaises(IndexError):
            d.var_from_domain(incomeA, True)
        with self.assertRaises(TypeError):
            d.var_from_domain(1, no_index=True)
        with self.assertRaises(TypeError):
            d.var_from_domain(-1, no_index=True)

    def test_contains(self):
        d = Domain((age, gender, income), metas=(ssn,))
        self.assertTrue("AGE" in d)
        self.assertTrue(age in d)
        self.assertTrue(0 in d)
        self.assertTrue("income" in d)
        self.assertTrue(income in d)
        self.assertTrue(2 in d)
        self.assertTrue("SSN" in d)
        self.assertTrue(ssn in d)
        self.assertTrue(-1 in d)

        self.assertFalse("no_such_thing" in d)
        self.assertFalse(race in d)
        self.assertFalse(3 in d)
        self.assertFalse(-2 in d)

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

    def test_get_conversion(self):
        d = Domain((age, gender, income), metas=(ssn, race))
        e = Domain((gender, race), None, metas=(age, gender, ssn))
        f = Domain((gender,), (race, income), metas=(age, income, ssn))
        g = Domain((), metas=(age, gender, ssn))

        d_to_e = e.get_conversion(d)
        self.assertIs(d_to_e.source, d)
        self.assertEqual(d_to_e.attributes, [1, -2])
        self.assertEqual(d_to_e.class_vars, [])
        self.assertEqual(d_to_e.metas, [0, 1, -1])

        d_to_e = e.get_conversion(d)
        self.assertIs(d_to_e.source, d)
        self.assertEqual(d_to_e.attributes, [1, -2])
        self.assertEqual(d_to_e.class_vars, [])
        self.assertEqual(d_to_e.metas, [0, 1, -1])

        d_to_f = f.get_conversion(d)
        self.assertIs(d_to_f.source, d)
        self.assertEqual(d_to_f.attributes, [1])
        self.assertEqual(d_to_f.class_vars, [-2, 2])
        self.assertEqual(d_to_f.metas, [0, 2, -1])

        d_to_e = e.get_conversion(d)
        self.assertIs(d_to_e.source, d)
        self.assertEqual(d_to_e.attributes, [1, -2])
        self.assertEqual(d_to_e.class_vars, [])
        self.assertEqual(d_to_e.metas, [0, 1, -1])

        d_to_f = f.get_conversion(d)
        self.assertIs(d_to_f.source, d)
        self.assertEqual(d_to_f.attributes, [1])
        self.assertEqual(d_to_f.class_vars, [-2, 2])
        self.assertEqual(d_to_f.metas, [0, 2, -1])

        f_to_g = g.get_conversion(f)
        self.assertIs(f_to_g.source, f)
        self.assertEqual(f_to_g.attributes, [])
        self.assertEqual(f_to_g.class_vars, [])
        self.assertEqual(f_to_g.metas, [-1, 0, -3])

        x = lambda: 42
        income.compute_value = x
        g_to_f = f.get_conversion(g)
        self.assertIs(g_to_f.source, g)
        self.assertEqual(g_to_f.attributes, [-2])
        self.assertEqual(g_to_f.class_vars, [Variable.compute_value, x])
        self.assertEqual(g_to_f.metas, [-1, x, -3])

    def test_conversion(self):
        domain = Domain([age, income], [race],
                        [gender, education, ssn])

        values, metas = domain.convert([42, 13, "White"])
        assert_array_equal(values, np.array([42, 13, 0]))
        assert_array_equal(metas, np.array([Unknown, Unknown, None]))

        values, metas = domain.convert([42, 13, "White", "M", "HS", "1234567"])
        assert_array_equal(values, np.array([42, 13, 0]))
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

    def test_unpickling_recreates_known_domains(self):
        domain = Domain([])
        unpickled_domain = pickle.loads(pickle.dumps(domain))
        self.assertTrue(hasattr(unpickled_domain, '_known_domains'))


if __name__ == "__main__":
    unittest.main()

import unittest
import pickle

from Orange.data import Domain

from Orange.testing import create_pickling_tests
from Orange import data

age = data.ContinuousVariable(name="AGE")
gender = data.DiscreteVariable(name="Gender",
                               values=["M", "F"])
incomeA = data.ContinuousVariable(name="AGE")
income = data.ContinuousVariable(name="income")
education = data.DiscreteVariable(name="education",
                                  values=["GS", "HS", "C"])
ssn = data.StringVariable(name="SSN")
race = data.DiscreteVariable(name="race",
                             values=["White", "Hypsanic", "African", "Other"])

PickleDomain = create_pickling_tests(
    "PickleDomain",
    ("empty_domain", lambda: data.Domain([])),
    ("with_continuous_variable", lambda: data.Domain([age])),
    ("with_discrete_variable", lambda: data.Domain([gender])),
    ("with_mixed_variables", lambda: data.Domain([age, gender])),
    ("with_continuous_class", lambda: data.Domain([age, gender],
                                                  [incomeA])),
    ("with_discrete_class", lambda: data.Domain([age, gender],
                                                [education])),
    ("with_multiple_classes", lambda: data.Domain([age, gender],
                                                  [incomeA, education])),
    ("with_metas", lambda: data.Domain([age, gender], metas=[ssn])),
    ("with_class_and_metas", lambda: data.Domain([age, gender],
                                                 [incomeA, education],
                                                 [ssn])),
)


class TestDomainInit(unittest.TestCase):
    def test_init_class(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes, race)
        self.assertEqual(d.variables, attributes + (race,))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race,))
        self.assertEqual(d.metas, ())
        self.assertEqual(d.indices,
                         {"AGE": 0, "Gender": 1, "income": 2, "race": 3})

    def test_init_class_list(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes, [race])
        self.assertEqual(d.variables, attributes + (race,))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race,))
        self.assertEqual(d.metas, ())
        self.assertEqual(d.indices,
                         {"AGE": 0, "Gender": 1, "income": 2, "race": 3})

    def test_init_no_class(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes)
        self.assertEqual(d.variables, attributes)
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, None)
        self.assertEqual(d.class_vars, ())
        self.assertEqual(d.metas, ())
        self.assertEqual(d.indices,
                         {"AGE": 0, "Gender": 1, "income": 2})

    def test_init_no_class_false(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes, None)
        self.assertEqual(d.variables, attributes)
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, None)
        self.assertEqual(d.class_vars, ())
        self.assertEqual(d.metas, ())
        self.assertEqual(d.indices,
                         {"AGE": 0, "Gender": 1, "income": 2})

    def test_init_multi_class(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes, (education, race))
        self.assertEqual(d.variables, attributes + (education, race))
        self.assertEqual(d.attributes, attributes)
        self.assertIsNone(d.class_var)
        self.assertEqual(d.class_vars, (education, race))
        self.assertEqual(d.metas, ())
        self.assertEqual(d.indices,
                         {"AGE": 0, "Gender": 1, "income": 2,
                          "education": 3, "race": 4})

    def test_init_source(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes, (education, race))
        d2 = data.Domain(["Gender", 0, income], source=d)
        self.assertEqual(d2.variables, (gender, age, income))

    def test_init_source_class(self):
        attributes = (age, gender, income)
        d = data.Domain(attributes, (education, race))
        d2 = data.Domain(["Gender", 0], "income", source=d)
        self.assertEqual(d2.variables, (gender, age, income))

    def test_init_metas(self):
        attributes = (age, gender, income)
        metas = (ssn, race)
        d = data.Domain(attributes, race, metas=metas)
        self.assertEqual(d.variables, attributes + (race, ))
        self.assertEqual(d.attributes, attributes)
        self.assertEqual(d.class_var, race)
        self.assertEqual(d.class_vars, (race, ))
        self.assertEqual(d.metas, metas)
        self.assertEqual(d.indices, {"AGE": 0, "Gender": 1, "income": 2,
                                     "SSN": -1, "race": -2})

    def test_wrong_vartypes(self):
        attributes = (age, gender, income)
        with self.assertRaises(TypeError):
            data.Domain(attributes, ssn)
        with self.assertRaises(TypeError):
            data.Domain(attributes + (ssn,))
        with self.assertRaises(TypeError):
            data.Domain((ssn, ) + attributes)

    def test_wrong_vartypes_w_source(self):
        d = data.Domain((age, gender), metas=(ssn,))
        with self.assertRaises(TypeError):
            data.Domain(-1, source=d)

    def test_wrong_types(self):
        with self.assertRaises(TypeError):
            data.Domain((age, []))
        with self.assertRaises(TypeError):
            data.Domain((age, "income"))
        with self.assertRaises(TypeError):
            data.Domain(([], age))
        with self.assertRaises(TypeError):
            data.Domain(("income", age))
        with self.assertRaises(TypeError):
            data.Domain((age,), self)
        with self.assertRaises(TypeError):
            data.Domain((age,), metas=("income",))

    def test_get_item(self):
        d = data.Domain((age, gender, income), metas=(ssn, race))
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
        d = data.Domain((age, gender, income), metas=(ssn, race))
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
        d = data.Domain((age, gender, income, race), metas=(ssn, race))
        self.assertEqual(d[:2], (age, gender))
        self.assertEqual(d[1:3], (gender, income))
        self.assertEqual(d[2:], (income, race))

    def test_get_item_error(self):
        d = data.Domain((age, gender, income), metas=(ssn, race))
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
        d = data.Domain((age, gender, income), metas=(ssn, race))
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
        d = data.Domain((age, gender, income), metas=(ssn, race))
        self.assertEqual(d.var_from_domain(incomeA), incomeA)
        self.assertEqual(d.var_from_domain(incomeA, False), incomeA)
        with self.assertRaises(IndexError):
            d.var_from_domain(incomeA, True)
        with self.assertRaises(TypeError):
            d.var_from_domain(1, no_index=True)
        with self.assertRaises(TypeError):
            d.var_from_domain(-1, no_index=True)

    def test_contains(self):
        d = data.Domain((age, gender, income), metas=(ssn,))
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
        d = data.Domain((age, gender, income), metas=(ssn,))
        self.assertEqual([var for var in d], [age, gender, income])

        d = data.Domain((age, ), metas=(ssn,))
        self.assertEqual([var for var in d], [age])

        d = data.Domain((), metas=(ssn,))
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
            self.assertEqual(str(data.Domain(*args)), printout)

    def test_has_discrete(self):
        self.assertFalse(
            data.Domain([]).has_discrete_attributes())
        self.assertFalse(
            data.Domain([], [age]).has_discrete_attributes())
        self.assertFalse(
            data.Domain([], race).has_discrete_attributes())

        self.assertFalse(
            data.Domain([age], None).has_discrete_attributes())
        self.assertTrue(
            data.Domain([race], None).has_discrete_attributes())
        self.assertTrue(
            data.Domain([age, race], None).has_discrete_attributes())
        self.assertTrue(
            data.Domain([race, age], None).has_discrete_attributes())

        self.assertFalse(
            data.Domain([], [age]).has_discrete_attributes(True))
        self.assertTrue(
            data.Domain([], [race]).has_discrete_attributes(True))
        self.assertFalse(
            data.Domain([age], None).has_discrete_attributes(True))
        self.assertTrue(
            data.Domain([race], None).has_discrete_attributes(True))
        self.assertTrue(
            data.Domain([age], race).has_discrete_attributes(True))
        self.assertTrue(
            data.Domain([race], age).has_discrete_attributes(True))
        self.assertTrue(
            data.Domain([], [race, age]).has_discrete_attributes(True))

    def test_has_continuous(self):
        self.assertFalse(
            data.Domain([]).has_continuous_attributes())
        self.assertFalse(
            data.Domain([], [age]).has_continuous_attributes())
        self.assertFalse(
            data.Domain([], [race]).has_continuous_attributes())

        self.assertTrue(
            data.Domain([age], None).has_continuous_attributes())
        self.assertFalse(
            data.Domain([race], None).has_continuous_attributes())
        self.assertTrue(
            data.Domain([age, race], None).has_continuous_attributes())
        self.assertTrue(
            data.Domain([race, age], None).has_continuous_attributes())

        self.assertTrue(
            data.Domain([], [age]).has_continuous_attributes(True))
        self.assertFalse(
            data.Domain([], [race]).has_continuous_attributes(True))
        self.assertTrue(
            data.Domain([age], None).has_continuous_attributes(True))
        self.assertFalse(
            data.Domain([race], None).has_continuous_attributes(True))
        self.assertTrue(
            data.Domain([age], race).has_continuous_attributes(True))
        self.assertTrue(
            data.Domain([race], age).has_continuous_attributes(True))
        self.assertTrue(
            data.Domain([], [race, age]).has_continuous_attributes(True))

    def test_get_conversion(self):
        d = data.Domain((age, gender, income), metas=(ssn, race))
        e = data.Domain((gender, race), None, metas=(age, gender, ssn))
        f = data.Domain((gender,), (race, income), metas=(age, income, ssn))
        g = data.Domain((), metas=(age, gender, ssn))

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
        income.get_value_from = x
        g_to_f = f.get_conversion(g)
        self.assertIs(g_to_f.source, g)
        self.assertEqual(g_to_f.attributes, [-2])
        self.assertEqual(g_to_f.class_vars, [None, x])
        self.assertEqual(g_to_f.metas, [-1, x, -3])

    def test_unpickling_recreates_known_domains(self):
        domain = Domain([])
        unpickled_domain = pickle.loads(pickle.dumps(domain))
        self.assertTrue(hasattr(unpickled_domain, 'known_domains'))


if __name__ == "__main__":
    unittest.main()

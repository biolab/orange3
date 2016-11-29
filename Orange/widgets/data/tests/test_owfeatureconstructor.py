import unittest
import ast
import sys

from Orange.data import (Table, Domain, StringVariable,
                         ContinuousVariable, DiscreteVariable)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.data.owfeatureconstructor import (DiscreteDescriptor,
                                                      ContinuousDescriptor,
                                                      StringDescriptor,
                                                      construct_variables, OWFeatureConstructor)

from Orange.widgets.data.owfeatureconstructor import (
    freevars, make_lambda, validate_exp
)

import dill as pickle  # Import dill after Orange because patched


class FeatureConstructorTest(unittest.TestCase):
    def test_construct_variables_discrete(self):
        data = Table("iris")
        name = 'Discrete Variable'
        expression = "iris_one if iris == 'Iris-setosa' else iris_two " \
                     "if iris == 'Iris-versicolor' else iris_three"
        values = ('iris one', 'iris two', 'iris three')
        desc = PyListModel(
            [DiscreteDescriptor(name=name, expression=expression,
                                values=values, base_value=-1, ordered=False)]
        )
        data = Table(Domain(list(data.domain.attributes) +
                            construct_variables(desc, data.domain),
                            data.domain.class_vars,
                            data.domain.metas), data)
        self.assertTrue(isinstance(data.domain[name], DiscreteVariable))
        self.assertEqual(data.domain[name].values, list(values))
        for i in range(3):
            self.assertEqual(data[i * 50, name], values[i])

    def test_construct_variables_continuous(self):
        data = Table("iris")
        name = 'Continuous Variable'
        expression = "pow(sepal_length + sepal_width, 2)"
        featuremodel = PyListModel(
            [ContinuousDescriptor(name=name, expression=expression,
                                  number_of_decimals=2)]
        )
        data = Table(Domain(list(data.domain.attributes) +
                            construct_variables(featuremodel, data.domain),
                            data.domain.class_vars,
                            data.domain.metas), data)
        self.assertTrue(isinstance(data.domain[name], ContinuousVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             pow(data[i * 50, 0] + data[i * 50, 1], 2))

    def test_construct_variables_string(self):
        data = Table("iris")
        name = 'String Variable'
        expression = "str(iris) + '_name'"
        desc = PyListModel(
            [StringDescriptor(name=name, expression=expression)]
        )
        data = Table(Domain(data.domain.attributes,
                            data.domain.class_vars,
                            list(data.domain.metas) +
                            construct_variables(desc, data.domain)),
                     data)
        self.assertTrue(isinstance(data.domain[name], StringVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             str(data[i * 50, "iris"]) + "_name")


GLOBAL_CONST = 2


class PicklingTest(unittest.TestCase):
    CLASS_CONST = 3

    def test_lambdas_pickle(self):
        NONLOCAL_CONST = 5

        lambda_func = lambda x, LOCAL_CONST=7: \
            x * LOCAL_CONST * NONLOCAL_CONST * self.CLASS_CONST * GLOBAL_CONST

        def nested_func(x, LOCAL_CONST=7):
            return x * LOCAL_CONST * NONLOCAL_CONST * self.CLASS_CONST * GLOBAL_CONST

        self.assertEqual(lambda_func(11),
                         pickle.loads(pickle.dumps(lambda_func))(11))
        self.assertEqual(nested_func(11),
                         pickle.loads(pickle.dumps(nested_func))(11))


class TestTools(unittest.TestCase):
    def test_free_vars(self):
        stmt = ast.parse("foo", "", "single")
        with self.assertRaises(ValueError):
            freevars(stmt, [])

        suite = ast.parse("foo; bar();", "exec")
        with self.assertRaises(ValueError):
            freevars(suite, [])

        def freevars_(source, env=[]):
            return freevars(ast.parse(source, "", "eval"), env)

        self.assertEqual(freevars_("1"), [])
        self.assertEqual(freevars_("..."), [])
        self.assertEqual(freevars_("a"), ["a"])
        self.assertEqual(freevars_("a", ["a"]), [])
        self.assertEqual(freevars_("f(1)"), ["f"])
        self.assertEqual(freevars_("f(x)"), ["f", "x"])
        self.assertEqual(freevars_("f(x)", ["f"]), ["x"])
        self.assertEqual(freevars_("a + 1"), ["a"])
        self.assertEqual(freevars_("a + b"), ["a", "b"])
        self.assertEqual(freevars_("a + b", ["a", "b"]), [])
        self.assertEqual(freevars_("a[b]"), ["a", "b"])
        self.assertEqual(freevars_("a[b]", ["a", "b"]), [])
        self.assertEqual(freevars_("f(x, *a)", ["f"]), ["x", "a"])
        self.assertEqual(freevars_("f(x, *a, y=1)", ["f"]), ["x", "a"])
        self.assertEqual(freevars_("f(x, *a, y=1, **k)", ["f"]),
                         ["x", "a", "k"])
        if sys.version_info >= (3, 5):
            self.assertEqual(freevars_("f(*a, *b, k=c, **d, **e)", ["f"]),
                             ["a", "b", "c", "d", "e"])

        self.assertEqual(freevars_("True"), [])
        self.assertEqual(freevars_("'True'"), [])
        self.assertEqual(freevars_("None"), [])
        self.assertEqual(freevars_("b'None'"), [])

        self.assertEqual(freevars_("a < b"), ["a", "b"])
        self.assertEqual(freevars_("a < b <= c"), ["a", "b", "c"])
        self.assertEqual(freevars_("1 < a <= 3"), ["a"])

        self.assertEqual(freevars_("{}"), [])
        self.assertEqual(freevars_("[]"), [])
        self.assertEqual(freevars_("()"), [])
        self.assertEqual(freevars_("[a, 1]"), ["a"])
        self.assertEqual(freevars_("{a: b}"), ["a", "b"])
        self.assertEqual(freevars_("{a, b}"), ["a", "b"])
        self.assertEqual(freevars_("0 if abs(a) < 0.1 else b", ["abs"]),
                         ["a", "b"])
        self.assertEqual(freevars_("lambda a: b + 1"), ["b"])
        self.assertEqual(freevars_("lambda a: b + 1", ["b"]), [])
        self.assertEqual(freevars_("lambda a: a + 1"), [])
        self.assertEqual(freevars_("(lambda a: a + 1)(a)"), ["a"])
        self.assertEqual(freevars_("lambda a, *arg: arg + (a,)"), [])
        self.assertEqual(freevars_("lambda a, *arg, **kwargs: arg + (a,)"), [])

        self.assertEqual(freevars_("[a for a in b]"), ["b"])
        self.assertEqual(freevars_("[1 + a for c in b if c]"), ["a", "b"])
        self.assertEqual(freevars_("{a for _ in [] if b}"), ["a", "b"])
        self.assertEqual(freevars_("{a for _ in [] if b}", ["a", "b"]), [])

    def test_validate_exp(self):

        stmt = ast.parse("1", mode="single")
        with self.assertRaises(ValueError):
            validate_exp(stmt)
        suite = ast.parse("a; b", mode="exec")
        with self.assertRaises(ValueError):
            validate_exp(suite)

        def validate_(source):
            return validate_exp(ast.parse(source, mode="eval"))

        self.assertTrue(validate_("a"))
        self.assertTrue(validate_("a + 1"))
        self.assertTrue(validate_("a < 1"))
        self.assertTrue(validate_("1 < a"))
        self.assertTrue(validate_("1 < a < 10"))
        self.assertTrue(validate_("a and b"))
        self.assertTrue(validate_("not a"))
        self.assertTrue(validate_("a if b else c"))

        self.assertTrue(validate_("f(x)"))
        self.assertTrue(validate_("f(g(x)) + g(x)"))

        self.assertTrue(validate_("f(x, r=b)"))
        self.assertTrue(validate_("a[b]"))

        self.assertTrue(validate_("a in {'a', 'b'}"))
        self.assertTrue(validate_("{}"))
        self.assertTrue(validate_("{'a': 1}"))
        self.assertTrue(validate_("()"))
        self.assertTrue(validate_("[]"))

        with self.assertRaises(ValueError):
            validate_("[a for a in s]")

        with self.assertRaises(ValueError):
            validate_("(a for a in s)")

        with self.assertRaises(ValueError):
            validate_("{a for a in s}")

        with self.assertRaises(ValueError):
            validate_("{a:1 for a in s}")


class OWFeatureConstructorTests(WidgetTest):
    def setUp(self):
        self.widget = OWFeatureConstructor()

    def test_create_variable_with_no_data(self):
        self.widget.addFeature(
            ContinuousDescriptor("X1", "", 3))

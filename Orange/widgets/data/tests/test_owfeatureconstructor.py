import unittest
import ast
import sys
import math
import pickle
import copy

import numpy as np

from Orange.data import (Table, Domain, StringVariable,
                         ContinuousVariable, DiscreteVariable)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.data.owfeatureconstructor import (
    DiscreteDescriptor, ContinuousDescriptor, StringDescriptor,
    construct_variables, OWFeatureConstructor,
    FeatureEditor, DiscreteFeatureEditor)

from Orange.widgets.data.owfeatureconstructor import (
    freevars, validate_exp, FeatureFunc
)


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
                            construct_variables(desc, data.domain.variables),
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
                            construct_variables(featuremodel, data.domain.variables),
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
                            construct_variables(desc, data.domain.variables)),
                     data)
        self.assertTrue(isinstance(data.domain[name], StringVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             str(data[i * 50, "iris"]) + "_name")

    def test_construct_numeric_names(self):
        data = Table("iris")
        data.domain.attributes[0].name = "0.1"
        data.domain.attributes[1].name = "1"
        desc = PyListModel(
            [ContinuousDescriptor(name="S",
                                  expression="_0_1 + _1",
                                  number_of_decimals=3)]
        )
        nv = construct_variables(desc, data.domain.variables)
        ndata = Table(Domain(nv, None), data)
        np.testing.assert_array_equal(ndata.X[:, 0],
                                      data.X[:, :2].sum(axis=1))
        ContinuousVariable._clear_all_caches()


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


class FeatureFuncTest(unittest.TestCase):
    def test_reconstruct(self):
        f = FeatureFunc("a * x + c", [("x", "x")], {"a": 2, "c": 10})
        self.assertEqual(f({"x": 2}), 14)
        f1 = pickle.loads(pickle.dumps(f))
        self.assertEqual(f1({"x": 2}), 14)
        fc = copy.copy(f)
        self.assertEqual(fc({"x": 3}), 16)

    def test_repr(self):
        self.assertEqual(repr(FeatureFunc("a + 1", [("a", 2)])),
                         "FeatureFunc('a + 1', [('a', 2)], {})")

    def test_call(self):
        f = FeatureFunc("a + 1", [("a", "a")])
        self.assertEqual(f({"a": 2}), 3)

        iris = Table("iris")
        f = FeatureFunc("sepal_width + 10",
                        [("sepal_width", iris.domain["sepal width"])])
        r = f(iris)
        np.testing.assert_array_equal(r, iris.X[:, 1] + 10)


class OWFeatureConstructorTests(WidgetTest):
    def setUp(self):
        self.widget = OWFeatureConstructor()

    def test_create_variable_with_no_data(self):
        self.widget.addFeature(
            ContinuousDescriptor("X1", "", 3))

    def test_error_invalid_expression(self):
        data = Table("iris")
        self.widget.setData(data)
        self.widget.addFeature(
            ContinuousDescriptor("X", "0", 3)
        )
        self.widget.apply()
        self.assertFalse(self.widget.Error.invalid_expressions.is_shown())
        self.widget.addFeature(
            ContinuousDescriptor("X", "0a", 3)
        )
        self.widget.apply()
        self.assertTrue(self.widget.Error.invalid_expressions.is_shown())

    def test_discrete_no_values(self):
        """
        Should not fail when there are no values set.
        GH-2417
        """
        data = Table("iris")
        self.widget.setData(data)
        discreteFeatureEditor = DiscreteFeatureEditor()

        discreteFeatureEditor.valuesedit.setText("")
        discreteFeatureEditor.nameedit.setText("D1")
        discreteFeatureEditor.expressionedit.setText("iris")
        self.widget.addFeature(
            discreteFeatureEditor.editorData()
        )
        self.assertFalse(self.widget.Error.more_values_needed.is_shown())
        self.widget.apply()
        self.assertTrue(self.widget.Error.more_values_needed.is_shown())


class TestFeatureEditor(unittest.TestCase):
    def test_has_functions(self):
        self.assertIs(FeatureEditor.FUNCTIONS["abs"], abs)
        self.assertIs(FeatureEditor.FUNCTIONS["sqrt"], math.sqrt)

# pylint: disable=unsubscriptable-object
import unittest
import ast
import sys
import math
import pickle
import copy
from unittest.mock import patch, Mock

import numpy as np

from orangewidget.settings import Context

from Orange.data import (Table, Domain, StringVariable,
                         ContinuousVariable, DiscreteVariable, TimeVariable)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils import vartype
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.data.owfeatureconstructor import (
    DiscreteDescriptor, ContinuousDescriptor, StringDescriptor,
    construct_variables, OWFeatureConstructor,
    FeatureEditor, DiscreteFeatureEditor, FeatureConstructorHandler,
    DateTimeDescriptor)

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
                                values=values, ordered=True)]
        )
        data = data.transform(Domain(list(data.domain.attributes) +
                                     construct_variables(desc, data),
                                     data.domain.class_vars,
                                     data.domain.metas))
        self.assertTrue(isinstance(data.domain[name], DiscreteVariable))
        self.assertEqual(data.domain[name].values, values)
        for i in range(3):
            self.assertEqual(data[i * 50, name], values[i])

    def test_construct_variables_discrete_no_values(self):
        data = Table("iris")
        name = 'Discrete Variable'
        expression = "str(iris)[-1]"  # last letter - a or r
        values = ()
        desc = PyListModel(
            [DiscreteDescriptor(name=name, expression=expression,
                                values=values, ordered=False)]
        )
        data = data.transform(Domain(list(data.domain.attributes) +
                                     construct_variables(desc, data),
                                     data.domain.class_vars,
                                     data.domain.metas))
        newvar = data.domain[name]
        self.assertTrue(isinstance(newvar, DiscreteVariable))
        self.assertEqual(set(data.domain[name].values), set("ar"))
        for i in range(3):
            inst = data[i * 50]
            self.assertEqual(str(inst[name]), str(inst["iris"])[-1])

    def test_construct_variables_continuous(self):
        data = Table("iris")
        name = 'Continuous Variable'
        expression = "pow(sepal_length + sepal_width, 2)"
        featuremodel = PyListModel(
            [ContinuousDescriptor(name=name, expression=expression,
                                  number_of_decimals=2)]
        )
        data = data.transform(Domain(list(data.domain.attributes) +
                                     construct_variables(featuremodel, data),
                                     data.domain.class_vars,
                                     data.domain.metas))
        self.assertTrue(isinstance(data.domain[name], ContinuousVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             pow(data[i * 50, 0] + data[i * 50, 1], 2))

    def test_construct_variables_datetime(self):
        data = Table("housing")
        name = 'Date'
        expression = '"2019-07-{:02}".format(int(MEDV/3))'
        featuremodel = PyListModel(
            [DateTimeDescriptor(name=name, expression=expression)]
        )
        data = data.transform(Domain(list(data.domain.attributes) +
                                     construct_variables(featuremodel, data),
                                     data.domain.class_vars,
                                     data.domain.metas))
        self.assertTrue(isinstance(data.domain[name], TimeVariable))
        for row in data:
            self.assertEqual(f"2019-07-{int(row['MEDV'] / 3):02}",
                             str(row["Date"])[:10])

    def test_construct_variables_string(self):
        data = Table("iris")
        name = 'String Variable'
        expression = "str(iris) + '_name'"
        desc = PyListModel(
            [StringDescriptor(name=name, expression=expression)]
        )
        data = data.transform(Domain(data.domain.attributes,
                                     data.domain.class_vars,
                                     list(data.domain.metas) +
                                     construct_variables(desc, data)))
        self.assertTrue(isinstance(data.domain[name], StringVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             str(data[i * 50, "iris"]) + "_name")

    @staticmethod
    def test_construct_numeric_names():
        data = Table("iris")
        newdomain = Domain((ContinuousVariable("0.1"), ContinuousVariable("1"))
                           + data.domain.attributes[2:], data.domain.class_var)
        data = Table.from_numpy(newdomain, data.X, data.Y)
        desc = PyListModel(
            [ContinuousDescriptor(name="S",
                                  expression="_0_1 + _1",
                                  number_of_decimals=3)]
        )
        nv = construct_variables(desc, data)
        ndata = data.transform(Domain(nv))
        np.testing.assert_array_equal(ndata.X[:, 0],
                                      data.X[:, :2].sum(axis=1))

    @staticmethod
    def test_unicode_normalization():
        micro = "\u00b5"
        domain = Domain([ContinuousVariable(micro)])
        name = 'Micro Variable'
        expression = micro
        desc = PyListModel(
            [ContinuousDescriptor(name=name, expression=expression,
                                  number_of_decimals=2)]
        )
        data = Table.from_numpy(domain, np.arange(5).reshape(5, 1))
        data = data.transform(Domain(data.domain.attributes,
                                     [],
                                     construct_variables(desc, data)))
        np.testing.assert_equal(data.X, data.metas)


class TestTools(unittest.TestCase):
    def test_free_vars(self):
        stmt = ast.parse("foo", "", "single")
        with self.assertRaises(ValueError):
            freevars(stmt, [])

        suite = ast.parse("foo; bar();", "exec")
        with self.assertRaises(ValueError):
            freevars(suite, [])

        def freevars_(source, env=None):
            return freevars(ast.parse(source, "", "eval"), env or [])

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
        iris = Table("iris")
        inst1 = iris[0]
        val1 = 2 * inst1["sepal width"] + 10
        inst2 = iris[100]
        val2 = 2 * inst2["sepal width"] + 10

        f = FeatureFunc("a * sepal_width + c",
                        [("sepal_width", iris.domain["sepal width"])],
                        {"a": 2, "c": 10})
        self.assertAlmostEqual(f(inst1), val1)
        f1 = pickle.loads(pickle.dumps(f))
        self.assertAlmostEqual(f1(inst1), val1)
        fc = copy.copy(f)
        self.assertEqual(fc(inst2), val2)

    def test_repr(self):
        self.assertEqual(repr(FeatureFunc("a + 1", [("a", 2)])),
                         "FeatureFunc('a + 1', [('a', 2)], {}, None)")

    def test_call(self):
        iris = Table("iris")
        f = FeatureFunc("sepal_width + 10",
                        [("sepal_width", iris.domain["sepal width"])])
        r = f(iris)
        np.testing.assert_array_equal(r, iris.X[:, 1] + 10)
        self.assertEqual(f(iris[0]), iris[0]["sepal width"] + 10)

    def test_string_casting(self):
        zoo = Table("zoo")
        f = FeatureFunc("name[0]",
                        [("name", zoo.domain["name"])])
        r = f(zoo)
        self.assertEqual(r, [x[0] for x in zoo.metas[:, 0]])
        self.assertEqual(f(zoo[0]), str(zoo[0, "name"])[0])

    def test_missing_variable(self):
        zoo = Table("zoo")
        assert zoo.domain.class_var.name == "type"
        f = FeatureFunc("type[0]",
                        [("type", zoo.domain["type"])])
        no_class = Domain(zoo.domain.attributes, None, zoo.domain.metas)
        data2 = zoo.transform(no_class)
        r = f(data2)
        self.assertTrue(np.all(np.isnan(r)))
        self.assertTrue(np.isnan(f(data2[0])))

    def test_invalid_expression_variable(self):
        iris = Table("iris")
        f = FeatureFunc("1 / petal_length",
                        [("petal_length", iris.domain["petal length"])])
        with iris.unlocked():
            iris[0]["petal length"] = 0

        f.mask_exceptions = False
        self.assertRaises(Exception, f, iris)
        self.assertRaises(Exception, f, iris[0])
        _ = f(iris[1])

        f.mask_exceptions = True
        r = f(iris)
        self.assertTrue(np.isnan(r[0]))
        self.assertFalse(np.isnan(r[1]))
        self.assertTrue(np.isnan(f(iris[0])))
        self.assertFalse(np.isnan(f(iris[1])))


class OWFeatureConstructorTests(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWFeatureConstructor)

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

    def test_renaming_duplicate_vars(self):
        data = Table("iris")
        self.widget.setData(data)
        self.widget.addFeature(
            ContinuousDescriptor("iris", "0", 3)
        )
        self.widget.apply()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(set(var.name for var in output.domain.variables)),
                         len(output.domain.variables))

    def test_discrete_no_values(self):
        """
        Should not fail when there are no values set.
        GH-2417
        """
        data = Table("iris")
        self.widget.setData(data)
        discreteFeatureEditor = DiscreteFeatureEditor()

        discreteFeatureEditor.valuesedit.setText("A")
        discreteFeatureEditor.nameedit.setText("D1")
        discreteFeatureEditor.expressionedit.setText("1")
        self.widget.addFeature(
            discreteFeatureEditor.editorData()
        )
        self.assertFalse(self.widget.Error.more_values_needed.is_shown())
        self.widget.apply()
        self.assertTrue(self.widget.Error.more_values_needed.is_shown())

    @patch("Orange.widgets.data.owfeatureconstructor.QMessageBox")
    def test_fix_values(self, msgbox):
        w = self.widget

        msgbox.ApplyRole, msgbox.RejectRole = object(), object()
        msgbox.return_value = Mock()
        dlgexec = msgbox.return_value.exec = Mock()

        v = [DiscreteVariable(name, values=tuple("abc"))
             for name in ("ana", "berta", "cilka")]
        domain = Domain(v, [])
        self.send_signal(w.Inputs.data, Table.from_numpy(domain, [[0, 1, 2]]))

        w.descriptors = [StringDescriptor(
            "y", "ana.value + berta.value + cilka.value")]

        # Reject fixing - no changes
        dlgexec.return_value=msgbox.RejectRole
        w.fix_expressions()
        self.assertEqual(w.descriptors[0].expression,
                         "ana.value + berta.value + cilka.value")

        dlgexec.return_value = Mock(return_value=msgbox.AcceptRole)

        w.fix_expressions()
        self.assertEqual(w.descriptors[0].expression, "ana + berta + cilka")

        w.descriptors = [StringDescriptor(
            "y", "ana.value + dani.value + cilka.value")]
        with patch.object(w, "apply"):  # dani doesn't exist and will fail
            w.fix_expressions()
        self.assertEqual(w.descriptors[0].expression,
                         "ana + dani.value + cilka")

        w.descriptors = [ContinuousDescriptor("y", "sqrt(berta)", 1)]
        w.fix_expressions()
        self.assertEqual(w.descriptors[0].expression,
                         "sqrt({'a': 0, 'b': 1, 'c': 2}[berta])")

    def test_migration_discrete_strings(self):
        v = [DiscreteVariable("Ana", values=tuple("012")),
             ContinuousVariable("Cilka")]
        domain = Domain(v)
        data = Table.from_numpy(domain, [[1, 3.14]])

        settings_w_discrete = {
            "context_settings":
            [Context(
                attributes=dict(Ana=1, Cilka=2), metas={},
                values=dict(
                    descriptors=[
                        ContinuousDescriptor("y", "Ana + int(Cilka)", 1),
                        StringDescriptor("u", "Ana.value + 'X'")
                    ],
                    currentIndex=0)
             )]
        }
        widget = self.create_widget(OWFeatureConstructor, settings_w_discrete)
        self.send_signal(widget.Inputs.data, data)
        self.assertTrue(widget.expressions_with_values)
        self.assertFalse(widget.fix_button.isHidden())
        out = self.get_output(widget.Outputs.data)
        np.testing.assert_almost_equal(out.X, [[1, 3.14, 4]])
        np.testing.assert_equal(out.metas, [["1X"]])

        settings_no_discrete = {
            "context_settings":
            [Context(
                attributes=dict(Ana=1, Cilka=2), metas={},
                values=dict(
                    descriptors=[
                        ContinuousDescriptor("y", "int(Cilka)", 1),
                    ],
                    currentIndex=0)
             )]
        }
        widget = self.create_widget(OWFeatureConstructor, settings_no_discrete)
        self.send_signal(widget.Inputs.data, data)
        self.assertFalse(widget.expressions_with_values)
        self.assertTrue(widget.fix_button.isHidden())
        out = self.get_output(widget.Outputs.data)
        np.testing.assert_almost_equal(out.X, [[1, 3.14, 3]])

        widget = self.create_widget(OWFeatureConstructor, settings_w_discrete)
        self.send_signal(widget.Inputs.data, data)
        self.assertTrue(widget.expressions_with_values)
        self.assertFalse(widget.fix_button.isHidden())
        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.expressions_with_values)
        self.assertTrue(widget.fix_button.isHidden())

    def test_report(self):
        settings = {
            "context_settings":
                [Context(
                    attributes=dict(x=2, y=2, z=2), metas={},
                    values=dict(
                        descriptors=[
                            ContinuousDescriptor("a", "x + 2", 1),
                            DiscreteDescriptor("b", "x < 3", (), False),
                            DiscreteDescriptor("c", "x > 15", (), True),
                            DiscreteDescriptor("d", "y > x", ("foo", "bar"), False),
                            DiscreteDescriptor("e", "x ** 2 + y == 5", ("foo", "bar"), True),
                            StringDescriptor("f", "str(x)"),
                            DateTimeDescriptor("g", "z")
                        ],
                        currentIndex=0)
                )]
        }

        w = self.create_widget(OWFeatureConstructor, settings)
        v = [ContinuousVariable(name) for name in "xyz"]
        domain = Domain(v, [])
        self.send_signal(w.Inputs.data, Table.from_numpy(domain, [[0, 1, 2]]))
        w.report_items = Mock()
        w.send_report()
        args = w.report_items.call_args[0][1]
        self.assertEqual(list(args), list("abcdefg"))


class TestFeatureEditor(unittest.TestCase):
    def test_has_functions(self):
        self.assertIs(FeatureEditor.FUNCTIONS["abs"], abs)
        self.assertIs(FeatureEditor.FUNCTIONS["sqrt"], math.sqrt)


class FeatureConstructorHandlerTests(unittest.TestCase):
    def test_handles_builtins_in_expression(self):
        self.assertTrue(
            FeatureConstructorHandler().is_valid_item(
                OWFeatureConstructor.descriptors,
                StringDescriptor("X", "str(A) + str(B)"),
                {"A": vartype(DiscreteVariable)},
                {"B": vartype(DiscreteVariable)}
            )
        )

        # no variables is also ok
        self.assertTrue(
            FeatureConstructorHandler().is_valid_item(
                OWFeatureConstructor.descriptors,
                StringDescriptor("X", "str('foo')"),
                {},
                {}
            )
        )

        # should fail on unknown variables
        self.assertFalse(
            FeatureConstructorHandler().is_valid_item(
                OWFeatureConstructor.descriptors,
                StringDescriptor("X", "str(X)"),
                {},
                {}
            )
        )

    def test_handles_special_characters_in_var_names(self):
        self.assertTrue(
            FeatureConstructorHandler().is_valid_item(
                OWFeatureConstructor.descriptors,
                StringDescriptor("X", "A_2_f"),
                {"A.2 f": vartype(DiscreteVariable)},
                {}
            )
        )


if __name__ == "__main__":
    unittest.main()

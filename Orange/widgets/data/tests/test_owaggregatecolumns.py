# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, abstract-method, protected-access
import unittest
from itertools import chain

from unittest.mock import Mock

import numpy as np

from Orange.data import (
    Table, Domain,
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
)
from Orange.widgets.data.owaggregatecolumns import OWAggregateColumns
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.signals import AttributeList


class TestOWAggregateColumn(WidgetTest):
    def setUp(self):
        #: OWAggregateColumns
        self.widget = self.create_widget(OWAggregateColumns)
        c1, c2, c3 = map(ContinuousVariable, "c1 c2 c3".split())
        t1, t2 = map(TimeVariable, "t1 t2".split())
        d1, d2, d3 = (DiscreteVariable(n, values=("a", "b", "c"))
                      for n in "d1 d2 d3".split())
        s1 = StringVariable("s1")
        domain1 = Domain([c1, c2, d1, d2, t1], [d3], [s1, c3, t2])
        self.data1 = Table.from_list(domain1,
                                     [[0, 1, 0, 1, 2, 0, "foo", 0, 3],
                                      [3, 1, 0, 1, 42, 0, "bar", 0, 4]])

        domain2 = Domain([ContinuousVariable("c4")])
        self.data2 = Table.from_list(domain2, [[4], [5]])

    def test_no_input(self):
        widget = self.widget
        domain = self.data1.domain

        self.send_signal(widget.Inputs.data, self.data1)
        self.assertEqual(widget.variables, [])
        widget.commit.now()
        output = self.get_output(self.widget.Outputs.data)
        self.assertIs(output, self.data1)

        widget.variables = [domain[n] for n in "c1 c2 t2".split()]
        widget.commit.now()
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(output)

        self.send_signal(widget.Inputs.data, None)
        widget.commit.now()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_compute_data(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        self.widget.variables = [domain[n] for n in "c1 c2 t2".split()]

        self.widget.operation = "Sum"
        output = self.widget._compute_data()
        self.assertEqual(output.domain.attributes[:-1], domain.attributes)
        np.testing.assert_equal(output.X[:, -1], [4, 8])

        self.widget.operation = "Max"
        output = self.widget._compute_data()
        self.assertEqual(output.domain.attributes[:-1], domain.attributes)
        np.testing.assert_equal(output.X[:, -1], [3, 4])

    def test_var_name(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        self.widget.selection_method = self.widget.SelectAllAndMeta

        self.widget.var_name = "test"
        output = self.widget._compute_data()
        self.assertEqual(output.domain.attributes[-1].name, "test")

        self.widget.var_name = "d1"
        output = self.widget._compute_data()
        self.assertNotIn(
            output.domain.attributes[-1].name,
            [var.name for var in chain(domain.variables, domain.metas)])

    def test_var_types(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        variables = [domain[n] for n in "t1 c2 t2".split()]

        for self.widget.operation in self.widget.Operations:
            self.assertIsInstance(self.widget._new_var(variables),
                                  ContinuousVariable)

        variables = [domain[n] for n in "t1 t2".split()]
        for self.widget.operation in self.widget.Operations:
            self.assertIsInstance(
                self.widget._new_var(variables),
                TimeVariable
                if self.widget.operation in ("Min", "Max", "Mean", "Median")
                else ContinuousVariable)

    def test_operations(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        variables = [domain[n] for n in "c1 c2 t2".split()]

        m1, m2 = 4 / 3, 8 / 3
        for self.widget.operation, expected in {
                "Sum": [4, 8], "Product": [0, 12],
                "Min": [0, 1], "Max": [3, 4],
                "Mean": [m1, m2],
                "Variance": [(m1 ** 2 + (m1 - 1) ** 2 + (m1 - 3) ** 2) / 3,
                             ((m2 - 3) ** 2 + (m2 - 1) ** 2 + (m2 - 4) ** 2) / 3],
                "Median": [1, 3]}.items():
            np.testing.assert_equal(
                self.widget._compute_column(variables), expected,
                err_msg=f"error in '{self.widget.operation}'")

    def test_operations_with_nan(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        with self.data1.unlocked():
            self.data1.X[1, 0] = np.nan
        variables = [domain[n] for n in "c1 c2 t2".split()]

        m1, m2 = 4 / 3, 5 / 2
        for self.widget.operation, expected in {
                "Sum": [4, 5], "Product": [0, 4],
                "Min": [0, 1], "Max": [3, 4],
                "Mean": [m1, m2],
                "Variance": [(m1 ** 2 + (m1 - 1) ** 2 + (m1 - 3) ** 2) / 3,
                             ((m2 - 1) ** 2 + (m2 - 4) ** 2) / 2],
                "Median": [1, 2.5]}.items():
            np.testing.assert_equal(
                self.widget._compute_column(variables), expected,
                err_msg=f"error in '{self.widget.operation}'")

    def test_contexts(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        self.widget.variables = [domain[n] for n in "c1 c2 t2".split()]
        saved = self.widget.variables[:]

        self.send_signal(self.widget.Inputs.data, self.data2)
        self.assertEqual(self.widget.variables, [])

        self.send_signal(self.widget.Inputs.data, self.data1)
        self.assertEqual(self.widget.variables, saved)

    def test_selection_in_context(self):
        widget = self.widget

        self.send_signal(widget.Inputs.data, self.data1)
        self.widget.variables[:] = self.data1.domain.variables[1:3]

        self.send_signal(widget.Inputs.data, self.data2)
        self.assertEqual(widget.variables, [])

        self.send_signal(widget.Inputs.data, self.data1)
        self.assertSequenceEqual(self.widget.variables[:],
                                 self.data1.domain.variables[1:3])

    def test_features_signal(self):
        widget = self.widget
        widget.selection_method = widget.SelectAll
        self.send_signal(widget.Inputs.data, self.data1)

        self.assertEqual([attr.name for attr in widget._variables()],
                         "c1 c2 t1".split())

        attr_list = [self.data1.domain[attr] for attr in "c1 t2".split()]
        self.send_signal(widget.Inputs.features, AttributeList(attr_list))
        self.assertEqual(widget._variables(), attr_list)
        self.assertFalse(widget.Warning.missing_features.is_shown())
        self.assertFalse(widget.Warning.discrete_features.is_shown())
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 7])

        attr_list = [self.data1.domain[attr] for attr in "c1 t2 d1".split()]
        self.send_signal(widget.Inputs.features, AttributeList(attr_list))
        self.assertEqual(widget._variables(), attr_list[:2])
        self.assertFalse(widget.Warning.missing_features.is_shown())
        self.assertTrue(widget.Warning.discrete_features.is_shown())
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 7])

        attr_list.append(ContinuousVariable("foo"))
        self.send_signal(widget.Inputs.features, AttributeList(attr_list))
        self.assertEqual(widget._variables(), attr_list[:2])
        self.assertTrue(widget.Warning.missing_features.is_shown())
        self.assertTrue(widget.Warning.discrete_features.is_shown())
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 7])

        self.send_signal(widget.Inputs.features, None)
        self.assertFalse(widget.Warning.missing_features.is_shown())
        self.assertFalse(widget.Warning.discrete_features.is_shown())

        del attr_list[2]  # discrete variable
        attr_list.append(ContinuousVariable("foo"))
        self.send_signal(widget.Inputs.features, AttributeList(attr_list))
        self.assertEqual(widget._variables(), attr_list[:2])
        self.assertTrue(widget.Warning.missing_features.is_shown())
        self.assertFalse(widget.Warning.discrete_features.is_shown())
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 7])

        self.assertEqual(widget.selection_group.checkedId(),
                         widget.InputFeatures)
        self.assertTrue(all(
            button.isEnabled() is (i == widget.InputFeatures)
            for i, button in enumerate(widget.selection_group.buttons())))
        self.assertFalse(widget.controls.variables.isEnabled())

        self.send_signal(widget.Inputs.features, None)
        self.assertEqual([attr.name for attr in widget._variables()],
                         "c1 c2 t1".split())
        self.assertEqual(widget.selection_group.checkedId(), widget.SelectAll)
        self.assertTrue(all(
            button.isEnabled() is (i != widget.InputFeatures)
            for i, button in enumerate(widget.selection_group.buttons())))
        self.assertFalse(widget.controls.variables.isEnabled())

        self.send_signal(widget.Inputs.features, AttributeList())
        self.assertEqual(widget.selection_group.checkedId(), widget.InputFeatures)
        self.assertTrue(all(button.isDisabled())
                        for button in widget.selection_group.buttons())
        self.assertFalse(widget.controls.variables.isEnabled())

        attr_list = [self.data1.domain[attr] for attr in "d1 d2".split()]
        self.send_signal(widget.Inputs.features, AttributeList(attr_list))
        self.assertEqual(widget.selection_group.checkedId(), widget.InputFeatures)
        self.assertTrue(all(button.isDisabled())
                        for button in widget.selection_group.buttons())
        self.assertFalse(widget.controls.variables.isEnabled())
        self.assertNotIn(
            "agg",
            [var.name for var in self.get_output(widget.Outputs.data).domain])

    def test_selection_radios(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.data1)
        widget.variables = [self.data1.domain[attr] for attr in "c1 t2".split()]

        widget.selection_group.button(widget.SelectAll).click()
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 46])

        widget.selection_group.button(widget.SelectAllAndMeta).click()
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [6, 50])

        widget.selection_group.button(widget.SelectManually).click()
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 7])

    def test_operation_changed(self):
        widget = self.widget
        self.send_signal(widget.Inputs.data, self.data1)
        widget.selection_group.button(widget.SelectAll).click()
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [3, 46])

        oper = widget.Operations["Max"].name
        widget.operation_combo.setCurrentText(oper)
        widget.operation_combo.textActivated[str].emit(oper)
        np.testing.assert_equal(
            self.get_output(widget.Outputs.data).get_column_view("agg")[0],
            [2, 42])

    def test_and_others(self):
        self.assertEqual(
            self.widget._and_others(self.data1.domain.variables[:1], 1),
            "'c1'")
        self.assertEqual(
            self.widget._and_others(self.data1.domain.variables[:1], 10),
            "'c1'")
        self.assertEqual(
            self.widget._and_others(self.data1.domain.variables, 20),
            "'c1', 'c2', 'd1', 'd2', 't1' and 'd3'")
        self.assertEqual(
            self.widget._and_others(self.data1.domain.variables, 6),
            "'c1', 'c2', 'd1', 'd2', 't1' and 'd3'")
        self.assertEqual(
            self.widget._and_others(self.data1.domain.variables, 5),
            "'c1', 'c2', 'd1', 'd2', 't1' and 1 more")
        self.assertEqual(
            self.widget._and_others(self.data1.domain.variables, 2),
            "'c1', 'c2' and 4 more")

    def test_missing(self):
        attrs = self.data1.domain.attributes
        self.assertEqual(self.widget._missing(attrs, attrs), "")

        self.assertEqual(self.widget._missing(attrs, attrs[1:]),
                         f"'{attrs[0].name}'")
        self.assertEqual(self.widget._missing(attrs, attrs[2:]),
                         f"'{attrs[0].name}' and '{attrs[1].name}'")

    def test_report(self):
        self.widget.send_report()

        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        self.widget.variables = [domain[n] for n in "c1 c2 t2".split()]
        self.widget.send_report()

        domain3 = Domain([ContinuousVariable(f"c{i:02}") for i in range(100)])
        data3 = Table.from_numpy(domain3, np.zeros((2, 100)))
        self.send_signal(self.widget.Inputs.data, data3)
        self.widget.variables[:] = self.widget.variable_model[:]
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()

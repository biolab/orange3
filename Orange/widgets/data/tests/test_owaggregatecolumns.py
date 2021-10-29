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
        self.widget.variables = self.widget.variable_model[:]

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

        self.widget.variables = [domain[n] for n in "t1 c2 t2".split()]
        for self.widget.operation in self.widget.Operations:
            self.assertIsInstance(self.widget._new_var(), ContinuousVariable)

        self.widget.variables = [domain[n] for n in "t1 t2".split()]
        for self.widget.operation in self.widget.Operations:
            self.assertIsInstance(
                self.widget._new_var(),
                TimeVariable
                if self.widget.operation in ("Min", "Max", "Mean", "Median")
                else ContinuousVariable)

    def test_operations(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        self.widget.variables = [domain[n] for n in "c1 c2 t2".split()]

        m1, m2 = 4 / 3, 8 / 3
        for self.widget.operation, expected in {
                "Sum": [4, 8], "Product": [0, 12],
                "Min": [0, 1], "Max": [3, 4],
                "Mean": [m1, m2],
                "Variance": [(m1 ** 2 + (m1 - 1) ** 2 + (m1 - 3) ** 2) / 3,
                             ((m2 - 3) ** 2 + (m2 - 1) ** 2 + (m2 - 4) ** 2) / 3],
                "Median": [1, 3]}.items():
            np.testing.assert_equal(
                self.widget._compute_column(), expected,
                err_msg=f"error in '{self.widget.operation}'")

    def test_operations_with_nan(self):
        domain = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        with self.data1.unlocked():
            self.data1.X[1, 0] = np.nan
        self.widget.variables = [domain[n] for n in "c1 c2 t2".split()]

        m1, m2 = 4 / 3, 5 / 2
        for self.widget.operation, expected in {
                "Sum": [4, 5], "Product": [0, 4],
                "Min": [0, 1], "Max": [3, 4],
                "Mean": [m1, m2],
                "Variance": [(m1 ** 2 + (m1 - 1) ** 2 + (m1 - 3) ** 2) / 3,
                             ((m2 - 1) ** 2 + (m2 - 4) ** 2) / 2],
                "Median": [1, 2.5]}.items():
            np.testing.assert_equal(
                self.widget._compute_column(), expected,
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

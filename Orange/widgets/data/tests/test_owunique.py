# Tests test protected methods
# pylint: disable=protected-access
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.widgets.tests.base import WidgetTest

from Orange.widgets.data import owunique


class TestOWUnique(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(owunique.OWUnique)  #: OWUnique

        self.domain = Domain(
            [DiscreteVariable(name, values=("a", "b", "c")) for name in "abcd"],
            [ContinuousVariable("e")],
            [DiscreteVariable(name, values=("a", "b", "c")) for name in "fg"])
        self.table = Table.from_numpy(
            self.domain,
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 2, 0, 0],
             [1, 2, 0, 0]],
            np.arange(6),
            np.zeros((6, 2)))

    def test_settings(self):
        w = self.widget
        domain = self.domain
        w.unconditional_commit = Mock()

        self.send_signal(w.Inputs.data, self.table)
        w.selected_vars = [w.var_model[2]]

        self.send_signal(w.Inputs.data, None)
        self.assertEqual(w.selected_vars, [])

        domain = Domain(domain.attributes[2:], domain.class_vars, domain.metas)
        table = self.table.transform(domain)
        self.send_signal(w.Inputs.data, table)
        self.assertEqual(w.selected_vars, [self.domain[2]])

    def test_unconditional_commit(self):
        w = self.widget
        w.autocommit = False

        w._compute_unique_data = cud = Mock()
        cud.return_value = self.table

        self.send_signal(w.Inputs.data, self.table)
        out = self.get_output(w.Outputs.data)
        self.assertIs(out, cud.return_value)

        self.send_signal(w.Inputs.data, None)
        out = self.get_output(w.Outputs.data)
        self.assertIs(out, None)

    def test_compute(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.table)
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, self.table.Y)

        w.selected_vars = w.var_model[:2]

        w.tiebreaker = "Last instance"
        w.commit.now()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [2, 3, 4, 5])

        w.tiebreaker = "First instance"
        w.commit.now()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [0, 3, 4, 5])

        w.tiebreaker = "Middle instance"
        w.commit.now()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [1, 3, 4, 5])

        w.tiebreaker = "Discard non-unique instances"
        w.commit.now()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.Y, [3, 4, 5])

    def test_use_all_when_non_selected(self):
        w = self.widget
        w.tiebreaker = "First instance"

        data = self.table.transform(Domain(self.table.domain.attributes))

        self.send_signal(w.Inputs.data, data)
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.X, data.X[2:])

        w.selected_vars.clear()
        w.commit.now()
        out = self.get_output(w.Outputs.data)
        np.testing.assert_equal(out.X, data.X[2:])

    def test_no_output_on_no_unique(self):
        w = self.widget
        w.tiebreaker = "Discard non-unique instances"

        attrs = self.table.domain.attributes
        data = Table.from_numpy(Domain(attrs), np.zeros((5, len(attrs))))
        self.send_signal(w.Inputs.data, data)
        self.assertIsNone(self.get_output(w.Outputs.data))


if __name__ == "__main__":
    unittest.main()

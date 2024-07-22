import unittest
from unittest.mock import patch, Mock

import numpy as np

from orangewidget.tests.utils import simulate

from Orange.data import Domain, DiscreteVariable, ContinuousVariable, Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owcolumn import OWColumn


class OWColumnTest(WidgetTest):
    def setUp(self):
        self.widget: OWColumn = self.create_widget(OWColumn)

        self.domain = Domain([DiscreteVariable("d1", values=["a", "b"]),
                              DiscreteVariable("d2", values=["c", "d"]),
                              DiscreteVariable("d3", values=["d", "c"]),
                              ContinuousVariable("c1"),
                              ContinuousVariable("c2")
                              ],
                             DiscreteVariable("cls", values=["c", "d"]),
                             [DiscreteVariable("m1", values=["a", "b"]),
                              DiscreteVariable("m2", values=["d"]),
                              ContinuousVariable("c3")]
                             )
        self.data = Table.from_numpy(
            self.domain,
            np.array([[0, 0, 0, 1, 0.5],
                      [0, 1, 1, 0.25, -3],
                      [1, 0, np.nan, np.nan, np.nan]]),
            np.array([0, 1, 1]),
            np.array([[0, 0, 2],
                      [1, 0, 8],
                      [np.nan, np.nan, 5]])
        )

    def test_set_data(self):
        def proper_data():
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertEqual({var.name for var in self.widget.column_model},
                             {"d2", "d3", "c1", "c2", "m2", "c3"})
            self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
            self.assertIsNotNone(self.get_output(self.widget.Outputs.model))

        proper_data()
        self.assertIs(self.widget.column, self.widget.column_model[0])

        simulate.combobox_activate_item(self.widget.column_combo, "d3")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.model))

        # No class - show error
        self.send_signal(
            self.widget.Inputs.data,
            self.data.transform(Domain(self.domain.attributes))
        )
        self.assertEqual(len(self.widget.column_model), 0)
        self.assertIsNone(self.widget.column)
        self.assertTrue(self.widget.Error.no_class.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

        # Alles gut - no error, recover setting
        proper_data()
        self.assertIs(self.widget.column, self.widget.column_model[1])

        # No data - no column
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.column)
        self.assertEqual(len(self.widget.column_model), 0)
        self.assertFalse(self.widget.Error.no_class.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

        proper_data()

        # No class - error
        self.send_signal(
            self.widget.Inputs.data,
            self.data.transform(Domain(self.domain.attributes))
        )
        self.assertTrue(self.widget.Error.no_class.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

        proper_data()

        # No data - no column, no error
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.column)
        self.assertEqual(len(self.widget.column_model), 0)
        self.assertFalse(self.widget.Error.no_class.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

        proper_data()

        # No suitable columns - no column, error
        self.send_signal(
            self.widget.Inputs.data,
            self.data.transform(
                Domain(self.domain.attributes,
                       DiscreteVariable("cls", values=["e", "f", "g"])
                       )
            )
        )
        self.assertIsNone(self.widget.column)
        self.assertEqual(len(self.widget.column_model), 0)
        self.assertTrue(self.widget.Error.no_variables.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

        # Binary class: allow continuous attrs
        self.send_signal(
            self.widget.Inputs.data,
            self.data.transform(
                Domain(self.domain.attributes,
                       DiscreteVariable("cls", values=["e", "f"]),
                       self.domain.metas)
            )
        )
        self.assertEqual({var.name for var in self.widget.column_model},
                         {"c1", "c2", "c3"})
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.model))
        self.assertIs(self.widget.column, self.widget.column_model[0])

    def test_gui_update(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        simulate.combobox_activate_item(self.widget.column_combo, "d2")
        self.assertFalse(self.widget.options.isEnabled())

        simulate.combobox_activate_item(self.widget.column_combo, "c3")
        self.assertTrue(self.widget.options.isEnabled())

    @patch("Orange.widgets.evaluate.owcolumn.ColumnLearner")
    def test_commit(self, learner_class):
        learner = learner_class.return_value = Mock()

        self.widget.apply_logistic = 0
        self.widget.offset = 3.5
        self.widget.k = 0.5
        self.send_signal(self.widget.Inputs.data, self.data)

        domain = self.domain
        class_var = domain.class_var

        simulate.combobox_activate_item(self.widget.column_combo, "d2")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.model))
        self.assertFalse(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["d2"], None, None)
        learner.assert_called_with(self.data)
        learner.reset_mock()

        simulate.combobox_activate_item(self.widget.column_combo, "c3")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.assertTrue(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["c3"], None, None)
        learner.assert_not_called()

        simulate.combobox_activate_item(self.widget.column_combo, "c1")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.model))
        self.assertFalse(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["c1"], None, None)
        learner.assert_called_with(self.data)
        learner.reset_mock()

        simulate.combobox_activate_item(self.widget.column_combo, "c2")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.assertTrue(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["c2"], None, None)
        learner.assert_not_called()

        simulate.combobox_activate_item(self.widget.column_combo, "d2")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.model))
        self.assertFalse(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["d2"], None, None)
        learner.assert_called_with(self.data)
        learner.reset_mock()

        simulate.combobox_activate_item(self.widget.column_combo, "c3")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.assertTrue(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["c3"], None, None)
        learner.assert_not_called()

        self.widget.apply_logistic = 1
        self.widget.on_apply_logistic_changed()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.model))
        self.assertFalse(self.widget.Error.invalid_probabilities.is_shown())
        learner_class.assert_called_with(class_var, domain["c3"], 3.5, 0.5)
        learner.assert_called_with(self.data)
        learner.reset_mock()

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.apply_logistic = 0
        self.widget.send_report()
        self.widget.apply_logistic = 1
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()

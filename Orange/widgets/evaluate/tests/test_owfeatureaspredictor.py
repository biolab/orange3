import unittest
from unittest.mock import patch, Mock

import numpy as np

from Orange.data import Table, Domain, \
    StringVariable, DiscreteVariable, ContinuousVariable
from Orange.widgets.evaluate.owfeatureaspredictor import OWFeatureAsPredictor
from Orange.widgets.tests.base import WidgetTest


class OWFeatureAsPredictorTest(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWFeatureAsPredictor)
        self.disc_a = DiscreteVariable("a", values=("a", "b", "c"))
        self.disc_b = DiscreteVariable("b", values=("c", "a", "b"))
        self.disc_c = DiscreteVariable("c", values=("c", "a", "b", "d"))
        self.disc_d = DiscreteVariable("d", values=("c", "b"))
        self.disc_de = DiscreteVariable("de", values=("b", "c"))
        self.cont_e = ContinuousVariable("e")
        self.cont_f = ContinuousVariable("f")
        self.cont_g = ContinuousVariable("g")

        attrs = [self.disc_b, self.disc_c, self.disc_d,
                 self.cont_e]
        meta_attrs = [self.cont_f, StringVariable("s"), self.disc_de]
        x = np.array([[0, 1, 0, 0.1],
                      [1, 1, 1, 0.6],
                      [2, 0, np.nan, 0.2],
                      [0, np.nan, 1, np.nan],
                      [np.nan, 0, 6, 0.8]])
        y = np.array([0, 1, 0, 1, 0])
        metas = np.array([[-0.1, "a", 0],
                          [0.3, "b", 1],
                          [np.nan, "c", 1],
                          [0.9, "d", 1],
                          [0.24, "e", 0]])

        self.class_data = Table.from_numpy(
            Domain(attrs, self.disc_a, meta_attrs),
            x, y, metas
        )

        self.bin_data = Table.from_numpy(
            Domain(attrs, self.disc_de, meta_attrs[:-1]),
            x, y, metas[:, :-1]
        )

        self.regr_data = Table.from_numpy(
            Domain(attrs[1:], self.cont_g, meta_attrs),
            x[:, 1:], y, metas
        )

    def set_column(self, var):
        combo = self.widget.column_combo
        index = combo.model().indexOf(var)
        self.widget.column_combo.setCurrentIndex(index)
        self.widget.column_combo.activated.emit(index)

    def test_model(self):
        w = self.widget
        model = w.column_combo.model()
        self.send_signal(self.class_data)
        self.assertEqual(list(model), [self.disc_b, self.disc_d, self.disc_de])
        self.assertIsNotNone(self.get_output(w.Outputs.model))
        self.assertIsNotNone(self.get_output(w.Outputs.learner))

        self.send_signal(None)
        self.assertIsNone(self.get_output(w.Outputs.model))
        self.assertIsNone(self.get_output(w.Outputs.learner))
        self.assertEqual(list(model), [])

        self.send_signal(self.bin_data)
        self.assertEqual(list(model), [self.disc_d, self.cont_e, self.cont_f])
        self.assertIsNotNone(self.get_output(w.Outputs.model))
        self.assertIsNotNone(self.get_output(w.Outputs.learner))

        self.send_signal(self.regr_data)
        self.assertEqual(list(model), [self.cont_e, self.cont_f])
        self.assertIsNotNone(self.get_output(w.Outputs.model))
        self.assertIsNotNone(self.get_output(w.Outputs.learner))

        self.send_signal(
            self.bin_data.transform(Domain(self.bin_data.domain.attributes)))
        self.assertIsNone(self.get_output(w.Outputs.model))
        self.assertIsNone(self.get_output(w.Outputs.learner))
        self.assertTrue(w.Error.no_class.is_shown())
        self.assertFalse(w.Error.no_variables.is_shown())

        self.send_signal(self.regr_data)
        self.assertIsNotNone(self.get_output(w.Outputs.model))
        self.assertIsNotNone(self.get_output(w.Outputs.learner))
        self.assertFalse(w.Error.no_class.is_shown())
        self.assertFalse(w.Error.no_variables.is_shown())

        self.send_signal(
            self.regr_data.transform(
                Domain([self.disc_b, self.disc_c, self.disc_d], self.cont_g)))
        self.assertIsNone(self.get_output(w.Outputs.model))
        self.assertIsNone(self.get_output(w.Outputs.learner))
        self.assertFalse(w.Error.no_class.is_shown())
        self.assertTrue(w.Error.no_variables.is_shown())

    def test_combo_hint(self):
        self.send_signal(self.bin_data)
        self.assertEqual(self.widget.column, self.disc_d)

        self.send_signal(self.regr_data)
        self.assertEqual(self.widget.column, self.cont_e)

        self.set_column(self.cont_f)
        # Keep f, because it exists
        self.send_signal(self.bin_data)
        self.assertEqual(self.widget.column, self.cont_f)

        self.set_column(self.disc_d)
        # Can't keep
        self.send_signal(self.regr_data)
        self.assertEqual(self.widget.column, self.cont_e)

        # Keep hint when there is no data
        self.set_column(self.cont_f)
        self.send_signal(None)
        self.send_signal(self.regr_data)
        self.assertEqual(self.widget.column, self.cont_f)

    def set_checked(self, checked):
        cb = self.widget.cb_transformation
        if cb.isChecked() != checked:
            cb.click()

    def test_update_transform_checkbox(self):
        check = self.widget.cb_transformation
        # No data: button is enabled
        self.assertTrue(check.isEnabled())

        # Check the checkbox so that we see it behaves properly
        # when disabled, unchecked and re-enabled
        self.set_checked(True)

        with self.subTest("Multinomial target"):
            self.send_signal(self.class_data)
            # Discrete target: button is disabled and unchecked
            # transformation not applied
            self.assertFalse(check.isEnabled())
            self.assertFalse(check.isChecked())
            self.assertFalse(self.widget.apply_transformation)

            # No data: re-enabled and re-checked
            self.send_signal(None)
            self.assertTrue(check.isEnabled())
            self.assertTrue(check.isChecked())

            self.send_signal(self.bin_data)
            # Binary target, discrete column: button is disabled and unchecked
            # transformation not applied
            self.assertIs(self.widget.column, self.disc_d)
            self.assertFalse(check.isEnabled())
            self.assertFalse(check.isChecked())
            self.assertFalse(self.widget.apply_transformation)

        with self.subTest("Binary target, numeric column within range"):
            # Binary target, numeric column within range:
            # enabled, checked (because of setting)
            self.set_column(self.cont_e)
            self.assertTrue(check.isEnabled())
            self.assertTrue(check.isChecked())
            self.assertTrue(self.widget.apply_transformation)
            # Binary target, numeric column outside range:
            # disabled, unchecked (because of setting)
            self.set_column(self.cont_e)
            self.assertTrue(check.isEnabled())
            self.assertTrue(check.isChecked())
            self.assertTrue(self.widget.apply_transformation)
            # Go back to numeric withing range to verify that it is re-checked
            self.set_column(self.cont_e)
            self.assertTrue(check.isEnabled())
            self.assertTrue(check.isChecked())
            self.assertTrue(self.widget.apply_transformation)

            self.send_signal(None)
            self.set_checked(False)
            self.send_signal(self.bin_data)
            self.set_column(self.cont_e)
            self.assertTrue(check.isEnabled())
            self.assertFalse(check.isChecked())
            self.assertFalse(self.widget.apply_transformation)

        with self.subTest("Regression target"):
            self.send_signal(self.regr_data)
            # Regression target: button is enabled and checked
            self.assertTrue(check.isEnabled())
            self.assertFalse(check.isChecked())
            self.assertFalse(self.widget.apply_transformation)

            self.set_checked(True)
            self.assertTrue(self.widget.apply_transformation)

            self.send_signal(self.class_data)
            assert not check.isEnabled()
            assert not check.isChecked()

            self.send_signal(self.regr_data)
            self.assertTrue(check.isEnabled())
            self.assertTrue(check.isChecked())
            self.assertTrue(self.widget.apply_transformation)

            # Column that would be out of range for discrete,
            # but regression doesn't mind
            self.set_checked(False)
            self.set_column(self.cont_f)
            self.assertTrue(check.isEnabled())
            self.assertFalse(check.isChecked())
            self.assertFalse(self.widget.apply_transformation)

    def test_checkbox_text(self):
        check = self.widget.cb_transformation
        self.send_signal(self.bin_data)
        self.assertIn("logistic", check.text())
        self.assertIn("logistic", check.toolTip())
        self.send_signal(self.regr_data)
        self.assertIn("linear", check.text())
        self.assertIn("linear", check.toolTip())
        self.send_signal(self.class_data)
        self.assertIn("logistic", check.text())
        self.assertIn("logistic", check.toolTip())

    @patch("Orange.widgets.evaluate.owfeatureaspredictor.ColumnLearner")
    def test_commit(self, learner):
        model = self.widget.column_combo.model()
        extract = self.regr_data.domain.class_var

        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))
        self.set_checked(False)

        learner.reset_mock()
        self.send_signal(self.regr_data)
        learner.assert_called_once()
        assert not self.widget.apply_transformation
        self.assertEqual(learner.call_args, ((extract, model[0], False),))
        self.assertIs(self.get_output(
            self.widget.Outputs.learner), learner.return_value)
        self.assertIs(self.get_output(
            self.widget.Outputs.model), learner.return_value.return_value)

        learner.reset_mock()
        self.set_column(model[1])
        assert not self.widget.apply_transformation
        learner.assert_called_once()
        self.assertEqual(learner.call_args, ((extract, model[1], False),))
        self.assertIs(self.get_output(
            self.widget.Outputs.learner), learner.return_value)
        self.assertIs(self.get_output(
            self.widget.Outputs.model), learner.return_value.return_value)

        learner.reset_mock()
        self.set_checked(True)
        learner.assert_called_once()
        self.assertEqual(learner.call_args, ((extract, model[1], True),))
        self.assertIs(self.get_output(
            self.widget.Outputs.learner), learner.return_value)
        self.assertIs(self.get_output(
            self.widget.Outputs.model), learner.return_value.return_value)

    @patch("Orange.modelling.column.ColumnModel")
    def test_report_data(self, model):
        def assert_items(*expected):
            self.assertEqual(tuple(i[1] for i in items.call_args[0][0][1:]),
                             expected)

        items = self.widget.report_items = Mock()
        model.return_value.intercept = 1
        model.return_value.coefficient = 2

        self.widget.send_report()
        items.assert_not_called()

        self.send_signal(self.class_data)
        self.set_column(self.disc_d)
        self.widget.send_report()
        self.assertEqual(items.call_args[0][0][0][1], "d")
        assert_items(False, False, False)

        self.set_column(self.disc_b)
        self.widget.send_report()
        self.assertEqual(items.call_args[0][0][0][1], "b")
        assert_items(False, False, False)

        self.send_signal(self.regr_data)
        self.set_checked(False)
        self.widget.send_report()
        self.assertEqual(items.call_args[0][0][0][1], "e")
        assert_items(False, False, False)

        self.set_checked(True)
        self.widget.send_report()
        self.assertEqual(items.call_args[0][0][0][1], "e")
        assert_items("linear", 1, 2)

        self.send_signal(self.bin_data)
        self.set_column(self.disc_d)
        self.widget.send_report()
        self.assertEqual(items.call_args[0][0][0][1], "d")
        assert_items(False, False, False)

        self.set_column(self.cont_e)
        assert self.widget.apply_transformation
        self.widget.send_report()
        self.assertEqual(items.call_args[0][0][0][1], "e")
        assert_items("logistic", 1, 2)


if __name__ == "__main__":
    unittest.main()

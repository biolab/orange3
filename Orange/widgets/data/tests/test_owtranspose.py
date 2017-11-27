# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table, Domain
from Orange.widgets.data.owtranspose import OWTranspose
from Orange.widgets.tests.base import WidgetTest


class TestOWTranspose(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTranspose)
        self.zoo = Table("zoo")

    def test_transpose(self):
        widget = self.widget
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertEqual(widget.data, None)
        self.send_signal(widget.Inputs.data, self.zoo)
        output = self.get_output(widget.Outputs.data)
        transpose = Table.transpose(self.zoo)
        np.testing.assert_array_equal(output.X, transpose.X)
        np.testing.assert_array_equal(output.Y, transpose.Y)
        np.testing.assert_array_equal(output.metas, transpose.metas)
        self.send_signal(widget.Inputs.data, None)
        self.assertIsNone(self.get_output(widget.Outputs.data))

    def test_feature_type(self):
        widget = self.widget
        data = Table("conferences.tab")
        metas = data.domain.metas
        domain = data.domain
        # Put one non-string column to metas, so widget must skip it
        domain2 = Domain(domain.attributes[:-1],
                         domain.class_vars,
                         (domain.attributes[0], ) + domain.metas)
        data2 = Table(domain2, data)

        widget.feature_type = widget.GENERIC
        self.send_signal(widget.Inputs.data, data2)

        # By default, the widget switches from GENERIC to the first string meta
        self.assertEqual(widget.feature_type, widget.FROM_META_ATTR)
        self.assertIs(widget.feature_names_column, metas[0])
        output = self.get_output(widget.Outputs.data)
        self.assertListEqual(
            [a.name for a in output.domain.attributes],
            [metas[0].to_val(m) for m in data.metas[:, 0]])

        # Test that the widget takes the correct column
        widget.feature_names_column = metas[1]
        widget.apply()
        output = self.get_output(widget.Outputs.data)
        self.assertListEqual(
            [a.name for a in output.domain.attributes],
            [metas[1].to_val(m) for m in data.metas[:, 1]])

        # Switch to generic
        self.assertEqual(widget.DEFAULT_PREFIX, "Feature")
        widget.feature_type = widget.GENERIC
        widget.apply()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(
            all(x.name.startswith(widget.DEFAULT_PREFIX)
                for x in output.domain.attributes))

        # Check that the widget uses the supplied name
        widget.feature_name = "Foo"
        widget.apply()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(
            all(x.name.startswith("Foo ") for x in output.domain.attributes))

        # Check that the widget uses default when name is not given
        widget.feature_name = ""
        widget.apply()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(
            all(x.name.startswith(widget.DEFAULT_PREFIX)
                for x in output.domain.attributes))

    def test_send_report(self):
        widget = self.widget
        widget.feature_type = widget.FROM_META_ATTR
        widget.report_button.click()
        widget.feature_type = widget.GENERIC
        widget.report_button.click()

        self.send_signal(widget.Inputs.data, self.zoo)
        widget.feature_type = widget.FROM_META_ATTR
        widget.report_button.click()
        widget.feature_type = widget.GENERIC
        widget.report_button.click()

    def test_gui_behaviour(self):
        widget = self.widget
        widget.apply = unittest.mock.Mock()

        # widget.apply must be called
        widget.auto_apply = False

        # No data: type is generic, meta radio disabled
        self.assertEqual(widget.feature_type, widget.GENERIC)
        self.assertFalse(widget.meta_button.isEnabled())
        self.assertFalse(widget.apply.called)

        # Data with metas: default type is meta, radio enabled
        self.send_signal(widget.Inputs.data, self.zoo)
        self.assertTrue(widget.meta_button.isEnabled())
        self.assertEqual(widget.feature_type, widget.FROM_META_ATTR)
        self.assertIs(widget.feature_names_column, widget.feature_model[0])
        self.assertTrue(widget.apply.called)

        # Editing the line edit changes the radio button to generic
        widget.apply.reset_mock()
        widget.controls.feature_name.editingFinished.emit()
        self.assertEqual(widget.feature_type, widget.GENERIC)
        self.assertTrue(widget.apply.called)

        # Changing combo changes the radio button to meta
        widget.apply.reset_mock()
        widget.feature_combo.activated.emit(0)
        self.assertEqual(widget.feature_type, widget.FROM_META_ATTR)
        self.assertTrue(widget.apply.called)

    def test_all_whitespace(self):
        widget = self.widget
        widget.feature_name = "  "
        widget.controls.feature_name.editingFinished.emit()
        self.assertEqual(widget.feature_name, "")

    def test_error(self):
        widget = self.widget
        with unittest.mock.patch("Orange.data.Table.transpose",
                                 side_effect=ValueError("foo")):
            self.send_signal(widget.Inputs.data, self.zoo)
            self.assertTrue(widget.Error.value_error.is_shown())
        self.send_signal(widget.Inputs.data, self.zoo)
        self.assertFalse(widget.Error.value_error.is_shown())

if __name__ == "__main__":
    unittest.main()

# pylint: disable=missing-docstring
from unittest.mock import Mock

import numpy as np

from AnyQt.QtWidgets import QWidget

from orangewidget.tests.base import GuiTest
from Orange.data import Table
from Orange.widgets.data.owcreateinstance import OWCreateInstance, \
    DiscreteVariableEditor, ContinuousVariableEditor
from Orange.widgets.tests.base import WidgetTest, datasets
from Orange.widgets.utils.state_summary import format_summary_details, \
    format_multiple_summaries


class TestOWCreateInstance(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCreateInstance)

    def test_output(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 1)
        self.assertEqual(output.domain, data.domain)

    def test_summary(self):
        info = self.widget.info
        data, reference = Table("iris"), Table("iris")[:1]
        no_input, no_output = "No data on input", "No data on output"

        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        self.send_signal(self.widget.Inputs.data, data)
        data_list = [("Data", data), ("Reference", None)]
        summary, details = "150, 0", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        output = self.get_output(self.widget.Outputs.data)
        summary, details = f"{len(output)}", format_summary_details(output)
        self.assertEqual(info._StateInfo__output_summary.brief, summary)
        self.assertEqual(info._StateInfo__output_summary.details, details)

        self.send_signal(self.widget.Inputs.reference, reference)
        data_list = [("Data", data), ("Reference", reference)]
        summary, details = "150, 1", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None)
        data_list = [("Data", None), ("Reference", reference)]
        summary, details = "0, 1", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        self.send_signal(self.widget.Inputs.reference, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)

    def test_default_buttons(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        default_box = self.widget.controlArea.layout().itemAt(0).widget()
        buttons = default_box.children()[1:]
        buttons[0].click()
        buttons[1].click()
        buttons[2].click()

    def test_table(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(self.widget.view.model().rowCount(), 5)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

        data = Table("zoo")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(self.widget.view.model().rowCount(), 17)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.view.model().rowCount(), 0)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)


class TestDiscreteVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = DiscreteVariableEditor(
            self.parent, ["Foo", "Bar"], self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, 0)
        self.assertEqual(self.editor._combo.currentText(), "Foo")
        self.callback.assert_not_called()

    def test_edit(self):
        """ Edit combo by user. """
        self.editor._combo.setCurrentText("Bar")
        self.assertEqual(self.editor.value, 1)
        self.assertEqual(self.editor._combo.currentText(), "Bar")
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set combo box value. """
        self.editor.value = 1
        self.assertEqual(self.editor.value, 1)
        self.assertEqual(self.editor._combo.currentText(), "Bar")
        self.callback.assert_called_once()


class TestContinuousVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.data = Table("iris")
        self.variable = self.data.domain[0]
        values = self.data.get_column_view(self.variable)[0]
        self.min_value = np.nanmin(values)
        self.max_value = np.nanmax(values)
        self.editor = ContinuousVariableEditor(
            self.parent, self.variable, self.min_value,
            self.max_value, self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, self.min_value)
        self.assertEqual(self.editor._slider.value(), self.min_value * 10)
        self.assertEqual(self.editor._spin.value(), self.min_value)
        self.callback.assert_not_called()

    def test_edit_slider(self):
        """ Edit slider by user. """
        self.editor._slider.setValue(self.max_value * 10)
        self.assertEqual(self.editor.value, self.max_value)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 2
        self.editor._slider.setValue(value * 10)
        self.assertEqual(self.editor.value, value)
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.callback.assert_called_once()

    def test_edit_spin(self):
        """ Edit spin by user. """
        self.editor._spin.setValue(self.max_value)
        self.assertEqual(self.editor.value, self.max_value)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 2
        self.editor._spin.setValue(value)
        self.assertEqual(self.editor.value, value)
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set slider/spin value. """
        self.editor.value = -2
        self.assertEqual(self.editor._slider.value(), self.min_value * 10)
        self.assertEqual(self.editor._spin.value(), self.min_value)
        self.assertEqual(self.editor.value, self.min_value)
        self.callback.assert_not_called()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 4
        self.editor.value = value
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.assertEqual(self.editor.value, value)
        self.callback.assert_called_once()


if __name__ == "__main__":
    import unittest
    unittest.main()

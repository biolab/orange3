# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.data.owsimulator import OWCreateInstance
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


if __name__ == "__main__":
    import unittest
    unittest.main()

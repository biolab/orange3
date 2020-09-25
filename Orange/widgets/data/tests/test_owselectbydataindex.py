# pylint: disable=protected-access
from Orange.data import Table, Domain
from Orange.widgets.data.owselectbydataindex import OWSelectByDataIndex
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_FEATURE_NAME
from Orange.widgets.utils.state_summary import format_summary_details, \
    format_multiple_summaries


class TestOWSelectSubset(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWSelectByDataIndex)

    def apply_subset_20_40(self, data):
        data_subset = data[20:40].transform(Domain([]))  # destroy domain
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data_subset)

    def test_subset(self):
        data = Table("iris")
        self.apply_subset_20_40(data)
        out = self.get_output(self.widget.Outputs.matching_data)
        self.assertEqual(list(data[20:40]), list(out))

    def test_non_matching(self):
        data = Table("iris")
        self.apply_subset_20_40(data)
        out = self.get_output(self.widget.Outputs.non_matching_data)
        self.assertEqual(list(data[:20]) + list(data[40:]), list(out))

    def test_annotated(self):
        data = Table("iris")
        self.apply_subset_20_40(data)
        out = self.get_output(self.widget.Outputs.annotated_data)
        vals = [a[ANNOTATED_DATA_FEATURE_NAME].value for a in out]
        self.assertEqual(['No']*20 + ['Yes']*20 + ['No']*(len(data) - 40), vals)

    def test_subset_nosubset(self):
        data = Table("iris")
        data_subset = Table("titanic")
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data_subset)
        matching = self.get_output(self.widget.Outputs.matching_data)
        non_matching = self.get_output(self.widget.Outputs.non_matching_data)
        self.assertTrue(self.widget.Warning.instances_not_matching.is_shown())
        self.assertEqual([], list(matching))
        self.assertEqual(list(data), list(non_matching))

    def test_summary(self):
        """Check if the status bar is updated when data is received"""
        data = Table("iris")
        info = self.widget.info
        no_input, no_output = "No data on input", "No data on output"

        self.send_signal(self.widget.Inputs.data, data)
        data_list = [("Data", data), ("Data subset", None)]
        summary, details = f"{len(data)}, 0", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        self.assertEqual(info._StateInfo__output_summary.brief, "0")

        self.send_signal(self.widget.Inputs.data_subset, data)
        data_list = [("Data", data), ("Data subset", data)]
        summary = f"{len(data)}, {len(data)}"
        details = format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        output = self.get_output(self.widget.Outputs.matching_data)
        summary, details = f"{len(output)}", format_summary_details(output)
        self.assertEqual(info._StateInfo__output_summary.brief, summary)
        self.assertEqual(info._StateInfo__output_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None)
        data_list = [("Data", None), ("Data subset", data)]
        summary, details = f"0, {len(data)}", format_multiple_summaries(
            data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        self.assertEqual(info._StateInfo__output_summary.brief, "-")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        self.send_signal(self.widget.Inputs.data_subset, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "-")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertEqual(info._StateInfo__output_summary.brief, "-")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

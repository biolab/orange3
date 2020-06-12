# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import random
import unittest

from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owdistancemap import OWDistanceMap
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.utils.state_summary import format_summary_details


class TestOWDistanceMap(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Distances"
        cls.signal_data = Euclidean(cls.data)

    def setUp(self):
        self.widget = self.create_widget(OWDistanceMap)

    def _select_data(self):
        random.seed(42)
        selected_indices = random.sample(range(0, len(self.data)), 20)
        self.widget._selection = selected_indices
        self.widget.commit()
        return selected_indices

    def test_saved_selection(self):
        self.widget.settingsHandler.pack_data(self.widget)
        # no assert here, just test is doesn't crash on empty

        self.send_signal(self.signal_name, self.signal_data)
        random.seed(42)
        self.widget.matrix_item.set_selections([(range(5, 10), range(8, 15))])
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWDistanceMap, stored_settings=settings)
        self.send_signal(self.signal_name, self.signal_data, widget=w)
        self.assertEqual(len(self.get_output(w.Outputs.selected_data, widget=w)), 10)

    def test_summary(self):
        """Check if the status bar updates"""
        info = self.widget.info
        no_input, no_output = "No data on input", "No data on output"
        matrix = f"{len(self.signal_data)}"

        self.send_signal(self.widget.Inputs.distances, self.signal_data)
        self.assertEqual(info._StateInfo__input_summary.brief, matrix)
        self.assertEqual(info._StateInfo__input_summary.details, matrix)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)
        self._select_data()
        output = self.get_output(self.widget.Outputs.selected_data)
        summary, details = f"{len(output)}", format_summary_details(output)
        self.assertEqual(info._StateInfo__output_summary.brief, summary)
        self.assertEqual(info._StateInfo__output_summary.details, details)

        self.send_signal(self.widget.Inputs.distances, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)


if __name__ == "__main__":
    unittest.main()

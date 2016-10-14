# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
from Orange.data import Table
from Orange.preprocess import Continuize
from Orange.widgets.visualize.owheatmap import OWHeatMap
from Orange.widgets.tests.base import WidgetTest


class TestOWHeatMap(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWHeatMap)
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.titanic = Table("titanic")

    def test_input_data(self):
        """Check widget's data with data on the input"""
        for data in (self.iris, self.housing):
            self.assertEqual(self.widget.data, None)
            self.send_signal("Data", data)
            self.assertEqual(self.widget.data, data)
            self.assertFalse(self.widget.Error.active)
            self.assertFalse(self.widget.Warning.active)
            self.assertFalse(self.widget.Information.active)
            self.send_signal("Data", None)

    def test_error_message(self):
        self.send_signal("Data", self.titanic)
        self.assertTrue(self.widget.Error.active)
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Error.active)

    def test_information_message(self):
        self.widget.sort_rows = self.widget.OrderedClustering
        continuizer = Continuize()
        cont_titanic = continuizer(self.titanic)
        self.send_signal("Data", cont_titanic)
        self.assertTrue(self.widget.Information.active)
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Information.active)

    def test_settings_changed(self):
        self.send_signal("Data", self.iris)
        # check output when "Sorting Column" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.widget.colsortcb.activated.emit(1)
        self.widget.colsortcb.setCurrentIndex(1)
        self.assertIsNone(self.get_output("Selected Data"))
        # check output when "Sorting Row" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.widget.rowsortcb.activated.emit(1)
        self.widget.rowsortcb.setCurrentIndex(1)
        self.assertIsNone(self.get_output("Selected Data"))
        # check output when "Merge by k-means" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.widget.controlledAttributes["merge_kmeans"][0].control.setChecked(True)
        self.assertIsNone(self.get_output("Selected Data"))

    def _select_data(self):
        rows = random.sample(range(0, len(self.iris)), 20)
        self.widget.selection_manager.select_rows(rows)
        self.widget.on_selection_finished()

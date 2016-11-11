# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.preprocess import Continuize
from Orange.widgets.visualize.owheatmap import OWHeatMap
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWHeatMap(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.housing = Table("housing")
        cls.titanic = Table("titanic")

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWHeatMap)

    def test_input_data(self):
        """Check widget's data with data on the input"""
        for data in (self.data, self.housing):
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
        self.send_signal("Data", self.data)
        self.assertFalse(self.widget.Error.active)

    def test_information_message(self):
        self.widget.sort_rows = self.widget.OrderedClustering
        continuizer = Continuize()
        cont_titanic = continuizer(self.titanic)
        self.send_signal("Data", cont_titanic)
        self.assertTrue(self.widget.Information.active)
        self.send_signal("Data", self.data)
        self.assertFalse(self.widget.Information.active)

    def test_settings_changed(self):
        self.send_signal("Data", self.data)
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
        self.widget.controls.merge_kmeans.setChecked(True)
        self.assertIsNone(self.get_output("Selected Data"))

    def _select_data(self):
        selected_indices = list(range(10, 31))
        self.widget.selection_manager.select_rows(selected_indices)
        self.widget.on_selection_finished()
        return selected_indices

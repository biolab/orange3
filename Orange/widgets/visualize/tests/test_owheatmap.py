# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
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

        self.send_signal("Data", self.data[:0])

    def test_error_message(self):
        self.send_signal("Data", self.titanic)
        self.assertTrue(self.widget.Error.active)
        self.send_signal("Data", self.data)
        self.assertFalse(self.widget.Error.active)

    def test_information_message(self):
        self.widget.controls.row_clustering.setChecked(True)
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
        self.widget.controls.col_clustering.setChecked(True)
        self.assertIsNone(self.get_output("Selected Data"))
        # check output when "Sorting Row" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.widget.controls.row_clustering.setChecked(True)
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

    def test_not_enough_data_settings_changed(self):
        """Check widget for dataset with one feature or for one instance"""
        msg = self.widget.Error
        for kmeans_checked in (False, True):
            self.widget.controls.merge_kmeans.setChecked(kmeans_checked)
            for col_checked in (False, True):
                self.widget.controls.col_clustering.setChecked(col_checked)
                self.send_signal("Data", None)
                self.send_signal("Data", self.data[:, 0])
                if col_checked:
                    self.assertTrue(msg.not_enough_features.is_shown())
                for row_checked in (False, True):
                    self.widget.controls.row_clustering.setChecked(row_checked)
                    self.send_signal("Data", None)
                    self.send_signal("Data", self.data[0:1])
                    if row_checked:
                        self.assertTrue(msg.not_enough_instances.is_shown())
                    elif kmeans_checked and row_checked:
                        self.assertTrue(msg.not_enough_instances_k_means.is_shown())
            self.send_signal("Data", None)
            self.assertFalse(msg.not_enough_features.is_shown())
            self.assertFalse(msg.not_enough_instances.is_shown())
            self.assertFalse(msg.not_enough_instances_k_means.is_shown())

    def test_color_low_high(self):
        '''
        GH-2017
        '''
        self.widget.controls.threshold_low._send_value(4)
        self.widget.controls.threshold_high._send_value(2)
        self.assertGreater(self.widget.threshold_high, self.widget.threshold_low)

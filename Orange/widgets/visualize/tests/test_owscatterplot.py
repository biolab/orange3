# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.owscatterplot import OWScatterPlot


class TestOWScatterPlot(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWScatterPlot)
        self.data = Table("iris")

    def test_set_data(self):
        self.widget.set_data(self.data)
        self.assertEqual(self.widget.data, self.data)
        self.assertEqual(self.widget.subset_data, None)

    def test_subset_data(self):
        self.widget.set_subset_data(self.data[:30])
        self.assertEqual(len(self.widget.subset_data), 30)
        self.assertEqual(self.widget.data, None)
        np.testing.assert_array_equal(self.widget.subset_data, self.data[:30])

    def test_set_data_none(self):
        self.widget.set_data(None)
        self.assertEqual(self.widget.data, None)
        self.assertEqual(self.widget.subset_data, None)

    def test_subset_data_none(self):
        self.widget.set_subset_data(None)
        self.assertEqual(self.widget.subset_data, None)
        self.assertEqual(self.widget.data, None)

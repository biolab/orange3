# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.owscatterplot import OWScatterPlot


class TestOWScatterPlot(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWScatterPlot)
        self.iris = Table("iris")

    def test_set_data(self):
        # Connect iris to scatter plot
        self.send_signal("Data", self.iris)

        # First two attribute should be selected as x an y
        self.assertEqual(self.widget.attr_x, self.iris.domain[0].name)
        self.assertEqual(self.widget.attr_y, self.iris.domain[1].name)

        # Class var should be selected as color
        self.assertEqual(self.widget.graph.attr_color,
                         self.iris.domain.class_var.name)

        # Change which attributes are displayed
        self.widget.attr_x = self.iris.domain[2].name
        self.widget.attr_y = self.iris.domain[3].name

        # Disconnect the data
        self.send_signal("Data", None)

        # removing data should have cleared attributes
        self.assertEqual(self.widget.attr_x, None)
        self.assertEqual(self.widget.attr_y, None)
        self.assertEqual(self.widget.graph.attr_color, None)

        # and remove the legend
        self.assertEqual(self.widget.graph.legend, None)

        # Connect iris again
        # same attributes that were used last time should be selected
        self.send_signal("Data", self.iris)

        self.assertEqual(self.widget.attr_x, self.iris.domain[2].name)
        self.assertEqual(self.widget.attr_y, self.iris.domain[3].name)

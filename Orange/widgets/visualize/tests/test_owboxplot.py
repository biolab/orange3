# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest import skip

import numpy as np
from Orange.data import Table, ContinuousVariable
from Orange.widgets.visualize.owboxplot import OWBoxPlot
from Orange.widgets.tests.base import WidgetTest


class TestOWBoxPlot(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.zoo = Table("zoo")
        cls.housing = Table("housing")

    def setUp(self):
        self.widget = self.create_widget(OWBoxPlot)

    @skip("Known bug, FIXME!")
    def test_input_data(self):
        """Check widget's data"""
        self.send_signal("Data", self.iris)
        self.assertGreater(len(self.widget.attrs), 0)
        self.send_signal("Data", None)
        self.assertEqual(len(self.widget.attrs), 0)

    def test_input_data_missings_cont_group_var(self):
        """Check widget with continuous data with missing values and group variable"""
        data = self.iris
        data.X[:, 0] = np.nan
        self.send_signal("Data", data)
        # used to crash, see #1568

    def test_input_data_missings_cont_no_group_var(self):
        """Check widget with continuous data with missing values and no group variable"""
        data = self.housing
        data.X[:, 0] = np.nan
        self.send_signal("Data", data)
        # used to crash, see #1568

    def test_input_data_missings_disc_group_var(self):
        """Check widget with discrete data with missing values and group variable"""
        data = self.zoo
        data.X[:, 0] = np.nan
        self.send_signal("Data", data)

    def test_input_data_missings_disc_no_group_var(self):
        """Check widget discrete data with missing values and no group variable"""
        data = self.zoo
        data.domain.class_var = ContinuousVariable("cls")
        data.X[:, 0] = np.nan
        self.send_signal("Data", data)

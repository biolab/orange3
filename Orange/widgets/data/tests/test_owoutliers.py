# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table
from Orange.widgets.data.owoutliers import OWOutliers
from Orange.widgets.tests.base import WidgetTest


class TestOWOutliers(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWOutliers)
        self.iris = Table("iris")

    def test_data(self):
        """Check widget's data and the output with data on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)
        self.assertEqual(len(self.get_output(self.widget.Outputs.inliers)), 76)
        self.assertEqual(len(self.get_output(self.widget.Outputs.outliers)), 74)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.inliers))

    def test_memory_error(self):
        """
        Handling memory error.
        GH-2374
        """
        data = Table("iris")[::3]
        self.assertFalse(self.widget.Error.memory_error.is_shown())
        with unittest.mock.patch(
            "Orange.widgets.data.owoutliers.OWOutliers.detect_outliers",
            side_effect=MemoryError):
            self.send_signal("Data", data)
            self.assertTrue(self.widget.Error.memory_error.is_shown())

    def test_nans(self):
        """Widget does not crash with nans"""
        a = np.arange(20, dtype=float).reshape(4, 5)
        a[0, 0] = np.nan
        data = Table(a)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertIsNot(self.get_output(self.widget.Outputs.inliers), None)

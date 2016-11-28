# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import numpy as np

from Orange.data import Table
from Orange.widgets.data.owtranspose import OWTranspose
from Orange.widgets.tests.base import WidgetTest


class TestOWTranspose(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTranspose)
        self.zoo = Table("zoo")

    def test_data(self):
        """Check widget's data and the output with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal("Data", self.zoo)
        self.assertEqual(self.widget.data, self.zoo)
        output = self.get_output("Data")
        transpose = Table.transpose(self.zoo)
        np.testing.assert_array_equal(output.X, transpose.X)
        np.testing.assert_array_equal(output.Y, transpose.Y)
        np.testing.assert_array_equal(output.metas, transpose.metas)
        self.send_signal("Data", None)
        self.assertEqual(self.widget.data, None)
        self.assertIsNone(self.get_output("Data"))

    def test_parameters(self):
        """Check widget's output for all possible values of parameters"""
        self.send_signal("Data", self.zoo)
        self.assertListEqual(
            [a.name for a in self.get_output("Data").domain.attributes],
            [self.zoo.domain.metas[0].to_val(m) for m in self.zoo.metas[:, 0]])
        self.widget.feature_radio.buttons[0].click()
        self.widget.apply()
        self.assertTrue(all(["Feature" in x.name for x
                             in self.get_output("Data").domain.attributes]))

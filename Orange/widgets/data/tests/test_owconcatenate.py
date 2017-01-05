# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import numpy as np

from Orange.data import Table
from Orange.widgets.data.owconcatenate import OWConcatenate
from Orange.widgets.tests.base import WidgetTest

class TestOWConcatenate(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWConcatenate)
        self.iris = Table("iris")
        self.titanic = Table("titanic")

    def test_single_input(self):
        self.assertIsNone(self.get_output("Data"))
        self.send_signal("Primary Data", self.iris)
        output = self.get_output("Data")
        self.assertEqual(list(output), list(self.iris))
        self.send_signal("Primary Data", None)
        self.assertIsNone(self.get_output("Data"))
        self.send_signal("Additional Data", self.iris)
        output = self.get_output("Data")
        self.assertEqual(list(output), list(self.iris))
        self.send_signal("Additional Data", None)
        self.assertIsNone(self.get_output("Data"))

    def test_two_inputs_union(self):
        self.send_signal("Primary Data", self.iris)
        self.send_signal("Additional Data", self.titanic)
        output = self.get_output("Data")
        # needs to contain all instances
        self.assertEqual(len(output), len(self.iris) + len(self.titanic))
        # needs to contain all variables
        outvars = output.domain.variables
        self.assertLess(set(self.iris.domain.variables), set(outvars))
        self.assertLess(set(self.titanic.domain.variables), set(outvars))
        # the first part of the data set is iris, the second part is titanic
        np.testing.assert_equal(self.iris.X, output.X[:len(self.iris), :-3])
        self.assertTrue(np.isnan(output.X[:len(self.iris), -3:]).all())
        np.testing.assert_equal(self.titanic.X, output.X[len(self.iris):, -3:])
        self.assertTrue(np.isnan(output.X[len(self.iris):, :-3]).all())

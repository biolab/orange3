# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.data.owneighbors import OWNeighbors, METRICS
from Orange.widgets.tests.base import WidgetTest, ParameterMapping


class TestOWNeighbors(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNeighbors,
                                         stored_settings={"auto_apply": False})
        self.iris = Table("iris")

    def test_input_data(self):
        """Check widget's data with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.data, self.iris)

    def test_input_data_disconnect(self):
        """Check widget's data after disconnecting data on the input"""
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.data, self.iris)
        self.send_signal("Data", None)
        self.assertEqual(self.widget.data, None)

    def test_input_reference(self):
        """Check widget's reference with reference on the input"""
        self.assertEqual(self.widget.reference, None)
        self.send_signal("Reference", self.iris)
        self.assertEqual(self.widget.reference, self.iris)

    def test_input_reference_disconnect(self):
        """Check reference after disconnecting reference on the input"""
        self.send_signal("Data", self.iris)
        self.send_signal("Reference", self.iris)
        self.assertEqual(self.widget.reference, self.iris)
        self.send_signal("Reference", None)
        self.assertEqual(self.widget.reference, None)
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output("Neighbors"))

    def test_output_neighbors(self):
        """Check if neighbors are on the output after apply"""
        self.assertIsNone(self.get_output("Neighbors"))
        self.send_signal("Data", self.iris)
        self.send_signal("Reference", self.iris[:10])
        self.widget.apply_button.button.click()
        self.assertIsNotNone(self.get_output("Neighbors"))
        self.assertIsInstance(self.get_output("Neighbors"), Table)

    def test_settings(self):
        """Check neighbors for various distance metrics"""
        settings = [ParameterMapping("", self.widget.distance_combo, METRICS),
                    ParameterMapping("", self.widget.nn_spin)]
        for setting in settings:
            for val in setting.values:
                self.send_signal("Data", self.iris)
                self.send_signal("Reference", self.iris[:10])
                setting.set_value(val)
                self.widget.apply_button.button.click()
                self.assertIsNotNone(self.get_output("Neighbors"))

    def test_exclude_reference(self):
        """Check neighbors when reference is excluded"""
        reference = self.iris[:5]
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.reference, reference)
        self.widget.apply_button.button.click()
        neighbors = self.get_output(self.widget.Outputs.data)
        for inst in reference:
            self.assertNotIn(inst, neighbors)

    def test_include_reference(self):
        """Check neighbors when reference is included"""
        self.widget.controls.exclude_reference.setChecked(False)
        reference = self.iris[:5]
        self.send_signal("Data", self.iris)
        self.send_signal("Reference", reference)
        self.widget.apply_button.button.click()
        neighbors = self.get_output("Neighbors")
        for inst in reference:
            self.assertIn(inst, neighbors)

    def test_similarity(self):
        reference = self.iris[:10]
        self.send_signal("Data", self.iris)
        self.send_signal("Reference", reference)
        self.widget.apply_button.button.click()
        neighbors = self.get_output("Neighbors")
        self.assertEqual(self.iris.domain.attributes,
                         neighbors.domain.attributes)
        self.assertEqual(self.iris.domain.class_vars,
                         neighbors.domain.class_vars)
        self.assertIn("similarity", neighbors.domain)
        self.assertTrue(all(100 >= ins["similarity"] >= 0 for ins in neighbors))

    def test_missing_values(self):
        data = Table("iris")
        reference = data[:3]
        data.X[0:10, 0] = np.nan
        self.send_signal("Data", self.iris)
        self.send_signal("Reference", reference)
        self.widget.apply_button.button.click()
        self.assertIsNotNone(self.get_output("Neighbors"))

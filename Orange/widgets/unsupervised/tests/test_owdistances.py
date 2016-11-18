# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.distance import MahalanobisDistance, Mahalanobis
from Orange.widgets.unsupervised.owdistances import OWDistances, METRICS
from Orange.widgets.tests.base import WidgetTest


class TestOWDistances(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.titanic = Table("titanic")

    def setUp(self):
        self.widget = self.create_widget(OWDistances)

    def test_distance_combo(self):
        """Check distances when the metric changes"""
        self.assertEqual(self.widget.metrics_combo.count(), len(METRICS))
        self.send_signal("Data", self.iris)
        for i, metric in enumerate(METRICS):
            if isinstance(metric, MahalanobisDistance):
                metric.fit(self.iris)
            self.widget.metrics_combo.activated.emit(i)
            self.widget.metrics_combo.setCurrentIndex(i)
            self.send_signal("Data", self.iris)
            np.testing.assert_array_equal(
                metric(self.iris), self.get_output("Distances"))

    def test_error_message(self):
        """Check if error message appears and then disappears when
        data is removed from input"""
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Error.no_continuous_features.is_shown())
        self.send_signal("Data", self.titanic)
        self.assertTrue(self.widget.Error.no_continuous_features.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.no_continuous_features.is_shown())

    def test_mahalanobis_error(self):
        self.widget.axis = 1
        self.send_signal("Data", self.iris)
        self.widget.compute_distances(Mahalanobis, self.iris)
        self.assertTrue(self.widget.Error.too_few_observations.is_shown())
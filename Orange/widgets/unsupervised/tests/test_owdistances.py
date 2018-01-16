# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import Mock

import numpy as np

from Orange.data import Table
from Orange.widgets.unsupervised.owdistances import OWDistances, METRICS
from Orange.widgets.tests.base import WidgetTest


class TestOWDistances(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")[::5]
        cls.titanic = Table("titanic")[::10]

    def setUp(self):
        self.widget = self.create_widget(OWDistances)

    def test_distance_combo(self):
        """Check distances when the metric changes"""
        self.assertEqual(self.widget.metrics_combo.count(), len(METRICS))
        self.send_signal(self.widget.Inputs.data, self.iris)
        for i, (_, metric) in enumerate(METRICS):
            self.widget.metrics_combo.activated.emit(i)
            self.widget.metrics_combo.setCurrentIndex(i)
            self.send_signal(self.widget.Inputs.data, self.iris)
            if metric.supports_normalization:
                expected = metric(self.iris, normalize=self.widget.normalized_dist)
            else:
                expected = metric(self.iris)

            np.testing.assert_array_equal(
                expected, self.get_output(self.widget.Outputs.distances))

    def test_error_message(self):
        """Check if error message appears and then disappears when
        data is removed from input"""
        self.widget.metric_idx = 2
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertFalse(self.widget.Error.no_continuous_features.is_shown())
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertTrue(self.widget.Error.no_continuous_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_continuous_features.is_shown())

    def test_too_big_array(self):
        """
        Users sees an error message when calculating too large arrays and Orange
        does not crash.
        GH-2315
        """
        self.assertEqual(len(self.widget.Error.active), 0)
        self.send_signal(self.widget.Inputs.data, self.iris)

        mock = Mock(side_effect=ValueError)
        self.widget.compute_distances(mock, self.iris)
        self.assertTrue(self.widget.Error.distances_value_error.is_shown())

        mock = Mock(side_effect=MemoryError)
        self.widget.compute_distances(mock, self.iris)
        self.assertEqual(len(self.widget.Error.active), 1)
        self.assertTrue(self.widget.Error.distances_memory_error.is_shown())

    def test_migrates_normalized_dist(self):
        w = self.create_widget(OWDistances, stored_settings={"metric_idx": 0})
        self.assertFalse(w.normalized_dist)

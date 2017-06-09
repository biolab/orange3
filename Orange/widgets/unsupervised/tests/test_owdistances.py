# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import Mock

import numpy as np

from Orange.data import Table
from Orange.distance import MahalanobisDistance
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
        self.send_signal("Data", self.iris)
        for i, metric in enumerate(METRICS):
            if isinstance(metric, MahalanobisDistance):
                metric = MahalanobisDistance(self.iris)
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
        mah_index = [i for i, d in enumerate(METRICS)
                     if isinstance(d, MahalanobisDistance)][0]
        self.widget.metric_idx = mah_index
        self.widget.autocommit = True

        invalid = self.iris[:]
        invalid.X = np.vstack((invalid.X[0], invalid.X[0]))
        invalid.Y = np.vstack((invalid.Y[0], invalid.Y[0]))
        datasets = [self.iris, None, invalid]
        bad = [False, False, True]
        out = [True, False, False]

        for data1, bad1, out1 in zip(datasets, bad, out):
            for data2, bad2, out2 in zip(datasets, bad, out):
                self.send_signal("Data", data1)
                self.assertEqual(self.widget.Error.mahalanobis_error.is_shown(), bad1)
                self.assertEqual(self.get_output("Distances") is not None, out1)
                self.send_signal("Data", data2)
                self.assertEqual(self.widget.Error.mahalanobis_error.is_shown(), bad2)
                self.assertEqual(self.get_output("Distances") is not None, out2)

    def test_too_big_array(self):
        """
        Users sees an error message when calculating too large arrays and Orange
        does not crash.
        GH-2315
        """
        self.assertEqual(len(self.widget.Error.active), 0)
        self.send_signal("Data", self.iris)

        mock = Mock(side_effect=ValueError)
        self.widget.compute_distances(mock, self.iris)
        self.assertTrue(self.widget.Error.distances_value_error.is_shown())

        mock = Mock(side_effect=MemoryError)
        self.widget.compute_distances(mock, self.iris)
        self.assertEqual(len(self.widget.Error.active), 1)
        self.assertTrue(self.widget.Error.distances_memory_error.is_shown())

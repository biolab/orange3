# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random

import numpy as np

from AnyQt.QtCore import QEvent

from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owmds import OWMDS
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWMDS(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Distances"
        cls.signal_data = Euclidean(cls.data)
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWMDS)

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget.select_indices(points)
        self.widget.commit()
        return sorted(points)

    def test_pca_init(self):
        self.send_signal(self.signal_name, self.signal_data)
        self.widget.customEvent(QEvent(QEvent.User))
        self.widget.commit()
        output = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        np.testing.assert_array_almost_equal(
            output.X[0, 4:], np.array([-2.6928912, 0.32603512]))
        np.testing.assert_array_almost_equal(
            output.X[1, 4:], np.array([-2.72432089, -0.21129957]))
        np.testing.assert_array_almost_equal(
            output.X[2, 4:], np.array([-2.90231621, -0.13535431]))
        np.testing.assert_array_almost_equal(
            output.X[3, 4:], np.array([-2.75269913, -0.33885988]))

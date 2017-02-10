# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
from unittest.mock import patch, Mock

import numpy as np

from AnyQt.QtCore import QEvent

from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owmds import OWMDS
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets
from Orange.widgets.tests.utils import simulate


class TestOWMDS(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Distances"
        cls.signal_data = Euclidean(cls.data)
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWMDS)  # type: OWMDS

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

    def test_nan_plot(self):
        data = datasets.missing_data_1()
        self.send_signal("Data", data)

        simulate.combobox_run_through_all(self.widget.cb_color_value)
        simulate.combobox_run_through_all(self.widget.cb_color_value)
        simulate.combobox_run_through_all(self.widget.cb_shape_value)
        simulate.combobox_run_through_all(self.widget.cb_size_value)
        simulate.combobox_run_through_all(self.widget.cb_label_value)

        self.send_signal("Data", None)

        data.X[:, 0] = np.nan
        data.Y[:] = np.nan
        data.metas[:, 1] = np.nan

        self.send_signal("Data", data)

        simulate.combobox_run_through_all(self.widget.cb_color_value)
        simulate.combobox_run_through_all(self.widget.cb_shape_value)
        simulate.combobox_run_through_all(self.widget.cb_size_value)
        simulate.combobox_run_through_all(self.widget.cb_label_value)

    @patch("Orange.projection.MDS.__call__", Mock(side_effect=MemoryError))
    def test_out_of_memory(self):
        with patch("sys.excepthook", Mock()) as hook:
            self.send_signal("Data", self.data)
            self.process_events()
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())

    @patch("Orange.projection.MDS.__call__", Mock(side_effect=ValueError))
    def test_other_error(self):
        with patch("sys.excepthook", Mock()) as hook:
            self.send_signal("Data", self.data)
            self.process_events()
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.optimization_error.is_shown())


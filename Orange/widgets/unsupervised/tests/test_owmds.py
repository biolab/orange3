# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
from unittest.mock import patch, Mock

import numpy as np

from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owmds import OWMDS
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
        self.widget = self.create_widget(
            OWMDS, stored_settings={
                "max_iter": 10,
                "initialization": OWMDS.PCA,
            }
        )  # type: OWMDS

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget.select_indices(points)
        self.widget.commit()
        return sorted(points)

    def test_pca_init(self):
        self.send_signal(self.signal_name, self.signal_data, wait=1000)
        output = self.get_output(self.widget.Outputs.annotated_data)
        expected = np.array(
            [[-2.69304803, 0.32676458],
             [-2.7246721, -0.20921726],
             [-2.90244761, -0.13630526],
             [-2.75281107, -0.33854819]]
        )
        np.testing.assert_array_almost_equal(output.X[:4, 4:], expected)

    def test_nan_plot(self):
        data = datasets.missing_data_1()
        self.send_signal(self.widget.Inputs.data, data)

        simulate.combobox_run_through_all(self.widget.cb_color_value)
        simulate.combobox_run_through_all(self.widget.cb_color_value)
        simulate.combobox_run_through_all(self.widget.cb_shape_value)
        simulate.combobox_run_through_all(self.widget.cb_size_value)
        simulate.combobox_run_through_all(self.widget.cb_label_value)

        self.send_signal(self.widget.Inputs.data, None)

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
            self.send_signal(self.widget.Inputs.data, self.data)
            self.process_events()
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())

    @patch("Orange.projection.MDS.__call__", Mock(side_effect=ValueError))
    def test_other_error(self):
        with patch("sys.excepthook", Mock()) as hook:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.process_events()
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.optimization_error.is_shown())

    def test_distances_without_data_0(self):
        """
        Only distances and no data.
        GH-2335
        """
        signal_data = Euclidean(self.data, axis=0)
        signal_data.row_items = None
        self.send_signal("Distances", signal_data)

    def test_distances_without_data_1(self):
        """
        Only distances and no data.
        GH-2335
        """
        signal_data = Euclidean(self.data, axis=1)
        signal_data.row_items = None
        self.send_signal("Distances", signal_data)

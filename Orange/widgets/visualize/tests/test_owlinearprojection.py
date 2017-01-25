# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
import numpy as np

from Orange.data import Table
from Orange.widgets.visualize.owlinearprojection import OWLinearProjection
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets
from Orange.widgets.tests.utils import EventSpy, excepthook_catch, simulate


class TestOWLinearProjection(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWLinearProjection)  # type: OWLinearProjection

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget.select_indices(points)
        return sorted(points)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal("Data", Table(Table("iris").domain))

    def test_nan_plot(self):
        data = datasets.missing_data_1()
        espy = EventSpy(self.widget, OWLinearProjection.ReplotRequest)
        with excepthook_catch():
            self.send_signal("Data", data)
            # ensure delayed replot request is processed
            if not espy.events():
                assert espy.wait(1000)

        cb_color = self.widget.controls.color_index
        cb_size = self.widget.controls.size_index
        cb_shape = self.widget.controls.shape_index
        cb_jitter = self.widget.controls.jitter_value

        simulate.combobox_run_through_all(cb_color)
        simulate.combobox_run_through_all(cb_size)
        simulate.combobox_run_through_all(cb_shape)
        with excepthook_catch():
            simulate.combobox_activate_index(cb_jitter, 1, delay=1)

        data = data.copy()
        data.X[:, 0] = np.nan
        data.Y[:] = np.nan

        spy = EventSpy(self.widget, OWLinearProjection.ReplotRequest)
        self.send_signal("Data", data)
        self.send_signal("Data Subset", data[2:3])
        if not spy.events():
            assert spy.wait()

        with excepthook_catch():
            simulate.combobox_activate_item(cb_color, "X1")

        with excepthook_catch():
            simulate.combobox_activate_item(cb_size, "X1")

        with excepthook_catch():
            simulate.combobox_activate_item(cb_shape, "D")

        with excepthook_catch():
            simulate.combobox_activate_index(cb_jitter, 2, delay=1)

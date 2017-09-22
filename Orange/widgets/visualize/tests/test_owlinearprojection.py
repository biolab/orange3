# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
import numpy as np

from Orange.data import Table
from Orange.widgets.settings import Context
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
        self.send_signal(self.widget.Inputs.data, Table(Table("iris").domain))

    def test_nan_plot(self):
        data = datasets.missing_data_1()
        espy = EventSpy(self.widget, OWLinearProjection.ReplotRequest)
        with excepthook_catch():
            self.send_signal(self.widget.Inputs.data, data)
            # ensure delayed replot request is processed
            if not espy.events():
                assert espy.wait(1000)

        cb_color = self.widget.controls.attr_color
        cb_size = self.widget.controls.attr_size
        cb_shape = self.widget.controls.attr_shape
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
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data_subset, data[2:3])
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

    def test_points_combo_boxes(self):
        self.send_signal("Data", self.data)
        self.assertEqual(len(self.widget.controls.attr_color.model()), 8)
        self.assertEqual(len(self.widget.controls.attr_shape.model()), 3)
        self.assertEqual(len(self.widget.controls.attr_size.model()), 6)
        self.assertEqual(len(self.widget.controls.attr_label.model()), 8)

    def test_migrate_settings_from_version_1(self):
        # Settings from Orange 3.4.0
        settings = {
            '__version__': 1,
            'alpha_value': 255,
            'auto_commit': True,
            'class_density': False,
            'context_settings': [
                Context(attributes={'iris': 1,
                                    'petal length': 2, 'petal width': 2,
                                    'sepal length': 2, 'sepal width': 2},
                        metas={},
                        ordered_domain=[('sepal length', 2),
                                        ('sepal width', 2),
                                        ('petal length', 2),
                                        ('petal width', 2),
                                        ('iris', 1)],
                        time=1504865133.098991,
                        values={'__version__': 1,
                                'color_index': (5, -2),
                                'shape_index': (1, -2),
                                'size_index': (1, -2),
                                'variable_state': ({}, -2)})],
            'jitter_value': 0,
            'legend_anchor': ((1, 0), (1, 0)),
            'point_size': 8,
            'savedWidgetGeometry': None
        }
        w = self.create_widget(OWLinearProjection, stored_settings=settings)
        iris = Table("iris")
        self.send_signal(w.Inputs.data, iris, widget=w)
        self.assertEqual(w.point_width, 8)
        self.assertEqual(w.attr_color, iris.domain["iris"])
        self.assertEqual(w.attr_shape, iris.domain["iris"])
        self.assertEqual(w.attr_size, iris.domain["sepal length"])

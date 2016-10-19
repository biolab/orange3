# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random

from Orange.widgets.visualize.owlinearprojection import OWLinearProjection
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWLinearProjection(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWLinearProjection)

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget.select_indices(points)
        return sorted(points)

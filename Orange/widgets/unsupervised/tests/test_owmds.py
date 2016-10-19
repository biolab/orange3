# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random

from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owmds import OWMDS
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
        self.selected_indices = sorted(points)

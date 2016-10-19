# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owdistancemap import OWDistanceMap
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWDistanceMap(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Distances"
        cls.signal_data = Euclidean(cls.data)

    def setUp(self):
        self.widget = self.create_widget(OWDistanceMap)

    def _select_data(self):
        random.seed(42)
        selected_indices = random.sample(range(0, len(self.data)), 20)
        self.widget._selection = selected_indices
        self.widget.commit()
        return selected_indices

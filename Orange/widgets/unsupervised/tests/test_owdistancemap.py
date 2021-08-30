# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import random
import unittest

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
        self.widget.commit.now()
        return selected_indices

    def test_saved_selection(self):
        self.widget.settingsHandler.pack_data(self.widget)
        # no assert here, just test is doesn't crash on empty

        self.send_signal(self.signal_name, self.signal_data)
        random.seed(42)
        self.widget.matrix_item.set_selections([(range(5, 10), range(8, 15))])
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWDistanceMap, stored_settings=settings)
        self.send_signal(self.signal_name, self.signal_data, widget=w)
        self.assertEqual(len(self.get_output(w.Outputs.selected_data, widget=w)), 10)


if __name__ == "__main__":
    unittest.main()

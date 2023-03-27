# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import random
import unittest

from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets.unsupervised.owdistancemap import OWDistanceMap
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.tests.utils import simulate


class TestOWDistanceMap(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = OWDistanceMap.Inputs.distances
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

    def test_widget(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, self.signal_data)
        for i in range(w.sorting_cb.count()):
            simulate.combobox_activate_index(w.sorting_cb, i)
        for i in range(w.annot_combo.count()):
            simulate.combobox_activate_index(w.annot_combo, i)
        w.grab()
        self.send_signal(w.Inputs.distances, None)

    def test_not_symmetric(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(w.Error.not_symmetric.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.not_symmetric.is_shown())

    def test_empty_matrix(self):
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[]]))
        self.assertTrue(w.Error.empty_matrix.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.empty_matrix.is_shown())


if __name__ == "__main__":
    unittest.main()

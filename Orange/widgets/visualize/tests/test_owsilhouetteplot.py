# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random

import numpy as np

from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.visualize.owsilhouetteplot import OWSilhouettePlot
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWSilhouettePlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWSilhouettePlot,
                                         stored_settings={"auto_commit": True})

    def test_outputs_add_scores(self):
        # check output when appending scores
        self.send_signal("Data", self.data)
        self.widget.controlledAttributes["add_scores"][0].control.setChecked(1)
        selected_indices = self._select_data()
        name = "Silhouette ({})".format(self.data.domain.class_var.name)
        selected = self.get_output("Selected Data")
        annotated = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(name, selected.domain.metas[0].name)
        self.assertEqual(name, annotated.domain.metas[0].name)
        np.testing.assert_array_equal(selected.X, self.data.X[selected_indices])

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget._silplot.setSelection(points)
        return sorted(points)

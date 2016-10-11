# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np
from Orange.data import Table
from Orange.misc.flagged_data import FLAGGED_SIGNAL_NAME, FLAGGED_FEATURE_NAME
from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owhierarchicalclustering import \
    OWHierarchicalClustering
from Orange.widgets.tests.base import WidgetTest


class TestOWHierarchicalClustering(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWHierarchicalClustering)
        self.iris = Table("iris")
        self.iris_distances = Euclidean(self.iris)

    def test_outputs(self):
        self.send_signal("Distances", self.iris_distances)

        # check selected data output
        self.assertIsNone(self.get_output("Selected Data"))

        # check flagged data output
        flagged = self.get_output(FLAGGED_SIGNAL_NAME)
        self.assertEqual(0, np.sum([i[FLAGGED_FEATURE_NAME] for i in flagged]))

        # select a cluster
        items = self.widget.dendrogram._items
        cluster = items[next(iter(items))]
        self.widget.dendrogram.set_selected_items([cluster])

        # check selected data output
        selected = self.get_output("Selected Data")
        self.assertGreater(len(selected), 0)

        # check flagged data output
        flagged = self.get_output(FLAGGED_SIGNAL_NAME)
        self.assertEqual(len(selected),
                         np.sum([i[FLAGGED_FEATURE_NAME] for i in flagged]))

        # check output when data is removed
        self.send_signal("Distances", None)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(FLAGGED_SIGNAL_NAME))

    def test_selection_box_output(self):
        """Check output if Selection method changes"""
        self.send_signal("Distances", self.iris_distances)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNotNone(self.get_output(FLAGGED_SIGNAL_NAME))

        # change selection to 'Height ratio'
        self.widget.selection_box.buttons[1].click()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(FLAGGED_SIGNAL_NAME))

        # change selection to 'Top N'
        self.widget.selection_box.buttons[2].click()
        self.assertIsNotNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(FLAGGED_SIGNAL_NAME))

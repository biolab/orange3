# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
import numpy as np
from PyQt4.QtCore import QEvent
from Orange.data import Table
from Orange.misc.flagged_data import FLAGGED_SIGNAL_NAME, FLAGGED_FEATURE_NAME
from Orange.distance import Euclidean
from Orange.widgets.unsupervised.owmds import OWMDS
from Orange.widgets.tests.base import WidgetTest


class TestOWMDS(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWMDS)
        self.iris = Table("iris")
        self.iris_distances = Euclidean(self.iris)

    def test_outputs(self):
        self.send_signal("Distances", self.iris_distances)
        self.widget.customEvent(QEvent(QEvent.User))
        self.widget.commit()

        # check selected data output
        self.assertIsNone(self.get_output("Selected Data"))

        # check flagged data output
        flagged = self.get_output(FLAGGED_SIGNAL_NAME)
        self.assertEqual(0, np.sum([i[FLAGGED_FEATURE_NAME] for i in flagged]))

        # select data points
        points = random.sample(range(0, len(self.iris)), 20)
        self.widget.select_indices(points)
        self.widget.commit()

        # check selected data output
        selected = self.get_output("Selected Data")
        self.assertEqual(len(selected), len(points))

        # check flagged data output
        flagged = self.get_output(FLAGGED_SIGNAL_NAME)
        self.assertEqual(len(selected),
                         np.sum([i[FLAGGED_FEATURE_NAME] for i in flagged]))

        # compare selected and flagged data domains
        selected_vars = selected.domain.variables + selected.domain.metas
        flagged_vars = flagged.domain.variables + flagged.domain.metas
        self.assertTrue(all((var in flagged_vars for var in selected_vars)))

        # check output when data is removed
        self.send_signal("Distances", None)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(FLAGGED_SIGNAL_NAME))

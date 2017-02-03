import copy

import numpy as np

import Orange.data
import Orange.evaluation
import Orange.classification

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot


class TestOWCalibrationPlot(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lenses = data = Orange.data.Table("lenses")
        cls.res = Orange.evaluation.TestOnTestData(
            train_data=data[::2], test_data=data[1::2],
            learners=[Orange.classification.MajorityLearner(),
                      Orange.classification.KNNLearner()],
            store_data=True,
        )

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWCalibrationPlot)  # type: OWCalibrationPlot

    def test_basic(self):
        self.send_signal("Evaluation Results", self.res)
        self.widget.controls.display_rug.click()

    def test_empty(self):
        res = copy.copy(self.res)
        res.row_indices = res.row_indices[:0]
        res.actual = res.actual[:0]
        res.predicted = res.predicted[:, 0]
        res.probabilities = res.probabilities[:, :0, :]
        self.send_signal("Evaluation Results", res)

    def test_nan_input(self):
        res = copy.copy(self.res)
        res.actual = res.actual.copy()
        res.probabilities = res.probabilities.copy()

        res.actual[0] = np.nan
        res.probabilities[:, [0, 3], :] = np.nan
        self.send_signal("Evaluation Results", res)
        self.assertTrue(self.widget.Error.invalid_results.is_shown())
        self.send_signal("Evaluation Results", None)
        self.assertFalse(self.widget.Error.invalid_results.is_shown())

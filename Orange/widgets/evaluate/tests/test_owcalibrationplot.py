import copy
import warnings

from sklearn.exceptions import ConvergenceWarning

from Orange.data import Table
import Orange.evaluation
import Orange.classification

from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.evaluate.owcalibrationplot import OWCalibrationPlot
from Orange.widgets.tests.base import WidgetTest
from Orange.tests import test_filename


class TestOWCalibrationPlot(WidgetTest, EvaluateTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lenses = data = Table(test_filename("datasets/lenses.tab"))
        cls.res = Orange.evaluation.TestOnTestData(
            train_data=data[::2], test_data=data[1::2],
            learners=[Orange.classification.MajorityLearner(),
                      Orange.classification.KNNLearner()],
            store_data=True,
        )

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWCalibrationPlot)  # type: OWCalibrationPlot
        warnings.filterwarnings("ignore", ".*", ConvergenceWarning)

    def test_basic(self):
        self.send_signal(self.widget.Inputs.evaluation_results, self.res)
        self.widget.controls.display_rug.click()

    def test_empty(self):
        res = copy.copy(self.res)
        res.row_indices = res.row_indices[:0]
        res.actual = res.actual[:0]
        res.predicted = res.predicted[:, 0]
        res.probabilities = res.probabilities[:, :0, :]
        self.send_signal(self.widget.Inputs.evaluation_results, res)

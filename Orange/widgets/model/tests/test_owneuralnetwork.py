import warnings

from sklearn.exceptions import ConvergenceWarning

from Orange.widgets.model.owneuralnetwork import OWNNLearner
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWNeuralNetwork(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        warnings.filterwarnings("ignore", ".*", ConvergenceWarning)
        self.widget = self.create_widget(OWNNLearner,
                                         stored_settings={"auto_apply": False})
        self.init()

    def test_migrate_setting(self):
        settings = dict(alpha=2.9)
        OWNNLearner.migrate_settings(settings, None)
        self.assertEqual(OWNNLearner.alphas[settings["alpha_index"]], 3)

        settings = dict(alpha=103)
        OWNNLearner.migrate_settings(settings, None)
        self.assertEqual(OWNNLearner.alphas[settings["alpha_index"]], 100)

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
        OWNNLearner.migrate_settings(settings, 0)
        self.assertEqual(OWNNLearner.alphas[settings["alpha_index"]], 3)

        settings = dict(alpha=103)
        OWNNLearner.migrate_settings(settings, 0)
        self.assertEqual(OWNNLearner.alphas[settings["alpha_index"]], 100)

        settings = dict(alpha_index=0)
        OWNNLearner.migrate_settings(settings, version=1)
        self.assertEqual(OWNNLearner.alphas[settings["alpha_index"]], 0.0001)

    def test_no_layer_warning(self):
        self.assertFalse(self.widget.Warning.no_layers.is_shown())

        self.widget.hidden_layers_input = ""
        self.widget.apply_button.button.click()
        self.assertTrue(self.widget.Warning.no_layers.is_shown())

        self.widget.hidden_layers_input = "10,"
        self.widget.apply_button.button.click()
        self.assertFalse(self.widget.Warning.no_layers.is_shown())

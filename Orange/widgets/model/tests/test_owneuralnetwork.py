from Orange.widgets.model.owneuralnetwork import OWNNLearner
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWNeuralNetwork(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWNNLearner,
                                         stored_settings={"auto_apply": False})
        self.init()

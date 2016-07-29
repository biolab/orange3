# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owknn import OWKNNLearner
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       ParameterMapping)


class TestOWKNNLearner(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWKNNLearner,
                                         stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('metric', self.widget.metrics_combo,
                             self.widget.metrics),
            ParameterMapping('weights', self.widget.weights_combo,
                             self.widget.weights),
            ParameterMapping('n_neighbors', self.widget.n_neighbors_spin)]

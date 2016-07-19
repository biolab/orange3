# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owknnregression import OWKNNRegression
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       GuiToParam)


class TestOWKNNRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWKNNRegression,
                                         stored_settings={"auto_apply": False})
        self.init()

        def combo_set_value(i, x):
            x.activated.emit(i)
            x.setCurrentIndex(i)

        # FIXME: widget crashes when Mahalanobis metric is chosen.
        # When fixed, remove metrics slicing
        metrics = self.widget.metrics[:-1]
        weights = self.widget.weights
        nn_spin = self.widget.n_neighbors_spin
        nn_min_max = [nn_spin.minimum(), nn_spin.maximum()]
        self.gui_to_params = [
            GuiToParam('metric', self.widget.metrics_combo,
                       lambda x: x.currentText().lower(),
                       combo_set_value, metrics, list(range(len(metrics)))),
            GuiToParam('weights', self.widget.weights_combo,
                       lambda x: x.currentText().lower(),
                       combo_set_value, weights, list(range(len(weights)))),
            GuiToParam('n_neighbors', nn_spin, lambda x: x.value(),
                       lambda i, x: x.setValue(i), nn_min_max, nn_min_max)]

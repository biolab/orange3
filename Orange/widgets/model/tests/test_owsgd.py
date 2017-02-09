# pylint: disable=missing-docstring
from Orange.widgets.model.owsgd import OWSGD
from Orange.widgets.tests.base import (
    WidgetTest,
    WidgetLearnerTestMixin,
    ParameterMapping
)


class TestOWSGD(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWSGD, stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            # Loss params for classification
            ParameterMapping("loss", self.widget.cls_loss_function_combo,
                             list(zip(*self.widget.cls_losses))[1],
                             problem_type="classification"),
            ParameterMapping("epsilon", self.widget.cls_epsilon_spin,
                             problem_type="classification"),
            # Loss params for regression
            ParameterMapping("loss", self.widget.reg_loss_function_combo,
                             list(zip(*self.widget.reg_losses))[1],
                             problem_type="regression"),
            ParameterMapping("epsilon", self.widget.reg_epsilon_spin,
                             problem_type="regression"),
            # Shared params
            ParameterMapping("penalty", self.widget.penalty_combo,
                             list(zip(*self.widget.penalties))[1]),
            ParameterMapping.from_attribute(self.widget, "alpha"),
            ParameterMapping.from_attribute(self.widget, "l1_ratio"),
            ParameterMapping("learning_rate", self.widget.learning_rate_combo,
                             list(zip(*self.widget.learning_rates))[1]),
            ParameterMapping.from_attribute(self.widget, "eta0"),
            ParameterMapping.from_attribute(self.widget, "power_t"),
        ]

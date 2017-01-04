# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owsgdregression import OWSGD
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, \
    ParameterMapping


class TestOWSGDRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWSGD, stored_settings={"auto_apply": False})
        self.init()
        self.data = self.housing
        self.parameters = [
            ParameterMapping('loss', self.widget.reg_loss_function_combo,
                             list(zip(*self.widget.reg_losses))[1]),
            ParameterMapping.from_attribute(self.widget, 'reg_epsilon', 'epsilon'),
            ParameterMapping('penalty', self.widget.penalty_combo,
                             list(zip(*self.widget.penalties))[1]),
            ParameterMapping.from_attribute(self.widget, 'alpha'),
            ParameterMapping.from_attribute(self.widget, 'l1_ratio'),
            ParameterMapping('learning_rate', self.widget.learning_rate_combo,
                             list(zip(*self.widget.learning_rates))[1]),
            ParameterMapping.from_attribute(self.widget, 'eta0'),
            ParameterMapping.from_attribute(self.widget, 'power_t'),
        ]

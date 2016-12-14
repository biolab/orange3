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
            ParameterMapping('loss', self.widget.loss_function_combo,
                             list(zip(*self.widget.losses))[1]),
            ParameterMapping.from_attribute(self.widget, 'epsilon'),
            ParameterMapping('penalty', self.widget.penalty_combo,
                             list(zip(*self.widget.penalties))[1]),
            ParameterMapping.from_attribute(self.widget, 'alpha'),
            ParameterMapping.from_attribute(self.widget, 'l1_ratio'),
            ParameterMapping('learning_rate', self.widget.learning_rate_combo,
                             list(zip(*self.widget.learning_rates))[1]),
            ParameterMapping.from_attribute(self.widget, 'eta0'),
            ParameterMapping.from_attribute(self.widget, 'power_t'),
        ]

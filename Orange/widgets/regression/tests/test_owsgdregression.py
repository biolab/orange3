# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.regression.owsgdregression import OWSGD
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, \
    ParameterMapping


class TestOWSGDRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWSGD, stored_settings={"auto_apply": False})
        self.init()
        self.data = self.housing
        self.valid_datasets = (self.housing,)
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

    def test_output_coefficients(self):
        """Check if coefficients are on output after apply"""
        self.assertIsNone(self.get_output("Coefficients"))
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        coeffs = self.get_output("Coefficients")
        self.assertIsInstance(coeffs, Table)
        self.assertEqual(coeffs.X.shape,
                         (len(self.data.domain.attributes) + 1, 1))
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output("Coefficients"))

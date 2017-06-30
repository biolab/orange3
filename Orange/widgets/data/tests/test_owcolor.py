# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.owcolor import OWColor
from Orange.widgets.tests.base import WidgetTest


class TestOWColor(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWColor)

        self.iris = Table("iris")

    def test_reuse_old_settings(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        assert isinstance(self.widget, OWColor)
        self.widget.saveSettings()

        w = self.create_widget(OWColor, reset_default_settings=False)
        self.send_signal(self.widget.Inputs.data, self.iris, widget=w)

    def test_invalid_input_colors(self):
        a = ContinuousVariable("a")
        a.attributes["colors"] = "invalid"
        a.colors
        t = Table(Domain([a]))

        self.send_signal(self.widget.Inputs.data, t)

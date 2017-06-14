# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.data.owdiscretize import OWDiscretize
from Orange.widgets.tests.base import WidgetTest


class TestOWDiscretize(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDiscretize)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget
        widget.default_method = 3
        self.send_signal(self.widget.Inputs.data, Table(data.domain))
        widget.unconditional_commit()

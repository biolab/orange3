# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owmajority import OWMajority
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWMajority(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWMajority,
                                         stored_settings={"auto_apply": False})
        self.init()

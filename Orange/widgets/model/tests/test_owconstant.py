# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from Orange.widgets.model.owconstant import OWConstant
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWConstant(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWConstant, stored_settings={"auto_apply": False})
        self.init()

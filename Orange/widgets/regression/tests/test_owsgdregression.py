# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owsgdregression import OWSGDRegression
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWSGDRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWSGDRegression,
                                         stored_settings={"auto_apply": False})
        self.init()

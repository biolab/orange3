# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owrandomforestregression import \
    OWRandomForestRegression
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWRandomForestRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWRandomForestRegression,
                                         stored_settings={"auto_apply": False})
        self.init()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owsvmregression import OWSVMRegression
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWSVMRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWSVMRegression,
                                         stored_settings={"auto_apply": False})
        self.init()

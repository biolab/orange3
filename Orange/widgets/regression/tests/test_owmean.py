# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owmean import OWMean
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWMean(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWMean,
                                         stored_settings={"auto_apply": False})
        self.init()

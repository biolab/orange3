# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owrandomforest import OWRandomForest
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWRandomForest(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWRandomForest,
                                         stored_settings={"auto_apply": False})
        self.init()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.ownaivebayes import OWNaiveBayes
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWNaiveBayes(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWNaiveBayes,
                                         stored_settings={"auto_apply": False})
        self.init()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.regression.owregressiontree import OWRegressionTree
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWRegressionTree(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWRegressionTree,
                                         stored_settings={"auto_apply": False})
        self.init()

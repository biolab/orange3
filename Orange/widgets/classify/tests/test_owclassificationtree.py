# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.base import Model
from Orange.widgets.classify.owclassificationtree import OWClassificationTree
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWClassificationTree(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWClassificationTree,
                                         stored_settings={"auto_apply": False})
        self.init()

    def init(self):
        super().init()
        self.model_class = Model

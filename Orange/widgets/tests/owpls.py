from Orange.widgets.model.owpls import OWPLS
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWPLS(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWPLS,
                                         stored_settings={"auto_apply": False})
        self.init()
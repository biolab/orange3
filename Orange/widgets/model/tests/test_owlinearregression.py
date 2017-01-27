from Orange.widgets.model.owlinearregression import OWLinearRegression
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWLinearRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWLinearRegression,
                                         stored_settings={"auto_apply": False})
        self.init()

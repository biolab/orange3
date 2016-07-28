# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest

from Orange.data import Table
from Orange.statistics.util import stats
from Orange.classification import LogisticRegressionLearner
from Orange.widgets.classify.owlogisticregression import (create_coef_table,
                                                          OWLogisticRegression)
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       GuiToParam)


class LogisticRegressionTest(unittest.TestCase):
    def test_coef_table_single(self):
        data = Table("titanic")
        learn = LogisticRegressionLearner()
        classifier = learn(data)
        coef_table = create_coef_table(classifier)
        self.assertEqual(1, len(stats(coef_table.metas, None)))
        self.assertEqual(len(coef_table), len(classifier.domain.attributes) + 1)
        self.assertEqual(len(coef_table[0]), 1)

    def test_coef_table_multiple(self):
        data = Table("zoo")
        learn = LogisticRegressionLearner()
        classifier = learn(data)
        coef_table = create_coef_table(classifier)
        self.assertEqual(1, len(stats(coef_table.metas, None)))
        self.assertEqual(len(coef_table), len(classifier.domain.attributes) + 1)
        self.assertEqual(len(coef_table[0]),
                         len(classifier.domain.class_var.values))


class TestOWLogisticRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWLogisticRegression,
                                         stored_settings={"auto_apply": False})
        self.init()

        def combo_set_value(i, x):
            x.activated.emit(i)
            x.setCurrentIndex(i)

        pen_types = self.widget.penalty_types_short
        c_slider = self.widget.c_slider
        c_min_max = [c_slider.minimum(), c_slider.maximum()]
        c_min_max_values = [self.widget.C_s[0], self.widget.C_s[-1]]
        self.gui_to_params = [
            GuiToParam('penalty', self.widget.penalty_combo,
                       lambda x: pen_types[x.currentIndex()],
                       combo_set_value, pen_types, list(range(len(pen_types)))),
            GuiToParam('C', c_slider, lambda x: self.widget.C_s[x.value()],
                       lambda i, x: x.setValue(i), c_min_max_values, c_min_max)]

    def test_output_coefficients(self):
        """Check if coefficients are on output after apply"""
        self.assertIsNone(self.get_output("Coefficients"))
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        self.assertIsInstance(self.get_output("Coefficients"), Table)

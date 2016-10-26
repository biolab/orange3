# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest

from Orange.data import Table
from Orange.statistics.util import stats
from Orange.classification import LogisticRegressionLearner
from Orange.widgets.classify.owlogisticregression import (create_coef_table,
                                                          OWLogisticRegression)
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       ParameterMapping)


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
        c_slider = self.widget.c_slider

        def setter(val):
            index = self.widget.C_s.index(val)
            self.widget.C_s[c_slider.value()]
            c_slider.setValue(index)

        self.parameters = [
            ParameterMapping('penalty', self.widget.penalty_combo,
                             self.widget.penalty_types_short),
            ParameterMapping('C', c_slider,
                             values=[self.widget.C_s[0], self.widget.C_s[-1]],
                             getter=lambda: self.widget.C_s[c_slider.value()],
                             setter=setter)]

    def test_output_coefficients(self):
        """Check if coefficients are on output after apply"""
        self.assertIsNone(self.get_output("Coefficients"))
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        self.assertIsInstance(self.get_output("Coefficients"), Table)

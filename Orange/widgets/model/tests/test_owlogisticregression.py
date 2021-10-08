# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest

import numpy as np

from Orange.classification import LogisticRegressionLearner
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.statistics.util import stats
from Orange.widgets.model.owlogisticregression import (create_coef_table,
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
        self.assertIsNone(self.get_output(self.widget.Outputs.coefficients))
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        self.assertIsInstance(self.get_output(self.widget.Outputs.coefficients), Table)

    def test_domain_with_more_values_than_table(self):
        """
        When the data with a domain which has more values than
        a table was sent, the widget threw error Invalid number of variable columns.
        GH-2116
        """
        table = Table("iris")
        cases = [slice(80), slice(90, 140),
                 np.hstack((np.arange(30, dtype=int),
                            np.arange(120, 140, dtype=int)))]
        for case in cases:
            data = table[case, :]
            self.send_signal("Data", data)
            self.widget.apply_button.button.click()

    def test_coefficients_one_value(self):
        """
        In case we have only two values of a target we get coefficients of only value.
        Instead of writing "coef" or sth similar it is written a second value name.
        GH-2116
        """
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a"),
                 ContinuousVariable("b")],
                [DiscreteVariable("c", values=("yes", "no"))]
            ),
            list(zip(
                [1., 0.],
                [0., 1.],
                ["yes", "no"]))
        )
        self.send_signal("Data", table)
        self.widget.apply_button.button.click()
        coef = self.get_output(self.widget.Outputs.coefficients)
        self.assertEqual(coef.domain[0].name, "no")
        self.assertGreater(coef[2][0], 0.)

    def test_target_with_nan(self):
        """
        Rows with targets with nans are removed.
        GH-2392
        """
        table = Table("iris")
        with table.unlocked():
            table.Y[:5] = np.NaN
        self.send_signal("Data", table)
        coef1 = self.get_output("Coefficients")
        table = table[5:]
        self.send_signal("Data", table)
        coef2 = self.get_output("Coefficients")
        self.assertTrue(np.array_equal(coef1, coef2))

    def test_class_weights(self):
        table = Table("iris")
        self.send_signal("Data", table)
        self.assertFalse(self.widget.class_weight)
        self.widget.controls.class_weight.setChecked(True)
        self.assertTrue(self.widget.class_weight)
        self.widget.apply_button.button.click()
        self.assertEqual(self.widget.model.skl_model.class_weight, "balanced")
        self.assertTrue(self.widget.Warning.class_weights_used.is_shown())

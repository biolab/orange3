import unittest

import bottlechest as bn

from Orange.data import Table
from Orange.classification import LogisticRegressionLearner
from Orange.widgets.classify.owlogisticregression import create_coef_table, OWLogisticRegression
from Orange.widgets.tests.base import GuiTest


class LogisticRegressionTest(unittest.TestCase):
    def test_coef_table_single(self):
        data = Table("titanic")
        learn = LogisticRegressionLearner()
        classifier = learn(data)
        coef_table = create_coef_table(classifier)
        self.assertEqual(1, len(bn.stats(coef_table.metas, None)))
        self.assertEqual(len(coef_table), len(classifier.domain.attributes) + 1)
        self.assertEqual(len(coef_table[0]), 1)

    def test_coef_table_multiple(self):
        data = Table("zoo")
        learn = LogisticRegressionLearner()
        classifier = learn(data)
        coef_table = create_coef_table(classifier)
        self.assertEqual(1, len(bn.stats(coef_table.metas, None)))
        self.assertEqual(len(coef_table), len(classifier.domain.attributes) + 1)
        self.assertEqual(len(coef_table[0]),
                         len(classifier.domain.class_var.values))


class WidgetTests(GuiTest):
    def test_data_before_apply(self):
        widget = OWLogisticRegression()
        widget.set_data(Table("iris"))
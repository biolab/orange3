# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,protected-access
import unittest
from unittest.mock import Mock, patch

import numpy as np

from AnyQt.QtCore import QPoint, QPropertyAnimation

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.classification import (
    NaiveBayesLearner, LogisticRegressionLearner, MajorityLearner
)
from Orange.preprocess import Scale, Continuize
from Orange.tests import test_filename
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate, qbuttongroup_emit_clicked
from Orange.widgets.visualize.ownomogram import (
    OWNomogram, DiscreteFeatureItem, ContinuousFeatureItem, ProbabilitiesDotItem,
    MovableToolTip
)


class TestOWNomogram(WidgetTest):
    def setUp(self):
        super().setUp()
        self.data = Table("heart_disease")
        self.nb_cls = NaiveBayesLearner()(self.data)
        self.lr_cls = LogisticRegressionLearner()(self.data)
        self.titanic = Table("titanic")
        self.lenses = Table(test_filename("datasets/lenses.tab"))

        self.widget = self.create_widget(OWNomogram)  # type: OWNomogram

    def test_nomogram_nb(self):
        """Check probabilities for naive bayes classifier for various values
        of classes and radio buttons"""
        self._test_helper(self.nb_cls, [54, 46])

    def test_nomogram_lr(self):
        """Check probabilities for logistic regression classifier for various
        values of classes and radio buttons"""
        self.widget.display_index = 0  # show ALL features
        self._test_helper(self.lr_cls, [61, 39])


    def _test_helper(self, cls, values):
        self.send_signal(self.widget.Inputs.classifier, cls)

        # check for all class values
        for i in range(self.widget.class_combo.count()):
            self.widget.class_combo.setCurrentIndex(i)
            self.widget.class_combo.activated.emit(i)

            # check probabilities marker value
            self._test_helper_check_probability(values[i])

            # point scale
            self.widget.controls.scale.buttons[0].click()
            self._test_helper_check_probability(values[i])

            # best ranked
            self.widget.n_attributes = 5
            self.widget.controls.display_index.buttons[1].click()
            visible_items = [item for item in self.widget.scene.items() if
                             isinstance(item, (DiscreteFeatureItem,
                                               ContinuousFeatureItem)) and
                             item.isVisible()]
            self.assertGreaterEqual(5, len(visible_items))

            # 2D curve
            self.widget.n_attributes = 15
            self.widget.cont_feature_dim_combo.activated.emit(1)
            self.widget.cont_feature_dim_combo.setCurrentIndex(1)
            self._test_helper_check_probability(values[i])

            # initial state
            self.widget.controls.scale.buttons[1].click()
            self.widget.controls.display_index.buttons[0].click()
            self.widget.cont_feature_dim_combo.activated.emit(0)
            self.widget.cont_feature_dim_combo.setCurrentIndex(0)

    def _test_helper_check_probability(self, value):
        assert len([item for item in self.widget.scene.items() if
                       isinstance(item, ProbabilitiesDotItem)]) == 1
        prob_marker = [item for item in self.widget.scene.items() if
                       isinstance(item, ProbabilitiesDotItem)][0]
        self.assertIn("Probability: {}".format(value),
                      prob_marker.get_tooltip_text())

    def _check_values(self, attributes, data):
        for attr, item in zip(attributes, self.widget.feature_items.values()):
            assert attr.name == item.childItems()[0].toPlainText()
            value = data[0][attr.name].value
            value = "{}: 100%".format(value) if attr.is_discrete \
                else "Value: {}".format(value)
            self.assertIn(value, item.dot.get_tooltip_text())

    def _test_sort(self, names):
        for i in range(self.widget.sort_combo.count()):
            self.widget.sort_combo.activated.emit(i)
            self.widget.sort_combo.setCurrentIndex(i)
            ordered = [self.widget.nomogram_main.layout().itemAt(i).childItems()[0].toPlainText()
                       for i in range(self.widget.nomogram_main.layout().count())]
            self.assertListEqual(names[i], ordered)



if __name__ == "__main__":
    unittest.main()

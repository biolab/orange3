# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.classification import (
    NaiveBayesLearner, LogisticRegressionLearner, MajorityLearner
)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.ownomogram import (
    OWNomogram, DiscreteFeatureItem, ContinuousFeatureItem, ProbabilitiesDotItem
)


class TestOWNomogram(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = Table("heart_disease")
        cls.nb_cls = NaiveBayesLearner()(cls.data)
        cls.lr_cls = LogisticRegressionLearner()(cls.data)
        cls.titanic = Table("titanic")
        cls.lenses = Table("lenses")

    def setUp(self):
        self.widget = self.create_widget(OWNomogram)
        self.widget.repaint = True

    def test_input_nb_cls(self):
        """Check naive bayes classifier on input"""
        self.send_signal("Classifier", self.nb_cls)
        self.assertEqual(len([item for item in self.widget.scene.items() if
                              isinstance(item, DiscreteFeatureItem)]),
                         len([a for a in self.data.domain.attributes]))

    def test_input_lr_cls(self):
        """Check logistic regression classifier on input"""
        self.send_signal("Classifier", self.lr_cls)
        self.assertEqual(
            len([item for item in self.widget.scene.items() if
                 isinstance(item, DiscreteFeatureItem)]),
            len([a for a in self.data.domain.attributes if a.is_discrete]))
        self.assertEqual(
            len([item for item in self.widget.scene.items() if
                 isinstance(item, ContinuousFeatureItem)]),
            len([a for a in self.data.domain.attributes if a.is_continuous]))

    def test_input_invalid_cls(self):
        """Check any classifier on input"""
        majority_cls = MajorityLearner()(self.data)
        self.send_signal("Classifier", majority_cls)
        self.assertTrue(self.widget.Error.invalid_classifier.is_shown())
        self.send_signal("Classifier", None)
        self.assertFalse(self.widget.Error.invalid_classifier.is_shown())

    def test_input_instance(self):
        """ Check data instance on input"""
        self.send_signal("Data", self.data)
        self.assertIsNotNone(self.widget.instances)
        self.send_signal("Data", None)
        self.assertIsNone(self.widget.instances)

    def test_target_values(self):
        """Check Target class combo values"""
        self.send_signal("Classifier", self.nb_cls)
        for i, text in enumerate(self.data.domain.class_var.values):
            self.assertEqual(text, self.widget.class_combo.itemText(i))
        self.send_signal("Classifier", None)
        self.assertEqual(0, self.widget.class_combo.count())

    def test_nomogram_nb(self):
        """Check probabilities for naive bayes classifier for various values
        of classes and radio buttons"""
        self._test_helper(self.nb_cls, [54, 46])

    def test_nomogram_lr(self):
        """Check probabilities for logistic regression classifier for various
        values of classes and radio buttons"""
        self._test_helper(self.lr_cls, [9, 91])

    def test_nomogram_nb_multiclass(self):
        """Check probabilities for naive bayes classifier for various values
        of classes and radio buttons for multiclass data"""
        cls = NaiveBayesLearner()(self.lenses)
        self._test_helper(cls, [19, 53, 13])

    def test_nomogram_lr_multiclass(self):
        """Check probabilities for logistic regression classifier for various
        values of classes and radio buttons for multiclass data"""
        cls = LogisticRegressionLearner()(self.lenses)
        self._test_helper(cls, [9, 45, 52])

    def test_nomogram_with_instance_nb(self):
        """Check initialized marker values and feature sorting for naive bayes
        classifier and data on input"""
        cls = NaiveBayesLearner()(self.titanic)
        data = self.titanic[10:11]
        self.send_signal("Classifier", cls)
        self.send_signal("Data", data)
        self._check_values(data.domain.attributes, data)
        self._test_sort([["status", "age", "sex"],
                         ["age", "sex", "status"],
                         ["sex", "status", "age"],
                         ["sex", "status", "age"],
                         ["sex", "status", "age"]])

    def test_nomogram_with_instance_lr(self):
        """Check initialized marker values and feature sorting for logistic
        regression classifier and data on input"""
        cls = LogisticRegressionLearner()(self.titanic)
        data = self.titanic[10:11]
        self.send_signal("Classifier", cls)
        self.send_signal("Data", data)
        self._check_values(data.domain.attributes, data)
        self._test_sort([["status", "age", "sex"],
                         ["age", "sex", "status"],
                         ["sex", "status", "age"],
                         ["sex", "status", "age"],
                         ["sex", "status", "age"]])

    def test_constant_feature_disc(self):
        """Check nomogram for data with constant discrete feature"""
        domain = Domain([DiscreteVariable("d1", ("a", "c")),
                         DiscreteVariable("d2", ("b",))],
                        DiscreteVariable("cls", ("e", "d")))
        X = np.array([[0, 0], [1, 0], [0, 0], [1, 0]])
        data = Table(domain, X, np.array([0, 1, 1, 0]))
        cls = NaiveBayesLearner()(data)
        self._test_helper(cls, [50, 50])
        cls = LogisticRegressionLearner()(data)
        self._test_helper(cls, [50, 50])

    def test_constant_feature_cont(self):
        """Check nomogram for data with constant continuous feature"""
        domain = Domain([DiscreteVariable("d", ("a", "b")),
                         ContinuousVariable("c")],
                        DiscreteVariable("cls", ("c", "d")))
        X = np.array([[0, 0], [1, 0], [0, 0], [1, 0]])
        data = Table(domain, X, np.array([0, 1, 1, 0]))
        cls = NaiveBayesLearner()(data)
        self._test_helper(cls, [50, 50])
        cls = LogisticRegressionLearner()(data)
        self._test_helper(cls, [50, 50])

    def _test_helper(self, cls, values):
        self.send_signal("Classifier", cls)

        # check for all class values
        for i in range(self.widget.class_combo.count()):
            self.widget.class_combo.activated.emit(i)
            self.widget.class_combo.setCurrentIndex(i)

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
            self.widget.cont_feature_dim_combo.activated.emit(1)
            self.widget.cont_feature_dim_combo.setCurrentIndex(1)
            self._test_helper_check_probability(values[i])

            # initial state
            self.widget.controls.scale.buttons[1].click()
            self.widget.controls.display_index.buttons[0].click()
            self.widget.cont_feature_dim_combo.activated.emit(0)
            self.widget.cont_feature_dim_combo.setCurrentIndex(0)

    def _test_helper_check_probability(self, value):
        prob_marker = [item for item in self.widget.scene.items() if
                       isinstance(item, ProbabilitiesDotItem)][0]
        self.assertIn("Probability: {}".format(value),
                      prob_marker.get_tooltip_text())

    def _check_values(self, attributes, data):
        for attr, item in zip(attributes, self.widget.feature_items):
            value = data[0][attr.name].value
            value = "{}: 100%".format(value) if attr.is_discrete \
                else "Value: {}".format(value)
            self.assertIn(value, item.dot.get_tooltip_text())

    def _test_sort(self, names):
        for i in range(self.widget.sort_combo.count()):
            self.widget.sort_combo.activated.emit(i)
            self.widget.sort_combo.setCurrentIndex(i)
            ordered = [self.widget.nomogram_main.layout.itemAt(i).name
                       for i in range(self.widget.nomogram_main.layout.count())]
            self.assertListEqual(names[i], ordered)

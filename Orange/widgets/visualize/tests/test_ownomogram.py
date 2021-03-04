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
from Orange.widgets.visualize.ownomogram import (
    OWNomogram, DiscreteFeatureItem, ContinuousFeatureItem, ProbabilitiesDotItem,
    MovableToolTip
)


class TestOWNomogram(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = Table("heart_disease")
        cls.nb_cls = NaiveBayesLearner()(cls.data)
        cls.lr_cls = LogisticRegressionLearner()(cls.data)
        cls.titanic = Table("titanic")
        cls.lenses = Table(test_filename("datasets/lenses.tab"))

    def setUp(self):
        self.widget = self.create_widget(OWNomogram)  # type: OWNomogram

    def test_input_nb_cls(self):
        """Check naive bayes classifier on input"""
        self.send_signal(self.widget.Inputs.classifier, self.nb_cls)
        self.assertEqual(len([item for item in self.widget.scene.items() if
                              isinstance(item, DiscreteFeatureItem)]),
                         min(self.widget.n_attributes,
                             len(self.data.domain.attributes)))

    def test_input_lr_cls(self):
        """Check logistic regression classifier on input"""
        self.widget.display_index = 0  # display ALL features
        self.send_signal(self.widget.Inputs.classifier, self.lr_cls)
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
        self.send_signal(self.widget.Inputs.classifier, majority_cls)
        self.assertTrue(self.widget.Error.invalid_classifier.is_shown())
        self.send_signal(self.widget.Inputs.classifier, None)
        self.assertFalse(self.widget.Error.invalid_classifier.is_shown())

    def test_input_instance(self):
        """ Check data instance on input"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNotNone(self.widget.instances)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.instances)

    def test_target_values(self):
        """Check Target class combo values"""
        self.send_signal(self.widget.Inputs.classifier, self.nb_cls)
        for i, text in enumerate(self.data.domain.class_var.values):
            self.assertEqual(text, self.widget.class_combo.itemText(i))
        self.send_signal(self.widget.Inputs.classifier, None)
        self.assertEqual(0, self.widget.class_combo.count())

    def test_nomogram_nb(self):
        """Check probabilities for naive bayes classifier for various values
        of classes and radio buttons"""
        self._test_helper(self.nb_cls, [54, 46])

    def test_nomogram_lr(self):
        """Check probabilities for logistic regression classifier for various
        values of classes and radio buttons"""
        self.widget.display_index = 0  # show ALL features
        self._test_helper(self.lr_cls, [58, 42])

    def test_nomogram_nb_multiclass(self):
        """Check probabilities for naive bayes classifier for various values
        of classes and radio buttons for multiclass data"""
        cls = NaiveBayesLearner()(self.lenses)
        self._test_helper(cls, [19, 53, 13])

    def test_nomogram_lr_multiclass(self):
        """Check probabilities for logistic regression classifier for various
        values of classes and radio buttons for multiclass data"""
        cls = LogisticRegressionLearner(
            multi_class="ovr", solver="liblinear"
        )(self.lenses)
        self._test_helper(cls, [9, 45, 52])

    def test_nomogram_with_instance_nb(self):
        """Check initialized marker values and feature sorting for naive bayes
        classifier and data on input"""
        cls = NaiveBayesLearner()(self.titanic)
        data = self.titanic[10:11]
        self.send_signal(self.widget.Inputs.classifier, cls)
        self.send_signal(self.widget.Inputs.data, data)
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
        self.send_signal(self.widget.Inputs.classifier, cls)
        self.send_signal(self.widget.Inputs.data, data)
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
        self.send_signal(self.widget.Inputs.classifier, cls)

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

    def test_tooltip(self):
        # had problems on PyQt4
        m = MovableToolTip()
        m.show(QPoint(0, 0), "Some text.")

    def test_output(self):
        cls = LogisticRegressionLearner()(self.titanic)
        data = self.titanic[10:11]
        status, age, sex = data.domain.attributes
        self.send_signal(self.widget.Inputs.classifier, cls)
        self.widget.sort_combo.setCurrentIndex(1)
        self.widget.sort_combo.activated.emit(1)

        # Output more attributer than there are -> output all
        self.widget.n_attributes = 5
        self.widget.n_spin.valueChanged.emit(5)
        attrs = self.get_output(self.widget.Outputs.features)
        self.assertEqual(attrs, [age, sex, status])

        # Output the first two
        self.widget.n_attributes = 2
        self.widget.n_spin.valueChanged.emit(2)
        attrs = self.get_output(self.widget.Outputs.features)
        self.assertEqual(attrs, [age, sex])

        # Set to output all
        self.widget.display_index = 0
        self.widget.controls.display_index.group.buttonClicked[int].emit(0)
        attrs = self.get_output(self.widget.Outputs.features)
        self.assertEqual(attrs, [age, sex, status])

        # Remove classifier -> output None
        self.send_signal(self.widget.Inputs.classifier, None)
        attrs = self.get_output(self.widget.Outputs.features)
        self.assertIsNone(attrs)

    def test_reset_settings(self):
        self.widget.n_attributes = 5
        self.widget.n_spin.valueChanged.emit(5)
        self.widget.reset_settings()
        self.assertEqual(10, self.widget.n_attributes)

    @patch("Orange.widgets.visualize.ownomogram.QGraphicsTextItem")
    def test_adjust_scale(self, mocked_item: Mock):
        def mocked_width():
            nonlocal ind
            ind += 1
            most_right = {4: 30, 9: 59, 14: 59}
            return [2, 30, 59, 2, most_right.get(ind)][ind % 5]

        ind = -1
        mocked_item().boundingRect().width.side_effect = mocked_width
        attrs = [DiscreteVariable("var1", values=("foo1", "foo2")),
                 DiscreteVariable("var2", values=("foo3", "foo4"))]
        points = [np.array([0, 1.8]), np.array([1.5, 2.0])]
        diff = np.max(points) - np.min(points)
        # foo3 eventually overcomes foo2, while the scale is getting smaller
        #
        #  0                        1              1.5         1.8     2
        # foo1                                          _______foo2_______
        #                               __________foo3__________      foo4
        self.widget._adjust_scale(attrs, points, 100, diff, [0, 1], [], 0)
        # most left text at 1. iteration
        self.assertEqual(mocked_item.call_args_list[5][0][0], "foo2")
        # most left text at 2. iteration
        self.assertEqual(mocked_item.call_args_list[10][0][0], "foo3")
        # most left text at 3. iteration is the same -> stop
        self.assertEqual(mocked_item.call_args_list[15][0][0], "foo3")

    def test_dots_stop_flashing(self):
        self.widget.set_data(self.data)
        self.widget.set_classifier(self.nb_cls)
        animator = self.widget.dot_animator
        dot = animator._GraphicsColorAnimator__items[0]
        dot._mousePressFunc()
        anim = animator._GraphicsColorAnimator__animation
        self.assertNotEqual(anim.state(), QPropertyAnimation.Running)

    def test_reconstruct_domain(self):
        data = Table("heart_disease")
        cls = LogisticRegressionLearner()(data)
        domain = OWNomogram.reconstruct_domain(cls.original_domain, cls.domain)
        transformed_data = cls.original_data.transform(domain)
        self.assertEqual(transformed_data.X.shape, data.X.shape)
        self.assertFalse(np.isnan(transformed_data.X[0]).any())

        scaled_data = Scale()(data)
        cls = LogisticRegressionLearner()(scaled_data)
        domain = OWNomogram.reconstruct_domain(cls.original_domain, cls.domain)
        transformed_data = cls.original_data.transform(domain)
        self.assertEqual(transformed_data.X.shape, scaled_data.X.shape)
        self.assertFalse(np.isnan(transformed_data.X[0]).any())

        disc_data = Continuize()(data)
        cls = LogisticRegressionLearner()(disc_data)
        domain = OWNomogram.reconstruct_domain(cls.original_domain, cls.domain)
        transformed_data = cls.original_data.transform(domain)
        self.assertEqual(transformed_data.X.shape, disc_data.X.shape)
        self.assertFalse(np.isnan(transformed_data.X[0]).any())


if __name__ == "__main__":
    unittest.main()

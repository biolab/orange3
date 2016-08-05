# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.regression import TreeRegressionLearner, KNNRegressionLearner
from Orange.widgets.regression.owadaboostregression import OWAdaBoostRegression
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       ParameterMapping)


class TestOWAdaBoostRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWAdaBoostRegression,
                                         stored_settings={"auto_apply": False})
        self.init()
        losses = [loss.lower() for loss in self.widget.losses]
        self.parameters = [
            ParameterMapping('loss', self.widget.loss_combo, losses),
            ParameterMapping('learning_rate', self.widget.learning_rate_spin),
            ParameterMapping('n_estimators', self.widget.n_estimators_spin)]

    def test_input_learner(self):
        """Check if base learner properly changes with learner on the input"""
        max_depth = 2
        default_base_est = self.widget.base_estimator
        self.assertIsInstance(default_base_est, TreeRegressionLearner)
        self.assertIsNone(default_base_est.params.get("max_depth"))
        self.send_signal("Learner", TreeRegressionLearner(max_depth=max_depth))
        self.assertEqual(self.widget.base_estimator.params.get("max_depth"),
                         max_depth)
        self.widget.apply_button.button.click()
        output_base_est = self.get_output("Learner").params.get("base_estimator")
        self.assertEqual(output_base_est.max_depth, max_depth)

    def test_input_learner_disconnect(self):
        """Check base learner after disconnecting learner on the input"""
        self.send_signal("Learner", KNNRegressionLearner())
        self.assertIsInstance(self.widget.base_estimator, KNNRegressionLearner)
        self.send_signal("Learner", None)
        self.assertEqual(self.widget.base_estimator,
                         self.widget.DEFAULT_BASE_ESTIMATOR)

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.classification import RandomForestLearner
from Orange.modelling import SklTreeLearner, KNNLearner
from Orange.widgets.model.owadaboost import OWAdaBoost
from Orange.widgets.tests.base import (
    WidgetTest, WidgetLearnerTestMixin, ParameterMapping
)


class TestOWAdaBoostClassification(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWAdaBoost, stored_settings={"auto_apply": False})
        self.init()
        losses = self.widget.losses
        self.parameters = [
            ParameterMapping('algorithm', self.widget.algorithm_combo, losses),
            ParameterMapping('learning_rate', self.widget.learning_rate_spin),
            ParameterMapping('n_estimators', self.widget.n_estimators_spin)]

    def test_input_learner(self):
        """Check if base learner properly changes with learner on the input"""
        default_base_est = self.widget.base_estimator
        new_learner = SklTreeLearner()
        # Check that the default learner is indeed a SklTreeLearner
        self.assertIsInstance(default_base_est, SklTreeLearner)
        # Sending a new valid learner should change the learner
        self.send_signal('Learner', new_learner)
        self.assertEqual(self.widget.base_estimator, new_learner)
        # Removing the new learner should change back to the default one
        self.send_signal('Learner', None)
        self.assertEqual(self.widget.base_estimator, default_base_est)

    def test_input_learner_disconnect(self):
        """Check base learner after disconnecting learner on the input"""
        self.send_signal("Learner", KNNLearner())
        self.assertIsInstance(self.widget.base_estimator, KNNLearner)
        self.send_signal("Learner", None)
        self.assertEqual(
            self.widget.base_estimator, self.widget.DEFAULT_BASE_ESTIMATOR)

    def test_input_learner_that_does_not_support_sample_weights(self):
        self.send_signal("Learner", KNNLearner())
        self.assertNotIsInstance(self.widget.base_estimator, KNNLearner)
        self.assertIsNone(self.widget.base_estimator)
        self.assertTrue(self.widget.Error.no_weight_support.is_shown())

    def test_error_message_cleared_when_valid_learner_on_input(self):
        # Disconnecting an invalid learner should use the default one and hide
        # the error
        self.send_signal("Learner", KNNLearner())
        self.send_signal('Learner', None)
        self.assertFalse(
            self.widget.Error.no_weight_support.is_shown(),
            'Error message was not hidden on input disconnect')
        # Connecting a valid learner should also reset the error message
        self.send_signal("Learner", KNNLearner())
        self.send_signal('Learner', RandomForestLearner())
        self.assertFalse(
            self.widget.Error.no_weight_support.is_shown(),
            'Error message was not hidden when a valid learner appeared on '
            'input')

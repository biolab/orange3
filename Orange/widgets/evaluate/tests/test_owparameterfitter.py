# pylint: disable=missing-docstring,protected-access
import unittest

from Orange.classification import NaiveBayesLearner
from Orange.data import Table
from Orange.regression import PLSRegressionLearner
from Orange.widgets.evaluate.owparameterfitter import OWParameterFitter
from Orange.widgets.tests.base import WidgetTest


class TestOWParameterFitter(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._heart = Table("heart_disease")
        cls._housing = Table("housing")
        cls._naive_bayes = NaiveBayesLearner()
        cls._pls = PLSRegressionLearner()

    def setUp(self):
        self.widget = self.create_widget(OWParameterFitter)

    def test_input(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        self.send_signal(self.widget.Inputs.data, self._heart)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

    def test_input_no_params(self):
        self.send_signal(self.widget.Inputs.data, self._heart)
        self.send_signal(self.widget.Inputs.learner, self._naive_bayes)
        self.wait_until_finished()
        self.assertTrue(self.widget.Warning.no_parameters.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())


if __name__ == "__main__":
    unittest.main()

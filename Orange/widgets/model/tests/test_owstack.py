# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.model.owstack import OWStackedLearner
from Orange.classification import LogisticRegressionLearner
from Orange.widgets.tests.base import WidgetTest


class TestOWStackedLearner(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWStackedLearner,
                                         stored_settings={"auto_apply": False})
        self.data = Table('iris')

    def test_input_data(self):
        """Check widget's data with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal("Data", self.data)
        self.assertEqual(self.widget.data, self.data)
        self.wait_until_stop_blocking()

    def test_output_learner(self):
        """Check if learner is on output after apply"""
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.send_signal("Learners", LogisticRegressionLearner(), 0)
        self.widget.apply_button.button.click()
        initial = self.get_output("Learner")
        self.assertIsNotNone(initial, "Does not initialize the learner output")
        self.widget.apply_button.button.click()
        newlearner = self.get_output("Learner")
        self.assertIsNot(initial, newlearner,
                         "Does not send a new learner instance on `Apply`.")
        self.assertIsNotNone(newlearner)
        self.assertIsInstance(newlearner, self.widget.LEARNER)

    def test_output_model(self):
        """Check if model is on output after sending data and apply"""
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.send_signal("Learners", LogisticRegressionLearner(), 0)
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.send_signal('Data', self.data)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, self.widget.LEARNER.__returns__)

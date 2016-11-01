# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.classification import LogisticRegressionLearner
from Orange.widgets.data.owpythonscript import OWPythonScript
from Orange.widgets.tests.base import WidgetTest


class TestOWPythonScript(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPythonScript)
        self.iris = Table("iris")
        self.learner = LogisticRegressionLearner()
        self.model = self.learner(self.iris)

    def test_inputs(self):
        """Check widget's inputs"""
        for _input, data in (("in_data", self.iris),
                             ("in_learner", self.learner),
                             ("in_classifier", self.model),
                             ("in_object", "object")):
            self.assertEqual(getattr(self.widget, _input), None)
            self.send_signal(_input, data)
            self.assertEqual(getattr(self.widget, _input), data)
            self.send_signal(_input, None)
            self.assertEqual(getattr(self.widget, _input), None)

    def test_outputs(self):
        """Check widget's outputs"""
        for _input, _output, data in (
                ("in_data", "out_data", self.iris),
                ("in_learner", "out_learner", self.learner),
                ("in_classifier", "out_data", self.model),
                ("in_object", "out_object", "object")):
            self.widget.text.setPlainText("{} = {}".format(_output, _input))
            self.send_signal(_input, data)
            self.assertEqual(self.get_output(_output), data)
            self.send_signal(_input, None)
            self.widget.text.setPlainText("print({})".format(_output))
            self.widget.execute_button.button.click()
            self.assertEqual(self.get_output(_output), None)

    def test_local_variable(self):
        """Check if variable remains in locals after removed from script"""
        self.widget.text.setPlainText("temp = 42\nprint(temp)")
        self.widget.execute_button.button.click()
        self.assertIn("42", self.widget.console.toPlainText())
        self.widget.text.setPlainText("print(temp)")
        self.widget.execute_button.button.click()
        self.assertNotIn("NameError: name 'temp' is not defined",
                         self.widget.console.toPlainText())

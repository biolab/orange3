# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.classification import LogisticRegressionLearner
from Orange.widgets.data.owpythonscript import OWPythonScript
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget


class TestOWPythonScript(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPythonScript)
        self.iris = Table("iris")
        self.learner = LogisticRegressionLearner()
        self.model = self.learner(self.iris)

    def test_inputs(self):
        """Check widget's inputs"""
        for input, data in (("Data", self.iris),
                            ("Learner", self.learner),
                            ("Classifier", self.model),
                            ("Object", "object")):
            self.assertEqual(getattr(self.widget, input.lower()), {})
            self.send_signal(input, data, (1,))
            self.assertEqual(getattr(self.widget, input.lower()), {1: data})
            self.send_signal(input, None, (1,))
            self.assertEqual(getattr(self.widget, input.lower()), {})

    def test_outputs(self):
        """Check widget's outputs"""
        for signal, data in (
                ("Data", self.iris),
                ("Learner", self.learner),
                ("Classifier", self.model)):
            lsignal = signal.lower()
            self.widget.text.setPlainText("out_{0} = in_{0}".format(lsignal))
            self.send_signal(signal, data, (1,))
            self.assertIs(self.get_output(signal), data)
            self.send_signal(signal, None, (1,))
            self.widget.text.setPlainText("print(in_{})".format(lsignal))
            self.widget.execute_button.click()
            self.assertIsNone(self.get_output(signal))

    def test_local_variable(self):
        """Check if variable remains in locals after removed from script"""
        self.widget.autobox.setCheckState(False)
        self.widget.text.setPlainText("temp = 42\nprint(temp)")
        self.widget.execute_button.click()
        self.assertIn("42", self.widget.console.toPlainText())
        self.widget.text.setPlainText("print(temp)")
        self.widget.execute_button.click()
        self.assertNotIn("NameError: name 'temp' is not defined",
                         self.widget.console.toPlainText())

    def test_wrong_outputs(self):
        """
        Error is shown when output variables are filled with wrong variable
        types and also output variable is set to None. (GH-2308)
        """
        self.widget.autobox.setCheckState(False)
        self.assertEqual(len(self.widget.Error.active), 0)
        for signal, data in (
                ("Data", self.iris),
                ("Learner", self.learner),
                ("Classifier", self.model)):
            lsignal = signal.lower()
            self.send_signal(signal, data, (1, ))
            self.widget.text.setPlainText("out_{} = 42".format(lsignal))
            self.widget.execute_button.click()
            self.assertEqual(self.get_output(signal), None)
            self.assertTrue(hasattr(self.widget.Error, lsignal))
            self.assertTrue(getattr(self.widget.Error, lsignal).is_shown())

            self.widget.text.setPlainText("out_{0} = in_{0}".format(lsignal))
            self.widget.execute_button.click()
            self.assertIs(self.get_output(signal), data)
            self.assertFalse(getattr(self.widget.Error, lsignal).is_shown())

    def test_owns_errors(self):
        self.assertIsNot(self.widget.Error, OWWidget.Error)

    def test_multiple_signals(self):
        self.widget.autobox.setCheckState(False)
        click = self.widget.execute_button.click
        console_locals = self.widget.console.locals

        titanic = Table("titanic")

        click()
        self.assertIsNone(console_locals["in_data"])
        self.assertEqual(console_locals["in_datas"], [])

        self.send_signal("Data", self.iris, (1, ))
        click()
        self.assertIs(console_locals["in_data"], self.iris)
        datas = console_locals["in_datas"]
        self.assertEqual(len(datas), 1)
        self.assertIs(datas[0], self.iris)

        self.send_signal("Data", titanic, (2, ))
        click()
        self.assertIsNone(console_locals["in_data"])
        self.assertEqual({id(obj) for obj in console_locals["in_datas"]},
                         {id(self.iris), id(titanic)})

        self.send_signal("Data", None, (2, ))
        click()
        self.assertIs(console_locals["in_data"], self.iris)
        datas = console_locals["in_datas"]
        self.assertEqual(len(datas), 1)
        self.assertIs(datas[0], self.iris)

        self.send_signal("Data", None, (1, ))
        click()
        self.assertIsNone(console_locals["in_data"])
        self.assertEqual(console_locals["in_datas"], [])

# pylint: disable=missing-docstring, protected-access
import ast
import pickle
import unittest

import numpy as np
from AnyQt.QtWidgets import QCheckBox, QLineEdit, QPushButton, QDoubleSpinBox

from Orange.data import Table
from Orange.regression import CurveFitLearner
from Orange.regression.curvefit import CurveFitModel
from Orange.widgets.model.owcurvefit import OWCurveFit, ParametersWidget, \
    Parameter, _create_lambda, FUNCTIONS
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin
from Orange.widgets.tests.utils import simulate


class TestFunctions(unittest.TestCase):
    def test_functions(self):
        a = np.full(5, 2)
        for f in FUNCTIONS:
            func = getattr(np, f)
            if isinstance(func, float):
                pass
            elif f in ["arctan2", "copysign", "fmod", "gcd", "hypot",
                       "isclose", "ldexp", "power", "remainder"]:
                self.assertIsInstance(func(a, 2), np.ndarray)
            else:
                self.assertIsInstance(func(a), np.ndarray)


class TestCreateLambda(unittest.TestCase):
    def test_create_lambda_simple(self):
        func, params = _create_lambda(ast.parse("a + b", mode="eval"), {}, [])
        self.assertTrue(callable(func))
        self.assertEqual(set(params), set(["a", "b"]))
        self.assertEqual(func(np.array([[1, 11], [2, 22]]), 1, 2), 3)

    def test_create_lambda_var(self):
        func, params = _create_lambda(
            ast.parse("var + a + b", mode="eval"), {"var": 0}, []
        )
        self.assertTrue(callable(func))
        self.assertEqual(set(params), set(["a", "b"]))
        np.testing.assert_array_equal(
            func(np.array([[1, 11], [2, 22]]), 1, 2),
            np.array([4, 5])
        )

    def test_create_lambda_fun(self):
        func, params = _create_lambda(
            ast.parse("pow(a, 2)", mode="eval"), {}, ["pow"]
        )
        self.assertTrue(callable(func))
        self.assertEqual(params, ["a"])
        np.testing.assert_array_equal(
            func(np.array([[1, 11], [2, 22]]), 3),
            np.array([9, 9])
        )

    def test_create_lambda_var_fun(self):
        func, params = _create_lambda(
            ast.parse("var1 + pow(a, 2) + pow(a, 2)", mode="eval"),
            {"var1": 0, "var2": 1}, ["pow"]
        )
        self.assertTrue(callable(func))
        self.assertEqual(params, ["a"])
        np.testing.assert_array_equal(
            func(np.array([[1, 11], [2, 22]]), 3),
            np.array([19, 20])
        )

    def test_create_lambda_x(self):
        func, params = _create_lambda(
            ast.parse("var + x", mode="eval"), {"var": 0}, []
        )
        self.assertTrue(callable(func))
        self.assertEqual(params, ["x"])
        np.testing.assert_array_equal(
            func(np.array([[1, 11], [2, 22]]), 3), np.array([4, 5])
        )

    def test_create_lambda(self):
        func, params = _create_lambda(
            ast.parse("a * var1 + b * exp(var2 * power(pi, 0))", mode="eval"),
            {"var1": 0, "var2": 1}, list(FUNCTIONS)
        )
        self.assertTrue(callable(func))
        self.assertEqual(set(params), set(["a", "b"]))
        np.testing.assert_allclose(
            func(np.array([[1, 2], [3, 4]]), 3, 2),
            np.array([17.778112, 118.1963])
        )


class TestParametersWidget(WidgetTest):
    def setUp(self):
        self._widget = ParametersWidget(None)

    def test_init(self):
        layout = self._widget._ParametersWidget__layout
        self.assertEqual(layout.rowCount(), 1)

    def test_add_row(self):
        self._widget._add_row()

        controls = self._widget._ParametersWidget__controls[0]
        self.assertIsInstance(controls[0], QPushButton)
        self.assertIsInstance(controls[1], QLineEdit)
        self.assertIsInstance(controls[2], QDoubleSpinBox)
        self.assertIsInstance(controls[3], QCheckBox)
        self.assertIsInstance(controls[4], QDoubleSpinBox)
        self.assertIsInstance(controls[5], QCheckBox)
        self.assertIsInstance(controls[6], QDoubleSpinBox)

        self.assertEqual(controls[1].text(), "p1")
        self.assertEqual(controls[2].value(), 1)
        self.assertFalse(controls[3].isChecked())
        self.assertFalse(controls[4].isEnabled())
        self.assertEqual(controls[4].value(), 0)
        self.assertFalse(controls[5].isChecked())
        self.assertFalse(controls[6].isEnabled())
        self.assertEqual(controls[6].value(), 100)

        data: Parameter = self._widget._ParametersWidget__data[0]
        self.assertEqual(data.name, "p1")
        self.assertEqual(data.initial, 1)
        self.assertFalse(data.use_lower)
        self.assertEqual(data.lower, 0)
        self.assertFalse(data.use_upper)
        self.assertEqual(data.upper, 100)

    def test_add_row_with_data(self):
        param = Parameter("a", 3, True, 2, False, 4)
        self._widget._add_row(param)

        controls = self._widget._ParametersWidget__controls[0]
        self.assertEqual(controls[1].text(), "a")
        self.assertEqual(controls[2].value(), 3)
        self.assertTrue(controls[3].isChecked())
        self.assertEqual(controls[4].value(), 2)
        self.assertTrue(controls[4].isEnabled())
        self.assertFalse(controls[5].isChecked())
        self.assertEqual(controls[6].value(), 4)
        self.assertFalse(controls[6].isEnabled())

        data = self._widget._ParametersWidget__data[0]
        self.assertEqual(data.name, "a")
        self.assertEqual(data.initial, 3)
        self.assertTrue(data.use_lower)
        self.assertEqual(data.lower, 2)
        self.assertFalse(data.use_upper)
        self.assertEqual(data.upper, 4)

    def test_set_data(self):
        data = [Parameter("a", 4, True, -2, True, 5),
                Parameter("b", 2, True, 0, False, 11)]
        self._widget.set_data(data)
        self.assertEqual(len(self._widget._ParametersWidget__controls), 2)

        controls = self._widget._ParametersWidget__controls
        self.assertEqual(controls[0][1].text(), "a")
        self.assertEqual(controls[0][2].value(), 4)
        self.assertTrue(controls[0][3].isChecked())
        self.assertEqual(controls[0][4].value(), -2)
        self.assertTrue(controls[0][5].isChecked())
        self.assertEqual(controls[0][6].value(), 5)
        self.assertEqual(controls[1][1].text(), "b")
        self.assertEqual(controls[1][2].value(), 2)
        self.assertTrue(controls[1][3].isChecked())
        self.assertEqual(controls[1][4].value(), 0)
        self.assertFalse(controls[1][5].isChecked())
        self.assertEqual(controls[1][6].value(), 11)

        data = self._widget._ParametersWidget__data
        self.assertEqual(data[0].name, "a")
        self.assertEqual(data[0].initial, 4)
        self.assertTrue(data[0].use_lower)
        self.assertEqual(data[0].lower, -2)
        self.assertTrue(data[0].use_upper)
        self.assertEqual(data[0].upper, 5)
        self.assertEqual(data[1].name, "b")
        self.assertEqual(data[1].initial, 2)
        self.assertTrue(data[1].use_lower)
        self.assertEqual(data[1].lower, 0)
        self.assertFalse(data[1].use_upper)
        self.assertEqual(data[1].upper, 11)

    def test_reset_data(self):
        self._widget.set_data([Parameter("a", 1, True, 2, True, 5)])
        self._widget.set_data([Parameter("a", 1, True, 3, True, 6)])
        self.assertEqual(len(self._widget._ParametersWidget__controls), 1)
        self.assertEqual(len(self._widget._ParametersWidget__data), 1)

    def test_clear_all(self):
        self._widget.set_data([Parameter("a", 1, True, 2, True, 5)])
        self._widget.clear_all()
        self.assertEqual(len(self._widget._ParametersWidget__controls), 0)
        self.assertEqual(len(self._widget._ParametersWidget__data), 0)


class TestOWCurveFit(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWCurveFit,
                                         stored_settings={"auto_apply": False})
        self.housing = Table("housing")
        self.init()
        self.__add_button = \
            self.widget._OWCurveFit__param_widget._ParametersWidget__button

    def __init_widget(self, data=None, widget=None):
        if data is None:
            data = self.housing
        if widget is None:
            widget = self.widget
        self.send_signal(widget.Inputs.data, data, widget=widget)
        widget._OWCurveFit__param_widget._ParametersWidget__button.click()
        widget._OWCurveFit__expression_edit.setText("p1")
        self.widget.apply_button.button.click()

    def test_input_data_learner_adequacy(self):  # overwritten
        for inadequate in self.inadequate_dataset:
            self.__init_widget(inadequate)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.Error.data_error.is_shown())
        for valid in self.valid_datasets:
            self.__init_widget(valid)
            self.wait_until_stop_blocking()
            self.assertFalse(self.widget.Error.data_error.is_shown())

    def test_input_preprocessor(self):
        self.__init_widget()
        super().test_input_preprocessor()

    def test_input_preprocessors(self):
        self.__init_widget()
        super().test_input_preprocessors()

    def test_output_learner(self):
        self.__init_widget()
        super().test_output_learner()

    def test_output_model(self):  # overwritten
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

        self.__init_widget()
        self.wait_until_stop_blocking()
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, self.widget.LEARNER.__returns__)
        self.assertIsInstance(model, self.model_class)

    def test_output_learner_name(self):
        self.__init_widget()
        super().test_output_learner_name()

    def test_output_model_name(self):  # overwritten
        new_name = "Model Name"
        self.widget.name_line_edit.setText(new_name)
        self.__init_widget()
        self.wait_until_stop_blocking()
        model_name = self.get_output(self.widget.Outputs.model).name
        self.assertEqual(model_name, new_name)

    def test_output_model_picklable(self):  # overwritten
        self.__init_widget()
        self.wait_until_stop_blocking()
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        pickle.dumps(model)

    def test_output_coefficients(self):
        self.__init_widget()
        coefficients = self.get_output(self.widget.Outputs.coefficients)
        self.assertTrue("coef" in coefficients.domain)
        self.assertTrue("name" in coefficients.domain)

    def test_output_mixed_features(self):
        self.__init_widget(self.data)
        learner = self.get_output(self.widget.Outputs.learner)
        self.assertIsInstance(learner, CurveFitLearner)
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsInstance(model, CurveFitModel)
        coef = self.get_output(self.widget.Outputs.coefficients)
        self.assertTrue("coef" in coef.domain)
        self.assertTrue("name" in coef.domain)

    def test_features_combo(self):
        combo = self.widget.controls._feature
        model = combo.model()
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(combo.currentText(), "Select Feature")

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(model.rowCount(), 14)
        self.assertEqual(combo.currentText(), "Select Feature")
        simulate.combobox_activate_index(combo, 1)
        self.assertEqual(self.widget._OWCurveFit__expression_edit.text(),
                         "CRIM")

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(combo.currentText(), "Select Feature")

    def test_parameters_combo(self):
        combo = self.widget.controls._parameter
        model = combo.model()

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(combo.currentText(), "Select Parameter")
        self.__add_button.click()
        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(combo.currentText(), "Select Parameter")
        simulate.combobox_activate_index(combo, 1)
        self.assertEqual(self.widget._OWCurveFit__expression_edit.text(), "p1")

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(combo.currentText(), "Select Parameter")

    def test_function_combo(self):
        combo = self.widget.controls._function
        model = combo.model()
        self.assertEqual(model.rowCount(), 44)
        self.assertEqual(combo.currentText(), "Select Function")

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(model.rowCount(), 44)
        self.assertEqual(combo.currentText(), "Select Function")
        simulate.combobox_activate_index(combo, 1)
        self.assertEqual(self.widget._OWCurveFit__expression_edit.text(),
                         "abs()")

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(model.rowCount(), 44)
        self.assertEqual(combo.currentText(), "Select Function")

    def test_expression(self):
        feature_combo = self.widget.controls._feature
        function_combo = self.widget.controls._function
        insert = self.widget._OWCurveFit__insert_into_expression
        for f in FUNCTIONS:
            self.__init_widget()
            insert(" + ")
            simulate.combobox_activate_item(function_combo, f)
            if isinstance(getattr(np, f), float):
                insert(" + ")
                simulate.combobox_activate_index(feature_combo, 1)
            elif f == "gcd":
                simulate.combobox_activate_index(feature_combo, 1)
                self.widget._OWCurveFit__expression_edit.cursorForward(0, 1)
                insert("2")
            elif f in ["arctan2", "copysign", "fmod", "gcd", "hypot",
                       "isclose", "ldexp", "power", "remainder"]:
                simulate.combobox_activate_index(feature_combo, 1)
                self.widget._OWCurveFit__expression_edit.cursorForward(0, 1)
                insert("2")
            else:
                simulate.combobox_activate_index(feature_combo, 1)
            self.widget.apply_button.button.click()

            self.assertIsNotNone(self.get_output(self.widget.Outputs.learner))
            self.assertFalse(self.widget.Error.no_parameter.is_shown())
            self.assertFalse(self.widget.Error.invalid_exp.is_shown())
            model = self.get_output(self.widget.Outputs.model)
            coefficients = self.get_output(self.widget.Outputs.coefficients)
            if f == "gcd":
                self.assertTrue(self.widget.Error.fitting_failed.is_shown())
                self.assertIsNone(model)
                self.assertIsNone(coefficients)
            else:
                self.assertIsNotNone(model)
                self.assertIsNotNone(coefficients)
                self.assertFalse(self.widget.Error.fitting_failed.is_shown())
            self.send_signal(self.widget.Inputs.data, None)
            self.assertIsNone(self.get_output(self.widget.Outputs.learner))
            self.assertIsNone(self.get_output(self.widget.Outputs.model))
            coefficients = self.get_output(self.widget.Outputs.coefficients)
            self.assertIsNone(coefficients)

    def test_enable_controls(self):
        function_box = self.widget._OWCurveFit__function_box
        self.assertFalse(self.__add_button.isEnabled())
        self.assertFalse(function_box.isEnabled())
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertTrue(self.__add_button.isEnabled())
        self.assertTrue(function_box.isEnabled())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.__add_button.isEnabled())
        self.assertFalse(function_box.isEnabled())

    def test_duplicated_parameter_name(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.__add_button.click()
        self.__add_button.click()
        param_controls = \
            self.widget._OWCurveFit__param_widget._ParametersWidget__controls
        param_controls[1][1].setText("p1")
        self.assertTrue(self.widget.Warning.duplicate_parameter.is_shown())
        param_controls[1][1].setText("p2")
        self.assertFalse(self.widget.Warning.duplicate_parameter.is_shown())
        param_controls[1][1].setText("p1")
        self.assertTrue(self.widget.Warning.duplicate_parameter.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.duplicate_parameter.is_shown())

    def test_unused_parameter(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.__add_button.click()
        self.__add_button.click()
        self.assertFalse(self.widget.Warning.unused_parameter.is_shown())

        self.widget._OWCurveFit__expression_edit.setText("p1 + LSTAT + p2")
        self.widget.apply_button.button.click()
        self.assertFalse(self.widget.Warning.unused_parameter.is_shown())

        self.widget._OWCurveFit__expression_edit.setText("p1 + LSTAT")
        self.widget.apply_button.button.click()
        self.assertTrue(self.widget.Warning.unused_parameter.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.unused_parameter.is_shown())

    def test_unknown_parameter(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.__add_button.click()

        self.widget._OWCurveFit__expression_edit.setText("p1 + LSTAT")
        self.widget.apply_button.button.click()
        self.assertFalse(self.widget.Error.unknown_parameter.is_shown())

        self.widget._OWCurveFit__expression_edit.setText("p2 + LSTAT")
        self.widget.apply_button.button.click()
        self.assertTrue(self.widget.Error.unknown_parameter.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.unknown_parameter.is_shown())

    def test_saved_parameters(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.__add_button.click()
        self.assertEqual(self.widget.controls._parameter.model().rowCount(), 2)
        param_controls = \
            self.widget._OWCurveFit__param_widget._ParametersWidget__controls
        param_controls[0][1].setText("a")
        param_controls[0][2].setValue(3)
        param_controls[0][3].setChecked(True)
        param_controls[0][4].setValue(-10)
        param_controls[0][5].setChecked(True)
        param_controls[0][6].setValue(10)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual(
            settings["parameters"],
            {"a": ("a", 3, True, -10, True, 10)}
        )

        widget = self.create_widget(OWCurveFit, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.housing, widget=widget)
        param_controls = \
            widget._OWCurveFit__param_widget._ParametersWidget__controls
        self.assertEqual(param_controls[0][1].text(), "a")
        self.assertEqual(param_controls[0][2].value(), 3)
        self.assertEqual(param_controls[0][3].isChecked(), True)
        self.assertEqual(param_controls[0][4].value(), -10)
        self.assertEqual(param_controls[0][5].isChecked(), True)
        self.assertEqual(param_controls[0][6].value(), 10)
        self.assertEqual(widget.controls._parameter.model().rowCount(), 2)

    def test_no_parameter(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.widget._OWCurveFit__expression_edit.setText("LSTAT + 1")
        self.widget.apply_button.button.click()
        self.assertTrue(self.widget.Error.no_parameter.is_shown())
        self.widget._OWCurveFit__expression_edit.setText("LSTAT + a")
        self.widget.apply_button.button.click()
        self.assertFalse(self.widget.Error.no_parameter.is_shown())

    def test_output(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        for _ in range(3):
            self.__add_button.click()
        exp = "p1 * exp(-p2 * LSTAT) + p3"
        self.widget._OWCurveFit__expression_edit.setText(exp)
        self.widget.apply_button.button.click()
        learner = self.get_output(self.widget.Outputs.learner)
        self.assertIsInstance(learner, CurveFitLearner)
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsInstance(model, CurveFitModel)
        coef = self.get_output(self.widget.Outputs.coefficients)
        self.assertTrue("coef" in coef.domain)
        self.assertTrue("name" in coef.domain)


if __name__ == "__main__":
    unittest.main()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
import time
from unittest.mock import Mock

from AnyQt.QtCore import QLocale, Qt
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QLineEdit, QComboBox

import numpy as np

from Orange.data import (
    Table, ContinuousVariable, StringVariable, DiscreteVariable, Domain)
from Orange.preprocess import discretize
from Orange.widgets.data.owselectrows import (
    OWSelectRows, FilterDiscreteType, SelectRowsContextHandler)
from Orange.widgets.tests.base import WidgetTest, datasets

from Orange.data.filter import FilterContinuous, FilterString
from Orange.widgets.tests.utils import simulate, override_locale
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_FEATURE_NAME
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.tests import test_filename
from orangewidget.settings import VERSION_KEY

CFValues = {
    FilterContinuous.Equal: ["5.4"],
    FilterContinuous.NotEqual: ["5.4"],
    FilterContinuous.Less: ["5.4"],
    FilterContinuous.LessEqual: ["5.4"],
    FilterContinuous.Greater: ["5.4"],
    FilterContinuous.GreaterEqual: ["5.4"],
    FilterContinuous.Between: ["5.4", "6.0"],
    FilterContinuous.Outside: ["5.4", "6.0"],
    FilterContinuous.IsDefined: [],
}


SFValues = {
    FilterString.Equal: ["aardwark"],
    FilterString.NotEqual: ["aardwark"],
    FilterString.Less: ["aardwark"],
    FilterString.LessEqual: ["aardwark"],
    FilterString.Greater: ["aardwark"],
    FilterString.GreaterEqual: ["aardwark"],
    FilterString.Between: ["aardwark", "cat"],
    FilterString.Outside: ["aardwark"],
    FilterString.Contains: ["aa"],
    FilterString.StartsWith: ["aa"],
    FilterString.EndsWith: ["ark"],
    FilterString.IsDefined: []
}

DFValues = {
    FilterDiscreteType.Equal: [0],
    FilterDiscreteType.NotEqual: [0],
    FilterDiscreteType.In: [0, 1],
    FilterDiscreteType.IsDefined: [],
}


class TestOWSelectRows(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSelectRows)  # type: OWSelectRows

    def test_filter_cont(self):
        iris = Table("iris")[::5]
        self.widget.auto_commit = True
        self.widget.set_data(iris)

        for i, (op, _) in enumerate(OWSelectRows.Operators[ContinuousVariable]):
            self.widget.remove_all()
            self.widget.add_row(1, i, CFValues[op])
            self.widget.conditions_changed()
            self.widget.unconditional_commit()

    def test_filter_str(self):
        zoo = Table("zoo")[::5]
        self.widget.auto_commit = False
        self.widget.set_data(zoo)
        for i, (op, _) in enumerate(OWSelectRows.Operators[StringVariable]):
            self.widget.remove_all()
            self.widget.add_row(1, i, SFValues[op])
            self.widget.conditions_changed()
            self.widget.unconditional_commit()

    def test_filter_disc(self):
        lenses = Table(test_filename("datasets/lenses.tab"))
        self.widget.auto_commit = False
        self.widget.set_data(lenses)

        for i, (op, _) in enumerate(OWSelectRows.Operators[DiscreteVariable]):
            self.widget.remove_all()
            self.widget.add_row(0, i, DFValues[op])
            self.widget.conditions_changed()
            self.widget.unconditional_commit()

    @override_locale(QLocale.C)  # Locale with decimal point
    def test_continuous_filter_with_c_locale(self):
        iris = Table("iris")[:5]
        self.send_signal(self.widget.Inputs.data, iris)

        # Validating with C locale should accept decimal point
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5.2")
        self.assertEqual(self.widget.conditions[0][2], ("5.2",))

        # but not decimal comma
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5,2")
        self.assertEqual(self.widget.conditions[0][2], ("52",))

    @override_locale(QLocale.Slovenian)  # Locale with decimal comma
    def test_continuous_filter_with_sl_SI_locale(self):
        iris = Table("iris")[:5]
        self.send_signal(self.widget.Inputs.data, iris)

        # sl_SI locale should accept decimal comma
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5,2")
        self.assertEqual(self.widget.conditions[0][2], ("5,2",))

        # but not decimal point
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5.2")
        self.assertEqual(self.widget.conditions[0][2], ("52",))

    @override_locale(QLocale.Slovenian)
    def test_stores_settings_in_invariant_locale(self):
        iris = Table("iris")[:5]
        self.send_signal(self.widget.Inputs.data, iris)

        # sl_SI locale should accept decimal comma
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5,2")
        self.assertEqual(self.widget.conditions[0][2], ("5,2",))

        context = self.widget.current_context
        self.send_signal(self.widget.Inputs.data, None)
        saved_condition = context.values["conditions"][0]
        self.assertEqual(saved_condition[3][0], 5.2)

    @override_locale(QLocale.C)
    def test_restores_continuous_filter_in_c_locale(self):
        iris = Table("iris")[:5]
        # Settings with string value
        self.widget = self.widget_with_context(
            iris.domain, [["sepal length", 102, 2, ("5.2",)]])
        self.send_signal(self.widget.Inputs.data, iris)

        values = self.widget.conditions[0][2]
        self.assertTrue(values[0].startswith("5.2"))

        # Settings with float value
        self.widget = self.widget_with_context(
            iris.domain, [["sepal length", 102, 2, (5.2,)]])
        self.send_signal(self.widget.Inputs.data, iris)

        values = self.widget.conditions[0][2]
        self.assertTrue(values[0].startswith("5.2"))

    @override_locale(QLocale.Slovenian)
    def test_restores_continuous_filter_in_sl_SI_locale(self):
        iris = Table("iris")[:5]
        # Settings with string value
        self.widget = self.widget_with_context(
            iris.domain, [["sepal length", 102, 2, ("5.2",)]])
        self.send_signal(self.widget.Inputs.data, iris)

        values = self.widget.conditions[0][2]
        self.assertTrue(values[0].startswith("5,2"))

        # Settings with float value
        self.widget = self.widget_with_context(
            iris.domain, [["sepal length", 102, 2, (5.2,)]])
        self.send_signal(self.widget.Inputs.data, iris)

        values = self.widget.conditions[0][2]
        self.assertTrue(values[0].startswith("5,2"))

    @override_locale(QLocale.C)
    def test_partial_matches(self):
        iris = Table("iris")
        domain = iris.domain
        self.widget = self.widget_with_context(
            domain, [[domain[0].name, 2, ("5.2",)]])
        iris2 = iris.transform(Domain(domain.attributes[:2], None))
        self.send_signal(self.widget.Inputs.data, iris2)
        condition = self.widget.conditions[0]
        self.assertEqual(condition[0], "sepal length")
        self.assertEqual(condition[1], 2)
        self.assertTrue(condition[2][0].startswith("5.2"))

    def test_load_settings(self):
        iris = Table("iris")[:5]
        self.send_signal(self.widget.Inputs.data, iris)

        sepal_length, sepal_width = iris.domain[:2]

        self.widget.remove_all_button.click()
        self.enterFilter(sepal_width, "is below", "5.2")
        self.enterFilter(sepal_length, "is at most", "4")
        data = self.widget.settingsHandler.pack_data(self.widget)

        w2 = self.create_widget(OWSelectRows, data)
        self.send_signal(self.widget.Inputs.data, iris, widget=w2)

        var_combo = w2.cond_list.cellWidget(0, 0)
        self.assertEqual(var_combo.currentText(), "sepal width")
        oper_combo = w2.cond_list.cellWidget(0, 1)
        self.assertEqual(oper_combo.currentText(), "is below")

        var_combo = w2.cond_list.cellWidget(1, 0)
        self.assertEqual(var_combo.currentText(), "sepal length")
        oper_combo = w2.cond_list.cellWidget(1, 1)
        self.assertEqual(oper_combo.currentText(), "is at most")

    def test_is_defined_on_continuous_variable(self):
        # gh-2054 regression

        data = Table(datasets.path("testing_dataset_cls"))
        self.send_signal(self.widget.Inputs.data, data)

        self.enterFilter(data.domain["c2"], "is defined")
        self.assertFalse(self.widget.Error.parsing_error.is_shown())
        self.assertEqual(len(self.get_output("Matching Data")), 3)
        self.assertEqual(len(self.get_output("Unmatched Data")), 1)
        self.assertEqual(len(self.get_output("Data")), len(data))

        # Test saving of settings
        self.widget.settingsHandler.pack_data(self.widget)

    def test_summary(self):
        """Check if status bar displays correct input/output summary"""
        input_sum = self.widget.info.set_input_summary = Mock()
        output_sum = self.widget.info.set_output_summary = Mock()

        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(len(data), format_summary_details(data))
        output = self.get_output("Matching Data")
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))

        self.enterFilter(data.domain["iris"], "is", "Iris-setosa")
        output = self.get_output("Matching Data")
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))
        input_sum.reset_mock()
        output_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertEqual(input_sum.call_args[0][0].brief, "")
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")

    def test_output_filter(self):
        """
        None on output when there is no data.
        GH-2726
        """
        data = Table("iris")[:10]
        len_data = len(data)
        self.send_signal(self.widget.Inputs.data, data)

        self.enterFilter(data.domain[0], "is below", "-1")
        self.assertIsNone(self.get_output("Matching Data"))
        self.assertEqual(len(self.get_output("Unmatched Data")), len_data)
        self.assertEqual(len(self.get_output("Data")), len_data)
        self.widget.remove_all_button.click()
        self.enterFilter(data.domain[0], "is below", "10")
        self.assertIsNone(self.get_output("Unmatched Data"))
        self.assertEqual(len(self.get_output("Matching Data")), len_data)
        self.assertEqual(len(self.get_output("Data")), len_data)

    def test_annotated_data(self):
        iris = Table("iris")
        self.send_signal(self.widget.Inputs.data, iris)

        self.enterFilter(iris.domain["iris"], "is", "Iris-setosa")

        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(annotated), 150)
        annotations = annotated.get_column_view(ANNOTATED_DATA_FEATURE_NAME)[0]
        np.testing.assert_equal(annotations[:50], True)
        np.testing.assert_equal(annotations[50:], False)

    def test_change_var_type(self):
        iris = Table("iris")
        domain = iris.domain

        self.send_signal(self.widget.Inputs.data, iris)
        self.widget.remove_all_button.click()
        self.enterFilter(domain[0], "is below", "5.2")

        var0vals = list({str(x) for x in iris.X[:, 0]})
        new_domain = Domain(
            (DiscreteVariable(domain[0].name, values=var0vals), )
            + domain.attributes[1:],
            domain.class_var)
        new_iris = iris.transform(new_domain)
        self.send_signal(self.widget.Inputs.data, new_iris)

    # Uncomment this on 2022/2/2
    #
    # def test_migration_to_version_1(self):
    #     iris = Table("iris")
    #
    #     ch = SelectRowsContextHandler()
    #     context = ch.new_context(iris.domain, *ch.encode_domain(iris.domain))
    #     context.values = dict(conditions=[["petal length", 2, (5.2,)]])
    #     settings = dict(context_settings=[context])
    #     widget = self.create_widget(OWSelectRows, settings)
    #     self.assertEqual(widget.conditions, [])

    @override_locale(QLocale.C)
    def test_support_old_settings(self):
        iris = Table("iris")
        self.widget = self.widget_with_context(
            iris.domain, [["sepal length", 2, ("5.2",)]])
        self.send_signal(self.widget.Inputs.data, iris)
        condition = self.widget.conditions[0]
        self.assertEqual(condition[0], "sepal length")
        self.assertEqual(condition[1], 2)
        self.assertTrue(condition[2][0].startswith("5.2"))

    def test_end_support_for_version_1(self):
        if time.gmtime() >= (2022, 2, 2):
            self.fail("""
Happy 22/2/2!

Now remove support for version==None settings in
SelectRowsContextHandler.decode_setting and SelectRowsContextHandler.match,
and uncomment OWSelectRows.migrate.

In tests, uncomment test_migration_to_version_1,
and remove test_support_old_settings and this test.

Basically, revert this commit.
""")

    def test_purge_discretized(self):
        housing = Table("housing")
        method = discretize.EqualFreq(n=3)
        discretizer = discretize.DomainDiscretizer(
            discretize_class=True, method=method)
        domain = discretizer(housing)
        data = housing.transform(domain)
        widget = self.widget_with_context(domain, [["MEDV", 2, (2, 3)]])
        widget.purge_classes = True
        self.send_signal(widget.Inputs.data, data)
        out = self.get_output(widget.Outputs.matching_data)
        expected = data.Y[(data.Y == 1) + (data.Y == 2)]
        expected = (expected == 2).astype(float)
        np.testing.assert_equal(out.Y, expected)

    def widget_with_context(self, domain, conditions):
        ch = SelectRowsContextHandler()
        context = ch.new_context(domain, *ch.encode_domain(domain))
        context.values = {"conditions": conditions,
                          VERSION_KEY: OWSelectRows.settings_version}
        settings = dict(context_settings=[context])

        return self.create_widget(OWSelectRows, settings)

    def enterFilter(self, variable, filter, value1=None, value2=None):
        # pylint: disable=redefined-builtin
        row = self.widget.cond_list.model().rowCount()
        self.widget.add_button.click()

        var_combo = self.widget.cond_list.cellWidget(row, 0)
        simulate.combobox_activate_item(var_combo, variable.name, delay=0)

        oper_combo = self.widget.cond_list.cellWidget(row, 1)
        simulate.combobox_activate_item(oper_combo, filter, delay=0)

        value_inputs = self.__get_value_widgets(row)
        for i, value in enumerate([value1, value2]):
            if value is None:
                continue
            self.__set_value(value_inputs[i], value)

    def __get_value_widgets(self, row):
        value_inputs = self.widget.cond_list.cellWidget(row, 2)
        if value_inputs:
            if isinstance(value_inputs, QComboBox):
                value_inputs = [value_inputs]
            else:
                value_inputs = [
                    w for w in value_inputs.children()
                    if isinstance(w, QLineEdit)]
        return value_inputs

    @staticmethod
    def __set_value(widget, value):
        if isinstance(widget, QLineEdit):
            QTest.mouseClick(widget, Qt.LeftButton)
            QTest.keyClicks(widget, value, delay=0)
            QTest.keyClick(widget, Qt.Key_Enter)
        elif isinstance(widget, QComboBox):
            simulate.combobox_activate_item(widget, value)
        else:
            raise ValueError("Unsupported widget {}".format(widget))

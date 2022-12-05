# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
import unittest
from unittest.mock import patch
import numpy as np

from AnyQt.QtCore import QLocale, Qt, QDate
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QLineEdit, QComboBox

from orangewidget.settings import VERSION_KEY

from Orange.data import (
    Table, Variable, ContinuousVariable, StringVariable, DiscreteVariable,
    Domain, TimeVariable)
from Orange.data.dask import DaskTable
from Orange.preprocess import discretize
from Orange.widgets.data import owselectrows
from Orange.widgets.data.owselectrows import (
    OWSelectRows, FilterDiscreteType, SelectRowsContextHandler, DateTimeWidget)
from Orange.widgets.tests.base import WidgetTest, datasets

from Orange.data.filter import FilterContinuous, FilterString
from Orange.widgets.tests.utils import simulate, override_locale
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_FEATURE_NAME
from Orange.tests import test_filename
from Orange.tests.test_dasktable import temp_dasktable


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
    FilterString.NotContain: ["bb"],
    FilterString.StartsWith: ["aa"],
    FilterString.NotStartsWith: ["bb"],
    FilterString.EndsWith: ["ark"],
    FilterString.NotEndsWith: ["zz"],
    FilterString.IsDefined: [],
    FilterString.NotIsDefined: [], # TODO: needs test data
}

DFValues = {
    FilterDiscreteType.Equal: [0],
    FilterDiscreteType.NotEqual: [0],
    FilterDiscreteType.In: [0, 1],
    FilterDiscreteType.IsDefined: [],
}

TFValues = {
    FilterContinuous.Equal: [QDate(2013, 5, 5)],
    FilterContinuous.NotEqual: [QDate(2013, 5, 5)],
    FilterContinuous.Less: [QDate(2013, 5, 5)],
    FilterContinuous.LessEqual: [QDate(2013, 5, 5)],
    FilterContinuous.Greater: [QDate(2013, 5, 5)],
    FilterContinuous.GreaterEqual: [QDate(2013, 5, 5)],
    FilterContinuous.Between: [QDate(2013, 5, 5), QDate(2015, 5, 5)],
    FilterContinuous.Outside: [QDate(2013, 5, 5), QDate(2015, 5, 5)],
    FilterContinuous.IsDefined: [],
}


class TestOWSelectRows(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.zoo = Table("zoo")
        cls.housing = Table("housing")
        cls.heart = Table("heart_disease")
        cls.lenses = Table(test_filename("datasets/lenses.tab"))
        cls.cyber = Table(test_filename("datasets/cyber-security-breaches.tab"))

    def setUp(self):
        self.widget = self.create_widget(OWSelectRows)  # type: OWSelectRows

    def test_filter_cont(self):
        iris = self.iris[::5]
        self.widget.auto_commit = True
        self.widget.set_data(iris)

        for i, (op, *_) in enumerate(OWSelectRows.Operators[ContinuousVariable]):
            self.widget.remove_all()
            self.widget.add_row(iris.domain[0], i, CFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

        # continuous var in metas
        iris = iris.transform(
            Domain([], metas=[iris.domain.attributes[0]])
        )
        self.widget.set_data(iris)
        for i, (op, *_) in enumerate(OWSelectRows.Operators[ContinuousVariable]):
            self.widget.remove_all()
            self.widget.add_row(iris.domain.metas[0], i, CFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

    def test_filter_str(self):
        zoo = self.zoo[::5]
        self.widget.auto_commit = False
        self.widget.set_data(zoo)
        for i, (op, *_) in enumerate(OWSelectRows.Operators[StringVariable]):
            self.widget.remove_all()
            self.widget.add_row(zoo.domain.metas[0], i, SFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

    def test_filter_disc(self):
        lenses = self.lenses
        self.widget.auto_commit = False
        self.widget.set_data(lenses)

        for i, (op, _) in enumerate(OWSelectRows.Operators[DiscreteVariable]):
            self.widget.remove_all()
            self.widget.add_row(0, i, DFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

        # discrete var in metas
        lenses = lenses.transform(
            Domain([], metas=[lenses.domain.attributes[0]])
        )
        self.widget.set_data(lenses)
        for i, (op, _) in enumerate(OWSelectRows.Operators[DiscreteVariable]):
            self.widget.remove_all()
            self.widget.add_row(lenses.domain.metas[0], i, DFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

    def test_filter_time(self):
        data = self.cyber
        self.widget.auto_commit = False
        self.widget.set_data(data)

        for i, (op, *_) in enumerate(OWSelectRows.Operators[TimeVariable]):
            self.widget.remove_all()
            self.widget.add_row(data.domain["breach_start"], i, TFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

        # time var in metas
        data = data.transform(
            Domain([], metas=[data.domain["breach_start"]])
        )
        self.widget.set_data(data)
        for i, (op, *_) in enumerate(OWSelectRows.Operators[TimeVariable]):
            self.widget.remove_all()
            self.widget.add_row(data.domain.metas[0], i, TFValues[op])
            self.widget.conditions_changed()
            self.widget.commit.now()

    @override_locale(QLocale.C)  # Locale with decimal point
    def test_continuous_filter_with_c_locale(self):
        iris = self.iris[:5]
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
        iris = self.iris[:5]
        self.send_signal(self.widget.Inputs.data, iris)

        # sl_SI locale should accept decimal comma
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5,2")
        self.assertEqual(self.widget.conditions[0][2], ("5,2",))

        # but not decimal point
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5.2")
        self.assertEqual(self.widget.conditions[0][2], ("52",))

    @override_locale(QLocale.C)  # Locale with decimal point
    def test_all_numeric_filter_with_c_locale_from_context(self):
        iris = self.iris[:5]
        widget = self.widget_with_context(
            iris.domain, [["All numeric variables", None, 0, (3.14, )]])
        self.send_signal(widget.Inputs.data, iris)
        self.assertTrue(widget.conditions[0][2][0].startswith("3.14"))

    @override_locale(QLocale.Slovenian)  # Locale with decimal comma
    def test_all_numeric_filter_with_sl_SI_locale(self):
        iris = self.iris[:5]
        widget = self.widget_with_context(
            iris.domain, [["All numeric variables", None, 0, (3.14, )]])
        self.send_signal(widget.Inputs.data, iris)
        self.assertTrue(widget.conditions[0][2][0].startswith("3,14"))

    @override_locale(QLocale.Slovenian)
    def test_stores_settings_in_invariant_locale(self):
        iris = self.iris[:5]
        self.send_signal(self.widget.Inputs.data, iris)

        # sl_SI locale should accept decimal comma
        self.widget.remove_all_button.click()
        self.enterFilter(iris.domain[2], "is below", "5,2")
        self.assertEqual(self.widget.conditions[0][2], ("5,2",))

        context = self.widget.current_context
        self.send_signal(self.widget.Inputs.data, None)
        saved_condition = context.values["conditions"][0]
        self.assertEqual(saved_condition[3][0], 5.2)

    @override_locale(QLocale.C)  # Locale with decimal point
    def test_store_all_numeric_filter_with_c_locale_to_context(self):
        iris = self.iris[:5]
        self.send_signal(self.widget.Inputs.data, iris)
        self.widget.remove_all_button.click()
        self.enterFilter("All numeric variables", "equal", "3.14")
        context = self.widget.current_context
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(context.values["conditions"][0][3], [3.14])

    @override_locale(QLocale.Slovenian)  # Locale with decimal comma
    def test_store_all_numeric_filter_with_sl_SI_locale_to_context(self):
        iris = self.iris[:5]
        self.send_signal(self.widget.Inputs.data, iris)
        self.widget.remove_all_button.click()
        self.enterFilter("All numeric variables", "equal", "3,14")
        context = self.widget.current_context
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(context.values["conditions"][0][3], [3.14])

    @override_locale(QLocale.C)
    def test_restores_continuous_filter_in_c_locale(self):
        iris = self.iris[:5]
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
        iris = self.iris[:5]
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
        iris = self.iris
        domain = iris.domain
        self.widget = self.widget_with_context(
            domain, [[domain[0].name, 2, 2, ("5.2",)]])
        iris2 = iris.transform(Domain(domain.attributes[:2], None))
        self.send_signal(self.widget.Inputs.data, iris2)
        condition = self.widget.conditions[0]
        self.assertEqual(condition[0], iris.domain[0])
        self.assertEqual(condition[1], 2)
        self.assertTrue(condition[2][0].startswith("5.2"))

    def test_partial_match_values(self):
        iris = self.iris
        domain = iris.domain
        class_var = domain.class_var
        self.widget = self.widget_with_context(
            domain, [[class_var.name, 1, 2,
                      (class_var.values[0], class_var.values[2])]])

        # sanity checks
        self.send_signal(self.widget.Inputs.data, iris)
        condition = self.widget.conditions[0]
        self.assertIs(condition[0], class_var)
        self.assertEqual(condition[1], 2)
        self.assertEqual(condition[2], (1, 3))  # indices of values + 1

        # actual test
        new_class_var = DiscreteVariable(class_var.name, class_var.values[1:])
        new_domain = Domain(domain.attributes, new_class_var)
        non0 = iris.Y != 0
        iris2 = Table.from_numpy(new_domain, iris.X[non0], iris.Y[non0] - 1)
        self.send_signal(self.widget.Inputs.data, iris2)
        condition = self.widget.conditions[0]
        self.assertIs(condition[0], new_class_var)
        self.assertEqual(condition[1], 2)
        self.assertEqual(condition[2], (2, ))  # index of value + 1

    @override_locale(QLocale.C)
    def test_partial_matches_with_missing_vars(self):
        iris = self.iris
        domain = iris.domain
        self.widget = self.widget_with_context(
            domain, [[domain[0].name, 2, 2, ("5.2",)],
                     [domain[2].name, 2, 2, ("4.2",)]])
        iris2 = iris.transform(Domain(domain.attributes[2:], None))
        self.send_signal(self.widget.Inputs.data, iris2)
        condition = self.widget.conditions[0]
        self.assertEqual(condition[0], domain[2])
        self.assertEqual(condition[1], 2)
        self.assertTrue(condition[2][0].startswith("4.2"))

    def test_load_settings(self):
        iris = self.iris[:5]
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
        outputs = self.widget.Outputs
        self.assertEqual(len(self.get_output(outputs.matching_data)), 3)
        self.assertEqual(len(self.get_output(outputs.unmatched_data)), 1)
        self.assertEqual(len(self.get_output(outputs.annotated_data)), len(data))

        # Test saving of settings
        self.widget.settingsHandler.pack_data(self.widget)

    def test_output_filter(self):
        """
        None on output when there is no data.
        GH-2726
        """
        data = self.iris[:10]
        len_data = len(data)
        self.send_signal(self.widget.Inputs.data, data)

        self.enterFilter(data.domain[0], "is below", "-1")
        outputs = self.widget.Outputs
        self.assertIsNone(self.get_output(outputs.matching_data))
        self.assertEqual(len(self.get_output(outputs.unmatched_data)), len_data)
        self.assertEqual(len(self.get_output(outputs.annotated_data)), len_data)
        self.widget.remove_all_button.click()
        self.enterFilter(data.domain[0], "is below", "10")
        self.assertIsNone(self.get_output(outputs.unmatched_data))
        self.assertEqual(len(self.get_output(outputs.matching_data)), len_data)
        self.assertEqual(len(self.get_output(outputs.annotated_data)), len_data)

    def test_annotated_data(self):
        iris = self.iris
        self.send_signal(self.widget.Inputs.data, iris)

        self.enterFilter(iris.domain["iris"], "is", "Iris-setosa")

        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(annotated), 150)
        annotations = annotated.get_column(ANNOTATED_DATA_FEATURE_NAME)
        np.testing.assert_equal(annotations[:50], True)
        np.testing.assert_equal(annotations[50:], False)

    def test_change_var_type(self):
        iris = self.iris
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

    def test_keep_operator(self):
        data = self.heart
        domain = data.domain

        self.send_signal(self.widget.Inputs.data, data)

        self.widget.remove_all_button.click()
        self.enterFilter(domain["age"], "is not", "42")
        simulate.combobox_activate_item(
            self.widget.cond_list.cellWidget(0, 0), "chest pain", delay=0)
        self.assertEqual(
            self.widget.cond_list.cellWidget(0, 1).currentText(), "is not")

        self.widget.remove_all_button.click()
        self.enterFilter(domain["age"], "is below", "42")
        simulate.combobox_activate_item(
            self.widget.cond_list.cellWidget(0, 0), "chest pain", delay=0)
        self.assertEqual(
            self.widget.cond_list.cellWidget(0, 1).currentText(), "is")

    def test_calendar_dates(self):
        data = Table(test_filename("datasets/cyber-security-breaches.tab"))
        self.send_signal(self.widget.Inputs.data, data)
        simulate.combobox_activate_item(
            self.widget.cond_list.cellWidget(0, 0), "Date_Posted_or_Updated",
            delay=0)
        value_combo = self.widget.cond_list.cellWidget(0, 2).children()[1]
        self.assertIsInstance(value_combo, DateTimeWidget)

        # first displayed date is min date
        self.assertEqual(value_combo.date(), QDate(2014, 1, 23))
        outputs = self.widget.Outputs
        self.assertEqual(len(self.get_output(outputs.matching_data)), 691)
        self.widget.remove_all_button.click()
        self.enterFilter("Date_Posted_or_Updated", "is below",
                         QDate(2014, 4, 17))
        self.assertEqual(len(self.get_output(outputs.matching_data)), 840)
        self.enterFilter("Date_Posted_or_Updated", "is greater than",
                         QDate(2014, 6, 30))
        self.assertIsNone(self.get_output(outputs.matching_data))
        self.widget.remove_all_button.click()
        # date is in range min-max date
        self.enterFilter("Date_Posted_or_Updated", "equals", QDate(2013, 1, 1))
        self.assertEqual(self.widget.conditions[0][2][0], QDate(2014, 1, 23))
        self.enterFilter("Date_Posted_or_Updated", "equals", QDate(2015, 1, 1))
        self.assertEqual(self.widget.conditions[1][2][0], QDate(2014, 6, 30))
        self.widget.remove_all_button.click()
        # no date crossings
        self.enterFilter("Date_Posted_or_Updated", "is between",
                         QDate(2014, 4, 17), QDate(2014, 1, 23))
        self.assertEqual(self.widget.conditions[0][2],
                         (QDate(2014, 4, 17), QDate(2014, 4, 17)))
        self.widget.remove_all_button.click()
        self.enterFilter("Date_Posted_or_Updated", "is between",
                         QDate(2014, 4, 17), QDate(2014, 4, 30))
        self.assertEqual(len(self.get_output(outputs.matching_data)), 58)

    @patch.object(owselectrows.QMessageBox, "question",
                  return_value=owselectrows.QMessageBox.Ok)
    def test_add_all(self, msgbox):
        iris = self.iris
        domain = iris.domain
        self.send_signal(self.widget.Inputs.data, iris)
        self.widget.add_all_button.click()
        msgbox.assert_called()
        self.assertEqual([cond[0] for cond in self.widget.conditions],
                         list(domain.class_vars + domain.attributes))

    @patch.object(owselectrows.QMessageBox, "question",
                  return_value=owselectrows.QMessageBox.Cancel)
    def test_add_all_cancel(self, msgbox):
        iris = self.iris
        domain = iris.domain
        self.send_signal(self.widget.Inputs.data, iris)
        self.assertEqual([cond[0] for cond in self.widget.conditions],
                         list(domain.class_vars))
        self.widget.add_all_button.click()
        msgbox.assert_called()
        self.assertEqual([cond[0] for cond in self.widget.conditions],
                         list(domain.class_vars))

    @patch.object(owselectrows.QMessageBox, "question",
                  return_value=owselectrows.QMessageBox.Ok)
    def test_report(self, _):
        zoo = self.zoo
        self.send_signal(self.widget.Inputs.data, zoo)
        self.widget.add_all_button.click()
        self.enterFilter("All numeric variables", "equal", "42")
        self.enterFilter(zoo.domain[0], "is defined")
        self.enterFilter(zoo.domain[1], "is one of")
        self.widget.send_report()  # don't crash

    def test_migration_to_version_1(self):
        iris = self.iris

        ch = SelectRowsContextHandler()
        context = ch.new_context(iris.domain, *ch.encode_domain(iris.domain))
        context.values = dict(conditions=[["petal length", 2, (5.2,)]])
        settings = dict(context_settings=[context])
        widget = self.create_widget(OWSelectRows, settings)
        self.assertEqual(widget.conditions, [])

    def test_purge_discretized(self):
        housing = self.housing
        method = discretize.EqualFreq(n=3)
        discretizer = discretize.DomainDiscretizer(
            discretize_class=True, method=method)
        domain = discretizer(housing)
        data = housing.transform(domain)
        widget = self.widget_with_context(
            domain, [["MEDV", 101, 2, domain.class_var.values[1:]]]
        )
        widget.purge_classes = True
        self.send_signal(widget.Inputs.data, data)
        out = self.get_output(widget.Outputs.matching_data)
        expected = data.Y[(data.Y == 1) + (data.Y == 2)]
        expected = (expected == 2).astype(float)
        np.testing.assert_equal(out.Y, expected)

    def test_meta_setting(self):
        """
        Test if all conditions from all segments (attributes, class, meta)
        stores correctly
        """
        data = self.iris
        data = data.transform(
            Domain(
                data.domain.attributes[:3],
                data.domain.class_var,
                data.domain.attributes[3:]
            ))
        self.send_signal(self.widget.Inputs.data, data)

        vars_ = [
            data.domain.metas[0],
            data.domain.attributes[0],
            data.domain.class_var
        ]
        cond = [0, 0, 0]
        val = [(0, ), (0, ), (1, )]
        conds = list(zip(vars_, cond, val))

        self.widget.conditions = conds
        self.assertListEqual([c[0] for c in self.widget.conditions], vars_)

        # when sending new-same data conditions are restored from the context
        self.send_signal(self.widget.Inputs.data, data)
        self.assertListEqual([c[0] for c in self.widget.conditions], vars_)

    def test_one_of_click(self):
        """Test items checked in is one of dropdown"""
        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.data, zoo)
        self.widget.remove_all_button.click()
        self.enterFilter(zoo.domain[1], "is one of")
        model = self.widget.cond_list.cellWidget(0, 2).popup.list_view.model()

        output = self.get_output(self.widget.Outputs.matching_data)
        self.assertEqual(len(zoo), len(output))

        # check second item (group 1) - only 20 elements in this group
        model.item(1).setCheckState(Qt.Checked)
        output = self.get_output(self.widget.Outputs.matching_data)
        self.assertEqual(20, len(output))

        # check first item (group 0) - now all elements should be at the output
        model.item(0).setCheckState(Qt.Checked)
        output = self.get_output(self.widget.Outputs.matching_data)
        self.assertEqual(len(zoo), len(output))

        # uncheck second element (group 1) - only elements fo group 0 at output
        model.item(1).setCheckState(Qt.Unchecked)
        output = self.get_output(self.widget.Outputs.matching_data)
        self.assertEqual(81, len(output))

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
        name = variable.name if isinstance(variable, Variable) else variable
        simulate.combobox_activate_item(var_combo, name, delay=0)

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
                value_input = []
                for widget in value_inputs.children():
                    if isinstance(widget, QLineEdit):
                        value_input.append(widget)
                    elif isinstance(widget, DateTimeWidget):
                        value_input.append(widget)
                return value_input
        return value_inputs

    @staticmethod
    def __set_value(widget, value):
        if isinstance(widget, QLineEdit):
            QTest.mouseClick(widget, Qt.LeftButton)
            QTest.keyClicks(widget, value, delay=0)
            QTest.keyClick(widget, Qt.Key_Enter)
        elif isinstance(widget, QComboBox):
            simulate.combobox_activate_item(widget, value)
        elif isinstance(widget, DateTimeWidget):
            widget.setDate(value)
        else:
            raise ValueError("Unsupported widget {}".format(widget))


class TestOWSelectRowsDask(TestOWSelectRows):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = temp_dasktable("iris")
        cls.zoo = temp_dasktable("zoo")
        cls.housing = temp_dasktable("housing")
        cls.heart = temp_dasktable("heart_disease")
        cls.lenses = temp_dasktable(cls.lenses)
        cls.cyber = temp_dasktable(cls.cyber)

    def test_dask_outputs(self):
        w = self.widget
        iris = self.iris
        self.widget.auto_commit = True
        self.widget.set_data(iris)
        self.widget.add_row(iris.domain[0], 0, CFValues[FilterContinuous.Equal])
        self.widget.conditions_changed()
        self.widget.commit.now()
        m = self.get_output(w.Outputs.matching_data)
        self.assertIsInstance(m, DaskTable)
        self.assertEqual(6, len(m))
        n = self.get_output(w.Outputs.unmatched_data)
        self.assertIsInstance(n, DaskTable)
        self.assertEqual(len(iris), len(m) + len(n))
        self.assertEqual(144, len(n))
        a = self.get_output(w.Outputs.annotated_data)
        self.assertEqual(len(iris), len(a))
        self.assertIsInstance(a, DaskTable)


if __name__ == "__main__":
    unittest.main()

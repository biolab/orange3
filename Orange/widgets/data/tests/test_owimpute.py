# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from AnyQt.QtCore import Qt, QItemSelection
from AnyQt.QtTest import QTest

from Orange.data import Table
from Orange.preprocess import impute
from Orange.widgets.data.owimpute import OWImpute, AsDefault
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_row


class TestOWImpute(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWImpute)  # type: OWImpute

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget
        widget.default_method_index = widget.MODEL_BASED_IMPUTER
        widget.default_method = widget.methods[widget.default_method_index]

        self.send_signal("Data", data)
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        np.testing.assert_equal(imp_data.X, data.X)
        np.testing.assert_equal(imp_data.Y, data.Y)

        self.send_signal("Data", Table(data.domain))
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        self.assertEqual(len(imp_data), 0)

    def test_no_features(self):
        widget = self.widget
        widget.default_method_index = widget.MODEL_BASED_IMPUTER
        widget.default_method = widget.methods[widget.default_method_index]

        self.send_signal("Data", Table("iris"))

        self.send_signal("Learner", lambda *_: 1/0)  # Learner fails
        widget.unconditional_commit()
        self.assertTrue(widget.Error.imputation_failed.is_shown())

        self.send_signal("Learner", lambda *_: (lambda *_: 1/0))  # Model fails
        widget.unconditional_commit()
        self.assertTrue(widget.Error.imputation_failed.is_shown())

    def test_select_method(self):
        data = Table("iris")[::5]

        self.send_signal("Data", data)
        method_types = [type(m) for m in OWImpute.METHODS]

        widget = self.widget
        model = widget.varmodel
        view = widget.varview
        defbg = widget.default_button_group
        varbg = widget.variable_button_group
        self.assertSequenceEqual(list(model), data.domain.variables)
        asdefid = method_types.index(AsDefault)
        leaveid = method_types.index(impute.DoNotImpute)
        avgid = method_types.index(impute.Average)
        defbg.button(avgid).click()
        self.assertEqual(widget.default_method_index, avgid)

        self.assertTrue(
            all(isinstance(m, AsDefault) and isinstance(m.method, impute.Average)
                for m in map(widget.get_method_for_column,
                             range(len(data.domain.variables)))))

        # change method for first variable
        select_row(view, 0)
        varbg.button(avgid).click()
        met = widget.get_method_for_column(0)
        self.assertIsInstance(met, impute.Average)

        # select a second var
        selmodel = view.selectionModel()
        selmodel.select(model.index(2), selmodel.Select)
        # the current checked button must unset
        self.assertEqual(varbg.checkedId(), -1)

        varbg.button(leaveid).click()
        self.assertIsInstance(widget.get_method_for_column(0),
                              impute.DoNotImpute)
        self.assertIsInstance(widget.get_method_for_column(2),
                              impute.DoNotImpute)
        # reset both back to default
        varbg.button(asdefid).click()
        self.assertIsInstance(widget.get_method_for_column(0), AsDefault)
        self.assertIsInstance(widget.get_method_for_column(2), AsDefault)

    def test_value_edit(self):
        data = Table("heart_disease")[::10]
        self.send_signal("Data", data)
        widget = self.widget
        model = widget.varmodel
        view = widget.varview
        selmodel = view.selectionModel()
        varbg = widget.variable_button_group
        method_types = [type(m) for m in OWImpute.METHODS]
        asdefid = method_types.index(AsDefault)
        valueid = method_types.index(impute.Default)

        def selectvars(varlist, command=selmodel.ClearAndSelect):
            indices = [data.domain.index(var) for var in varlist]
            itemsel = QItemSelection()
            for ind in indices:
                midx = model.index(ind)
                itemsel.select(midx, midx)
            selmodel.select(itemsel, command)

        def effective_method(var):
            return widget.get_method_for_column(data.domain.index(var))

        # select 'chest pain'
        selectvars(['chest pain'])
        self.assertTrue(widget.value_combo.isVisibleTo(widget) and
                        widget.value_combo.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), asdefid)

        simulate.combobox_activate_item(
            widget.value_combo, data.domain["chest pain"].values[1])
        # The 'Value' (impute.Default) should have been selected automatically
        self.assertEqual(varbg.checkedId(), valueid)
        imputer = effective_method('chest pain')
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, 1)

        # select continuous 'rest SBP' and 'cholesterol' variables
        selectvars(["rest SBP", "cholesterol"])
        self.assertTrue(widget.value_double.isVisibleTo(widget) and
                        widget.value_double.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), asdefid)
        widget.value_double.setValue(-1.0)
        QTest.keyClick(self.widget.value_double, Qt.Key_Enter)
        # The 'Value' (impute.Default) should have been selected automatically
        self.assertEqual(varbg.checkedId(), valueid)
        imputer = effective_method("rest SBP")
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, -1.0)
        imputer = effective_method("cholesterol")
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, -1.0)

        # Add 'chest pain' to the selection and ensure the value stack is
        # disabled
        selectvars(["chest pain"], selmodel.Select)
        self.assertEqual(varbg.checkedId(), -1)
        self.assertFalse(widget.value_combo.isEnabledTo(widget) and
                         widget.value_double.isEnabledTo(widget))

        # select 'chest pain' only and check that the selected value is
        # restored in the value combo
        selectvars(["chest pain"])
        self.assertTrue(widget.value_combo.isVisibleTo(widget) and
                        widget.value_combo.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), valueid)
        self.assertEqual(widget.value_combo.currentIndex(), 1)

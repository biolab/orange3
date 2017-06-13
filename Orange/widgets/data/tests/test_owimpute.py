# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,pointless-statement,blacklisted-name
import numpy as np

from AnyQt.QtCore import Qt, QItemSelection
from AnyQt.QtTest import QTest

from Orange.data import Table, Domain
from Orange.preprocess import impute
from Orange.widgets.data.owimpute import OWImpute, AsDefault, Learner, Method
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_row


class Foo(Learner):
    def __call__(self, *args, **kwargs):
        1/0


class Bar:
    def __call__(self, args, **kwargs):
        1/0


class FooBar(Learner):
    def __call__(self, data, *args, **kwargs):
        bar = Bar()
        bar.domain = data.domain
        return bar


class TestOWImpute(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWImpute)  # type: OWImpute

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")[::3]
        widget = self.widget
        widget.default_method_index = Method.Model

        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        imp_data = self.get_output(self.widget.Outputs.data)
        np.testing.assert_equal(imp_data.X, data.X)
        np.testing.assert_equal(imp_data.Y, data.Y)

        self.send_signal(self.widget.Inputs.data, Table(data.domain), wait=1000)
        imp_data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(imp_data), 0)

        # only meta columns
        data = data.transform(Domain([], [], data.domain.attributes))
        self.send_signal("Data", data, wait=1000)
        imp_data = self.get_output("Data")
        self.assertEqual(len(imp_data), len(data))
        self.assertEqual(imp_data.domain, data.domain)
        np.testing.assert_equal(imp_data.metas, data.metas)

    def test_model_error(self):
        widget = self.widget
        widget.default_method_index = Method.Model
        data = Table("brown-selected")[::4][:, :4]
        self.send_signal(self.widget.Inputs.data, data, wait=1000)

        self.send_signal(self.widget.Inputs.learner, Foo(), wait=1000)  # Learner fails
        self.assertTrue(widget.Error.imputation_failed.is_shown())

        self.send_signal(self.widget.Inputs.learner, FooBar(), wait=1000)  # Model fails
        self.assertTrue(widget.Error.imputation_failed.is_shown())

    def test_select_method(self):
        data = Table("iris")[::5]

        self.send_signal(self.widget.Inputs.data, data)

        widget = self.widget
        model = widget.varmodel
        view = widget.varview
        defbg = widget.default_button_group
        varbg = widget.variable_button_group
        self.assertSequenceEqual(list(model), data.domain.variables)
        defbg.button(Method.Average).click()
        self.assertEqual(widget.default_method_index, Method.Average)

        self.assertTrue(
            all(isinstance(m, AsDefault) and isinstance(m.method, impute.Average)
                for m in map(widget.get_method_for_column,
                             range(len(data.domain.variables)))))

        # change method for first variable
        select_row(view, 0)
        varbg.button(Method.Average).click()
        met = widget.get_method_for_column(0)
        self.assertIsInstance(met, impute.Average)

        # select a second var
        selmodel = view.selectionModel()
        selmodel.select(model.index(2), selmodel.Select)
        # the current checked button must unset
        self.assertEqual(varbg.checkedId(), -1)

        varbg.button(Method.Leave).click()
        self.assertIsInstance(widget.get_method_for_column(0),
                              impute.DoNotImpute)
        self.assertIsInstance(widget.get_method_for_column(2),
                              impute.DoNotImpute)
        # reset both back to default
        varbg.button(Method.AsAboveSoBelow).click()
        self.assertIsInstance(widget.get_method_for_column(0), AsDefault)
        self.assertIsInstance(widget.get_method_for_column(2), AsDefault)

    def test_value_edit(self):
        data = Table("heart_disease")[::10]
        self.send_signal(self.widget.Inputs.data, data)
        widget = self.widget
        model = widget.varmodel
        view = widget.varview
        selmodel = view.selectionModel()
        varbg = widget.variable_button_group

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
        self.assertEqual(varbg.checkedId(), Method.AsAboveSoBelow)

        simulate.combobox_activate_item(
            widget.value_combo, data.domain["chest pain"].values[1])
        # The 'Value' (impute.Default) should have been selected automatically
        self.assertEqual(varbg.checkedId(), Method.Default)
        imputer = effective_method('chest pain')
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, 1)

        # select continuous 'rest SBP' and 'cholesterol' variables
        selectvars(["rest SBP", "cholesterol"])
        self.assertTrue(widget.value_double.isVisibleTo(widget) and
                        widget.value_double.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), Method.AsAboveSoBelow)
        widget.value_double.setValue(-1.0)
        QTest.keyClick(self.widget.value_double, Qt.Key_Enter)
        # The 'Value' (impute.Default) should have been selected automatically
        self.assertEqual(varbg.checkedId(), Method.Default)
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
        self.assertEqual(varbg.checkedId(), Method.Default)
        self.assertEqual(widget.value_combo.currentIndex(), 1)

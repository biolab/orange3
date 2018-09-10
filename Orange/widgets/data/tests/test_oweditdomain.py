# Test methods with long descriptive names can omit docstrings
# pylint: disable=all
from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from AnyQt.QtCore import QModelIndex, QItemSelectionModel, Qt
from AnyQt.QtWidgets import QAction
from AnyQt.QtTest import QTest

from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable,
    Table, Domain
)
from Orange.preprocess.transformation import Identity, Lookup
from Orange.widgets.data.oweditdomain import (
    OWEditDomain,
    ContinuousVariableEditor, DiscreteVariableEditor, VariableEditor,
    TimeVariableEditor, Categorical, Real, Time, String,
    Rename, Annotate, CategoriesMapping, report_transform,
    apply_transform
)
from Orange.widgets.data.owcolor import OWColor, ColorRole
from Orange.widgets.tests.base import WidgetTest, GuiTest


class TestReport(TestCase):
    def test_rename(self):
        var = Real("X", (-1, ""), ())
        tr = Rename("Y")
        val = report_transform(var, [tr])
        self.assertIn("X", val)
        self.assertIn("Y", val)

    def test_annotate(self):
        var = Real("X", (-1, ""), (("a", "1"), ("b", "z")))
        tr = Annotate((("a", "2"), ("j", "z")))
        r = report_transform(var, [tr])
        self.assertIn("a", r)
        self.assertIn("b", r)

    def test_categories_mapping(self):
        var = Categorical("C", ("a", "b", "c"), None, ())
        tr = CategoriesMapping(
            (("a", "aa"),
             ("b", None),
             ("c", "cc"),
             (None, "ee")),
        )
        r = report_transform(var, [tr])
        self.assertIn("a", r)
        self.assertIn("aa", r)
        self.assertIn("b", r)
        self.assertIn("<s>", r)


class TestOWEditDomain(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWEditDomain)
        self.iris = Table("iris")

    def test_input_data(self):
        """Check widget's data with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)

    def test_input_data_disconnect(self):
        """Check widget's data after disconnecting data on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)

    def test_output_data(self):
        """Check data on the output after apply"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(output.X, self.iris.X)
        np.testing.assert_array_equal(output.Y, self.iris.Y)
        self.assertEqual(output.domain, self.iris.domain)

    def test_input_from_owcolor(self):
        """Check widget's data sent from OWColor widget"""
        owcolor = self.create_widget(OWColor)
        self.send_signal("Data", self.iris, widget=owcolor)
        owcolor.disc_model.setData(QModelIndex(), (250, 97, 70, 255), ColorRole)
        owcolor.cont_model.setData(
            QModelIndex(), ((255, 80, 114, 255), (255, 255, 0, 255), False),
            ColorRole)
        owcolor_output = self.get_output("Data", owcolor)
        self.send_signal("Data", owcolor_output)
        self.assertEqual(self.widget.data, owcolor_output)
        self.assertIsNotNone(self.widget.data.domain.class_vars[-1].colors)

    def test_list_attributes_remain_lists(self):
        a = ContinuousVariable("a")
        a.attributes["list"] = [1, 2, 3]
        d = Domain([a])
        t = Table(d)

        self.send_signal(self.widget.Inputs.data, t)

        assert isinstance(self.widget, OWEditDomain)
        # select first variable
        idx = self.widget.domain_view.model().index(0)
        self.widget.domain_view.setCurrentIndex(idx)

        # change first attribute value
        editor = self.widget.editor_stack.findChild(ContinuousVariableEditor)
        assert isinstance(editor, ContinuousVariableEditor)
        idx = editor.labels_model.index(0, 1)
        editor.labels_model.setData(idx, "[1, 2, 4]", Qt.EditRole)

        self.widget.commit()
        t2 = self.get_output(self.widget.Outputs.data)
        self.assertEqual(t2.domain["a"].attributes["list"], [1, 2, 4])

    def test_duplicate_names(self):
        """
        Tests if widget shows error when duplicate name is entered.
        And tests if widget sends None data when error is shown.
        GH-2143
        GH-2146
        """
        table = Table("iris")
        self.send_signal(self.widget.Inputs.data, table)
        self.assertFalse(self.widget.Error.duplicate_var_name.is_shown())

        idx = self.widget.domain_view.model().index(0)
        self.widget.domain_view.setCurrentIndex(idx)
        editor = self.widget.editor_stack.findChild(ContinuousVariableEditor)

        def enter_text(widget, text):
            # type: (QLineEdit, str) -> None
            widget.selectAll()
            QTest.keyClick(widget, Qt.Key_Delete)
            QTest.keyClicks(widget, text)
            QTest.keyClick(widget, Qt.Key_Return)

        enter_text(editor.name_edit, "iris")
        self.widget.commit()
        self.assertTrue(self.widget.Error.duplicate_var_name.is_shown())
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(output)

        enter_text(editor.name_edit, "sepal height")
        self.widget.commit()
        self.assertFalse(self.widget.Error.duplicate_var_name.is_shown())
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(output, Table)

    def test_time_variable_preservation(self):
        """Test if time variables preserve format specific attributes"""
        table = Table("cyber-security-breaches")
        self.send_signal(self.widget.Inputs.data, table)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(str(table[0, 4]), str(output[0, 4]))
        view = self.widget.variables_view
        view.setCurrentIndex(view.model().index(4))

        editor = self.widget.editor_stack.findChild(TimeVariableEditor)
        editor.name_edit.setText("Date")
        editor.variable_changed.emit()
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(str(table[0, 4]), str(output[0, 4]))


class TestEditors(GuiTest):
    def test_variable_editor(self):
        w = VariableEditor()
        self.assertEqual(w.get_data(), (None, []))

        v = String("S", (("A", "1"), ("B", "b")))
        w.set_data(v, [])

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(),
                         {"A": "1", "B": "b"})
        self.assertEqual(w.get_data(), (v, []))

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertEqual(w.get_data(), (None, []))

        w.set_data(v, [Rename("T"), Annotate((("a", "1"), ("b", "2")))])
        self.assertEqual(w.name_edit.text(), "T")
        self.assertEqual(w.labels_model.rowCount(), 2)
        add = w.findChild(QAction, "action-add-label")
        add.trigger()
        remove = w.findChild(QAction, "action-delete-label")
        remove.trigger()

    def test_continuous_editor(self):
        w = ContinuousVariableEditor()
        self.assertEqual(w.get_data(), (None, []))

        v = Real("X", (-1, ""), (("A", "1"), ("B", "b")))
        w.set_data(v, [])

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), dict(v.annotations))

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertEqual(w.get_data(), (None, []))

    def test_discrete_editor(self):
        w = DiscreteVariableEditor()
        self.assertEqual(w.get_data(), (None, []))

        v = Categorical("C", ("a", "b", "c"), None,
                        (("A", "1"), ("B", "b")))
        w.set_data(v)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), dict(v.annotations))
        self.assertEqual(w.get_data(), (v, []))
        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertEqual(w.get_data(), (None, []))
        mapping = [
            ("c", "C"),
            ("a", "A"),
            ("b", None),
            (None, "b")
        ]
        w.set_data(v, [CategoriesMapping(mapping)])
        w.grab()  # run delegate paint method
        self.assertEqual(w.get_data(), (v, [CategoriesMapping(mapping)]))

        # test selection/deselection in the view
        w.set_data(v)
        view = w.values_edit
        model = view.model()
        assert model.rowCount()
        sel_model = view.selectionModel()
        model = sel_model.model()
        sel_model.select(model.index(0, 0), QItemSelectionModel.Select)
        sel_model.select(model.index(0, 0), QItemSelectionModel.Deselect)

    def test_time_editor(self):
        w = TimeVariableEditor()
        self.assertEqual(w.get_data(), (None, []))

        v = Time("T", (("A", "1"), ("B", "b")))
        w.set_data(v,)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), dict(v.annotations))

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertEqual(w.get_data(), (None, []))


class TestTransforms(TestCase):
    def _test_common(self, var):
        tr = [Rename(var.name + "_copy"), Annotate((("A", "1"),))]
        XX = apply_transform(var, tr)
        self.assertEqual(XX.name, var.name + "_copy")
        self.assertEqual(XX.attributes, {"A": 1})
        self.assertIsInstance(XX.compute_value, Identity)
        self.assertIs(XX.compute_value.variable, var)

    def test_continous(self):
        X = ContinuousVariable("X")
        self._test_common(X)

    def test_string(self):
        X = StringVariable("S")
        self._test_common(X)

    def test_time(self):
        X = TimeVariable("X")
        self._test_common(X)

    def test_discrete(self):
        D = DiscreteVariable("D", values=("a", "b"))
        self._test_common(D)

    def test_discrete_rename(self):
        D = DiscreteVariable("D", values=("a", "b"))
        DD = apply_transform(D, [CategoriesMapping((("a", "A"), ("b", "B")))])
        self.assertSequenceEqual(DD.values, ["A", "B"])
        self.assertIs(DD.compute_value.variable, D)

    def test_discrete_reorder(self):
        D = DiscreteVariable("D", values=("2", "3", "1", "0"))
        DD = apply_transform(D, [CategoriesMapping((("0", "0"), ("1", "1"),
                                                    ("2", "2"), ("3", "3")))])
        self.assertSequenceEqual(DD.values, ["0", "1", "2", "3"])
        self._assertLookupEquals(
            DD.compute_value, Lookup(D, np.array([2, 3, 1, 0]))
        )

    def test_discrete_add_drop(self):
        D = DiscreteVariable("D", values=("2", "3", "1", "0"), base_value=1)
        mapping = (
            ("0", None),
            ("1", "1"),
            ("2", "2"),
            ("3", None),
            (None, "A"),
        )
        tr = [CategoriesMapping(mapping)]
        DD = apply_transform(D, tr)
        self.assertSequenceEqual(DD.values, ["1", "2", "A"])
        self._assertLookupEquals(
            DD.compute_value, Lookup(D, np.array([1, np.nan, 0, np.nan]))
        )
        self.assertEqual(DD.base_value, -1)

    def _assertLookupEquals(self, first, second):
        self.assertIsInstance(first, Lookup)
        self.assertIsInstance(second, Lookup)
        self.assertIs(first.variable, second.variable)
        assert_array_equal(first.lookup_table, second.lookup_table)
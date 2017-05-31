# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from unittest import TestCase
import numpy as np

from AnyQt.QtCore import QModelIndex, Qt

from Orange.data import ContinuousVariable, DiscreteVariable, \
    StringVariable, Table, Domain
from Orange.widgets.data.oweditdomain import EditDomainReport, OWEditDomain, \
    ContinuousVariableEditor, DiscreteVariableEditor, VariableEditor
from Orange.widgets.data.owcolor import OWColor, ColorRole
from Orange.widgets.tests.base import WidgetTest, GuiTest

SECTION_NAME = "NAME"


class TestEditDomainReport(TestCase):
    # This tests test private methods
    # pylint: disable=protected-access

    def setUp(self):
        self.report = EditDomainReport([], [])

    def test_section_yields_nothing_for_no_changes(self):
        result = self.report._section(SECTION_NAME, [])
        self.assertEmpty(result)

    def test_section_yields_header_for_changes(self):
        result = self.report._section(SECTION_NAME, ["a"])
        self.assertTrue(any(SECTION_NAME in item for item in result))

    def test_value_changes_yields_nothing_for_no_change(self):
        a = DiscreteVariable("a", values="abc")
        self.assertEmpty(self.report._value_changes(a, a))

    def test_value_changes_yields_nothing_for_continuous_variables(self):
        v1, v2 = ContinuousVariable("a"), ContinuousVariable("b")
        self.assertEmpty(self.report._value_changes(v1, v2))

    def test_value_changes_yields_changed_values(self):
        v1, v2 = DiscreteVariable("a", "ab"), DiscreteVariable("b", "ac")
        self.assertNotEmpty(self.report._value_changes(v1, v2))

    def test_label_changes_yields_nothing_for_no_change(self):
        v1 = ContinuousVariable("a")
        v1.attributes["a"] = "b"
        self.assertEmpty(self.report._value_changes(v1, v1))

    def test_label_changes_yields_added_labels(self):
        v1 = ContinuousVariable("a")
        v2 = v1.copy(None)
        v2.attributes["a"] = "b"
        self.assertNotEmpty(self.report._label_changes(v1, v2))

    def test_label_changes_yields_removed_labels(self):
        v1 = ContinuousVariable("a")
        v1.attributes["a"] = "b"
        v2 = v1.copy(None)
        del v2.attributes["a"]
        self.assertNotEmpty(self.report._label_changes(v1, v2))

    def test_label_changes_yields_modified_labels(self):
        v1 = ContinuousVariable("a")
        v1.attributes["a"] = "b"
        v2 = v1.copy(None)
        v2.attributes["a"] = "c"
        self.assertNotEmpty(self.report._label_changes(v1, v2))

    def assertEmpty(self, iterable):
        self.assertRaises(StopIteration, lambda: next(iter(iterable)))

    def assertNotEmpty(self, iterable):
        try:
            next(iter(iterable))
        except StopIteration:
            self.fail("Iterator did not produce any lines")


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

        self.widget.unconditional_commit()
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

        editor.name_edit.setText("iris")
        editor.commit()
        self.widget.commit()
        self.assertTrue(self.widget.Error.duplicate_var_name.is_shown())
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(output)

        editor.name_edit.setText("sepal height")
        editor.commit()
        self.widget.commit()
        self.assertFalse(self.widget.Error.duplicate_var_name.is_shown())
        output = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(output, Table)


class TestEditors(GuiTest):
    def test_variable_editor(self):
        w = VariableEditor()
        self.assertIs(w.get_data(), None)

        v = StringVariable(name="S")
        v.attributes.update({"A": 1, "B": "b"},)
        w.set_data(v)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), v.attributes)
        self.assertTrue(w.is_same())

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertIs(w.get_data(), None)

    def test_continuous_editor(self):
        w = ContinuousVariableEditor()
        self.assertIs(w.get_data(), None)

        v = ContinuousVariable("X", number_of_decimals=5)
        v.attributes.update({"A": 1, "B": "b"})
        w.set_data(v)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), v.attributes)
        self.assertTrue(w.is_same())

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertIs(w.get_data(), None)

    def test_discrete_editor(self):
        w = DiscreteVariableEditor()
        self.assertIs(w.get_data(), None)

        v = DiscreteVariable("C", values=["a", "b", "c"])
        v.attributes.update({"A": 1, "B": "b"})
        w.set_data(v)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), v.attributes)
        self.assertTrue(w.is_same())

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertIs(w.get_data(), None)

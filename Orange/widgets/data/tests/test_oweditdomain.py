# Test methods with long descriptive names can omit docstrings
# pylint: disable=all
import pickle
from itertools import product
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from AnyQt.QtCore import QItemSelectionModel, Qt, QItemSelection
from AnyQt.QtWidgets import QAction, QComboBox, QLineEdit, QStyleOptionViewItem
from AnyQt.QtTest import QTest, QSignalSpy

from Orange.widgets.utils import colorpalettes
from orangewidget.tests.utils import simulate
from orangewidget.utils.itemmodels import PyListModel

from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable,
    Table, Domain
)
from Orange.preprocess.transformation import Identity, Lookup
from Orange.widgets.data.oweditdomain import (
    OWEditDomain,
    ContinuousVariableEditor, DiscreteVariableEditor, VariableEditor,
    TimeVariableEditor, Categorical, Real, Time, String,
    Rename, Annotate, CategoriesMapping, ChangeOrdered, report_transform,
    apply_transform, apply_transform_var, apply_reinterpret, MultiplicityRole,
    AsString, AsCategorical, AsContinuous, AsTime,
    table_column_data, ReinterpretVariableEditor, CategoricalVector,
    VariableEditDelegate, TransformRole,
    RealVector, TimeVector, StringVector, make_dict_mapper, DictMissingConst,
    LookupMappingTransform, as_float_or_nan, column_str_repr
)
from Orange.widgets.data.owcolor import OWColor, ColorRole
from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.tests import test_filename, assert_array_nanequal
from Orange.widgets.utils.state_summary import format_summary_details

MArray = np.ma.MaskedArray


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
        var = Categorical("C", ("a", "b", "c"), ())
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

    def test_categorical_merge_mapping(self):
        var = Categorical("C", ("a", "b1", "b2"), ())
        tr = CategoriesMapping(
            (("a", "a"),
             ("b1", "b"),
             ("b2", "b"),
             (None, "c")),
        )
        r = report_transform(var, [tr])
        self.assertIn('b', r)

    def test_change_ordered(self):
        var = Categorical("C", ("a", "b"), ())
        tr = ChangeOrdered(True)
        r = report_transform(var, [tr])
        self.assertIn("ordered", r)

    def test_reinterpret(self):
        var = String("T", ())
        for tr in (AsContinuous(), AsCategorical(), AsTime()):
            t = report_transform(var, [tr])
            self.assertIn("→ (", t)


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
        disc_model = owcolor.disc_model
        disc_model.setData(disc_model.index(0, 1), (1, 2, 3), ColorRole)
        cont_model = owcolor.cont_model
        palette = list(colorpalettes.ContinuousPalettes.values())[-1]
        cont_model.setData(cont_model.index(1, 1), palette, ColorRole)
        owcolor_output = self.get_output("Data", owcolor)
        self.send_signal("Data", owcolor_output)
        self.assertEqual(self.widget.data, owcolor_output)
        np.testing.assert_equal(self.widget.data.domain.class_var.colors[0],
                                (1, 2, 3))
        self.assertIs(self.widget.data.domain.attributes[1].palette, palette)

    def test_list_attributes_remain_lists(self):
        a = ContinuousVariable("a")
        a.attributes["list"] = [1, 2, 3]
        d = Domain([a])
        t = Table.from_domain(d)

        self.send_signal(self.widget.Inputs.data, t)

        assert isinstance(self.widget, OWEditDomain)
        # select first variable
        idx = self.widget.domain_view.model().index(0)
        self.widget.domain_view.setCurrentIndex(idx)

        # change first attribute value
        editor = self.widget.findChild(ContinuousVariableEditor)
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
        editor = self.widget.findChild(ContinuousVariableEditor)

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
        table = Table(test_filename("datasets/cyber-security-breaches.tab"))
        self.send_signal(self.widget.Inputs.data, table)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(str(table[0, 4]), str(output[0, 4]))
        view = self.widget.variables_view
        view.setCurrentIndex(view.model().index(4))

        editor = self.widget.findChild(TimeVariableEditor)
        editor.name_edit.setText("Date")
        editor.variable_changed.emit()
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(str(table[0, 4]), str(output[0, 4]))

    def test_change_ordered(self):
        """Test categorical ordered flag change"""
        table = Table.from_domain(Domain(
            [DiscreteVariable("A", values=("a", "b"), ordered=True)]))
        self.send_signal(self.widget.Inputs.data, table)
        output = self.get_output(self.widget.Outputs.data)
        self.assertTrue(output.domain[0].ordered)

        editor = self.widget.findChild(DiscreteVariableEditor)
        assert isinstance(editor, DiscreteVariableEditor)
        editor.ordered_cb.setChecked(False)
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertFalse(output.domain[0].ordered)

    def test_restore(self):
        iris = self.iris
        viris = (
            "Categorical",
            ("iris", ("Iris-setosa", "Iris-versicolor", "Iris-virginica"), ())
        )
        w = self.widget

        def restore(state):
            w._domain_change_store = state
            w._restore()

        model = w.variables_model
        self.send_signal(w.Inputs.data, iris, widget=w)
        restore({viris: [("Rename", ("Z",))]})
        tr = model.data(model.index(4), TransformRole)
        self.assertEqual(tr, [Rename("Z")])

        restore({viris: [("AsString", ()), ("Rename", ("Z",))]})
        tr = model.data(model.index(4), TransformRole)
        self.assertEqual(tr, [AsString(), Rename("Z")])

    def test_summary(self):
        """Check if status bar is updated when data is received"""
        data = Table("iris")
        input_sum = self.widget.info.set_input_summary = Mock()
        output_sum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(len(data), format_summary_details(data))
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))

        def enter_text(widget, text):
            # type: (QLineEdit, str) -> None
            widget.selectAll()
            QTest.keyClick(widget, Qt.Key_Delete)
            QTest.keyClicks(widget, text)
            QTest.keyClick(widget, Qt.Key_Return)

        editor = self.widget.findChild(ContinuousVariableEditor)
        enter_text(editor.name_edit, "sepal height")
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))
        output_sum.reset_mock()
        enter_text(editor.name_edit, "sepal width")
        self.widget.commit()
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")

        input_sum.reset_mock()
        output_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertEqual(input_sum.call_args[0][0].brief, "")
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")


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

        v = Categorical("C", ("a", "b", "c"), (("A", "1"), ("B", "b")))
        w.set_data(v)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertFalse(w.ordered_cb.isChecked())
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

        w.set_data(v, [CategoriesMapping(mapping), ChangeOrdered(True)])
        self.assertTrue(w.ordered_cb.isChecked())
        self.assertEqual(
            w.get_data()[1], [CategoriesMapping(mapping), ChangeOrdered(True)]
        )
        # test selection/deselection in the view
        w.set_data(v)
        view = w.values_edit
        model = view.model()
        assert model.rowCount()
        sel_model = view.selectionModel()
        model = sel_model.model()
        sel_model.select(model.index(0, 0), QItemSelectionModel.Select)
        sel_model.select(model.index(0, 0), QItemSelectionModel.Deselect)

        # merge mapping
        mapping = [
            ("a", "a"),
            ("b", "b"),
            ("c", "b")
        ]
        w.set_data(v, [CategoriesMapping(mapping)])
        self.assertEqual(w.get_data()[1], [CategoriesMapping(mapping)])
        self.assertEqual(model.data(model.index(0, 0), MultiplicityRole), 1)
        self.assertEqual(model.data(model.index(1, 0), MultiplicityRole), 2)
        self.assertEqual(model.data(model.index(2, 0), MultiplicityRole), 2)
        w.grab()
        model.setData(model.index(0, 0), "b", Qt.EditRole)
        self.assertEqual(model.data(model.index(0, 0), MultiplicityRole), 3)
        self.assertEqual(model.data(model.index(1, 0), MultiplicityRole), 3)
        self.assertEqual(model.data(model.index(2, 0), MultiplicityRole), 3)
        w.grab()

    def test_discrete_editor_add_remove_action(self):
        w = DiscreteVariableEditor()
        v = Categorical("C", ("a", "b", "c"),
                        (("A", "1"), ("B", "b")))
        w.set_data(v)
        action_add = w.add_new_item
        action_remove = w.remove_item
        view = w.values_edit
        model, selection = view.model(), view.selectionModel()
        selection.clear()

        action_add.trigger()
        self.assertTrue(view.state() == view.EditingState)
        editor = view.focusWidget()
        assert isinstance(editor, QLineEdit)
        spy = QSignalSpy(model.dataChanged)
        QTest.keyClick(editor, Qt.Key_D)
        QTest.keyClick(editor, Qt.Key_Return)
        self.assertTrue(model.rowCount() == 4)
        # The commit to model is executed via a queued invoke
        self.assertTrue(bool(spy) or spy.wait())
        self.assertEqual(model.index(3, 0).data(Qt.EditRole), "d")
        # remove it
        spy = QSignalSpy(model.rowsRemoved)
        action_remove.trigger()
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(len(spy), 1)
        _, first, last = spy[0]
        self.assertEqual((first, last), (3, 3))
        # remove/drop and existing value
        selection.select(model.index(1, 0), QItemSelectionModel.ClearAndSelect)
        removespy = QSignalSpy(model.rowsRemoved)
        changedspy = QSignalSpy(model.dataChanged)
        action_remove.trigger()
        self.assertEqual(len(removespy), 0, "Should only mark item as removed")
        self.assertGreaterEqual(len(changedspy), 1, "Did not change data")
        w.grab()

    def test_discrete_editor_merge_action(self):
        w = DiscreteVariableEditor()
        v = Categorical("C", ("a", "b", "c"),
                        (("A", "1"), ("B", "b")))
        w.set_data(v)
        action = w.merge_items
        self.assertFalse(action.isEnabled())
        view = w.values_edit
        model = view.model()
        selmodel = view.selectionModel()  # type: QItemSelectionModel
        selmodel.select(
            QItemSelection(model.index(0, 0), model.index(1, 0)),
            QItemSelectionModel.ClearAndSelect
        )
        self.assertTrue(action.isEnabled())
        # trigger the action, then find the active popup, and simulate entry
        spy = QSignalSpy(w.variable_changed)
        w.merge_items.trigger()
        cb = w.findChild(QComboBox)
        cb.setCurrentText("BA")
        cb.activated[str].emit("BA")
        cb.close()
        self.assertEqual(model.index(0, 0).data(Qt.EditRole), "BA")
        self.assertEqual(model.index(1, 0).data(Qt.EditRole), "BA")

        self.assertSequenceEqual(
            list(spy), [[]], 'variable_changed should emit exactly once'
        )

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

    DataVectors = [
        CategoricalVector(
            Categorical("A", ("a", "aa"), ()), lambda:
                MArray([0, 1, 2], mask=[False, False, True])
        ),
        RealVector(
            Real("B", (6, "f"), ()), lambda:
                MArray([0.1, 0.2, 0.3], mask=[True, False, True])
        ),
        TimeVector(
            Time("T", ()), lambda:
                MArray([0, 100, 200], dtype="M8[us]", mask=[True, False, True])
        ),
        StringVector(
            String("S", ()), lambda:
                MArray(["0", "1", "2"], dtype=object, mask=[True, False, True])
        ),
    ]
    ReinterpretTransforms = {
        Categorical: AsCategorical, Real: AsContinuous, Time: AsTime,
        String: AsString
    }

    def test_reinterpret_editor(self):
        w = ReinterpretVariableEditor()
        self.assertEqual(w.get_data(), (None, []))
        data = self.DataVectors[0]
        w.set_data(data, )
        self.assertEqual(w.get_data(), (data.vtype, []))
        w.set_data(data, [Rename("Z")])
        self.assertEqual(w.get_data(), (data.vtype, [Rename("Z")]))

        for vec, tr in product(self.DataVectors, self.ReinterpretTransforms.values()):
            w.set_data(vec, [tr()])
            v, tr_ = w.get_data()
            self.assertEqual(v, vec.vtype)
            if not tr_:
                self.assertEqual(tr, self.ReinterpretTransforms[type(v)])
            else:
                self.assertEqual(tr_, [tr()])

    def test_reinterpret_editor_simulate(self):
        w = ReinterpretVariableEditor()
        tc = w.findChild(QComboBox, name="type-combo")

        def cb():
            var, tr = w.get_data()
            type_ = tc.currentData()
            if type_ is not type(var):
                self.assertEqual(tr, [self.ReinterpretTransforms[type_](), Rename("Z")])
            else:
                self.assertEqual(tr, [Rename("Z")])

        for vec in self.DataVectors:
            w.set_data(vec, [Rename("Z")])
            simulate.combobox_run_through_all(tc, callback=cb)


class TestDelegates(GuiTest):
    def test_delegate(self):
        model = PyListModel([None])

        def set_item(v: dict):
            model.setItemData(model.index(0),  v)

        def get_style_option() -> QStyleOptionViewItem:
            opt = QStyleOptionViewItem()
            delegate.initStyleOption(opt, model.index(0))
            return opt

        set_item({Qt.EditRole: Categorical("a", (), ())})
        delegate = VariableEditDelegate()
        opt = get_style_option()
        self.assertEqual(opt.text, "a")
        self.assertFalse(opt.font.italic())
        set_item({TransformRole: [Rename("b")]})
        opt = get_style_option()
        self.assertEqual(opt.text, "a \N{RIGHTWARDS ARROW} b")
        self.assertTrue(opt.font.italic())

        set_item({TransformRole: [AsString()]})
        opt = get_style_option()
        self.assertIn("reinterpreted", opt.text)
        self.assertTrue(opt.font.italic())


class TestTransforms(TestCase):
    def _test_common(self, var):
        tr = [Rename(var.name + "_copy"), Annotate((("A", "1"),))]
        XX = apply_transform_var(var, tr)
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
        DD = apply_transform_var(D, [CategoriesMapping((("a", "A"), ("b", "B")))])
        self.assertSequenceEqual(DD.values, ["A", "B"])
        self.assertIs(DD.compute_value.variable, D)

    def test_discrete_reorder(self):
        D = DiscreteVariable("D", values=("2", "3", "1", "0"))
        DD = apply_transform_var(D, [CategoriesMapping((("0", "0"), ("1", "1"),
                                                    ("2", "2"), ("3", "3")))])
        self.assertSequenceEqual(DD.values, ["0", "1", "2", "3"])
        self._assertLookupEquals(
            DD.compute_value, Lookup(D, np.array([2, 3, 1, 0]))
        )

    def test_ordered_change(self):
        D = DiscreteVariable("D", values=("a", "b"), ordered=True)
        Do = apply_transform_var(D, [ChangeOrdered(False)])
        self.assertFalse(Do.ordered)

    def test_discrete_add_drop(self):
        D = DiscreteVariable("D", values=("2", "3", "1", "0"))
        mapping = (
            ("0", None),
            ("1", "1"),
            ("2", "2"),
            ("3", None),
            (None, "A"),
        )
        tr = [CategoriesMapping(mapping)]
        DD = apply_transform_var(D, tr)
        self.assertSequenceEqual(DD.values, ["1", "2", "A"])
        self._assertLookupEquals(
            DD.compute_value, Lookup(D, np.array([1, np.nan, 0, np.nan]))
        )

    def test_discrete_merge(self):
        D = DiscreteVariable("D", values=("2", "3", "1", "0"))
        mapping = (
            ("0", "x"),
            ("1", "y"),
            ("2", "x"),
            ("3", "y"),
        )
        tr = [CategoriesMapping(mapping)]
        DD = apply_transform_var(D, tr)
        self.assertSequenceEqual(DD.values, ["x", "y"])
        self._assertLookupEquals(
            DD.compute_value, Lookup(D, np.array([0, 1, 1, 0]))
        )

    def _assertLookupEquals(self, first, second):
        self.assertIsInstance(first, Lookup)
        self.assertIsInstance(second, Lookup)
        self.assertIs(first.variable, second.variable)
        assert_array_equal(first.lookup_table, second.lookup_table)


class TestReinterpretTransforms(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        domain = Domain([
            DiscreteVariable("A", values=("a", "b", "c")),
            DiscreteVariable("B", values=("0", "1", "2")),
            ContinuousVariable("C"),
            TimeVariable("D", have_time=True),
        ],
        metas=[
            StringVariable("S")
        ])
        cls.data = Table.from_list(
            domain, [
                [0, 2, 0.25, 180],
                [1, 1, 1.25, 360],
                [2, 0, 0.20, 720],
                [1, 0, 0.00, 000],
            ]
        )
        cls.data_str = Table.from_list(
            Domain([], [], metas=[
                StringVariable("S"),
                StringVariable("T")
            ]),
            [["0.1", "2010"],
             ["1.0", "2020"]]
        )

    def test_as_string(self):
        table = self.data
        domain = table.domain

        tr = AsString()
        dtr = []
        for v in domain.variables:
            vtr = apply_reinterpret(v, tr, table_column_data(table, v))
            dtr.append(vtr)
        ttable = table.transform(Domain([], [], dtr))
        assert_array_equal(
            ttable.metas,
            np.array([
                ["a", "2", "0.25", "00:03:00"],
                ["b", "1", "1.25", "00:06:00"],
                ["c", "0", "0.2", "00:12:00"],
                ["b", "0", "0.0", "00:00:00"],
            ], dtype=object)
        )

    def test_as_discrete(self):
        table = self.data
        domain = table.domain

        tr = AsCategorical()
        dtr = []
        for v in domain.variables:
            vtr = apply_reinterpret(v, tr, table_column_data(table, v))
            dtr.append(vtr)
        tdomain = Domain(dtr)
        ttable = table.transform(tdomain)
        assert_array_equal(
            ttable.X,
            np.array([
                [0, 2, 2, 1],
                [1, 1, 3, 2],
                [2, 0, 1, 3],
                [1, 0, 0, 0],
            ], dtype=float)
        )
        self.assertEqual(tdomain["A"].values, ("a", "b", "c"))
        self.assertEqual(tdomain["B"].values, ("0", "1", "2"))
        self.assertEqual(tdomain["C"].values, ("0.0", "0.2", "0.25", "1.25"))
        self.assertEqual(
            tdomain["D"].values,
            ("1970-01-01 00:00:00", "1970-01-01 00:03:00",
             "1970-01-01 00:06:00", "1970-01-01 00:12:00")
        )

    def test_as_continuous(self):
        table = self.data
        domain = table.domain

        tr = AsContinuous()
        dtr = []
        for v in domain.variables:
            vtr = apply_reinterpret(v, tr, table_column_data(table, v))
            dtr.append(vtr)
        ttable = table.transform(Domain(dtr))
        assert_array_equal(
            ttable.X,
            np.array([
                [np.nan, 2, 0.25, 180],
                [np.nan, 1, 1.25, 360],
                [np.nan, 0, 0.20, 720],
                [np.nan, 0, 0.00, 000],
            ], dtype=float)
        )

    def test_as_time(self):
        table = self.data
        domain = table.domain

        tr = AsTime()
        dtr = []
        for v in domain.variables:
            vtr = apply_reinterpret(v, tr, table_column_data(table, v))
            dtr.append(vtr)

        ttable = table.transform(Domain(dtr))
        assert_array_equal(
            ttable.X,
            np.array([
                [np.nan, np.nan, 0.25, 180],
                [np.nan, np.nan, 1.25, 360],
                [np.nan, np.nan, 0.20, 720],
                [np.nan, np.nan, 0.00, 000],
            ], dtype=float)
        )

    def test_reinterpret_string(self):
        table = self.data_str
        domain = table.domain
        tvars = []
        for v in domain.metas:
            for tr in [AsContinuous(), AsCategorical(), AsTime(), AsString()]:
                tr = apply_reinterpret(v, tr, table_column_data(table, v))
                tvars.append(tr)
        tdomain = Domain([], metas=tvars)
        ttable = table.transform(tdomain)
        assert_array_nanequal(
            ttable.metas,
            np.array([
                [0.1, 0., np.nan, "0.1", 2010., 0., 1262304000., "2010"],
                [1.0, 1., np.nan, "1.0", 2020., 1., 1577836800., "2020"],
            ], dtype=object)
        )

    def test_compound_transform(self):
        table = self.data_str
        domain = table.domain
        v1 = domain.metas[0]
        v1.attributes["a"] = "a"
        tv1 = apply_transform(v1, table, [AsContinuous(), Rename("Z1")])
        tv2 = apply_transform(v1, table, [AsContinuous(), Rename("Z2"), Annotate((("a", "b"),))])

        self.assertIsInstance(tv1, ContinuousVariable)
        self.assertEqual(tv1.name, "Z1")
        self.assertEqual(tv1.attributes, {"a": "a"})

        self.assertIsInstance(tv2, ContinuousVariable)
        self.assertEqual(tv2.name, "Z2")
        self.assertEqual(tv2.attributes, {"a": "b"})

        tdomain = Domain([], metas=[tv1, tv2])
        ttable = table.transform(tdomain)

        assert_array_nanequal(
            ttable.metas,
            np.array([
                [0.1, 0.1],
                [1.0, 1.0],
            ], dtype=object)
        )

    def test_null_transform(self):
        table = self.data_str
        domain = table.domain
        v = apply_transform(domain.metas[0],table, [])
        self.assertIs(v, domain.metas[0])


class TestUtils(TestCase):
    def test_mapper(self):
        mapper = make_dict_mapper({"a": 1, "b": 2})
        r = mapper(["a", "a", "b"])
        assert_array_equal(r, [1, 1, 2])
        self.assertEqual(r.dtype, np.dtype("O"))
        r = mapper(["a", "a", "b"], dtype=float)
        assert_array_equal(r, [1, 1, 2])
        self.assertEqual(r.dtype, np.dtype(float))
        r = mapper(["a", "a", "b"], dtype=int)
        self.assertEqual(r.dtype, np.dtype(int))

        mapper = make_dict_mapper({"a": 1, "b": 2}, dtype=int)
        r = mapper(["a", "a", "b"])
        self.assertEqual(r.dtype, np.dtype(int))

        r = np.full(3, -1, dtype=float)
        r_ = mapper(["a", "a", "b"], out=r)
        self.assertIs(r, r_)
        assert_array_equal(r, [1, 1, 2])

    def test_dict_missing(self):
        d = DictMissingConst("<->", {1: 1, 2: 2})
        self.assertEqual(d[1], 1)
        self.assertEqual(d[-1], "<->")
        # must be sufficiently different from defaultdict to warrant existence
        self.assertEqual(d, {1: 1, 2: 2})

    def test_as_float_or_nan(self):
        a = np.array(["a", "1.1", ".2", "NaN"], object)
        r = as_float_or_nan(a)
        assert_array_equal(r, [np.nan, 1.1, .2, np.nan])

        a = np.array([1, 2, 3], dtype=int)
        r = as_float_or_nan(a)
        assert_array_equal(r, [1., 2., 3.])

        r = as_float_or_nan(r, dtype=np.float32)
        assert_array_equal(r, [1., 2., 3.])
        self.assertEqual(r.dtype, np.dtype(np.float32))

    def test_column_str_repr(self):
        v = StringVariable("S")
        d = column_str_repr(v, np.array(["A", "", "B"]))
        assert_array_equal(d, ["A", "?", "B"])
        v = ContinuousVariable("C")
        d = column_str_repr(v, np.array([0.1, np.nan, 1.0]))
        assert_array_equal(d, ["0.1", "?", "1"])
        v = DiscreteVariable("D", ("a", "b"))
        d = column_str_repr(v, np.array([0., np.nan, 1.0]))
        assert_array_equal(d, ["a", "?", "b"])
        v = TimeVariable("T", have_date=False, have_time=True)
        d = column_str_repr(v, np.array([0., np.nan, 1.0]))
        assert_array_equal(d, ["00:00:00", "?", "00:00:01"])


class TestLookupMappingTransform(TestCase):
    def setUp(self) -> None:
        self.lookup = LookupMappingTransform(
            StringVariable("S"),
            DictMissingConst(np.nan, {"": np.nan, "a": 0, "b": 1}),
            dtype=float,
        )

    def test_transform(self):
        r = self.lookup.transform(np.array(["", "a", "b", "c"]))
        assert_array_equal(r, [np.nan, 0, 1, np.nan])

    def test_pickle(self):
        lookup = self.lookup
        lookup_ = pickle.loads(pickle.dumps(lookup))
        c = np.array(["", "a", "b", "c"])
        r = lookup.transform(c)
        assert_array_equal(r, [np.nan, 0, 1, np.nan])
        r_ = lookup_.transform(c)
        assert_array_equal(r_, [np.nan, 0, 1, np.nan])

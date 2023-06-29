# Test methods with long descriptive names can omit docstrings
# pylint: disable=all
import pickle
import unittest
from functools import partial
from itertools import product, chain
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal

from AnyQt.QtCore import QItemSelectionModel, Qt, QItemSelection, QPoint
from AnyQt.QtGui import QPalette, QColor, QHelpEvent
from AnyQt.QtWidgets import QAction, QComboBox, QLineEdit, \
    QStyleOptionViewItem, QDialog, QMenu, QToolTip, QListView
from AnyQt.QtTest import QTest, QSignalSpy

from orangewidget.settings import Context
from orangewidget.tests.utils import simulate

from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable,
    Table, Domain
)
from Orange.preprocess.transformation import Identity, Lookup
from Orange.widgets.data.oweditdomain import (
    OWEditDomain,
    ContinuousVariableEditor, DiscreteVariableEditor, VariableEditor,
    TimeVariableEditor, Categorical, Real, Time, String,
    Rename, Annotate, Unlink, CategoriesMapping, report_transform,
    apply_transform, apply_transform_var, apply_reinterpret, MultiplicityRole,
    AsString, AsCategorical, AsContinuous, AsTime,
    table_column_data, ReinterpretVariableEditor, CategoricalVector,
    VariableEditDelegate, TransformRole,
    RealVector, TimeVector, StringVector, make_dict_mapper,
    LookupMappingTransform, as_float_or_nan, column_str_repr,
    GroupItemsDialog, VariableListModel, StrpTime, RestoreOriginal, BaseEditor
)
from Orange.widgets.data.owcolor import OWColor, ColorRole
from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.widgets.tests.utils import contextMenu
from Orange.widgets.utils import colorpalettes
from Orange.tests import test_filename, assert_array_nanequal

MArray = np.ma.MaskedArray


class TestReport(TestCase):
    def test_rename(self):
        var = Real("X", (-1, ""), (), False)
        tr = Rename("Y")
        val = report_transform(var, [tr])
        self.assertIn("X", val)
        self.assertIn("Y", val)

    def test_annotate(self):
        var = Real("X", (-1, ""), (("a", "1"), ("b", "z")), False)
        tr = Annotate((("a", "2"), ("j", "z")))
        r = report_transform(var, [tr])
        self.assertIn("a", r)
        self.assertIn("b", r)

    def test_unlinke(self):
        var = Real("X", (-1, ""), (("a", "1"), ("b", "z")), True)
        r = report_transform(var, [Unlink()])
        self.assertIn("unlinked", r)

    def test_categories_mapping(self):
        var = Categorical("C", ("a", "b", "c"), (), False)
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
        var = Categorical("C", ("a", "b1", "b2"), (), False)
        tr = CategoriesMapping(
            (("a", "a"),
             ("b1", "b"),
             ("b2", "b"),
             (None, "c")),
        )
        r = report_transform(var, [tr])
        self.assertIn('b', r)

    def test_reinterpret(self):
        var = String("T", (), False)
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

    def test_widget_state(self):
        """Check if widget clears its state when the input is disconnected"""
        editor = self.widget.findChild(ContinuousVariableEditor)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(editor.name_edit.text(), "sepal length")
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(editor.name_edit.text(), "")
        self.assertEqual(
            self.widget.variables_model.index(0).data(Qt.EditRole), None)

        editor = self.widget.findChild(DiscreteVariableEditor)
        self.send_signal(self.widget.Inputs.data, self.iris)
        index = self.widget.domain_view.model().index(4)
        self.widget.variables_view.setCurrentIndex(index)
        self.assertEqual(editor.name_edit.text(), "iris")
        self.assertEqual(editor.labels_model.get_dict(), {})
        self.assertNotEqual(
            self.widget.variables_model.index(0).data(Qt.EditRole), None)
        model = editor.values_edit.selectionModel().model()
        self.assertEqual(model.index(0).data(Qt.EditRole), "Iris-setosa")
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(editor.name_edit.text(), "")
        self.assertEqual(editor.labels_model.get_dict(), {})
        self.assertEqual(model.index(0).data(Qt.EditRole), None)
        self.assertEqual(
            self.widget.variables_model.index(0).data(Qt.EditRole), None)

        editor = self.widget.findChild(TimeVariableEditor)
        table = Table(test_filename("datasets/cyber-security-breaches.tab"))
        self.send_signal(self.widget.Inputs.data, table)
        index = self.widget.domain_view.model().index(4)
        self.widget.variables_view.setCurrentIndex(index)
        self.assertEqual(editor.name_edit.text(), "Date_Posted_or_Updated")
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(editor.name_edit.text(), "")
        self.assertEqual(
            self.widget.variables_model.index(0).data(Qt.EditRole), None)

        editor = self.widget.findChild(VariableEditor)
        self.send_signal(self.widget.Inputs.data, table)
        index = self.widget.domain_view.model().index(8)
        self.widget.variables_view.setCurrentIndex(index)
        self.assertEqual(editor.var.name, "Business_Associate_Involved")
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(editor.var, None)
        self.assertEqual(
            self.widget.variables_model.index(0).data(Qt.EditRole), None)

    def test_output_data(self):
        """Check data on the output after apply"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(output.X, self.iris.X)
        np.testing.assert_array_equal(output.Y, self.iris.Y)
        self.assertEqual(output.domain, self.iris.domain)

        self.widget.output_table_name = "Iris 2"
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(output.name, "Iris 2")

    def test_input_from_owcolor(self):
        """Check widget's data sent from OWColor widget"""
        owcolor = self.create_widget(OWColor)
        self.send_signal(owcolor.Inputs.data, self.iris)
        disc_model = owcolor.disc_model
        disc_model.setData(disc_model.index(0, 1), (1, 2, 3), ColorRole)
        cont_model = owcolor.cont_model
        palette = list(colorpalettes.ContinuousPalettes.values())[-1]
        cont_model.setData(cont_model.index(1, 1), palette, ColorRole)
        owcolor_output = self.get_output(owcolor.Outputs.data)
        self.send_signal(owcolor_output)
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

    def test_annotation_bool(self):
        """Check if bool labels remain bool"""
        a = ContinuousVariable("a")
        a.attributes["hidden"] = True
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
        editor.labels_model.setData(idx, "False", Qt.EditRole)

        self.widget.commit()
        t2 = self.get_output(self.widget.Outputs.data)
        self.assertFalse(t2.domain["a"].attributes["hidden"])

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

    def test_unlink(self):
        var0, var1, var2 = [ContinuousVariable("x", compute_value=Mock()),
                            ContinuousVariable("y", compute_value=Mock()),
                            ContinuousVariable("z")]
        domain = Domain([var0, var1, var2], None)
        table = Table.from_numpy(domain, np.zeros((5, 3)), np.zeros((5, 0)))
        self.send_signal(self.widget.Inputs.data, table)

        index = self.widget.domain_view.model().index
        for i in range(3):
            self.widget.domain_view.setCurrentIndex(index(i))
            editor = self.widget.findChild(ContinuousVariableEditor)
            self.assertIs(editor.unlink_var_cb.isEnabled(), i < 2)
            editor._set_unlink(i == 1)

        self.widget.commit()
        out = self.get_output(self.widget.Outputs.data)
        out0, out1, out2 = out.domain.variables
        self.assertIs(out0, domain[0])
        self.assertIsNot(out1, domain[1])
        self.assertIs(out2, domain[2])

        self.assertIsNotNone(out0.compute_value)
        self.assertIsNone(out1.compute_value)
        self.assertIsNone(out2.compute_value)

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

    def test_restore(self):
        iris = self.iris
        viris = (
            "Categorical",
            ("iris", ("Iris-setosa", "Iris-versicolor", "Iris-virginica"), (),
             False)
        )
        w = self.widget

        def restore(state):
            w._domain_change_hints = state
            w._restore()

        model = w.variables_model
        self.send_signal(w.Inputs.data, iris, widget=w)
        restore({viris: [("Rename", ("Z",))]})
        tr = model.data(model.index(4), TransformRole)
        self.assertEqual(tr, [Rename("Z")])

        restore({viris: [("AsString", ()), ("Rename", ("Z",))]})
        tr = model.data(model.index(4), TransformRole)
        self.assertEqual(tr, [AsString(), Rename("Z")])

    def test_reset_selected(self):
        w = self.widget
        model = w.domain_view.model()
        sel_model = w.domain_view.selectionModel()

        self.send_signal(self.iris)
        model.setData(model.index(1, 0), [Rename("foo")], TransformRole)
        model.setData(model.index(2, 0), [AsCategorical()], TransformRole)
        model.setData(model.index(3, 0), [Rename("bar")], TransformRole)
        w.commit()
        out = self.get_output()
        self.assertEqual([var.name for var in out.domain.attributes],
                         ["sepal length", "foo", "petal length", "bar"])
        self.assertIsInstance(out.domain[2], DiscreteVariable)

        sel_model.select(model.index(0, 0), QItemSelectionModel.Select)
        sel_model.select(model.index(2, 0), QItemSelectionModel.Select)
        sel_model.select(model.index(3, 0), QItemSelectionModel.Select)
        w.reset_selected()
        w.commit()
        out = self.get_output()
        self.assertEqual([var.name for var in out.domain.attributes],
                         ["sepal length", "foo", "petal length", "petal width"])
        self.assertIsInstance(out.domain[2], ContinuousVariable)

    @patch("Orange.widgets.data.oweditdomain.ReinterpretVariableEditor.set_data")
    def test_selection_sets_data(self, set_data):
        w = self.widget
        model = w.domain_view.model()
        sel_model = w.domain_view.selectionModel()
        tr = (Rename("x"), )

        iris = self.iris

        self.send_signal(iris)
        model.setData(model.index(1, 0), tr, TransformRole)

        sel_model.select(model.index(1, 0), QItemSelectionModel.ClearAndSelect)
        args, kwargs = set_data.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(len(args[0]), 1)
        self.assertEqual(args[0][0].vtype.name, iris.domain[1].name)
        self.assertEqual(kwargs["transforms"], [tr])

        sel_model.select(model.index(2, 0), QItemSelectionModel.Select)
        args, kwargs = set_data.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(len(args[0]), 2)
        self.assertEqual(args[0][0].vtype.name, iris.domain[1].name)
        self.assertEqual(args[0][1].vtype.name, iris.domain[2].name)
        self.assertEqual(kwargs["transforms"], [tr, ()])

    def test_selection_after_new_data(self):
        w = self.widget
        model = w.domain_view.model()
        sel_model = w.domain_view.selectionModel()
        iris = self.iris
        attrs = iris.domain.attributes

        self.send_signal(iris.transform(Domain(attrs[:3])))
        sel_model.select(model.index(1, 0), QItemSelectionModel.ClearAndSelect)
        sel_model.select(model.index(2, 0), QItemSelectionModel.Select)
        # Select #1 and #2, out of attributes 0, 1, 2
        self.assertEqual(w.selected_var_indices(), [1, 2])

        # Send attributes 1, 2, 3; #0 and #1 must be selected
        self.send_signal(iris.transform(Domain(attrs[1:])))
        self.assertEqual(w.selected_var_indices(), [0, 1])

        # Now send 0 and 2; only #1 (2) must be selected
        self.send_signal(iris.transform(Domain([attrs[0], attrs[2]])))
        self.assertEqual(w.selected_var_indices(), [1])

        # Send 0 and 3, first must be selected by default
        self.send_signal(iris.transform(Domain([attrs[0], attrs[3]])))
        self.assertEqual(w.selected_var_indices(), [0])

        # Send 1 and 2; first is selected by default
        self.send_signal(iris.transform(Domain([attrs[1], attrs[2]])))
        self.assertEqual(w.selected_var_indices(), [0])

    def test_hint_keeping(self):
        editor: ContinuousVariableEditor = self.widget.findChild(ContinuousVariableEditor)
        name_edit = editor.name_edit
        model = self.widget.domain_view.model()

        def rename(fr, to):
            for idx in range(fr, to):
                self.widget.domain_view.setCurrentIndex(model.index(idx))
                cur_text = name_edit.text()
                if cur_text[0] != "x":
                    name_edit.setText("x" + cur_text)
                editor.on_name_changed()

        def data(fr, to):
            return Table.from_numpy(Domain(vars[fr:to]),
                                    np.zeros((1, to - fr)))


        vars = [ContinuousVariable(f"v{i}") for i in range(1020)]
        self.send_signal(data(0, 5))
        rename(2, 4)

        self.send_signal(None)
        self.assertIsNone(self.get_output())

        self.send_signal(data(3, 7))
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes],
                         ["xv3", "v4", "v5", "v6"])

        self.send_signal(data(0, 5))
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes],
                         ["v0", "v1", "xv2", "xv3", "v4"])

        self.send_signal(data(3, 7))
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes],
                         ["xv3", "v4", "v5", "v6"])

        # This is too large: widget should retain just hints related to
        # the current data
        self.send_signal(data(3, 1020))
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes[:4]],
                         ["xv3", "v4", "v5", "v6"])
        rename(5, 1017)
        self.widget.commit()
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes[-3:]],
                         ["xv1017", "xv1018", "xv1019"])

        self.send_signal(None)
        self.assertIsNone(self.get_output())

        # Tests that hints for the current data are kept
        # - including the earliest (v3) and latest (v1019)
        self.send_signal(data(3, 1020))
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes[:4]],
                         ["xv3", "v4", "v5", "v6"])
        self.assertEqual([var.name for var in outp.domain.attributes[-3:]],
                         ["xv1017", "xv1018", "xv1019"])

        # Tests that older hints are dropped: v2 should be lost
        self.send_signal(data(0, 5))
        outp = self.get_output()
        self.assertEqual([var.name for var in outp.domain.attributes],
                         ["v0", "v1", "v2", "xv3", "v4"])

    def test_migrate_settings_hints_2_to_3(self):
        settings = {
            '__version__': 2,
            'context_settings':
                [Context(values={
                    '_domain_change_store': (
                        {('Categorical', ('a', ('mir1', 'mir4', 'mir2'), (), False)):
                            [('Rename', ('disease mir',))],
                         ('Categorical', ('b', ('mir4', 'mir1', 'mir2'), (), False)):
                             [('Rename', ('disease mirs',))]
                         },
                        -2),
                    '_merge_dialog_settings': ({}, -4),
                    '_selected_item': (('1', 0), -2),
                    'output_table_name': ('boo', -2),
                    '__version__': 2}),
                 Context(values={
                     '_domain_change_store': (
                         {('Categorical', ('b', ('mir4', 'mir1', 'mir2'), (), False)):
                             [('Rename', ('disease bmir',))],
                          ('Categorical', ('c', ('mir4', 'mir1', 'mir2'), (), False)):
                              [('Rename', ('disease mirs',))]
                         },
                         -2),
                      '_merge_dialog_settings': ({}, -4),
                      '_selected_item': (('1', 0), -2),
                      'output_table_name': ('far', -2),
                      '__version__': 2}),
                ]}
        migrated_hints = {
            ('Categorical', ('b', ('mir4', 'mir1', 'mir2'), (), False)):
                [('Rename', ('disease bmir',))],
            ('Categorical', ('c', ('mir4', 'mir1', 'mir2'), (), False)):
                [('Rename', ('disease mirs',))],
            ('Categorical', ('a', ('mir1', 'mir4', 'mir2'), (), False)):
                 [('Rename', ('disease mir',))],
        }
        widget = self.create_widget(OWEditDomain, stored_settings=settings)
        self.assertEqual(widget._domain_change_hints, migrated_hints)
        # order matters
        self.assertEqual(list(widget._domain_change_hints), list(migrated_hints))
        self.assertEqual(widget.output_table_name, "far")

    def test_migrate_settings_2_to_3_realworld(self):
        settings = {
            'controlAreaVisible': True,
            '__version__': 2,
            'context_settings': [Context(
                values={
                    '_domain_change_store':
                        ({('Real', ('sepal length', (1, 'f'), (), False)):
                              [('AsString', ())],
                          ('Real', ('sepal width', (1, 'f'), (), False)):
                              [('AsTime', ()), ('StrpTime', ('Detect automatically', None, 1, 1))],
                          ('Real', ('petal width', (1, 'f'), (), False)):
                              [('Annotate', ((('a', 'b'),),))]}, -2),
                    '_merge_dialog_settings': ({}, -4),
                    '_selected_item': (('petal width', 2), -2),
                    'output_table_name': ('', -2),
                    '__version__': 2},
                attributes={'sepal length': 2, 'sepal width': 2,
                            'petal length': 2, 'petal width': 2, 'iris': 1},
                metas={}
            )]
        }
        widget = self.create_widget(OWEditDomain, stored_settings=settings)
        self.assertEqual(
            widget._domain_change_hints,
            {('Real', ('sepal length', (1, 'f'), (), False)):
                 [('AsString', ())],
             ('Real', ('sepal width', (1, 'f'), (), False)):
                 [('AsTime', ()),
                  ('StrpTime', ('Detect automatically', None, 1, 1))],
             ('Real', ('petal width', (1, 'f'), (), False)):
                 [('Annotate', ((('a', 'b'),),))]}
        )

    def test_migrate_settings_name_2_to_3(self):
        settings = {
            '__version__': 2,
            'context_settings':
                [Context(values={
                    '_domain_change_store': ({}, -2),
                     'output_table_name': ('boo', -2),
                     '__version__': 2}),
                 Context(values={
                     '_domain_change_store': ({}, -2),
                     'output_table_name': ('far', -2),
                     '__version__': 2}),
                 Context(values={
                     '_domain_change_store': ({}, -2),
                     'output_table_name': ('', -2),
                     '__version__': 2}),
                 Context(values={
                     '_domain_change_store': ({}, -2),
                     '__version__': 2})
                ]
        }
        widget = self.create_widget(OWEditDomain, stored_settings=settings)
        self.assertEqual(widget.output_table_name, "far")


class TestEditors(GuiTest):
    def test_variable_editor(self):
        w = VariableEditor()
        self.assertEqual(w.get_data(), (None, []))

        v = String("S", (("A", "1"), ("B", "b")), False)
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

        v = Real("X", (-1, ""), (("A", "1"), ("B", "b")), False)
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

        v = Categorical("C", ("a", "b", "c"), (("A", "1"), ("B", "b")), False)
        values = [0, 0, 0, 1, 1, 2]
        w.set_data_categorical(v, values)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), dict(v.annotations))
        self.assertEqual(w.get_data(), (v, []))
        w.set_data_categorical(None, None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertEqual(w.get_data(), (None, []))
        mapping = [
            ("c", "C"),
            ("a", "A"),
            ("b", None),
            (None, "b")
        ]
        w.set_data_categorical(v, values, [CategoriesMapping(mapping)])
        w.grab()  # run delegate paint method
        self.assertEqual(w.get_data(), (v, [CategoriesMapping(mapping)]))

        # test selection/deselection in the view
        w.set_data_categorical(v, values)
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
        w.set_data_categorical(v, values, [CategoriesMapping(mapping)])
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
                        (("A", "1"), ("B", "b")), False)
        values = [0, 0, 0, 1, 1, 2]
        w.set_data_categorical(v, values)
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

    # mocking exec make dialog never show - dialog blocks running until closed
    @patch(
        "Orange.widgets.data.oweditdomain.GroupItemsDialog.exec",
        Mock(side_effect=lambda: QDialog.Accepted)
    )
    def test_discrete_editor_merge_action(self):
        """
        This function check whether results of dialog have effect on
        merging the attributes. The dialog itself is tested separately.
        """
        w = DiscreteVariableEditor()
        v = Categorical("C", ("a", "b", "c"),
                        (("A", "1"), ("B", "b")), False)

        w.set_data_categorical(
            v, [0, 0, 0, 1, 1, 2],
            [CategoriesMapping([
                ("a", "AA"), ("b", "BB"), ("c", "CC"),
            ])]
        )

        view = w.values_edit
        model = view.model()
        selmodel = view.selectionModel()  # type: QItemSelectionModel
        selmodel.select(
            QItemSelection(model.index(0, 0), model.index(1, 0)),
            QItemSelectionModel.ClearAndSelect
        )
        spy = QSignalSpy(w.variable_changed)
        w.merge_items.trigger()

        self.assertEqual(model.index(0, 0).data(Qt.EditRole), "other")
        self.assertEqual(model.index(1, 0).data(Qt.EditRole), "other")
        self.assertEqual(model.index(2, 0).data(Qt.EditRole), "CC")

        self.assertSequenceEqual(
            list(spy), [[]], 'variable_changed should emit exactly once'
        )

    def test_discrete_editor_rename_selected_items_action(self):
        w = DiscreteVariableEditor()
        v = Categorical("C", ("a", "b", "c"),
                        (("A", "1"), ("B", "b")), False)
        w.set_data_categorical(v, [])
        action = w.rename_selected_items
        view = w.values_edit
        model = view.model()
        selmodel = view.selectionModel()  # type: QItemSelectionModel
        selmodel.select(
            QItemSelection(model.index(0, 0), model.index(1, 0)),
            QItemSelectionModel.ClearAndSelect
        )
        # trigger the action, then find the active popup, and simulate entry
        spy = QSignalSpy(w.variable_changed)
        with patch.object(QComboBox, "setVisible", return_value=None) as m:
            action.trigger()
            m.assert_called()
        cb = view.findChild(QComboBox)
        cb.setCurrentText("BA")
        view.commitData(cb)
        self.assertEqual(model.index(0, 0).data(Qt.EditRole), "BA")
        self.assertEqual(model.index(1, 0).data(Qt.EditRole), "BA")
        self.assertSequenceEqual(
            list(spy), [[]], 'variable_changed should emit exactly once'
        )

    def test_discrete_editor_context_menu(self):
        w = DiscreteVariableEditor()
        v = Categorical("C", ("a", "b", "c"),
                        (("A", "1"), ("B", "b")), False)
        w.set_data_categorical(v, [])
        view = w.values_edit
        model = view.model()

        pos = view.visualRect(model.index(0, 0)).center()
        with patch.object(QMenu, "setVisible", return_value=None) as m:
            contextMenu(view.viewport(), pos)
            m.assert_called()

        menu = view.findChild(QMenu)
        self.assertIsNotNone(menu)
        menu.close()

    def test_time_editor(self):
        w = TimeVariableEditor()
        self.assertEqual(w.get_data(), (None, []))

        v = Time("T", (("A", "1"), ("B", "b")), False)
        w.set_data(v,)

        self.assertEqual(w.name_edit.text(), v.name)
        self.assertEqual(w.labels_model.get_dict(), dict(v.annotations))

        w.set_data(None)
        self.assertEqual(w.name_edit.text(), "")
        self.assertEqual(w.labels_model.get_dict(), {})
        self.assertEqual(w.get_data(), (None, []))

    DataVectors = [
        CategoricalVector(
            Categorical("A", ("a", "aa"), (), False), lambda:
                MArray([0, 1, 2], mask=[False, False, True])
        ),
        RealVector(
            Real("B", (6, "f"), (), False), lambda:
                MArray([0.1, 0.2, 0.3], mask=[True, False, True])
        ),
        TimeVector(
            Time("T", (), False), lambda:
                MArray([0, 100, 200], dtype="M8[us]", mask=[True, False, True])
        ),
        StringVector(
            String("S", (), False), lambda:
                MArray(["0", "1", "2"], dtype=object, mask=[True, False, True])
        ),
    ]
    ReinterpretTransforms = {
        Categorical: [AsCategorical], Real: [AsContinuous],
        Time: [AsTime, partial(StrpTime, 'Detect automatically', None, 1, 1)],
        String: [AsString]
    }

    def test_reinterpret_editor(self):
        w = ReinterpretVariableEditor()
        self.assertEqual(w.get_data(), ((None, ), ([], )))
        data = self.DataVectors[0]
        w.set_data((data, ))
        self.assertEqual(w.get_data(), ((data.vtype, ), ([], )))
        w.set_data((data, ), ([Rename("Z")], ))
        self.assertEqual(w.get_data(), ((data.vtype, ), ([Rename("Z")], )))

        for vec, tr in product(self.DataVectors, self.ReinterpretTransforms.values()):
            w.set_data((vec, ), ([t() for t in tr], ))
            v, tr_ = w.get_data()
            self.assertEqual(*v, vec.vtype)
            if not tr_[0]:
                self.assertEqual(tr, self.ReinterpretTransforms[type(*v)])
            else:
                self.assertListEqual(*tr_, [t() for t in tr])

    def test_reinterpret_editor_simulate(self):
        w = ReinterpretVariableEditor()

        def cb():
            var, tr = w.get_data()
            var, tr = var[0], tr[0]
            type_ = tc.currentData()
            if type_ is not type(var):
                self.assertEqual(
                    tr, [t() for t in self.ReinterpretTransforms[type_]] + [Rename("Z")]
                )
            else:
                self.assertEqual(tr, [Rename("Z")])

        for vec in self.DataVectors:
            w.set_data((vec, ), ([Rename("Z")], ))
            tc = w.layout().currentWidget().findChild(QComboBox,
                                                      name="type-combo")
            simulate.combobox_run_through_all(tc, callback=cb)

    def test_multiple_editor_init(self):
        w = ReinterpretVariableEditor()
        w.set_data(self.DataVectors, [()] * 4)
        cw = w.layout().currentWidget()
        tc = cw.findChild(QComboBox, name="type-combo")
        self.assertIs(type(cw), BaseEditor)
        self.assertEqual(tc.count(), 6)

        w.set_data(self.DataVectors[:1], [()])
        cw = w.layout().currentWidget()
        tc = cw.findChild(QComboBox, name="type-combo")
        self.assertIsNot(type(cw), BaseEditor)
        self.assertEqual(tc.count(), 4)

    def test_reinterpret_set_data_multiple_transforms(self):
        w = ReinterpretVariableEditor()

        w.set_data((Mock(), ) * 4,
                   [[AsContinuous()] for _ in range(4)])
        cw = w.layout().currentWidget()
        self.assertIs(type(cw), BaseEditor)
        tc = cw.findChild(QComboBox, name="type-combo")

        self.assertIsInstance(
            w.__dict__["_ReinterpretVariableEditor__transform"],
            AsContinuous)
        self.assertEqual(tc.currentData(), Real)

        w.set_data((Mock(), ) * 3,
                   [[AsContinuous(), Rename("x")],
                    [AsContinuous(), Rename("y")],
                    [AsContinuous()]
                    ]
                   )
        self.assertIsInstance(
            w.__dict__["_ReinterpretVariableEditor__transform"],
            AsContinuous)
        self.assertEqual(tc.currentData(), Real)

        w.set_data((Mock(), ) * 3,
                   [[AsContinuous(), Rename("x")],
                    [Rename("y")],
                    [AsContinuous()]
                    ]
                   )
        self.assertIsNone(w.__dict__["_ReinterpretVariableEditor__transform"])
        self.assertIsNone(tc.currentData())

        w.set_data((Mock(), ) * 3,
                   [[AsContinuous(), Rename("x")],
                    [],
                    [AsContinuous()]
                    ]
                   )
        self.assertIsNone(w.__dict__["_ReinterpretVariableEditor__transform"])
        self.assertIsNone(tc.currentData())

        w.set_data((Mock(),) * 3,
                   [[AsContinuous(), Rename("x")],
                    [AsTime()],
                    [AsContinuous()]
                    ]
                   )
        self.assertIsNone(w.__dict__["_ReinterpretVariableEditor__transform"])
        self.assertIsNone(tc.currentData())

    def test_reinterpret_multiple(self):
        def cb():
            for var, tr, v in zip(*w.get_data(), "SPQR"):
                type_ = tc.currentData()
                if type_ is not type(var) \
                        and type_ not in (RestoreOriginal, None):
                    self.assertSequenceEqual(
                        tr, [t() for t in self.ReinterpretTransforms[type_][:1]]
                            + [Rename(v)],
                        f"type: {type_}"
                    )
                else:
                    self.assertSequenceEqual(tr, (Rename(v), ), f"type: {type_}")

        w = ReinterpretVariableEditor()
        w.set_data(self.DataVectors, tuple([Rename(c)] for c in "SPQR"))
        tc = w.layout().currentWidget().findChild(QComboBox, name="type-combo")
        simulate.combobox_run_through_all(tc, callback=cb)

    def test_reinterpret_remove_specific(self):
        def cb():
            for var, tr, v in zip(*w.get_data(), "SPQR"):
                type_ = tc.currentData()
                if type_ is not type(var) \
                        and type_ not in (RestoreOriginal, None):
                    self.assertSequenceEqual(
                        tr, [t() for t in self.ReinterpretTransforms[type_][:1]]
                            + [Rename(v)],
                        f"type: {type_}"
                    )
                else:
                    self.assertSequenceEqual(tr, (Rename(v), ), f"type: {type_}")

        w = ReinterpretVariableEditor()
        transforms = (
            [CategoriesMapping([("a", "b")])],
            [AsCategorical(), Rename("xx")],
            [AsCategorical(), CategoriesMapping([("c", "d")])])
        w.set_data(self.DataVectors[:3], transforms)
        tc = w.layout().currentWidget().findChild(QComboBox, name="type-combo")

        tc.setCurrentIndex(0)  # Categorical
        tc.activated[int].emit(0)
        self.assertSequenceEqual(w.get_data()[1], transforms)

        tc.setCurrentIndex(1)  # Numeric
        tc.activated[int].emit(1)
        self.assertEqual(w.get_data()[1],
                         [[AsContinuous()],
                          [Rename("xx")],
                          [AsContinuous()]])

        tc.setCurrentIndex(4)  # Restore original
        tc.activated[int].emit(4)
        self.assertEqual(w.get_data()[1],
                         [[CategoriesMapping([("a", "b")])],
                          [Rename("xx")],
                          []])

        tc.setCurrentIndex(5)  # None
        tc.activated[int].emit(5)
        self.assertSequenceEqual(w.get_data()[1], transforms)

        # We don't have this situation, but simulate a situation in which the
        # target type has the same (specific) transformation
        with patch.dict(w.Specific, {Real: (CategoriesMapping, )}):
            tc.setCurrentIndex(1)  # Numeric
            tc.activated[int].emit(1)
            self.assertSequenceEqual(
                w.get_data()[1],
                ([AsContinuous(), CategoriesMapping([("a", "b")])],
                 [Rename("xx")],
                 [AsContinuous(), CategoriesMapping([("c", "d")])]
                 )
            )

    def test_reinterpret_multiple_keep_and_restore(self):
        w = ReinterpretVariableEditor()
        transforms = tuple([AsString(), Rename(c)] for c in "SPQR")
        w.set_data(self.DataVectors, tuple([AsString(), Rename(c)] for c in "SPQR"))
        tc = w.layout().currentWidget().findChild(QComboBox, name="type-combo")

        tc.setCurrentIndex(4)  # Restore original
        tc.activated[int].emit(4)
        self.assertSequenceEqual(
            [list(tr) for tr in w.get_data()[1]],
            [[Rename(c)] for c in "SPQR"])

        tc.setCurrentIndex(5)  # Keep
        tc.activated[int].emit(5)
        self.assertSequenceEqual(
            [list(tr) for tr in w.get_data()[1]],
            transforms)

    def test_unlink(self):
        w = ContinuousVariableEditor()
        cbox = w.unlink_var_cb
        self.assertEqual(w.get_data(), (None, []))

        v = Real("X", (-1, ""), (("A", "1"), ("B", "b")), False)
        w.set_data(v, [])
        self.assertFalse(cbox.isEnabled())

        v = Real("X", (-1, ""), (("A", "1"), ("B", "b")), True)
        w.set_data(v, [Unlink()])
        self.assertTrue(cbox.isEnabled())
        self.assertTrue(cbox.isChecked())

        v = Real("X", (-1, ""), (("A", "1"), ("B", "b")), True)
        w.set_data(v, [])
        self.assertTrue(cbox.isEnabled())
        self.assertFalse(cbox.isChecked())

        cbox.setChecked(True)
        self.assertEqual(w.get_data()[1], [Unlink()])

        w.set_data(v, [Unlink()])
        self.assertTrue(cbox.isChecked())

        cbox.setChecked(False)
        self.assertEqual(w.get_data()[1], [])

        cbox.setChecked(True)
        w.clear()
        self.assertFalse(cbox.isChecked())
        self.assertEqual(w.get_data()[1], [])

        w._set_unlink(True)
        self.assertTrue(cbox.isChecked())
        w._set_unlink(False)
        self.assertFalse(cbox.isChecked())


class TestModels(GuiTest):
    def test_variable_model(self):
        model = VariableListModel()
        self.assertEqual(model.effective_name(model.index(-1, -1)), None)

        def data(row, role):
            return model.data(model.index(row,), role)

        def set_data(row, data, role):
            model.setData(model.index(row), data, role)

        model[:] = [
            RealVector(Real("A", (3, "g"), (), False), lambda: MArray([])),
            RealVector(Real("B", (3, "g"), (), False), lambda: MArray([])),
        ]
        self.assertEqual(data(0, Qt.DisplayRole), "A")
        self.assertEqual(data(1, Qt.DisplayRole), "B")
        self.assertEqual(model.effective_name(model.index(1)), "B")
        set_data(1, [Rename("A")], TransformRole)
        self.assertEqual(model.effective_name(model.index(1)), "A")
        self.assertEqual(data(0, MultiplicityRole), 2)
        self.assertEqual(data(1, MultiplicityRole), 2)
        set_data(1, [], TransformRole)
        self.assertEqual(data(0, MultiplicityRole), 1)
        self.assertEqual(data(1, MultiplicityRole), 1)


class TestDelegates(GuiTest):
    def test_delegate(self):
        model = VariableListModel([None, None])

        def set_item(row: int, v: dict):
            model.setItemData(model.index(row),  v)

        def get_style_option(row: int) -> QStyleOptionViewItem:
            opt = QStyleOptionViewItem()
            delegate.initStyleOption(opt, model.index(row))
            return opt

        set_item(0, {Qt.EditRole: Categorical("a", (), (), False)})
        delegate = VariableEditDelegate()
        opt = get_style_option(0)
        self.assertEqual(opt.text, "a")
        self.assertFalse(opt.font.italic())
        set_item(0, {TransformRole: [Rename("b")]})
        opt = get_style_option(0)
        self.assertEqual(opt.text, "a \N{RIGHTWARDS ARROW} b")
        self.assertTrue(opt.font.italic())

        set_item(0, {TransformRole: [AsString()]})
        opt = get_style_option(0)
        self.assertIn("reinterpreted", opt.text)
        self.assertTrue(opt.font.italic())
        set_item(1, {
            Qt.EditRole: String("b", (), False),
            TransformRole: [Rename("a")]
        })
        opt = get_style_option(1)
        self.assertEqual(opt.palette.color(QPalette.Text), QColor(Qt.red))
        view = QListView()
        with patch.object(QToolTip, "showText") as p:
            delegate.helpEvent(
                QHelpEvent(QHelpEvent.ToolTip, QPoint(0, 0), QPoint(0, 0)),
                view, opt, model.index(1),
            )
            p.assert_called_once()


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
        # this test only test type of format that can be string, continuous and discrete
        # correctness of time formats is already tested in TimeVariable module
        d = TimeVariable("_").parse_exact_iso
        times = (
            ["07.02.2022", "18.04.2021"],  # date only
            ["07.02.2022 01:02:03", "18.04.2021 01:02:03"],  # datetime
            # datetime with timezone
            ["2021-02-08 01:02:03+01:00", "2021-02-07 01:02:03+01:00"],
            ["010203", "010203"],  # time
            ["02-07", "04-18"],
        )
        formats = [
            "25.11.2021", "25.11.2021 00:00:00", "2021-11-25 00:00:00", "000000", "11-25"
        ]
        expected = [
            [d("2022-02-07"), d("2021-04-18")],
            [d("2022-02-07 01:02:03"), d("2021-04-18 01:02:03")],
            [d("2021-02-08 01:02:03+0100"), d("2021-02-07 01:02:03+0100")],
            [d("01:02:03"), d("01:02:03")],
            [d("1900-02-07"), d("1900-04-18")],
        ]
        variables = [StringVariable(f"s{i}") for i in range(len(times))]
        variables += [DiscreteVariable(f"d{i}", values=t) for i, t in enumerate(times)]
        domain = Domain([], metas=variables)
        metas = [t for t in times] + [list(range(len(x))) for x in times]
        table = Table(domain, np.empty((len(times[0]), 0)), metas=np.array(metas).transpose())

        tr = AsTime()
        dtr = []
        for v, f in zip(domain.metas, chain(formats, formats)):
            strp = StrpTime(f, *TimeVariable.ADDITIONAL_FORMATS[f])
            vtr = apply_transform_var(
                apply_reinterpret(v, tr, table_column_data(table, v)), [strp]
            )
            dtr.append(vtr)

        ttable = table.transform(Domain([], metas=dtr))
        assert_array_equal(
            ttable.metas,
            np.array(list(chain(expected, expected)), dtype=float).transpose()
        )

    def test_reinterpret_string(self):
        table = self.data_str
        domain = table.domain
        tvars = []
        for v in domain.metas:
            for i, tr in enumerate(
                [AsContinuous(), AsCategorical(), AsTime(), AsString()]
            ):
                vtr = apply_reinterpret(v, tr, table_column_data(table, v)).renamed(
                    f"{v.name}_{i}"
                )
                if isinstance(tr, AsTime):
                    strp = StrpTime("Detect automatically", None, 1, 1)
                    vtr = apply_transform_var(vtr, [strp])
                tvars.append(vtr)
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

    def test_to_time_variable(self):
        table = self.data
        tr = AsTime()
        dtr = []
        for v in table.domain:
            strp = StrpTime("Detect automatically", None, 1, 1)
            vtr = apply_transform_var(
                apply_reinterpret(v, tr, table_column_data(table, v)), [strp]
            )
            dtr.append(vtr)
        ttable = table.transform(Domain([], metas=dtr))
        for var in ttable.domain:
            self.assertTrue(var.have_date or var.have_time)


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
            {"": np.nan, "a": 0, "b": 1},
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

    def test_equality(self):
        v1 = DiscreteVariable("v1", values=tuple("abc"))
        v2 = DiscreteVariable("v1", values=tuple("abc"))
        v3 = DiscreteVariable("v3", values=tuple("abc"))

        map1 = {"a": 2, "b": 0, "c": 1}
        map2 = {"a": 2, "b": 0, "c": 1}
        map3 = {"a": 2, "b": 0, "c": 1}

        t1 = LookupMappingTransform(v1, map1, float)
        t1a = LookupMappingTransform(v2, map2, float)
        t2 = LookupMappingTransform(v3, map3, float)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

        map1a = {"a": 2, "b": 1, "c": 0}
        t1 = LookupMappingTransform(v1, map1, float)
        t1a = LookupMappingTransform(v1, map1a, float)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))

        map1a = {"a": 2, "b": 0, "c": 1}
        t1 = LookupMappingTransform(v1, map1, float)
        t1a = LookupMappingTransform(v1, map1a, float, unknown=2)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))

        t1 = LookupMappingTransform(v1, map1, float)
        t1a = LookupMappingTransform(v1, map1, int)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))


class TestGroupLessFrequentItemsDialog(GuiTest):
    def setUp(self) -> None:
        self.v = Categorical("C", ("a", "b", "c"),
                        (("A", "1"), ("B", "b")), False)
        self.data = [0, 0, 0, 1, 1, 2]

    def test_dialog_open(self):
        dialog = GroupItemsDialog(self.v, self.data, ["a", "b"], {})
        self.assertTrue(dialog.selected_radio.isChecked())
        self.assertFalse(dialog.frequent_abs_radio.isChecked())
        self.assertFalse(dialog.frequent_rel_radio.isChecked())
        self.assertFalse(dialog.n_values_radio.isChecked())

        dialog = GroupItemsDialog(self.v, self.data, [], {})
        self.assertFalse(dialog.selected_radio.isChecked())
        self.assertTrue(dialog.frequent_abs_radio.isChecked())
        self.assertFalse(dialog.frequent_rel_radio.isChecked())
        self.assertFalse(dialog.n_values_radio.isChecked())

    def test_group_selected(self):
        dialog = GroupItemsDialog(self.v, self.data, ["a", "b"], {})
        dialog.selected_radio.setChecked(True)
        dialog.new_name_line_edit.setText("BA")

        self.assertListEqual(dialog.get_merge_attributes(), ["a", "b"])
        self.assertEqual(dialog.get_merged_value_name(), "BA")

    def test_group_less_frequent_abs(self):
        dialog = GroupItemsDialog(self.v, self.data, ["a", "b"], {})
        dialog.frequent_abs_radio.setChecked(True)
        dialog.frequent_abs_spin.setValue(3)
        dialog.new_name_line_edit.setText("BA")

        self.assertListEqual(dialog.get_merge_attributes(), ["b", "c"])
        self.assertEqual(dialog.get_merged_value_name(), "BA")

        dialog.frequent_abs_spin.setValue(2)
        self.assertListEqual(dialog.get_merge_attributes(), ["c"])

        dialog.frequent_abs_spin.setValue(1)
        self.assertListEqual(dialog.get_merge_attributes(), [])

    def test_group_less_frequent_rel(self):
        dialog = GroupItemsDialog(self.v, self.data, ["a", "b"], {})
        dialog.frequent_rel_radio.setChecked(True)
        dialog.frequent_rel_spin.setValue(50)
        dialog.new_name_line_edit.setText("BA")

        self.assertListEqual(dialog.get_merge_attributes(), ["b", "c"])
        self.assertEqual(dialog.get_merged_value_name(), "BA")

        dialog.frequent_rel_spin.setValue(20)
        self.assertListEqual(dialog.get_merge_attributes(), ["c"])

        dialog.frequent_rel_spin.setValue(15)
        self.assertListEqual(dialog.get_merge_attributes(), [])

    def test_group_keep_n(self):
        dialog = GroupItemsDialog(self.v, self.data, ["a", "b"], {})
        dialog.n_values_radio.setChecked(True)
        dialog.n_values_spin.setValue(1)
        dialog.new_name_line_edit.setText("BA")

        self.assertListEqual(dialog.get_merge_attributes(), ["b", "c"])
        self.assertEqual(dialog.get_merged_value_name(), "BA")

        dialog.n_values_spin.setValue(2)
        self.assertListEqual(dialog.get_merge_attributes(), ["c"])

        dialog.n_values_spin.setValue(3)
        self.assertListEqual(dialog.get_merge_attributes(), [])

    def test_group_less_frequent_missing(self):
        """
        Widget gives MaskedArray to GroupItemsDialog which can have missing
        values.
        gh-4599
        """
        def _test_correctness():
            dialog.frequent_abs_radio.setChecked(True)
            dialog.frequent_abs_spin.setValue(3)
            self.assertListEqual(dialog.get_merge_attributes(), ["b", "c"])

            dialog.frequent_rel_radio.setChecked(True)
            dialog.frequent_rel_spin.setValue(50)
            self.assertListEqual(dialog.get_merge_attributes(), ["b", "c"])

            dialog.n_values_radio.setChecked(True)
            dialog.n_values_spin.setValue(1)
            self.assertListEqual(dialog.get_merge_attributes(), ["b", "c"])

        # masked array
        data_masked = np.ma.array(
            [0, 0, np.nan, 0, 1, 1, 2], mask=[0, 0, 1, 0, 0, 0, 0]
        )
        dialog = GroupItemsDialog(self.v, data_masked, [], {})
        _test_correctness()

        data_array = np.array([0, 0, np.nan, 0, 1, 1, 2])
        dialog = GroupItemsDialog(self.v, data_array, [], {})
        _test_correctness()

        data_list = [0, 0, None, 0, 1, 1, 2]
        dialog = GroupItemsDialog(self.v, data_list, [], {})
        _test_correctness()


if __name__ == '__main__':
    unittest.main()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object,protected-access
import unittest
from functools import partial
from unittest.mock import patch, Mock

import numpy as np

from AnyQt.QtCore import QPoint, Qt, QModelIndex
from AnyQt.QtWidgets import QWidget, QApplication, QStyleOptionViewItem
from AnyQt.QtGui import QIcon

from orangewidget.settings import Context

from Orange.data import Table, ContinuousVariable, TimeVariable, Domain
from Orange.preprocess.discretize import TooManyIntervals
from Orange.widgets.data.owdiscretize import OWDiscretize, \
    IncreasingNumbersListValidator, VarHint, Methods, DefaultKey, \
    _fixed_width_discretization, _fixed_time_width_discretization, \
    _custom_discretization, variable_key, Options, DefaultHint, \
    _mdl_discretization, ListViewSearch, format_desc, DefaultDiscModel, \
    DiscDomainModel, DiscDesc
from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.widgets.utils.itemmodels import select_rows


class DataMixin:
    def prepare_data(self):
        self.domain = Domain([ContinuousVariable("x"),
                              ContinuousVariable("y"),
                              ContinuousVariable("z"),
                              TimeVariable("t"),
                              TimeVariable("u")])
        self.data = Table.from_numpy(self.domain, np.arange(20).reshape(4, 5))
        self.var_hints = {
            DefaultKey: VarHint(Methods.Keep, ()),
            ("x", False): VarHint(Methods.EqualFreq, (3, )),
            ("y", False): VarHint(Methods.Keep, ()),
            ("z", False): VarHint(Methods.Remove, ()),
            ("t", True): VarHint(Methods.Binning, (2, ))
        }
        # Copy the following line to tests, for reference:
        # Def: Keep, x: EqFreq 3, y: Keep, z: Remove, t (time): Bin 2, u (time):

class TestOWDiscretize(WidgetTest, DataMixin):
    def setUp(self):
        super().setUp()
        self.prepare_data()
        self.widget = self.create_widget(OWDiscretize)


    def test_empty_data(self):
        data = Table("iris")
        widget = self.widget
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))
        for m in range(len(Methods)):
            widget.var_hints = {DefaultKey: VarHint(m, ())}
            widget.commit.now()
            self.assertIsNotNone(self.get_output(widget.Outputs.data))

    def test_report(self):
        data = Table("brown-selected")

        w = self.create_widget(
            OWDiscretize,
            {"var_hints":
                {None: VarHint(Methods.EqualFreq, (3,)),
                 ('alpha 0', False): VarHint(Methods.Keep, ()),
                 ('alpha 7', False): VarHint(Methods.Remove, ()),
                 ('alpha 14', False): VarHint(Methods.Binning, (2, )),
                 ('alpha 21', False): VarHint(Methods.FixedWidth, ("0.05", )),
                 ('alpha 28', False): VarHint(Methods.EqualFreq, (4, )),
                 ('alpha 35', False): VarHint(Methods.MDL, ()),
                 ('alpha 42', False): VarHint(Methods.Custom, ("0, 0.125", )),
                 ('alpha 49', False): VarHint(Methods.MDL, ())},
             "__version__": 3})
        self.send_signal(w.Inputs.data, data)

        self.widget.send_report()

    def test_all(self):
        data = Table("brown-selected")

        w = self.create_widget(
            OWDiscretize,
            {"var_hints":
                {None: VarHint(Methods.EqualFreq, (3,)),
                 ('alpha 0', False): VarHint(Methods.Keep, ()),
                 ('alpha 7', False): VarHint(Methods.Remove, ()),
                 ('alpha 14', False): VarHint(Methods.Binning, (2, )),
                 ('alpha 21', False): VarHint(Methods.FixedWidth, ("0.05", )),
                 ('alpha 28', False): VarHint(Methods.EqualFreq, (4, )),
                 ('alpha 35', False): VarHint(Methods.MDL, ()),
                 ('alpha 42', False): VarHint(Methods.Custom, ("0, 0.125", )),
                 ('alpha 49', False): VarHint(Methods.MDL, ())},
             "__version__": 3})

        self.send_signal(w.Inputs.data, data)

        self.assertTrue(w.button_group.button(Methods.MDL).isEnabled())
        self.assertEqual(w.varview.default_view.model().hint,
                         VarHint(Methods.EqualFreq, (3, )))

        out = self.get_output(w.Outputs.data)
        dom = out.domain
        self.assertIsInstance(dom["alpha 0"], ContinuousVariable)
        self.assertNotIn("alpha 7", dom)
        self.assertEqual(dom["alpha 14"].values, ('< 0', '≥ 0'))
        self.assertEqual(dom["alpha 21"].values,
                         ('< -0.15', "-0.15 - -0.10", "-0.10 - -0.05",
                          "-0.05 - 0.00", "0.00 - 0.05", "0.05 - 0.10",
                          '≥ 0.10'))
        self.assertEqual(len(dom["alpha 28"].values), 4)
        self.assertNotIn("alpha 35", dom)  # removed by MDL
        self.assertEqual(dom["alpha 42"].values, ('< 0', '0 - 0.125', '≥ 0.125'))
        self.assertEqual(len(dom["alpha 49"].values), 2)

        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.data))
        self.assertIsNone(w.data)
        self.assertEqual(w.discretized_vars, {})
        self.assertEqual(len(w.varview.model()), 0)

        self.send_signal(w.Inputs.data, data)
        self.assertIsNotNone(self.get_output(w.Outputs.data))
        w.button_group.button(Methods.MDL).setChecked(True)
        self.assertTrue(w.button_group.button(Methods.MDL).isEnabled())
        self.assertTrue(w.button_group.button(Methods.MDL).isChecked())

        self.send_signal(w.Inputs.data, data[:, 0])
        self.assertFalse(w.button_group.button(Methods.MDL).isEnabled())
        self.assertFalse(w.button_group.button(Methods.MDL).isChecked())

        self.send_signal(w.Inputs.data, data)
        self.assertTrue(w.button_group.button(Methods.MDL).isEnabled())

    def test_get_values(self):
        w = self.widget

        w.binning_spin.setValue(5)
        w.width_line.setText("6")
        w.width_time_line.setText("7")
        w.width_time_unit.setCurrentIndex(1)
        w.freq_spin.setValue(8)
        w.width_spin.setValue(9)
        w.threshold_line.setText("1, 2, 3, 4, 5")

        self.assertEqual(w._get_values(Methods.Keep), ())
        self.assertEqual(w._get_values(Methods.Remove), ())
        self.assertEqual(w._get_values(Methods.Binning), (5, ))
        self.assertEqual(w._get_values(Methods.FixedWidth), ("6", ))
        self.assertEqual(w._get_values(Methods.FixedWidthTime), ("7", 1))
        self.assertEqual(w._get_values(Methods.EqualFreq), (8, ))
        self.assertEqual(w._get_values(Methods.EqualWidth), (9, ))
        self.assertEqual(w._get_values(Methods.MDL), ())
        self.assertEqual(w._get_values(Methods.Custom), ("1, 2, 3, 4, 5", ))

    def test_set_values(self):
        w = self.widget

        w._set_values(Methods.Keep, ())
        w._set_values(Methods.Remove, ())
        w._set_values(Methods.Binning, (5,))
        w._set_values(Methods.FixedWidth, ("6",))
        w._set_values(Methods.FixedWidthTime, ("7", 1))
        w._set_values(Methods.EqualFreq, (8,))
        w._set_values(Methods.EqualWidth, (9,))
        w._set_values(Methods.MDL, ())
        w._set_values(Methods.Custom, ("1, 2, 3, 4, 5",))

        self.assertEqual(w.binning_spin.value(), 5)
        self.assertEqual(w.width_line.text(), "6")
        self.assertEqual(w.width_time_line.text(), "7")
        self.assertEqual(w.width_time_unit.currentIndex(), 1)
        self.assertEqual(w.freq_spin.value(), 8)
        self.assertEqual(w.width_spin.value(), 9)
        self.assertEqual(w.threshold_line.text(), "1, 2, 3, 4, 5")

    def test_varkeys_for_selection(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        select_rows(w.varview, (0, 4))
        self.assertEqual(w.varkeys_for_selection(), [("x", False), ("u", True)])

    def test_change_selection_update_interface(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        w.var_hints = {
            DefaultKey: DefaultHint,
            ("x", False): VarHint(Methods.FixedWidth, ("10", )),
            ("y", False): VarHint(Methods.FixedWidth, ("10", )),
            ("z", False): VarHint(Methods.FixedWidth, ("5", )),
            ("t", False): VarHint(Methods.Binning, (5, ))
        }

        select_rows(w.varview, (0, 1))
        self.assertTrue(w.button_group.button(Methods.FixedWidth).isChecked())
        self.assertTrue(w.button_group.button(Methods.FixedWidth).isEnabled())
        self.assertFalse(w.button_group.button(Methods.FixedWidthTime).isEnabled())
        self.assertTrue(w.button_group.button(Methods.Custom).isEnabled())
        self.assertTrue(w.copy_to_custom.isEnabled())
        self.assertEqual(w.width_line.text(), "10")

        select_rows(w.varview, (1, 2))
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isChecked())
        self.assertTrue(w.button_group.button(Methods.FixedWidth).isEnabled())
        self.assertFalse(w.button_group.button(Methods.FixedWidthTime).isEnabled())
        self.assertTrue(w.button_group.button(Methods.Custom).isEnabled())
        self.assertTrue(w.copy_to_custom.isEnabled())

        select_rows(w.varview, (2, 4))
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isChecked())
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isEnabled())
        self.assertFalse(w.button_group.button(Methods.FixedWidthTime).isEnabled())
        self.assertFalse(w.button_group.button(Methods.Custom).isEnabled())

        select_rows(w.varview, (3, 4))
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isChecked())
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isEnabled())
        self.assertTrue(w.button_group.button(Methods.FixedWidthTime).isEnabled())
        self.assertFalse(w.button_group.button(Methods.Custom).isEnabled())
        self.assertFalse(w.copy_to_custom.isEnabled())

        select_rows(w.varview.default_view, (0, ))
        self.assertEqual(len(w.varview.selectionModel().selectedIndexes()), 0)
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isChecked())
        self.assertTrue(w.button_group.button(Methods.FixedWidth).isEnabled())
        self.assertTrue(w.button_group.button(Methods.FixedWidthTime).isEnabled())
        self.assertTrue(w.button_group.button(Methods.Custom).isEnabled())
        self.assertFalse(w.copy_to_custom.isEnabled())
        self.assertFalse(w.button_group.button(Methods.Default).isEnabled())
        w._check_button(Methods.FixedWidth, True)
        self.assertTrue(w.button_group.button(Methods.FixedWidth).isChecked())

        select_rows(w.varview, (3, ))
        self.assertEqual(len(w.varview.default_view.selectionModel().selectedIndexes()), 0)
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isChecked())
        self.assertFalse(w.button_group.button(Methods.FixedWidth).isEnabled())
        self.assertTrue(w.button_group.button(Methods.FixedWidthTime).isEnabled())
        self.assertFalse(w.button_group.button(Methods.Custom).isEnabled())
        self.assertTrue(w.button_group.button(Methods.Default).isEnabled())

    def test_update_hints(self):
        w = self.widget
        update_disc = w._update_discretizations
        w._update_discretizations = Mock()
        w.width_line.setText("10")
        self.send_signal(w.Inputs.data, self.data)
        w.var_hints = {
            DefaultKey: DefaultHint,
            ("x", False): VarHint(Methods.EqualFreq, (3, )),
            ("y", False): VarHint(Methods.EqualFreq, (3, )),
            ("z", False): VarHint(Methods.EqualFreq, (4, )),
            ("t", True): VarHint(Methods.Binning, (5, ))
        }
        update_disc()
        self.assertEqual(len(w.discretized_vars), 5)

        select_rows(w.varview, (0, ))
        w.button_group.button(Methods.Default).click()
        self.assertNotIn(("x", False), w.var_hints)
        # Check that "x" is invalidated
        self.assertEqual(len(w.discretized_vars), 4)
        self.assertNotIn(("x", False), w.discretized_vars)
        update_disc()
        self.assertEqual(len(w.discretized_vars), 5)
        self.assertIn(("x", False), w.discretized_vars)

        select_rows(w.varview, (0, 1))
        w.button_group.button(Methods.FixedWidth).click()
        self.assertEqual(w.var_hints[("x", False)],
                         VarHint(Methods.FixedWidth, ("10", )))
        self.assertEqual(w.var_hints[("y", False)],
                         VarHint(Methods.FixedWidth, ("10", )))
        # Check that "x" and "y" are invalidated
        self.assertEqual(len(w.discretized_vars), 3)
        self.assertNotIn(("x", False), w.discretized_vars)
        self.assertNotIn(("y", False), w.discretized_vars)
        update_disc()
        self.assertEqual(len(w.discretized_vars), 5)
        self.assertIn(("x", False), w.discretized_vars)
        self.assertIn(("y", False), w.discretized_vars)

        w.width_line.setText("5")
        self.assertEqual(w.var_hints[("x", False)],
                         VarHint(Methods.FixedWidth, ("5", )))
        self.assertEqual(w.var_hints[("y", False)],
                         VarHint(Methods.FixedWidth, ("5", )))
        # Check that "x" and "y" are invalidated
        self.assertEqual(len(w.discretized_vars), 3)
        self.assertNotIn(("x", False), w.discretized_vars)
        self.assertNotIn(("y", False), w.discretized_vars)
        update_disc()
        self.assertEqual(len(w.discretized_vars), 5)
        self.assertIn(("x", False), w.discretized_vars)
        self.assertIn(("y", False), w.discretized_vars)

        select_rows(w.varview.default_view, (0, ))
        w.button_group.button(Methods.FixedWidth).click()
        self.assertEqual(len(w.discretized_vars), 4)
        self.assertNotIn(("u", True), w.discretized_vars)
        update_disc()
        self.assertEqual(len(w.discretized_vars), 5)
        self.assertIn(("u", True), w.discretized_vars)

    def test_discretize_var(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)

        x = self.data.domain["x"]
        t = self.data.domain["t"]

        s, dvar = w._discretize_var(x, VarHint(Methods.FixedWidthTime, ("10", 0)))
        self.assertIn("keep", s)
        self.assertIs(dvar, x)

        s, dvar = w._discretize_var(t, VarHint(Methods.FixedWidth, ("10", )))
        self.assertIn("keep", s)
        self.assertIs(dvar, t)

        try:
            Options[42] = Mock()

            # Errored
            # Unit test - mocked function
            Options[42].function = lambda *_: "foo error"
            s, dvar = w._discretize_var(t, VarHint(42, ()))
            self.assertIn("foo error", s)
            self.assertIsNone(dvar)
            # Real error
            s, dvar = w._discretize_var(t, VarHint(Methods.MDL, ()))
            self.assertIn("<", s)
            self.assertIsNone(dvar)

            # Removed attribute
            Options[42].function = lambda *_: None
            s, dvar = w._discretize_var(t, VarHint(42, ()))
            self.assertEqual("", s)
            self.assertIsNone(dvar)
            # Really removed
            s, dvar = w._discretize_var(t, VarHint(Methods.Remove, ()))
            self.assertEqual("", s)
            self.assertIsNone(dvar)

            # No intervals
            var = Mock(compute_value=Mock(points=[]))
            Options[42].function = lambda *_: var
            s, dvar = w._discretize_var(t, VarHint(42, ()))
            self.assertIn("removed", s)
            self.assertIsNone(dvar)
            s, dvar = w._discretize_var(x, VarHint(Methods.FixedWidth, ("1000", )))
            self.assertIn("removed", s)
            self.assertIsNone(dvar)

            # All fine
            var = Mock(compute_value=Mock(points=[1, 2, 3]))
            Options[42].function = lambda *_: var
            s, dvar = w._discretize_var(t, VarHint(42, ()))
            self.assertIn("1, 2, 3", s)
            self.assertIs(dvar, var)
            s, dvar = w._discretize_var(x, VarHint(Methods.EqualWidth, (3, )))
            self.assertEqual(dvar.compute_value.points, [5, 10])

        finally:
            del Options[42]

    def test_update_discretizations(self):
        w = self.widget
        # Def: Keep, x: EqFreq 3, y: Keep, z: Remove, t (time): Bin 2, u (time):
        w.var_hints = self.var_hints
        y, t, u = map(self.domain.__getitem__, "ytu")

        # no data: do nothing, but don't crash
        w._update_discretizations()

        self.send_signal(w.Inputs.data, self.data)
        d = w.discretized_vars
        self.assertEqual(len(d), 5)
        self.assertEqual(len(d[("x", False)].values), 3)
        self.assertIs(d[("y", False)], y)
        self.assertIsNone(d[("z", False)])
        self.assertIsNot(d[("t", True)], t)
        self.assertIsNotNone(d[("t", True)], t)
        self.assertIs(d[("u", True)], u)

        d[("t", True)] = t
        del d[("x", False)]
        del d[("u", True)]
        w._update_discretizations()
        self.assertEqual(len(d[("x", False)].values), 3)
        self.assertIs(d[("t", True)], t)
        self.assertIs(d[("u", True)], u)

        w.var_hints[None] = VarHint(Methods.Remove, ())
        del d[("u", True)]
        w._update_discretizations()
        self.assertIsNone(d[("u", True)])

    def test_copy_to_manual(self):
        w = self.widget
        w.var_hints = { DefaultKey: VarHint(Methods.EqualFreq, (5, )) }
        self.send_signal(w.Inputs.data, self.data)
        w.button_group.button(Methods.MDL).setChecked(True)

        select_rows(w.varview, (0, 2))
        self.assertTrue(w.copy_to_custom.isEnabled())
        w.copy_to_custom.click()
        self.assertFalse(any(w.button_group.button(i).isChecked()
                             for i in Methods))
        self.assertEqual(w.var_hints[("x", False)],
                         VarHint(Methods.Custom, ('2.5, 7.5, 12.5', )))
        self.assertEqual(w.var_hints[("z", False)],
                         VarHint(Methods.Custom, ('4.5, 9.5, 14.5', )))
        self.assertNotIn(("y", False), w.var_hints)

        select_rows(w.varview, (1, ))
        self.assertTrue(w.copy_to_custom.isEnabled())
        w.copy_to_custom.click()
        self.assertTrue(w.button_group.button(Methods.Custom).isChecked())
        self.assertEqual(w.var_hints[("x", False)],
                         VarHint(Methods.Custom, ('2.5, 7.5, 12.5', )))
        self.assertEqual(w.var_hints[("z", False)],
                         VarHint(Methods.Custom, ('4.5, 9.5, 14.5', )))
        self.assertEqual(w.var_hints[("y", False)],
                         VarHint(Methods.Custom, ('3.5, 8.5, 13.5', )))
        self.assertEqual(w.threshold_line.text(), '3.5, 8.5, 13.5')

        select_rows(w.varview, (1, 4))
        w.copy_to_custom.click()
        self.assertNotIn(("u", False), w.var_hints)

    def test_migration_2_3(self):
        # Obsolete, don't want to cause confusion by public import
        # pylint: disable=import-outside-toplevel
        from Orange.widgets.data.owdiscretize import \
            Default, EqualFreq, Leave, Custom, MDL, EqualWidth, DState
        context_values = {
            'saved_var_states':
                ({(2, 'age'): DState(method=Leave()),
                  (2, 'rest SBP'): DState(method=EqualWidth(k=4)),
                  (2, 'cholesterol'): DState(method=EqualFreq(k=6)),
                  (4, 'max HR'): DState(
                      method=Custom(points=(1.0, 2.0, 3.0))),
                  (2, 'ST by exercise'): DState(method=MDL()),
                  (2, 'major vessels colored'):
                      DState(method=Default(method=EqualFreq(k=3)))}, -2),
            '__version__': 2}

        settings = {'autosend': True, 'controlAreaVisible': True,
                    'default_cutpoints': (), 'default_k': 3,
                    'default_method_name': 'EqualFreq',
                    '__version__': 2,
                    "context_settings": [Context(values=context_values)]}

        OWDiscretize.migrate_settings(settings, 2)
        self.assertNotIn("default_method_name", settings)
        self.assertNotIn("default_k", settings)
        self.assertNotIn("default_cutpoints", settings)
        self.assertNotIn("context_settings", settings)
        self.assertEqual(
            settings["var_hints"],
            {None: VarHint(Methods.EqualFreq, (3,)),
             ('ST by exercise', False): VarHint(Methods.MDL, ()),
             ('age', False): VarHint(Methods.Keep, ()),
             ('cholesterol', False): VarHint(Methods.EqualFreq, (6,)),
             ('max HR', True): VarHint(Methods.Custom, (('1, 2, 3'),)),
             ('rest SBP', False): VarHint(Methods.EqualWidth, (4,))})


class TestValidator(unittest.TestCase):
    def test_validate(self):
        v = IncreasingNumbersListValidator()
        self.assertEqual(v.validate("", 0), (v.Intermediate, '', 0))
        self.assertEqual(v.validate("1", 1), (v.Acceptable, '1', 1))
        self.assertEqual(v.validate(",", 0), (v.Intermediate, ',', 0))
        self.assertEqual(v.validate("-", 0), (v.Intermediate, '-', 0))
        self.assertEqual(v.validate("1,,", 1), (v.Intermediate, '1,,', 1))
        self.assertEqual(v.validate("1,a,", 1), (v.Invalid, '1,a,', 3))
        self.assertEqual(v.validate("a", 1), (v.Invalid, 'a', 1))
        self.assertEqual(v.validate("1,1", 0), (v.Intermediate, '1,1', 0))
        self.assertEqual(v.validate("1,12", 0), (v.Acceptable, '1,12', 0))

        self.assertEqual(v.validate("1, 2 ", 5), (v.Intermediate, "1, 2, ", 6))


class TestModels(WidgetTest, DataMixin):
    def setUp(self):
        self.prepare_data()
        self.widget = self.create_widget(OWDiscretize)

    def test_delegate(self):
        self.prepare_data()
        w = self.widget
        w.var_hints = self.var_hints
        # Def: Keep, x: EqFreq 3, y: Keep, z: Remove, t (time): Bin 2, u (time):
        self.send_signal(w.Inputs.data, self.data)

        model = w.varview.model()
        delegate: ListViewSearch.DiscDelegate = w.varview.itemDelegate()
        option = QStyleOptionViewItem()
        delegate.initStyleOption(option, model.index(0))
        self.assertTrue(option.font.bold())

        option = QStyleOptionViewItem()
        delegate.initStyleOption(option, model.index(4))
        self.assertFalse(option.font.bold())

    def test_layout(self):
        # Not much to test, just don't crash
        self.widget.varview.updateGeometries()

    def test_model(self):
        self.prepare_data()
        w = self.widget
        w.var_hints = self.var_hints
        # Def: Keep, x: EqFreq 3, y: Keep, z: Remove, t (time): Bin 2, u (time):
        self.send_signal(w.Inputs.data, self.data)

        model = w.varview.model()
        display = model.index(0).data()
        self.assertIn("x", display)
        self.assertIn("freq", display)
        self.assertIn("3", display)
        self.assertIn(
            str(w.discretized_vars[("x", False)].compute_value.points[0])[:3],
            display)

        tooltip = model.index(0).data(Qt.ToolTipRole)
        self.assertIn("x", tooltip)
        self.assertIn(
            str(w.discretized_vars[("x", False)].compute_value.points[0])[:3],
            tooltip)

        display = model.index(1).data()
        self.assertIn("y", display)
        self.assertIn("keep", display)

        self.assertIsNone(model.index(1).data(Qt.ToolTipRole))

        w.var_hints[("x", False)] = VarHint(Methods.EqualWidth, (7, ))
        del w.discretized_vars[("x", False)]
        w._update_discretizations()
        display = model.index(0).data()
        self.assertIn("x", display)
        self.assertIn("width", display)
        self.assertIn("3", display)
        self.assertIn(
            str(w.discretized_vars[("x", False)].compute_value.points[0])[:3],
            display)


class TestDiscModel(GuiTest, DataMixin):
    def setUp(self) -> None:
        super().setUp()
        self.prepare_data()

    def test_model(self):
        model = DiscDomainModel()
        model.set_domain(self.domain)
        index = model.index(0)
        self.assertEqual(index.data(Qt.DisplayRole), "x")
        self.assertIn("x", index.data(Qt.ToolTipRole), "x")
        model.setData(
            index,
            DiscDesc(
                VarHint(Methods.EqualFreq, (3, )), "1, 2", ("1", "2")),
            Qt.UserRole
        )
        self.assertTrue(index.data(Qt.DisplayRole).startswith("x "))
        self.assertIn("2", index.data(Qt.ToolTipRole))


class TestDefaultDiscModel(GuiTest):
    def test_counts(self):
        model = DefaultDiscModel()
        self.assertEqual(model.rowCount(QModelIndex()), 1)
        self.assertEqual(model.rowCount(model.index(0)), 0)

        self.assertEqual(model.columnCount(QModelIndex()), 1)
        self.assertEqual(model.columnCount(model.index(0)), 0)

    def test_data(self):
        model = DefaultDiscModel()
        self.assertIn(format_desc(DefaultHint), model.index(0).data())
        self.assertIsInstance(model.index(0).data(Qt.DecorationRole), QIcon)
        self.assertIsInstance(model.index(0).data(Qt.ToolTipRole), str)

        hint = VarHint(Methods.FixedWidth, ("314", ))
        model.setData(model.index(0), hint, Qt.UserRole)
        self.assertIn(format_desc(hint), model.index(0).data())
        self.assertIsInstance(model.index(0).data(Qt.DecorationRole), QIcon)
        self.assertIsInstance(model.index(0).data(Qt.ToolTipRole), str)



class TestUtils(GuiTest):
    def test_show_tip(self):
        w = QWidget()
        show_tip = IncreasingNumbersListValidator.show_tip
        show_tip(w, QPoint(100, 100), "Ha Ha")
        app = QApplication.instance()
        windows = app.topLevelWidgets()
        label = [tl for tl in windows
                 if tl.parent() is w and tl.objectName() == "tip-label"][0]
        self.assertTrue(label.isVisible())
        self.assertTrue(label.text() == "Ha Ha")

        show_tip(w, QPoint(100, 100), "Ha")
        self.assertTrue(label.text() == "Ha")
        show_tip(w, QPoint(100, 100), "")
        self.assertFalse(label.isVisible())

    def test_format_desc(self):
        self.assertEqual(format_desc(VarHint(Methods.MDL, ())),
                         Options[Methods.MDL].short_desc)
        self.assertEqual(format_desc(VarHint(Methods.EqualWidth, ("10", ))),
                         Options[Methods.EqualWidth].short_desc.format(10))
        self.assertEqual(format_desc(None),
                         Options[Methods.Default].short_desc)

        fwt = Methods.FixedWidthTime
        desc = Options[fwt].short_desc.format
        self.assertEqual(format_desc(VarHint(fwt, ("1", 0))), desc("1", "year"))
        self.assertEqual(format_desc(VarHint(fwt, ("2", 0))), desc("2", "years"))
        self.assertEqual(format_desc(VarHint(fwt, ("1", 2))), desc("1", "day"))
        self.assertEqual(format_desc(VarHint(fwt, ("2", 2))), desc("2", "days"))
        self.assertEqual(format_desc(VarHint(fwt, ("x", 2))), desc("x", "day(s)"))
        self.assertEqual(format_desc(VarHint(fwt, ("", 2))), desc("", "day(s)"))

    def test_fixed_width_disc(self):
        fw = partial(_fixed_width_discretization, None, None)
        for arg in ("", "5.3.1", "abc", "-5", "0"):
            self.assertIsInstance(fw(arg), str)

        with patch("Orange.preprocess.discretize.FixedWidth") as disc:
            self.assertNotIsInstance(fw("5.13"), str)
            disc.assert_called_with(5.13, 2)

            self.assertNotIsInstance(fw("5"), str)
            disc.assert_called_with(5, 0)

        with patch("Orange.preprocess.discretize.FixedWidth",
                   side_effect=TooManyIntervals):
            self.assertIsInstance(fw("42"), str)

    def test_fixed_time_width_disc(self):
        ftw = partial(_fixed_time_width_discretization, None, None)

        for arg in ("", "5.3.1", "5.3", "abc", "-5", "0"):
            self.assertIsInstance(ftw(arg, 1), str)

        with patch("Orange.preprocess.discretize.FixedTimeWidth") as disc:
            self.assertNotIsInstance(ftw("5", 2), str)
            disc.assert_called_with(5, 2)

            self.assertNotIsInstance(ftw("5", 3), str)
            disc.assert_called_with(35, 2)

            self.assertNotIsInstance(ftw("5", 4), str)
            disc.assert_called_with(5, 3)

        with patch("Orange.preprocess.discretize.FixedTimeWidth",
                   side_effect=TooManyIntervals):
            self.assertIsInstance(ftw("42", 3), str)

    def test_custom_discretization(self):
        cd = partial(_custom_discretization, None, None)

        for arg in ("", "4 5", "2, 1, 5", "1, foo, 13"):
            self.assertIsInstance(cd(arg), str)

        with patch("Orange.preprocess.discretize.Discretizer."
                   "create_discretized_var") as disc:
            cd("1, 1.25, 1.5, 4")
            disc.assert_called_with(None, [1, 1.25, 1.5, 4])

    def test_mdl_discretization(self):
        mdl = _mdl_discretization
        data = Table("iris")[::10]
        var = data.domain[0]
        with patch("Orange.preprocess.discretize.EntropyMDL") as mdldisc:
            mdl(data, var)
            mdldisc.return_value.assert_called_with(data, var)
            mdldisc.reset_mock()

            data = data[:, :4]
            self.assertIsInstance(mdl(data, var), str)
            mdldisc.assert_not_called()

            data = data.transform(Domain(data.domain[:3], data.domain[3]))
            self.assertIsInstance(mdl(data, var), str)
            mdldisc.assert_not_called()

    def test_var_key(self):
        self.assertEqual(variable_key(ContinuousVariable("foo")),
                         ("foo", False))
        self.assertEqual(variable_key(TimeVariable("bar")),
                         ("bar", True))


if __name__ == "__main__":
    unittest.main()

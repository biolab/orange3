# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object,protected-access
import unittest
from functools import partial
from unittest.mock import patch

from AnyQt.QtCore import QPoint
from AnyQt.QtWidgets import QWidget, QApplication

from orangewidget.settings import Context

from Orange.data import Table, ContinuousVariable, TimeVariable
from Orange.preprocess.discretize import TooManyIntervals
from Orange.widgets.data.owdiscretize import OWDiscretize, \
    IncreasingNumbersListValidator, VarHint, Methods, DefaultKey, \
    _fixed_width_discretization, _fixed_time_width_discretization, \
    _custom_discretization, variable_key, DefaultHint
from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.widgets.utils.itemmodels import select_row


class TestOWDiscretize(WidgetTest):
    def setUp(self):
        super().setUp()
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

    """
    def test_select_method(self):
        widget = self.widget
        data = Table("iris")[::5]
        self.send_signal(self.widget.Inputs.data, data)

        model = widget.varmodel
        view = widget.varview
        defbg = widget.default_button_group
        varbg = widget.variable_button_group
        self.assertSequenceEqual(list(model), data.domain.attributes)
        defbg.button(OWDiscretize.EqualFreq).click()
        self.assertEqual(widget.default_method, OWDiscretize.EqualFreq)
        self.assertTrue(
            all(isinstance(m, Default) and isinstance(m.method, EqualFreq)
                for m in map(widget.method_for_index,
                             range(len(data.domain.attributes)))))

        # change method for first variable
        select_row(view, 0)
        varbg.button(OWDiscretize.Remove).click()
        met = widget.method_for_index(0)
        self.assertIsInstance(met, Remove)

        # select a second var
        selmodel = view.selectionModel()
        selmodel.select(model.index(2), selmodel.Select)
        # the current checked button must unset
        self.assertEqual(varbg.checkedId(), -1)

        varbg.button(OWDiscretize.Leave).click()
        self.assertIsInstance(widget.method_for_index(0), Leave)
        self.assertIsInstance(widget.method_for_index(2), Leave)
        # reset both back to default
        varbg.button(OWDiscretize.Default).click()
        self.assertIsInstance(widget.method_for_index(0), Default)
        self.assertIsInstance(widget.method_for_index(2), Default)

    def test_manual_cuts_edit(self):
        widget = self.widget
        data = Table("iris")[::5]
        self.send_signal(self.widget.Inputs.data, data)
        view = widget.varview
        varbg = widget.variable_button_group
        widget.set_default_method(OWDiscretize.Custom)
        widget.default_cutpoints = (0, 2, 4)
        ledit = widget.manual_cuts_edit
        self.assertEqual(ledit.text(), "0, 2, 4")
        ledit.setText("3, 4, 5")
        ledit.editingFinished.emit()
        self.assertEqual(widget.default_cutpoints, (3, 4, 5))
        self.assertEqual(widget._current_default_method(), Custom((3, 4, 5)))
        self.assertTrue(
            all(widget.method_for_index(i) == Default(Custom((3, 4, 5)))
                for i in range(len(data.domain.attributes)))
        )
        select_row(view, 0)
        varbg.button(OWDiscretize.Custom).click()
        ledit = widget.manual_cuts_specific
        ledit.setText("1, 2, 3")
        ledit.editingFinished.emit()
        self.assertEqual(widget.method_for_index(0), Custom((1, 2, 3)))
        ledit.setText("")
        ledit.editingFinished.emit()
        self.assertEqual(widget.method_for_index(0), Custom(()))
"""
    def test_report(self):
        widget = self.widget
        data = Table("iris")[::5]
        self.send_signal(widget.Inputs.data, data)
        widget.send_report()

    def test_all(self):
        data = Table("brown-selected")

        w = self.create_widget(
            OWDiscretize,
            dict(context_settings=[Context(values={"var_hints":
                {None: VarHint(Methods.EqualFreq, (3,)),
                 ('alpha 0', False): VarHint(Methods.Keep, ()),
                 ('alpha 7', False): VarHint(Methods.Remove, ()),
                 ('alpha 14', False): VarHint(Methods.Binning, (2, )),
                 ('alpha 21', False): VarHint(Methods.FixedWidth, ("0.05", )),
                 ('alpha 28', False): VarHint(Methods.EqualFreq, (4, )),
                 ('alpha 35', False): VarHint(Methods.MDL, ()),
                 ('alpha 42', False): VarHint(Methods.Custom, ("0, 0.125", )),
                 ('alpha 49', False): VarHint(Methods.MDL, ())}})]))

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
        self.assertEqual(list(w.var_hints), [DefaultKey])
        self.assertEqual(w.discretized_vars, {})
        self.assertEqual(w.varview.default_view.model().hint, DefaultHint)
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
                  (4, 'max HR'): DState(method=Custom(points=(1.0, 2.0, 3.0))),
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
        values = settings["context_settings"][0].values
        self.assertNotIn("saved_var_states", values)
        self.assertEqual(
            values["var_hints"],
            {None: VarHint(Methods.EqualFreq, (3,)),
             ('ST by exercise', False): VarHint(Methods.MDL, ()),
             ('age', False): VarHint(Methods.Keep, ()),
             ('cholesterol', False): VarHint(Methods.EqualFreq, (6, )),
             ('max HR', True): VarHint(Methods.Custom, (('1, 2, 3'), )),
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

    def test_fixed_width_disc(self):
        fw = partial(_fixed_width_discretization, None, None)
        for arg in ("", "5.3.1", "abc"):
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

        for arg in ("", "5.3.1", "5.3", "abc"):
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

    def test_var_key(self):
        self.assertEqual(variable_key(ContinuousVariable("foo")),
                         ("foo", False))
        self.assertEqual(variable_key(TimeVariable("bar")),
                         ("bar", True))


if __name__ == "__main__":
    unittest.main()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object,protected-access
import unittest

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtWidgets import QWidget, QApplication, QStyleOptionViewItem

from Orange.data import Table, DiscreteVariable
from Orange.widgets.data.owdiscretize import OWDiscretize, Default, EqualFreq, \
    Remove, Leave, Custom, IncreasingNumbersListValidator, DiscDelegate, MDL, \
    EqualWidth, DState, show_tip
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.itemmodels import select_row, VariableListModel


class TestOWDiscretize(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWDiscretize)

    def test_empty_data(self):
        data = Table("iris")
        widget = self.widget
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))
        for m in (OWDiscretize.Leave, OWDiscretize.MDL, OWDiscretize.EqualFreq,
                  OWDiscretize.EqualWidth, OWDiscretize.Remove,
                  OWDiscretize.Custom):
            widget.default_method = m
            widget.commit.now()
            self.assertIsNotNone(self.get_output(widget.Outputs.data))

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

    def test_migration(self):
        w = self.create_widget(OWDiscretize, stored_settings={
            "default_method": 0
        })
        self.assertEqual(w.default_method, OWDiscretize.Leave)

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

    def test_manual_cuts_copy(self):
        widget = self.widget
        data = Table("iris")[::5]
        self.send_signal(self.widget.Inputs.data, data)
        view = widget.varview
        select_row(view, 0)
        varbg = widget.variable_button_group
        varbg.button(OWDiscretize.EqualWidth).click()
        v = widget.discretized_var(0)
        points = tuple(v.compute_value.points)
        cc_button = widget.copy_current_to_manual_button
        cc_button.click()
        self.assertEqual(widget.method_for_index(0), Custom(points))
        self.assertEqual(varbg.checkedId(), OWDiscretize.Custom)

    def test_report(self):
        widget = self.widget
        data = Table("iris")[::5]
        self.send_signal(widget.Inputs.data, data)
        widget.send_report()


class TestValidator(unittest.TestCase):
    def test_validate(self):
        v = IncreasingNumbersListValidator()
        self.assertEqual(v.validate("", 0), (v.Acceptable, '', 0))
        self.assertEqual(v.validate("1", 1), (v.Acceptable, '1', 1))
        self.assertEqual(v.validate(",", 0), (v.Acceptable, ',', 0))
        self.assertEqual(v.validate("-", 0), (v.Intermediate, '-', 0))
        self.assertEqual(v.validate("1,,", 1), (v.Acceptable, '1,,', 1))
        self.assertEqual(v.validate("1,a,", 1), (v.Invalid, '1,a,', 1))
        self.assertEqual(v.validate("a", 1), (v.Invalid, 'a', 1))
        self.assertEqual(v.validate("1,1", 0), (v.Intermediate, '1,1', 0))
        self.assertEqual(v.validate("1,12", 0), (v.Acceptable, '1,12', 0))

    def test_fixup(self):
        v = IncreasingNumbersListValidator()
        self.assertEqual(v.fixup(""), "")
        self.assertEqual(v.fixup("1,,2"), "1, 2")
        self.assertEqual(v.fixup("1,,"), "1")
        self.assertEqual(v.fixup("1,"), "1")
        self.assertEqual(v.fixup(",1"), "1")
        self.assertEqual(v.fixup(","), "")


class TestDelegate(GuiTest):
    def test_delegate(self):
        cases = (
            (DState(Default(Leave()), None, None), ""),
            (DState(Leave(), None, None), "(leave)"),
            (DState(MDL(), [1], None), "(entropy)"),
            (DState(MDL(), [], None), "<removed>"),
            (DState(EqualFreq(2), [1], None), "(equal frequency k=2)"),
            (DState(EqualWidth(2), [1], None), "(equal width k=2)"),
            (DState(Remove(), None, None), "(removed)"),
            (DState(Custom([1]), None, None), "(custom)"),
        )
        delegate = DiscDelegate()
        var = DiscreteVariable("C", ("a", "b"))
        model = VariableListModel()
        model.append(var)
        for state, text in cases:
            model.setData(model.index(0), state, Qt.UserRole)
            option = QStyleOptionViewItem()
            delegate.initStyleOption(option, model.index(0))
            self.assertIn(text, option.text)


class TestShowTip(GuiTest):
    def test_show_tip(self):
        w = QWidget()
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

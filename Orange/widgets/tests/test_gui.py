import unittest
from unittest.mock import patch

import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.gui import BarRatioTableModel
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget


class TestDoubleSpin(GuiTest):
    # make sure that the gui element does not crash when
    # 'checked' parameter is forwarded, ie. is not None
    def test_checked_extension(self):
        widget = OWWidget()
        widget.some_param = 0
        widget.some_option = False
        gui.doubleSpin(widget=widget, master=widget, value="some_param",
                       minv=1, maxv=10, checked="some_option")


class TestListModel(GuiTest):
    def setUp(self):
        self.widget = OWWidget()
        self.widget.foo = None
        self.attrs = VariableListModel()
        self.view = gui.listView(
            self.widget.controlArea, self.widget, "foo", model=self.attrs)

    def tearDown(self) -> None:
        self.widget.deleteLater()
        del self.widget

    def test_select_callback(self):
        widget = self.widget
        view = self.view

        self.assertIsNone(widget.foo)

        a, b, c = (ContinuousVariable(x) for x in "abc")
        self.attrs[:] = [a, b, c]

        view.setCurrentIndex(self.attrs.index(0, 0))
        self.assertIs(widget.foo, a)
        view.setCurrentIndex(self.attrs.index(2, 0))
        self.assertIs(widget.foo, c)

        view.setSelectionMode(view.MultiSelection)
        sel_model = view.selectionModel()
        sel_model.clear()
        view.setCurrentIndex(self.attrs.index(1, 0))
        self.assertEqual(widget.foo, [b])

        # unselect all
        sel_model.clear()
        self.assertEqual(widget.foo, [])

    def test_select_callfront(self):
        widget = self.widget
        view = self.view

        a, b, c = (ContinuousVariable(x) for x in "abc")
        self.attrs[:] = [a, b, c]

        widget.foo = b
        selection = view.selectedIndexes()
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0].row(), 1)

        view.setSelectionMode(view.MultiSelection)
        widget.foo = [a, c]
        selection = view.selectedIndexes()
        self.assertEqual(len(selection), 2)
        self.assertEqual({selection[0].row(), selection[1].row()}, {0, 2})

        widget.foo = []
        selection = view.selectedIndexes()
        self.assertEqual(len(selection), 0)

        widget.foo = [2, "b"]
        selection = view.selectedIndexes()
        self.assertEqual(len(selection), 2)
        self.assertEqual({selection[0].row(), selection[1].row()}, {1, 2})


class TestFloatSlider(GuiTest):

    def test_set_value(self):
        w = gui.FloatSlider(Qt.Horizontal, 0., 1., 0.5)
        w.setValue(1)
        # Float slider returns value divided by step
        # 1/0.5 = 2
        self.assertEqual(w.value(), 2)
        w = gui.FloatSlider(Qt.Horizontal, 0., 1., 0.05)
        w.setValue(1)
        # 1/0.05 = 20
        self.assertEqual(w.value(), 20)


class ComboBoxTest(GuiTest):
    def test_set_initial_value(self):
        widget = OWWidget()
        variables = [ContinuousVariable(x) for x in "abc"]
        model = VariableListModel(variables)
        widget.foo = variables[1]
        combo = gui.comboBox(widget.controlArea, widget, "foo", model=model)
        self.assertEqual(combo.currentIndex(), 1)

    @patch("Orange.widgets.gui.gui_comboBox")
    def test_warn_value_type(self, gui_combobox):
        with self.assertWarns(DeprecationWarning):
            gui.comboBox(None, None, "foo", valueType=int, editable=True)


class TestRankModel(GuiTest):
    @staticmethod
    def test_argsort():
        func = BarRatioTableModel()._argsortData  # pylint: disable=protected-access
        assert_equal = np.testing.assert_equal

        test_array = np.array([4.2, 7.2, np.nan, 1.3, np.nan])
        assert_equal(func(test_array, Qt.AscendingOrder)[:3], [3, 0, 1])
        assert_equal(func(test_array, Qt.DescendingOrder)[:3], [1, 0, 3])

        test_array = np.array([4, 7, 2])
        assert_equal(func(test_array, Qt.AscendingOrder), [2, 0, 1])
        assert_equal(func(test_array, Qt.DescendingOrder), [1, 0, 2])

        test_array = np.array(["Bertha", "daniela", "ann", "Cecilia"])
        assert_equal(func(test_array, Qt.AscendingOrder), [2, 0, 3, 1])
        assert_equal(func(test_array, Qt.DescendingOrder), [1, 3, 0, 2])


if __name__ == "__main__":
    unittest.main()

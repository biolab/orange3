from AnyQt.QtCore import Qt

from Orange.data import ContinuousVariable
from Orange.widgets import gui
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
        self.assertEqual(w.value(), 2)
        w = gui.FloatSlider(Qt.Horizontal, 0., 1., 0.05)
        w.setValue(1)
        self.assertEqual(w.value(), 20)


class ComboBoxTest(GuiTest):
    def test_set_initial_value(self):
        widget = OWWidget()
        variables = [ContinuousVariable(x) for x in "abc"]
        model = VariableListModel(variables)
        widget.foo = variables[1]
        combo = gui.comboBox(widget.controlArea, widget, "foo", model=model)
        self.assertEqual(combo.currentIndex(), 1)

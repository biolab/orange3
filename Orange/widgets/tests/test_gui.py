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
    def test_select(self):
        widget = OWWidget()
        widget.foo = None
        self.attrs = VariableListModel()
        view = gui.listView(widget.controlArea, widget, "foo", model=self.attrs)
        self.assertIsNone(widget.foo)
        a, b, c = (ContinuousVariable(x) for x in "abc")
        self.attrs[:] = [a, b, c]
        view.setCurrentIndex(self.attrs.index(0, 0))
        self.assertIs(widget.foo, a)
        view.setCurrentIndex(self.attrs.index(2, 0))
        self.assertIs(widget.foo, c)

        widget.foo = b
        selection = view.selectedIndexes()
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0].row(), 1)

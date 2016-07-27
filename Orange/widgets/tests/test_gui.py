from Orange.widgets import gui
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.widget import OWWidget


class OWTestDoubleSpin(GuiTest):
    some_param = 0
    some_option = False

    # make sure that the gui element does not crash when
    # 'checked' parameter is forwarded, ie. is not None
    def test_checked_extension(self):
        widget = OWWidget()
        gui.doubleSpin(widget=widget, master=self, value="some_param",
                       minv=1, maxv=10, checked="some_option")

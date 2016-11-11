# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets import gui
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import SettingProvider
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget


class DummyComponent(OWComponent):
    foo = True


class MyWidget(OWWidget):
    foo = True

    component = SettingProvider(DummyComponent)

    def __init__(self):
        super().__init__()

        self.component = DummyComponent(self)
        self.foo_control = gui.checkBox(self.controlArea, self, "foo", "")
        self.component_foo_control = \
            gui.checkBox(self.controlArea, self, "component.foo", "")


class ControlGetterTests(WidgetTest):
    def test_getter(self):
        widget = self.create_widget(MyWidget)
        self.assertIs(widget.controls.foo, widget.foo_control)
        self.assertIs(widget.controls.component.foo,
                      widget.component_foo_control)

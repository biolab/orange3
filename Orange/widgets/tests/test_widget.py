# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.gui import CONTROLLED_ATTRIBUTES, OWComponent
from Orange.widgets.settings import Setting
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget


class DummyComponent(OWComponent):
    b = None


class MyWidget(OWWidget):
    name = "Dummy"

    field = Setting(42)

    def __init__(self):
        super().__init__()

        self.component = DummyComponent(self)
        self.widget = None


class WidgetTestCase(WidgetTest):
    def test_setattr(self):
        widget = self.create_widget(MyWidget)
        widget.widget = self.create_widget(MyWidget)

        setattr(widget, 'field', 1)
        self.assertEqual(widget.field, 1)

        setattr(widget, 'component.b', 2)
        self.assertEqual(widget.component.b, 2)

        setattr(widget, 'widget.field', 3)
        self.assertEqual(widget.widget.field, 3)

        setattr(widget, 'unknown_field', 4)
        self.assertEqual(widget.unknown_field, 4)

        with self.assertRaises(AttributeError):
            setattr(widget, 'widget.widget.field', 5)

        with self.assertRaises(AttributeError):
            setattr(widget, 'unknown_field2.field', 6)

    def test_notify_controller_on_attribute_change(self):
        widget = self.create_widget(MyWidget)
        widget.widget = self.create_widget(MyWidget)
        widget.widget.widget = self.create_widget(MyWidget)
        widget.widget.widget.widget = self.create_widget(MyWidget)

        delattr(widget.widget, CONTROLLED_ATTRIBUTES)
        delattr(widget.widget.widget, CONTROLLED_ATTRIBUTES)
        delattr(widget.widget.widget.widget, CONTROLLED_ATTRIBUTES)
        calls = []

        def callback(*args, **kwargs):
            calls.append((args, kwargs))

        getattr(widget, CONTROLLED_ATTRIBUTES)['field'] = callback
        getattr(widget, CONTROLLED_ATTRIBUTES)['component.b'] = callback
        getattr(widget, CONTROLLED_ATTRIBUTES)['widget.field'] = callback
        getattr(widget, CONTROLLED_ATTRIBUTES)['widget.widget.component.b'] = callback
        getattr(widget, CONTROLLED_ATTRIBUTES)['widget.widget.widget.field'] = callback

        widget.field = 5
        widget.component.b = 5
        widget.widget.field = 5
        widget.widget.widget.component.b = 5
        widget.widget.widget.widget.field = 5

        self.assertEqual(len(calls), 5)

    def test_widget_tests_do_not_use_stored_settings(self):
        widget = self.create_widget(MyWidget)

        widget.field = 5
        widget.saveSettings()

        widget2 = self.create_widget(MyWidget)
        self.assertEqual(widget2.field, 42)

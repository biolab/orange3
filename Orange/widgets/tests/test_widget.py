from unittest import TestCase
from Orange.widgets.gui import CONTROLLED_ATTRIBUTES, ATTRIBUTE_CONTROLLERS, OWComponent
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.widget import OWWidget


class DummyComponent(OWComponent):
    b = None


class MyWidget(OWWidget):
    def __init__(self, depth=1):
        super().__init__()

        self.field = 42
        self.component = DummyComponent(self)
        if depth:
            self.widget = MyWidget(depth=depth-1)
        else:
            self.widget = None


class WidgetTestCase(GuiTest):
    def test_setattr(self):
        widget = MyWidget()

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
        widget = MyWidget(depth=3)
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




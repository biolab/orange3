# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from unittest.mock import patch, MagicMock

from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.widget import OWWidget, Msg


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

        callback = MagicMock()
        callback2 = MagicMock()

        widget.connect_control('field', callback)
        widget.connect_control('field', callback2)
        widget.field = 5
        self.assertTrue(callback.called)
        self.assertTrue(callback2.called)

    def test_widget_tests_do_not_use_stored_settings(self):
        widget = self.create_widget(MyWidget)

        widget.field = 5
        widget.saveSettings()

        widget2 = self.create_widget(MyWidget)
        self.assertEqual(widget2.field, 42)


class WidgetMsgTestCase(WidgetTest):

    class TestWidget(OWWidget):
        name = "Test"

        class Information(OWWidget.Information):
            hello = Msg("A message")

        def __init__(self):
            super().__init__()

            self.Information.hello()

    @staticmethod
    def active_messages(widget):
        """Return all active messages in a widget"""
        return [m for g in widget.message_groups for m in g.active]

    def test_widget_emits_messages(self):
        """Widget emits messageActivates/messageDeactivated signals"""

        w = WidgetMsgTestCase.TestWidget()
        messages = set(self.active_messages(w))

        self.assertEqual(len(messages), 1, )

        w.messageActivated.connect(messages.add)
        w.messageDeactivated.connect(messages.remove)

        w.Information.hello()
        self.assertEqual(len(messages), 1)
        self.assertSetEqual(messages, set(self.active_messages(w)))

        w.Information.hello.clear()
        self.assertEqual(len(messages), 0)
        self.assertSetEqual(set(self.active_messages(w)), set())

        with patch.object(
                WidgetMsgTestCase.TestWidget,
                "want_basic_layout", False):
            # OWWidget without a basic layout (completely empty; no default
            # message bar)
            w = WidgetMsgTestCase.TestWidget()

        messages = set(self.active_messages(w))

        w.messageActivated.connect(messages.add)
        w.messageDeactivated.connect(messages.remove)

        self.assertEqual(len(messages), 1)

        w.Information.hello.clear()
        self.assertEqual(len(messages), 0)

    def test_old_style_messages(self):
        w = WidgetMsgTestCase.TestWidget()
        w.Information.clear()

        messages = set(self.active_messages(w))

        w.messageActivated.connect(messages.add)
        w.messageDeactivated.connect(messages.remove)

        w.error(1, "A")

        self.assertEqual(len(w.Error.active), 1)
        self.assertEqual(len(messages), 1)

        w.error(1)

        self.assertEqual(len(messages), 0)
        self.assertEqual(len(w.Error.active), 0)

        w.error(2, "B")
        self.assertEqual(len(messages), 1)

        w.Error.clear()
        self.assertEqual(len(messages), 0)

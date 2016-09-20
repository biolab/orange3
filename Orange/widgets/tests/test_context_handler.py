from copy import copy
from unittest import TestCase
from unittest.mock import Mock
from Orange.widgets.settings import ContextHandler, Setting, ContextSetting, Context

__author__ = 'anze'


class SimpleWidget:
    setting = Setting(42)

    context_setting = ContextSetting(42)


class TestContextHandler(TestCase):
    def test_initialize(self):
        handler = ContextHandler()
        handler.provider = Mock()

        # Context settings from data
        widget = SimpleWidget()
        handler.initialize(widget, {'context_settings': 5})
        self.assertTrue(hasattr(widget, 'context_settings'))
        self.assertEqual(widget.context_settings, 5)

        # Default (global) context settings
        widget = SimpleWidget()
        handler.initialize(widget)
        self.assertTrue(hasattr(widget, 'context_settings'))
        self.assertEqual(widget.context_settings, handler.global_contexts)

    def test_fast_save(self):
        handler = ContextHandler()
        handler.bind(SimpleWidget)

        widget = SimpleWidget()
        handler.initialize(widget)

        context = widget.current_context = handler.new_context()
        handler.fast_save(widget, 'context_setting', 55)
        self.assertEqual(context.values['context_setting'], 55)
        self.assertEqual(handler.known_settings['context_setting'].default,
                         SimpleWidget.context_setting.default)

    def test_find_or_create_context(self):
        widget = SimpleWidget()
        handler = ContextHandler()
        handler.match = lambda context, i: (context.i == i) * 2
        handler.clone_context = lambda context, i: copy(context)

        c1, c2, c3, c4, c5, c6, c7, c8, c9 = (Context(i=i)
                                              for i in range(1, 10))

        # finding a perfect match in global_contexts should copy it to
        # the front of context_settings (and leave globals as-is)
        widget.context_settings = [c2, c5]
        handler.global_contexts = [c3, c7]
        context, new = handler.find_or_create_context(widget, 7)
        self.assertEqual(context.i, 7)
        self.assertEqual([c.i for c in widget.context_settings], [7, 2, 5])
        self.assertEqual([c.i for c in handler.global_contexts], [3, 7])

        # finding a perfect match in context_settings should move it to
        # the front of the list
        widget.context_settings = [c2, c5]
        handler.global_contexts = [c3, c7]
        context, new = handler.find_or_create_context(widget, 5)
        self.assertEqual(context.i, 5)
        self.assertEqual([c.i for c in widget.context_settings], [5, 2])
        self.assertEqual([c.i for c in handler.global_contexts], [3, 7])

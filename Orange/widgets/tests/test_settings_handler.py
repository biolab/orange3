from io import BytesIO
import os
import pickle
from tempfile import mkstemp
import unittest
from unittest.mock import patch, Mock
import warnings
from Orange.widgets.settings import SettingsHandler, Setting, SettingProvider


class SettingHandlerTestCase(unittest.TestCase):
    @patch('Orange.widgets.settings.SettingProvider', create=True)
    def test_create(self, SettingProvider):
        """:type SettingProvider: unittest.mock.Mock"""

        with patch.object(SettingsHandler, 'read_defaults'):
            handler = SettingsHandler.create(SimpleWidget)

            self.assertEqual(handler.widget_class, SimpleWidget)
            # create needs to create a SettingProvider which traverses
            # the widget definition and collects all settings and read
            # all settings and for widget class
            SettingProvider.assert_called_once_with(SimpleWidget)
            SettingsHandler.read_defaults.assert_called_once_with()

    def test_create_uses_template_if_provided(self):
        template = SettingsHandler()
        template.read_defaults = lambda: None
        template.a = 'a'
        template.b = 'b'
        handler = SettingsHandler.create(SimpleWidget, template)
        self.assertEqual(handler.a, 'a')
        self.assertEqual(handler.b, 'b')

        # create should copy the template
        handler.b = 'B'
        self.assertEqual(template.b, 'b')

    @patch('Orange.widgets.settings.store_settings', True)
    def test_read_defaults(self):
        default_settings = {'a': 5, 'b': {1: 5}}
        f, settings_file = mkstemp(suffix='.ini')
        with open(settings_file, 'wb') as f:
            pickle.dump(default_settings, f)

        handler = SettingsHandler()
        handler._get_settings_filename = lambda: settings_file
        handler.read_defaults()

        self.assertEqual(handler.defaults, default_settings)

        os.remove(settings_file)

    @patch('Orange.widgets.settings.store_settings', True)
    def test_write_defaults(self):
        f, settings_file = mkstemp(suffix='.ini')

        handler = SettingsHandler()
        handler.defaults = {'a': 5, 'b': {1: 5}}
        handler._get_settings_filename = lambda: settings_file
        handler.write_defaults()

        with open(settings_file, 'rb') as f:
            default_settings = pickle.load(f)

        self.assertEqual(handler.defaults, default_settings)

        os.remove(settings_file)

    def test_initialize_widget(self):
        handler = SettingsHandler()
        handler.defaults = {'default': 42, 'setting': 1}
        handler.provider = provider = Mock()
        provider.get_provider.return_value = provider
        widget = SimpleWidget()

        def reset_provider():
            provider.get_provider.return_value = None
            provider.reset_mock()
            provider.get_provider.return_value = provider

        # No data
        handler.initialize(widget)
        provider.initialize.assert_called_once_with(widget, {'default': 42,
                                                             'setting': 1})

        # Dictionary data
        reset_provider()
        handler.initialize(widget, {'setting': 5})
        provider.initialize.assert_called_once_with(widget, {'default': 42,
                                                             'setting': 5})

        # Pickled data
        reset_provider()
        handler.initialize(widget, pickle.dumps({'setting': 5}))
        provider.initialize.assert_called_once_with(widget, {'default': 42,
                                                             'setting': 5})

    def test_initialize_component(self):
        handler = SettingsHandler()
        handler.defaults = {'default': 42}
        provider = Mock()
        handler.provider = Mock(get_provider=Mock(return_value=provider))
        widget = SimpleWidget()

        # No data
        handler.initialize(widget)
        provider.initialize.assert_called_once_with(widget, None)

        # Dictionary data
        provider.reset_mock()
        handler.initialize(widget, {'setting': 5})
        provider.initialize.assert_called_once_with(widget, {'setting': 5})

        # Pickled data
        provider.reset_mock()
        handler.initialize(widget, pickle.dumps({'setting': 5}))
        provider.initialize.assert_called_once_with(widget, {'setting': 5})

    @patch('Orange.widgets.settings.SettingProvider', create=True)
    def test_initialize_with_no_provider(self, SettingProvider):
        """:type SettingProvider: unittest.mock.Mock"""
        handler = SettingsHandler()
        handler.provider = Mock(get_provider=Mock(return_value=None))
        provider = Mock()
        SettingProvider.return_value = provider
        widget = SimpleWidget()

        # initializing an undeclared provider should display a warning
        with warnings.catch_warnings(record=True) as w:
            handler.initialize(widget)

            self.assertEqual(1, len(w))

        SettingProvider.assert_called_once_with(SimpleWidget)
        provider.initialize.assert_called_once_with(widget, None)

    def test_fast_save(self):
        handler = SettingsHandler()
        handler.read_defaults = lambda: None
        handler.bind(SimpleWidget)

        widget = SimpleWidget()

        handler.fast_save(widget, 'component.int_setting', 5)
        self.assertEqual(Component.int_setting.default, 5)

        handler.fast_save(widget, 'non_setting', 4)


class Component:
    int_setting = Setting(42)


class SimpleWidget:
    setting = Setting(42)
    non_setting = 5

    component = SettingProvider(Component)


class WidgetWithNoProviderDeclared:
    def __init__(self):
        self.undeclared_component = Component()



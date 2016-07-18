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

    def test_read_defaults(self):
        default_settings = {'a': 5, 'b': {1: 5}}
        fd, settings_file = mkstemp(suffix='.ini')
        with open(settings_file, 'wb') as f:
            pickle.dump(default_settings, f)
        os.close(fd)

        handler = SettingsHandler()
        handler._get_settings_filename = lambda: settings_file
        handler.read_defaults()

        self.assertEqual(handler.defaults, default_settings)

        os.remove(settings_file)

    def test_write_defaults(self):
        fd, settings_file = mkstemp(suffix='.ini')

        handler = SettingsHandler()
        handler.defaults = {'a': 5, 'b': {1: 5}}
        handler._get_settings_filename = lambda: settings_file
        handler.write_defaults()

        with open(settings_file, 'rb') as f:
            default_settings = pickle.load(f)
        os.close(fd)

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

        self.assertEqual(
            handler.known_settings['component.int_setting'].default, 5)

        self.assertEqual(Component.int_setting.default, 42)

        handler.fast_save(widget, 'non_setting', 4)

    def test_fast_save_siblings_spill(self):
        handler_mk1 = SettingsHandler()
        handler_mk1.read_defaults = lambda: None
        handler_mk1.bind(SimpleWidgetMk1)

        widget_mk1 = SimpleWidgetMk1()

        handler_mk1.fast_save(widget_mk1, "setting", -1)
        handler_mk1.fast_save(widget_mk1, "component.int_setting", 1)

        self.assertEqual(
            handler_mk1.known_settings['setting'].default, -1)
        self.assertEqual(
            handler_mk1.known_settings['component.int_setting'].default, 1)

        handler_mk1.initialize(widget_mk1, data=None)
        handler_mk1.provider.providers["component"].initialize(
            widget_mk1.component, data=None)

        self.assertEqual(widget_mk1.setting, -1)
        self.assertEqual(widget_mk1.component.int_setting, 1)

        handler_mk2 = SettingsHandler()
        handler_mk2.read_defaults = lambda: None
        handler_mk2.bind(SimpleWidgetMk2)

        widget_mk2 = SimpleWidgetMk2()

        handler_mk2.initialize(widget_mk2, data=None)
        handler_mk2.provider.providers["component"].initialize(
            widget_mk2.component, data=None)

        self.assertEqual(widget_mk2.setting, 42,
                         "spils defaults into sibling classes")

        self.assertEqual(Component.int_setting.default, 42)

        self.assertEqual(widget_mk2.component.int_setting, 42,
                         "spils defaults into sibling classes")

    def test_schema_only_settings(self):
        handler = SettingsHandler()
        handler.read_defaults = lambda: None
        handler.bind(SimpleWidget)

        # fast_save should not update defaults
        widget = SimpleWidget()
        handler.fast_save(widget, 'schema_only_setting', 5)
        self.assertEqual(
            handler.known_settings['schema_only_setting'].default, None)

        # update_defaults should not update defaults
        widget.schema_only_setting = 5
        handler.update_defaults(widget)
        self.assertEqual(
            handler.known_settings['schema_only_setting'].default, None)

        # pack_data should pack setting
        widget.schema_only_setting = 5
        data = handler.pack_data(widget)
        self.assertEqual(data['schema_only_setting'], 5)


class Component:
    int_setting = Setting(42)


class SimpleWidget:
    setting = Setting(42)
    schema_only_setting = Setting(None, schema_only=True)
    non_setting = 5

    component = SettingProvider(Component)

    def __init__(self):
        self.component = Component()


class SimpleWidgetMk1(SimpleWidget):
    pass


class SimpleWidgetMk2(SimpleWidget):
    pass


class WidgetWithNoProviderDeclared:
    def __init__(self):
        self.undeclared_component = Component()

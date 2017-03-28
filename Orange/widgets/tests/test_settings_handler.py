# pylint: disable=protected-access
from contextlib import contextmanager
import os
import pickle
from tempfile import mkstemp
from sys import version_info
from platform import system
import unittest
from unittest.mock import patch, Mock, MagicMock
import warnings
from Orange.widgets.settings import SettingsHandler, Setting, SettingProvider,\
    VERSION_KEY, rename_setting, Context, migrate_str_to_variable


class SettingHandlerTestCase(unittest.TestCase):
    @patch('Orange.widgets.settings.SettingProvider', create=True)
    def test_create(self, SettingProvider):
        """:type SettingProvider: unittest.mock.Mock"""

        mock_read_defaults = Mock()
        with patch.object(SettingsHandler, 'read_defaults', mock_read_defaults):
            handler = SettingsHandler.create(SimpleWidget)

            self.assertEqual(handler.widget_class, SimpleWidget)
            # create needs to create a SettingProvider which traverses
            # the widget definition and collects all settings and read
            # all settings and for widget class
            SettingProvider.assert_called_once_with(SimpleWidget)
            mock_read_defaults.assert_called_once_with()

    def test_create_uses_template_if_provided(self):
        template = SettingsHandler()
        template.a = 'a'
        template.b = 'b'
        with self.override_default_settings(SimpleWidget):
            handler = SettingsHandler.create(SimpleWidget, template)
        self.assertEqual(handler.a, 'a')
        self.assertEqual(handler.b, 'b')

        # create should copy the template
        handler.b = 'B'
        self.assertEqual(template.b, 'b')

    def test_read_defaults(self):
        handler = SettingsHandler()
        handler.widget_class = SimpleWidget

        defaults = {'a': 5, 'b': {1: 5}}
        with self.override_default_settings(SimpleWidget, defaults):
            handler.read_defaults()

        self.assertEqual(handler.defaults, defaults)

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

    def test_write_PermissionError(self):
        """
        Error may occur while trying to open file. Therefore file
        cannot be closed. Tests if error is handled.
        GH-2147
        """
        handler = SettingsHandler()
        handler._get_settings_filename = lambda: ""
        patch_target = "Orange.widgets.settings.open" if version_info >= (3, 5) \
            else "builtins.open"
        patch_target_2 = "Orange.widgets.settings.SettingsHandler.write_defaults_file"
        patch_target_3 = "os.makedirs"

        mock2 = MagicMock()
        mock = MagicMock(side_effect=PermissionError)
        with patch(patch_target, mock),\
                patch(patch_target_2, mock2),\
                patch(patch_target_3, mock2):
            handler.write_defaults()

    def test_write_EOF_IO_Pickling_Errors(self):
        """
        Tests if errors are handled.
        GH-2147
        """
        patch_target_1 = "Orange.widgets.settings.open" if version_info >= (3, 5) \
            else "builtins.open"
        patch_target_2 = "os.remove"
        patch_target_3 = "os.makedirs"
        patch_target_4 = "Orange.widgets.settings.SettingsHandler.write_defaults_file"
        handler = SettingsHandler()
        handler._get_settings_filename = lambda: ""

        mock = MagicMock()
        with patch(patch_target_1, mock), patch(patch_target_2, mock), \
                patch(patch_target_3, mock):
            with patch(patch_target_4, side_effect=EOFError):
                handler.write_defaults()
            with patch(patch_target_4, side_effect=IOError):
                handler.write_defaults()
            with patch(patch_target_4, side_effect=pickle.PicklingError):
                handler.write_defaults()

    def test_initialize_widget(self):
        handler = SettingsHandler()
        handler.defaults = {'default': 42, 'setting': 1}
        handler.provider = provider = Mock()
        handler.widget_class = SimpleWidget
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
        handler.widget_class = SimpleWidget
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
        handler.widget_class = SimpleWidget
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

        with self.override_default_settings(SimpleWidget):
            handler.bind(SimpleWidget)

        widget = SimpleWidget()

        handler.fast_save(widget, 'component.int_setting', 5)

        self.assertEqual(
            handler.known_settings['component.int_setting'].default, 5)

        self.assertEqual(Component.int_setting.default, 42)

        handler.fast_save(widget, 'non_setting', 4)

    def test_fast_save_siblings_spill(self):
        handler_mk1 = SettingsHandler()
        with self.override_default_settings(SimpleWidgetMk1):
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
        with self.override_default_settings(SimpleWidgetMk2):
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
        with self.override_default_settings(SimpleWidget):
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

    def test_read_defaults_migrates_settings(self):
        handler = SettingsHandler()
        handler.widget_class = SimpleWidget

        migrate_settings = Mock()
        with patch.object(SimpleWidget, "migrate_settings", migrate_settings):
            # Old settings without version
            settings = {"value": 5}
            with self.override_default_settings(SimpleWidget, settings):
                handler.read_defaults()
            migrate_settings.assert_called_with(settings, 0)

            migrate_settings.reset()
            # Settings with version
            settings_with_version = dict(settings)
            settings_with_version[VERSION_KEY] = 1
            with self.override_default_settings(SimpleWidget, settings_with_version):
                handler.read_defaults()
            migrate_settings.assert_called_with(settings, 1)

    def test_initialize_migrates_settings(self):
        handler = SettingsHandler()
        with self.override_default_settings(SimpleWidget):
            handler.bind(SimpleWidget)

        widget = SimpleWidget()

        migrate_settings = Mock()
        with patch.object(SimpleWidget, "migrate_settings", migrate_settings):
            # Old settings without version
            settings = {"value": 5}

            handler.initialize(widget, settings)
            migrate_settings.assert_called_with(settings, 0)

            migrate_settings.reset_mock()
            # Settings with version

            settings_with_version = dict(settings)
            settings_with_version[VERSION_KEY] = 1
            handler.initialize(widget, settings_with_version)
            migrate_settings.assert_called_with(settings, 1)

    def test_pack_settings_stores_version(self):
        handler = SettingsHandler()
        handler.bind(SimpleWidget)

        widget = SimpleWidget()

        settings = handler.pack_data(widget)
        self.assertIn(VERSION_KEY, settings)

    def test_initialize_copies_mutables(self):
        handler = SettingsHandler()
        handler.bind(SimpleWidget)
        handler.defaults = dict(list_setting=[])

        widget = SimpleWidget()
        handler.initialize(widget)

        widget2 = SimpleWidget()
        handler.initialize(widget2)

        self.assertNotEqual(id(widget.list_setting), id(widget2.list_setting))

    @contextmanager
    def override_default_settings(self, widget, defaults=None):
        if defaults is None:
            defaults = {}

        h = SettingsHandler()
        h.widget_class = widget
        filename = h._get_settings_filename()
        with open(filename, "wb") as f:
            pickle.dump(defaults, f)

        yield

        if os.path.isfile(filename):
            os.remove(filename)


class Component:
    int_setting = Setting(42)


class SimpleWidget:
    settings_version = 1

    setting = Setting(42)
    schema_only_setting = Setting(None, schema_only=True)
    list_setting = Setting([])
    non_setting = 5

    component = SettingProvider(Component)

    def __init__(self):
        self.component = Component()

    migrate_settings = Mock()
    migrate_context = Mock()


class SimpleWidgetMk1(SimpleWidget):
    pass


class SimpleWidgetMk2(SimpleWidget):
    pass


class WidgetWithNoProviderDeclared:
    def __init__(self):
        self.undeclared_component = Component()


class MigrationsTestCase(unittest.TestCase):
    def test_rename_settings(self):
        some_settings = dict(foo=42, bar=13)
        rename_setting(some_settings, "foo", "baz")
        self.assertDictEqual(some_settings, dict(baz=42, bar=13))

        self.assertRaises(KeyError, rename_setting, some_settings, "qux", "quux")

        context = Context(values=dict(foo=42, bar=13))
        rename_setting(context, "foo", "baz")
        self.assertDictEqual(context.values, dict(baz=42, bar=13))

    def test_migrate_str_to_variable(self):
        values = dict(foo=("foo", 1), baz=("baz", 2), qux=("qux", 102), bar=13)

        context = Context(values=values.copy())
        migrate_str_to_variable(context)
        self.assertDictEqual(
            context.values,
            dict(foo=("foo", 101), baz=("baz", 102), qux=("qux", 102), bar=13))

        context = Context(values=values.copy())
        migrate_str_to_variable(context, ("foo", "qux"))
        self.assertDictEqual(
            context.values,
            dict(foo=("foo", 101), baz=("baz", 2), qux=("qux", 102), bar=13))

        context = Context(values=values.copy())
        migrate_str_to_variable(context, "foo")
        self.assertDictEqual(
            context.values,
            dict(foo=("foo", 101), baz=("baz", 2), qux=("qux", 102), bar=13))

        self.assertRaises(KeyError, migrate_str_to_variable, context, "quuux")

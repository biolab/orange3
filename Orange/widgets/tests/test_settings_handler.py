from io import BytesIO
import pickle
import unittest
import warnings
from Orange.widgets.settings import SettingsHandler, Setting
from Orange.widgets.tests.test_settings import MockWidget, MockComponent


class TestSettingsHandler(SettingsHandler):
    def __init__(self, saved_defaults={}):
        """
        Setting handler that overrides default settings with the values passed
        in the parameter saved_defaults (and not from disk).

        :param saved_defaults: a dict of default values that will override the
        defaults defined on settings.
        """

        super().__init__()
        self.saved_defaults = saved_defaults

    def read_defaults(self):
        settings_file = BytesIO(pickle.dumps(self.saved_defaults))
        self.read_defaults_file(settings_file)

    def write_defaults(self):
        settings_file = BytesIO()
        self.write_defaults_file(settings_file)
        settings_file.seek(0)
        self.saved_defaults = pickle.load(settings_file)


class SettingHandlerTestCase(unittest.TestCase):
    def test_initialization_of_not_declared_provider(self):
        widget = WidgetWithNoProviderDeclared()
        handler = SettingsHandler.create(WidgetWithNoProviderDeclared)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            handler.initialize(widget)
            handler.initialize(widget.undeclared_component)

        self.assertIsInstance(widget.undeclared_component.int_setting, int)

    def test_initialization_of_child_provider_with_default_data(self):
        handler = self.handler_with_defaults({'subprovider': {'setting': "12345"}})

        widget = MockWidget()
        handler.initialize(widget)

        self.assertEqual(widget.subprovider.setting, "12345")

    def test_delayed_initialization_of_child_provider_with_default_data(self):
        handler = self.handler_with_defaults({'subprovider': {'setting': "12345"}})

        widget = MockWidget.__new__(MockWidget)
        handler.initialize(widget)
        widget.subprovider = MockComponent()
        handler.initialize(widget.subprovider)

        self.assertEqual(widget.subprovider.setting, "12345")

    def test_reading_defaults(self):
        handler = self.handler_with_defaults({"string_setting": "12345"})

        widget = MockWidget()
        handler.initialize(widget)
        self.assertEqual(widget.string_setting, "12345")

    def test_writing_defaults(self):
        handler = self.handler_with_defaults({})

        widget = MockWidget()
        handler.initialize(widget)
        handler.initialize(widget.subprovider)
        widget.string_setting = "12345"
        handler.update_defaults(widget)
        handler.write_defaults()
        self.assertEqual(handler.saved_defaults["string_setting"], "12345")

    def handler_with_defaults(self, defaults):
        handler = TestSettingsHandler()
        handler.saved_defaults = defaults
        handler.bind(MockWidget)
        return handler


class WidgetWithNoProviderDeclared:
    name = "WidgetWithNoProviderDeclared"

    def __init__(self):
        super().__init__()

        self.undeclared_component = UndeclaredComponent()


class UndeclaredComponent:
    int_setting = Setting(42)

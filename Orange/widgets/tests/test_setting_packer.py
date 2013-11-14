import unittest
from Orange.widgets.settings import Setting, SettingProvider

SHOW_ZOOM_TOOLBAR = "show_zoom_toolbar"
SHOW_GRAPH = "show_graph"
GRAPH = "graph"
ZOOM_TOOLBAR = "zoom_toolbar"
SHOW_LABELS = "show_labels"
SHOW_X_AXIS = "show_x_axis"
SHOW_Y_AXIS = "show_y_axis"
ALLOW_ZOOMING = "allow_zooming"


default_provider = None


def initialize_settings(instance):
    """This is usually done in Widget's new,
    but we avoide all that complications for tests."""
    provider = default_provider.get_provider(instance.__class__)
    if provider:
        provider.initialize(instance)


class BaseGraph:
    show_labels = Setting(True)

    def __init__(self):
        initialize_settings(self)


class Graph(BaseGraph):
    show_x_axis = Setting(True)
    show_y_axis = Setting(True)

    def __init__(self):
        super().__init__()
        initialize_settings(self)


class ExtendedGraph(Graph):
    pass


class ZoomToolbar:
    allow_zooming = Setting(True)

    def __init__(self):
        initialize_settings(self)


class BaseWidget:
    settingsHandler = None

    show_graph = Setting(True)

    graph = SettingProvider(Graph)

    def __init__(self):
        initialize_settings(self)
        self.graph = Graph()


class Widget(BaseWidget):
    show_zoom_toolbar = Setting(True)

    zoom_toolbar = SettingProvider(ZoomToolbar)

    def __init__(self):
        super().__init__()
        initialize_settings(self)

        self.zoom_toolbar = ZoomToolbar()


class SettingProviderTestCase(unittest.TestCase):
    def setUp(self):
        global default_provider
        default_provider = SettingProvider(Widget)

    def tearDown(self):
        default_provider.settings[SHOW_GRAPH].default = True
        default_provider.settings[SHOW_ZOOM_TOOLBAR].default = True
        default_provider.providers[GRAPH].settings[SHOW_LABELS].default = True
        default_provider.providers[GRAPH].settings[SHOW_X_AXIS].default = True
        default_provider.providers[GRAPH].settings[SHOW_Y_AXIS].default = True
        default_provider.providers[ZOOM_TOOLBAR].settings[ALLOW_ZOOMING].default = True

    def test_registers_all_settings(self):
        self.assertDefaultSettingsEqual(default_provider, {
            GRAPH: {
                SHOW_LABELS: True,
                SHOW_X_AXIS: True,
                SHOW_Y_AXIS: True,
            },
            ZOOM_TOOLBAR: {
                ALLOW_ZOOMING: True,
            },
            SHOW_ZOOM_TOOLBAR: True,
            SHOW_GRAPH: True,
        })

    def test_sets_defaults(self):
        default_provider.set_defaults({
            SHOW_ZOOM_TOOLBAR: Setting(False),
            SHOW_GRAPH: Setting(False),
            GRAPH: {
                SHOW_LABELS: Setting(False),
                SHOW_X_AXIS: Setting(False),
            },
            ZOOM_TOOLBAR: {
                ALLOW_ZOOMING: Setting(False),
            }
        })

        self.assertDefaultSettingsEqual(default_provider, {
            GRAPH: {
                SHOW_LABELS: False,
                SHOW_X_AXIS: False,
                SHOW_Y_AXIS: True,
            },
            ZOOM_TOOLBAR: {
                ALLOW_ZOOMING: False,
            },
            SHOW_ZOOM_TOOLBAR: False,
            SHOW_GRAPH: False,
        })

    def test_gets_defaults(self):
        default_provider.settings[SHOW_GRAPH].default = False
        default_provider.providers[GRAPH].settings[SHOW_LABELS].default = False

        defaults = default_provider.get_defaults()

        self.assertEqual(defaults[SHOW_GRAPH].default, False)
        self.assertEqual(defaults[SHOW_ZOOM_TOOLBAR].default, True)
        self.assertEqual(defaults[GRAPH][SHOW_LABELS].default, False)
        self.assertEqual(defaults[GRAPH][SHOW_X_AXIS].default, True)
        self.assertEqual(defaults[GRAPH][SHOW_Y_AXIS].default, True)
        self.assertEqual(defaults[ZOOM_TOOLBAR][ALLOW_ZOOMING].default, True)

    def test_updates_defaults(self):
        widget = Widget()

        widget.show_graph = False
        widget.graph.show_y_axis = False

        default_provider.update_defaults(widget)

        self.assertDefaultSettingsEqual(default_provider, {
            GRAPH: {
                SHOW_LABELS: True,
                SHOW_X_AXIS: True,
                SHOW_Y_AXIS: False,
            },
            ZOOM_TOOLBAR: {
                ALLOW_ZOOMING: True,
            },
            SHOW_ZOOM_TOOLBAR: True,
            SHOW_GRAPH: False,
        })

    def test_initialize_sets_defaults(self):
        widget = Widget()

        self.assertEqual(widget.show_graph, True)
        self.assertEqual(widget.show_zoom_toolbar, True)
        self.assertEqual(widget.graph.show_labels, True)
        self.assertEqual(widget.graph.show_x_axis, True)
        self.assertEqual(widget.graph.show_y_axis, True)
        self.assertEqual(widget.zoom_toolbar.allow_zooming, True)

    def test_initialize_with_data_sets_values_from_data(self):
        widget = Widget()
        default_provider.initialize(widget, {
            SHOW_GRAPH: False,
            GRAPH: {
                SHOW_Y_AXIS: False
            }
        })

        self.assertEqual(widget.show_graph, False)
        self.assertEqual(widget.show_zoom_toolbar, True)
        self.assertEqual(widget.graph.show_labels, True)
        self.assertEqual(widget.graph.show_x_axis, True)
        self.assertEqual(widget.graph.show_y_axis, False)
        self.assertEqual(widget.zoom_toolbar.allow_zooming, True)

    def test_initialize_with_data_stores_initial_values_until_instance_is_connected(self):
        widget = Widget.__new__(Widget)

        default_provider.initialize(widget, {
            SHOW_GRAPH: False,
            GRAPH: {
                SHOW_Y_AXIS: False
            }
        })

        self.assertFalse(hasattr(widget.graph, SHOW_Y_AXIS))
        widget.graph = Graph()
        self.assertEqual(widget.graph.show_y_axis, False)

    def test_get_provider(self):
        self.assertEqual(default_provider.get_provider(BaseWidget), None)
        self.assertEqual(default_provider.get_provider(Widget), default_provider)
        self.assertEqual(default_provider.get_provider(BaseGraph), None)
        self.assertEqual(default_provider.get_provider(Graph), default_provider.providers[GRAPH])
        self.assertEqual(default_provider.get_provider(ExtendedGraph), default_provider.providers[GRAPH])
        self.assertEqual(default_provider.get_provider(ZoomToolbar), default_provider.providers[ZOOM_TOOLBAR])


    def assertDefaultSettingsEqual(self, provider, defaults):
        for name, value in defaults.items():
            if isinstance(value, dict):
                self.assertIn(name, provider.providers)
                self.assertDefaultSettingsEqual(provider.providers[name], value)
            else:
                self.assertEqual(provider.settings[name].default, value)

if __name__ == '__main__':
    unittest.main(verbosity=2)

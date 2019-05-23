from itertools import chain
from os import listdir, environ
from os.path import isfile, join, dirname
import unittest

from orangecanvas.registry import WidgetRegistry

from Orange.canvas.config import Config
from Orange.canvas import workflows
from Orange.canvas import widgetsscheme

from Orange.widgets.tests.base import GuiTest


def discover_workflows(dir):
    ows_files = [f for f in listdir(dir)
                 if isfile(join(dir, f)) and f.endswith(".ows")]
    for ows_file in ows_files:
        yield join(dir, ows_file)


def registry():
    d = Config.widget_discovery(WidgetRegistry())
    d.run(Config.widgets_entry_points())
    return d.registry


@unittest.skipIf(environ.get("SKIP_EXAMPLE_WORKFLOWS", False),
                 "Example workflows inflate coverage")
class TestWorkflows(GuiTest):
    def test_scheme_examples(self):
        """
        Test if Orange workflow examples can be opened. Examples in canvas
        and also those placed "workflows" subfolder.
        GH-2240
        """
        reg = registry()
        test_workflows = chain(
            discover_workflows(dirname(workflows.__file__)),
            discover_workflows(join(dirname(__file__), "workflows"))
        )
        for ows_file in test_workflows:
            new_scheme = widgetsscheme.WidgetsScheme()
            new_scheme.widget_manager.set_creation_policy(
                new_scheme.widget_manager.Immediate
            )
            with open(ows_file, "rb") as f:
                try:
                    new_scheme.load_from(f, registry=reg)
                except Exception as e:
                    self.fail("Old workflow '{}' could not be loaded\n'{}'".
                              format(ows_file, str(e)))
                finally:
                    new_scheme.clear()

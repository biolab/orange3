from itertools import chain
from os import listdir, environ
from os.path import isfile, join, dirname
import unittest

from orangecanvas.registry import WidgetRegistry
from orangecanvas.scheme.readwrite import scheme_load

from Orange.canvas.conf import orangeconfig
from Orange.canvas import workflows
from Orange.canvas import widgetsscheme

from Orange.widgets.tests.base import WidgetTest


def discover_workflows(tests_dir):
    ows_path = join(tests_dir, "workflows")
    ows_files = [f for f in listdir(ows_path)
                 if isfile(join(ows_path, f)) and f.endswith(".ows")]
    for ows_file in ows_files:
        yield join(ows_path, ows_file)

TEST_WORKFLOWS = chain(
    [t.abspath() for t in workflows.example_workflows()],
    discover_workflows(dirname(__file__))
)


def registry():
    d = orangeconfig.widget_discovery(WidgetRegistry())
    d.run(orangeconfig.widgets_entry_points())
    return d.registry


@unittest.skipIf(environ.get("SKIP_EXAMPLE_WORKFLOWS", False),
                 "Example workflows inflate coverage")
class TestWorkflows(WidgetTest):
    def test_scheme_examples(self):
        """
        Test if Orange workflow examples can be opened. Examples in canvas
        and also those placed "workflows" subfolder.
        GH-2240
        """
        reg = registry()
        for ows_file in TEST_WORKFLOWS:
            new_scheme = widgetsscheme.WidgetsScheme()
            with open(ows_file, "rb") as f:
                try:
                    scheme_load(new_scheme, f, registry=reg)
                except Exception as e:
                    self.fail("Old workflow '{}' could not be loaded\n'{}'".
                              format(ows_file, str(e)))

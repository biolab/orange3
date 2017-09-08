from itertools import chain
from os import listdir, environ
from os.path import isfile, join, dirname
import unittest

from Orange.canvas.application import workflows
from Orange.canvas.scheme import widgetsscheme
from Orange.canvas.scheme.readwrite import scheme_load
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


@unittest.skipIf(environ.get("SKIP_EXAMPLE_WORKFLOWS", False),
                 "Example workflows inflate coverage")
class TestWorkflows(WidgetTest):
    def test_scheme_examples(self):
        """
        Test if Orange workflow examples can be opened. Examples in canvas
        and also those placed "workflows" subfolder.
        GH-2240
        """
        for ows_file in TEST_WORKFLOWS:
            new_scheme = widgetsscheme.WidgetsScheme()
            with open(ows_file, "rb") as f:
                try:
                    scheme_load(new_scheme, f)
                except Exception as e:
                    self.fail("Old workflow '{}' could not be loaded\n'{}'".
                              format(ows_file, str(e)))

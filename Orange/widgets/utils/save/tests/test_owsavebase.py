# Proper tests of OWSaveBase would require too much mocking, so we test most
# OWSaveBase's methods within the test for OWSave widget.
# The test for pure OWSaveBase just check a few methods that do not require
# extensive mocking

# pylint: disable=missing-docstring, protected-access, unsubscriptable-object
import unittest
from unittest.mock import Mock
import os
import collections

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils import getmembers
from Orange.widgets.utils.save.owsavebase import OWSaveBase
from orangewidget.widget import Input


class SaveWidgetsTestBaseMixin:
    def test_input_handler(self):
        widget = self.widget
        if not widget:
            return
        widget.on_new_input = Mock()

        inputs = getmembers(widget.Inputs, Input)
        self.assertGreaterEqual(len(inputs), 1, msg="Widget defines no inputs")
        if len(inputs) > 1:
            self.skipTest(
                "widget has multiple inputs; input handler can't be tested")
            return

        handler = getattr(widget, inputs[0][1].handler)
        data = Mock()
        handler(data)
        self.assertIs(widget.data, data)
        widget.on_new_input.assert_called()

    def test_filters(self):
        self.assertGreaterEqual(len(self.widget.filters), 1,
                                msg="Widget defines no filters")
        if type(self.widget).do_save is OWSaveBase.do_save:
            self.assertIsInstance(self.widget.filters, collections.abc.Mapping)


class TestOWSaveBaseWithWriters(WidgetTest):
    # Tests for OWSaveBase methods that require filters to be dictionaries
    # with with writers as keys in `filters`.
    def setUp(self):
        class OWSaveMockWriter(OWSaveBase):
            name = "Mock save"
            writer = Mock()
            writer.EXTENSIONS = [".csv"]
            writer.SUPPORT_COMPRESSED = True
            writer.SUPPORT_SPARSE_DATA = False
            writer.OPTIONAL_TYPE_ANNOTATIONS = False
            writers = [writer]
            filters = {"csv (*.csv)": writer}

        self.widget = self.create_widget(OWSaveMockWriter)

    def test_no_data_no_save(self):
        widget = self.widget

        write = widget.writer.write = Mock()
        widget.save_file_as = Mock()

        widget.filename = "foo.tab"
        widget.save_file()
        write.assert_not_called()

        widget.filename = ""
        widget.save_file()
        widget.save_file_as.assert_called()
        write.assert_not_called()

    def test_save_calls_writer(self):
        widget = self.widget

        widget.writer = Mock()
        write = widget.writer.write = Mock()
        widget.filename = "foo"
        widget.data = object()

        widget.save_file()
        write.assert_called_with(widget.filename, widget.data)

    def test_base_methods(self):
        """Default methods do not crash and do something sensible"""
        widget = self.widget

        widget.update_status()
        self.assertEqual(widget.initial_start_dir(),
                         os.path.expanduser(f"~{os.sep}"))
        self.assertEqual(widget.suggested_name(), "")
        self.assertIs(widget.valid_filters(), widget.filters)
        self.assertIs(widget.default_valid_filter(), widget.filter)


class TestOWSaveBase(WidgetTest):
    # Tests for OWSaveBase methods with filters as list
    def setUp(self):
        class OWSaveMockWriter(OWSaveBase):
            name = "Mock save"
            filters = ["csv (*.csv)"]

            do_save = Mock()

        self.widget = self.create_widget(OWSaveMockWriter)

    def test_no_data_no_save(self):
        widget = self.widget

        widget.save_file_as = Mock()

        widget.filename = "foo.tab"
        widget.save_file()
        widget.do_save.assert_not_called()

        widget.filename = ""
        widget.data = Mock()
        widget.save_file()
        widget.do_save.assert_not_called()

        widget.filename = "foo.tab"
        widget.save_file()
        widget.do_save.assert_called()

    def test_base_methods(self):
        """Default methods do not crash and do something sensible"""
        widget = self.widget

        widget.update_status()
        self.assertEqual(widget.initial_start_dir(),
                         os.path.expanduser(f"~{os.sep}"))
        self.assertEqual(widget.suggested_name(), "")
        self.assertIs(widget.valid_filters(), widget.filters)
        self.assertIs(widget.default_valid_filter(), widget.filter)


class TestOWSaveUtils(unittest.TestCase):
    def test_replace_extension(self):
        class OWMockSaveBase(OWSaveBase):
            filters = ["Tab delimited (*.tab)",
                       "Compressed tab delimitd (*.gz.tab)",
                       "Comma separated (*.csv)",
                       "Compressed comma separated (*.csv.gz)",
                       "Excel File (*.xlsx)"]

        replace = OWMockSaveBase._replace_extension
        fname = "/bing.bada.boom/foo.1942.tab"
        self.assertEqual(
            replace(fname, ".tab"), "/bing.bada.boom/foo.1942.tab")
        self.assertEqual(
            replace(fname, ".tab.gz"), "/bing.bada.boom/foo.1942.tab.gz")
        self.assertEqual(
            replace(fname, ".xlsx"), "/bing.bada.boom/foo.1942.xlsx")

        fname = "foo.tab.gz"
        self.assertEqual(replace(fname, ".tab"), "foo.tab")
        self.assertEqual(replace(fname, ".tab.gz"), "foo.tab.gz")
        self.assertEqual(replace(fname, ".csv"), "foo.csv")
        self.assertEqual(replace(fname, ".csv.gz"), "foo.csv.gz")

        fname = "/bing.bada.boom/foo"
        self.assertEqual(replace(fname, ".tab"), fname + ".tab")
        self.assertEqual(replace(fname, ".tab.gz"), fname + ".tab.gz")

    def test_extension_from_filter(self):
        eff = OWSaveBase._extension_from_filter
        self.assertEqual(eff("Description (*.ext)"), ".ext")
        self.assertEqual(eff("Description (*.foo.ba)"), ".foo.ba")
        self.assertEqual(eff("Description (.ext)"), ".ext")
        self.assertEqual(eff("Description (.foo.bar)"), ".foo.bar")


if __name__ == "__main__":
    unittest.main()

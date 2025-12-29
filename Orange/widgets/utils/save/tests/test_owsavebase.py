# Proper tests of OWSaveBase would require too much mocking, so we test most
# OWSaveBase's methods within the test for OWSave widget.
# The test for pure OWSaveBase just check a few methods that do not require
# extensive mocking

# pylint: disable=missing-docstring, protected-access, unsubscriptable-object
import unittest
from unittest.mock import Mock, patch
import sys
import os
import collections

from orangewidget.widget import Input
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils import getmembers
from Orange.widgets.utils.save.owsavebase import OWSaveBase, _userhome


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
        filters = self.widget.get_filters()
        self.assertGreaterEqual(len(filters), 1,
                                msg="Widget defines no filters")
        if type(self.widget).do_save is OWSaveBase.do_save:
            self.assertIsInstance(filters, collections.abc.Mapping)


class TestOWSaveBaseWithWriters(WidgetTest):
    # Tests for OWSaveBase methods that require filters to be dictionaries
    # with with writers as keys in `filters`.
    class OWSaveMockWriter(OWSaveBase):
        name = "Mock save"
        keywords = "mock save"
        writer = Mock()
        writer.EXTENSIONS = [".csv"]
        writer.SUPPORT_COMPRESSED = True
        writer.SUPPORT_SPARSE_DATA = False
        writer.OPTIONAL_TYPE_ANNOTATIONS = False
        writers = [writer]
        filters = {"csv (*.csv)": writer}

    def setUp(self):
        self.widget = self.create_widget(self.OWSaveMockWriter)

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
        self.assertIs(widget.valid_filters(), widget.get_filters())
        self.assertIs(widget.default_valid_filter(), widget.filter)

    def assertPathEqual(self, a, b):
        if sys.platform == "win32":
            a = a.replace("\\", "/")
            b = b.replace("\\", "/")
        self.assertEqual(a.rstrip("/"), b.rstrip("/"))

    @patch("os.path.exists",
           lambda name: name in ["/home/u/orange/a/b", "/foo/bar"])
    @patch("os.path.isabs",
           lambda name: name in ["/a/d", "/foo/bar"])  # Python 3.13+ made that False on Windows
    def test_open_moved_workflow(self):
        """Stored relative paths are properly changed on load"""
        home = _userhome
        home_c_foo = os.path.join(_userhome, "c.foo")
        with patch("Orange.widgets.widget.OWWidget.workflowEnv",
                   Mock(return_value={})):
            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="a/b",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, home)
            self.assertPathEqual(w.filename, home_c_foo)
            self.assertFalse(w.auto_save)

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="/a/d",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, home)
            self.assertPathEqual(w.filename, home_c_foo)
            self.assertFalse(w.auto_save)

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="/foo/bar",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, "/foo/bar")
            self.assertPathEqual(w.filename, "/foo/bar/c.foo")
            self.assertFalse(w.auto_save)

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path=".",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, home)
            self.assertPathEqual(w.filename, home_c_foo)
            self.assertFalse(w.auto_save)

        with patch("Orange.widgets.widget.OWWidget.workflowEnv",
                   Mock(return_value={"basedir": "/home/u/orange/"})):
            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="a/b",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, "/home/u/orange/a/b")
            self.assertPathEqual(w.filename, "/home/u/orange/a/b/c.foo")
            self.assertTrue(w.auto_save)
            self.assertFalse(w.Warning.auto_save_disabled.is_shown())

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="a/b",
                                     stored_name="c.foo",
                                     auto_save=False))
            self.assertPathEqual(w.last_dir, "/home/u/orange/a/b")
            self.assertPathEqual(w.filename, "/home/u/orange/a/b/c.foo")
            self.assertFalse(w.auto_save)
            self.assertFalse(w.Warning.auto_save_disabled.is_shown())

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="a/d",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, "/home/u/orange/a/d")
            self.assertPathEqual(w.filename, "/home/u/orange/a/d/c.foo")
            self.assertTrue(w.auto_save)
            self.assertFalse(w.Warning.auto_save_disabled.is_shown())

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="/a/d",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, "/home/u/orange/")
            self.assertPathEqual(w.filename, "/home/u/orange/c.foo")
            self.assertFalse(w.auto_save)
            self.assertFalse(w.Warning.auto_save_disabled.is_shown())

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path=".",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, "/home/u/orange/")
            self.assertPathEqual(w.filename, "/home/u/orange/c.foo")
            self.assertTrue(w.auto_save)
            self.assertFalse(w.Warning.auto_save_disabled.is_shown())

            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(stored_path="",
                                     stored_name="c.foo",
                                     auto_save=True))
            self.assertPathEqual(w.last_dir, "/home/u/orange/")
            self.assertPathEqual(w.filename, "/home/u/orange/c.foo")
            self.assertTrue(w.auto_save)
            self.assertFalse(w.Warning.auto_save_disabled.is_shown())

    def test_move_workflow(self):
        """Widget correctly stores relative paths"""
        w = self.widget
        w._try_save = Mock()
        w.update_messages = Mock()
        env = {}

        with patch("Orange.widgets.widget.OWWidget.workflowEnv",
                   Mock(return_value=env)):
            # File is save to subdirectory of workflow path
            env["basedir"] = "/home/u/orange/"

            w.get_save_filename = \
                Mock(return_value=("/home/u/orange/a/b/c.foo", ""))
            w.save_file_as()
            self.assertPathEqual(w.last_dir, "/home/u/orange/a/b")
            self.assertPathEqual(w.stored_path, "a/b/")
            self.assertEqual(w.stored_name, "c.foo")

            # Workflow path changes: relative path is changed to absolute
            env["basedir"] = "/tmp/u/work/"
            w.workflowEnvChanged("basedir", "/tmp/u/work", "/home/u/orange")
            self.assertPathEqual(w.last_dir, "/home/u/orange/a/b/")
            self.assertPathEqual(w.stored_path, "/home/u/orange/a/b/")
            self.assertEqual(w.stored_name, "c.foo")

            # Workflow path changes back: absolute path is again relative
            env["basedir"] = "/home/u/orange/"
            w.workflowEnvChanged("basedir", "/home/u/orange", "/tmp/u/work")
            self.assertPathEqual(w.last_dir, "/home/u/orange/a/b")
            self.assertPathEqual(w.stored_path, "a/b/")
            self.assertEqual(w.stored_name, "c.foo")

            # File is saved to an unrelated directory: path is absolute
            w.get_save_filename = \
                Mock(return_value=("/tmp/u/work/a/b/c.foo", ""))
            w.save_file_as()
            self.assertPathEqual(w.last_dir, "/tmp/u/work/a/b/")
            self.assertPathEqual(w.stored_path, "/tmp/u/work/a/b/")
            self.assertEqual(w.stored_name, "c.foo")

            # File is saved to the workflow's directory: path is relative
            w.get_save_filename = \
                Mock(return_value=("/home/u/orange/c.foo", ""))
            w.save_file_as()
            self.assertPathEqual(w.last_dir, "/home/u/orange/")
            self.assertPathEqual(w.stored_path, ".")
            self.assertEqual(w.stored_name, "c.foo")

    @patch("os.path.isabs",
           lambda name: name in ["/a/b"])  # Python 3.13+ made that False on Windows
    def test_migrate_pre_relative_settings(self):
        with patch("os.path.exists", lambda name: name == "/a/b"):
            w = self.create_widget(
                self.OWSaveMockWriter,
                stored_settings=dict(last_dir="/a/b", filename="/a/b/c.foo"))
            self.assertPathEqual(w.last_dir, "/a/b")
            self.assertPathEqual(w.filename, "/a/b/c.foo")
            self.assertPathEqual(w.stored_path, "/a/b")
            self.assertPathEqual(w.stored_name, "c.foo")

        w = self.create_widget(
            self.OWSaveMockWriter,
            stored_settings=dict(last_dir="/a/b", filename="/a/b/c.foo"))
        self.assertPathEqual(w.last_dir, _userhome)
        self.assertPathEqual(w.filename, os.path.join(_userhome, "c.foo"))
        self.assertPathEqual(w.stored_path, _userhome)
        self.assertPathEqual(w.stored_name, "c.foo")

    def test_save_button_label(self):
        w = self.create_widget(
            self.OWSaveMockWriter,
            stored_settings=dict(stored_path="", stored_name="c.foo"))
        self.assertTrue(w.bt_save.text().endswith(" c.foo"))

    def test_invalid_filter(self):
        writer = Mock()

        class OWSaveNoWriter(OWSaveBase):
            name = "Mock save"
            keywords = "mock save"
            writers = {}
            filters = {"csv (*.csv)": writer}

        w = self.create_widget(
            OWSaveNoWriter,
            stored_settings=dict(
                filter="Unsupported format (*.foo)", stored_path='test.foo')
        )
        w.data = Mock()
        self.assertIsNone(w.writer)

        w.do_save()
        self.assertTrue(w.Error.unsupported_format.is_shown())

        name = "/home/u/orange/a/b/c.csv"
        w.get_save_filename = Mock(return_value=(name, "csv (*.csv)"))
        w.save_file_as()
        self.assertFalse(w.Error.unsupported_format.is_shown())
        call_name, _ = writer.write.call_args[0]
        self.assertPathEqual(call_name, name)

    def test_default_filter(self):
        class OWSave(OWSaveBase):
            name = "Mock save"
            filters = {"csv (*.csv)": Mock(), "txt (*.txt)": Mock()}

        widget = self.create_widget(OWSave)
        self.assertEqual(widget.default_filter(), "csv (*.csv)")


class TestOWSaveBase(WidgetTest):
    # Tests for OWSaveBase methods with filters as list
    def setUp(self):
        class OWSaveMockWriter(OWSaveBase):
            name = "Mock save"
            filters = ["csv (*.csv)"]
            keywords = "mock save"

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
        self.assertIs(widget.valid_filters(), widget.get_filters())
        self.assertIs(widget.default_valid_filter(), widget.filter)

    def test_default_filter(self):
        class OWSave(OWSaveBase):
            name = "Mock save"
            filters = ["csv (*.csv)", "txt (*.txt)"]

        widget = self.create_widget(OWSave)
        self.assertEqual(widget.default_filter(), OWSave.filters[0])

    @unittest.skipUnless(sys.platform.startswith("win"), "windows path tests")
    def test_paths_win(self):
        class OWSave(OWSaveBase):
            name = "Mock save"
            filters = ["csv (*.csv)", "txt (*.txt)"]
        widget = self.create_widget(OWSave)
        # relative stored paths
        for workflow_dir, filename in [("C:/Temp", "C:/Temp/abc.csv"),
                                       ("C:/Temp", "C:/Temp/Project/abc.csv"),
                                       ("C:/Temp/", "C:/Temp/abc.csv"),
                                       ("C:/Temp/", "C:/Temp/Project/abc.csv"),
                                       ("C:/Temp", "c:\\Temp\\Project\\abc.csv"),
                                       ("c:\\Temp", "c:/Temp/Project\\abc.csv")]:
            widget.workflowEnv = lambda bd=workflow_dir: {"basedir": bd}
            widget.filename = filename
            self.assertFalse(os.path.isabs(widget.stored_path))
        # absolute stored paths
        for workflow_dir, filename in [("C:/Temp", "C:/Folder/abc.csv"),
                                       ("C:/Temp/Project", "C:/Temp/abc.csv"),
                                       ("C:\\Temp\\Project", "C:\\Temp\\abc.csv"),
                                       ("C:/Temp", "D:/Folder/abc.csv"),
                                       ("C:\\Temp\\Project", "D:\\Temp\\abc.csv")]:
            widget.workflowEnv = lambda bd=workflow_dir: {"basedir": bd}
            widget.filename = filename
            self.assertTrue(os.path.isabs(widget.stored_path))

    @unittest.skipIf(sys.platform.startswith("win"), "unix path tests")
    def test_paths_unix(self):
        class OWSave(OWSaveBase):
            name = "Mock save"
            filters = ["csv (*.csv)", "txt (*.txt)"]
        widget = self.create_widget(OWSave)
        # relative stored paths
        for workflow_dir, filename in [("/temp", "/temp/abc.csv"),
                                       ("/temp", "/temp/project/abc.csv"),
                                       ("/temp/", "/temp/abc.csv"),
                                       ("/temp/", "/temp/project/abc.csv")]:
            widget.workflowEnv = lambda bd=workflow_dir: {"basedir": bd}
            widget.filename = filename
            self.assertFalse(os.path.isabs(widget.stored_path))
        # absolute stored paths
        for workflow_dir, filename in [("/temp", "/folder/abc.csv"),
                                       ("/temp/project", "/temp/abc.csv")]:
            widget.workflowEnv = lambda bd=workflow_dir: {"basedir": bd}
            widget.filename = filename
            self.assertTrue(os.path.isabs(widget.stored_path))


class TestOWSaveUtils(unittest.TestCase):
    def test_replace_extension(self):
        class OWMockSaveBase(OWSaveBase):
            filters = ["Tab delimited (*.tab)",
                       "Compressed tab delimited (*.gz.tab)",
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

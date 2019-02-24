# pylint: disable=missing-docstring, protected-access
import unittest
from unittest.mock import patch, Mock
import os
import sys

import scipy.sparse as sp
from AnyQt.QtWidgets import QFileDialog

from Orange.data import Table
from Orange.data.io import TabReader, PickleReader, ExcelReader
from Orange.tests import named_file
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.data.owsave import OWSave


# Yay, MS Windows!
# This is not the proper general way to do it, but it's simplest and sufficient
# Short name is suitable for the function's purpose
def _w(s):  # pylint: disable=invalid-name
    return s.replace("/", os.sep)


class TestOWSave(WidgetTest):
    def setUp(self):
        class OWSaveMockWriter(OWSave):
            writer = Mock()
            writer.EXTENSIONS = [".csv"]
            writer.SUPPORT_COMPRESSED = True
            writer.SUPPORT_SPARSE_DATA = False
            writer.OPTIONAL_TYPE_ANNOTATIONS = False

        self.widget = self.create_widget(OWSaveMockWriter)  # type: OWSave
        self.iris = Table("iris")

    def test_dataset(self):
        widget = self.widget
        widget.auto_save = True
        insum = widget.info.set_input_summary = Mock()
        savefile = widget.save_file = Mock()

        datasig = widget.Inputs.data
        self.send_signal(datasig, self.iris)
        self.assertEqual(insum.call_args[0][0], "150")
        insum.reset_mock()
        savefile.reset_mock()

        widget.filename = "foo.tab"
        widget.writer = TabReader
        widget.auto_save = False
        self.send_signal(datasig, self.iris)
        self.assertEqual(insum.call_args[0][0], "150")
        savefile.assert_not_called()

        widget.auto_save = True
        self.send_signal(datasig, self.iris)
        self.assertEqual(insum.call_args[0][0], "150")
        savefile.assert_called()

        self.send_signal(datasig, None)
        insum.assert_called_with(widget.info.NoInput)

    @unittest.skipUnless(sys.platform == "linux", "Test Qt's non-native dialog")
    def test_get_save_filename_non_native(self):
        widget = self.widget
        widget._initial_start_dir = lambda: "baz"
        widget.filters = dict.fromkeys("abc")
        widget.filter = "b"
        dlg = widget.SaveFileDialog = Mock()  # pylint: disable=invalid-name
        instance = dlg.return_value
        instance.selectedFiles.return_value = ["foo"]
        instance.selectedNameFilter.return_value = "bar"
        self.assertEqual(widget.get_save_filename(), ("foo", "bar"))
        self.assertEqual(dlg.call_args[0][2], "baz")
        self.assertEqual(dlg.call_args[0][3], "a;;b;;c")
        instance.selectNameFilter.assert_called_with("b")

        instance.exec.return_value = QFileDialog.Rejected
        self.assertEqual(widget.get_save_filename(), ("", ""))

    @unittest.skipUnless(sys.platform == "linux", "Test Qt's non-native dialog")
    def test_save_file_dialog_enforces_extension(self):
        dialog = OWSave.SaveFileDialog(
            None, "Save File", "high.txt",
            "Bar files (*.bar);;Low files (*.low)")

        dialog.selectNameFilter("Low files (*.low)")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.low"))

        dialog.selectFile("high.txt")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.low"))

        dialog.selectNameFilter("Bar files (*.bar)")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.bar"))

        dialog.selectFile("middle.txt")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/middle.bar"))

        dialog.filterSelected.emit("Low files (*.low)")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/middle.low"))

        dialog.selectFile("high.txt")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.low"))

    def test_initial_start_dir(self):
        widget = self.widget
        widget.filename = _w("/usr/foo/bar.csv")
        self.assertEqual(widget._initial_start_dir(),
                         _w(os.path.expanduser("~/")))

        with patch("os.path.exists", return_value=True):
            widget.filename = _w("/usr/foo/bar.csv")
            self.assertEqual(widget._initial_start_dir(), widget.filename)

            widget.filename = ""
            widget.last_dir = _w("/usr/bar")
            self.assertEqual(widget._initial_start_dir(), _w("/usr/bar/"))

            widget.last_dir = _w("/usr/bar")
            self.send_signal(widget.Inputs.data, self.iris)
            self.assertEqual(widget._initial_start_dir(),
                             _w("/usr/bar/iris.csv"))

            widget.last_dir = ""
            self.assertEqual(widget._initial_start_dir(),
                             os.path.expanduser(_w("~/iris.csv")))

    @patch("Orange.widgets.data.owsave.QFileDialog.getSaveFileName")
    def test_save_file_sets_name(self, filedialog):
        widget = self.widget
        filters = iter(widget.filters)
        filter1 = next(filters)
        filter2 = next(filters)

        widget.filename = _w("/usr/foo/bar.csv")
        widget.last_dir = _w("/usr/foo/")
        widget.filter = filter1

        widget._update_messages = Mock()
        widget.save_file = Mock()

        widget.get_save_filename = Mock(return_value=("", filter2))
        widget.save_file_as()
        self.assertEqual(widget.filename, _w("/usr/foo/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/usr/foo/"))
        self.assertEqual(widget.filter, filter1)
        widget._update_messages.assert_not_called()
        widget.save_file.assert_not_called()

        widget.get_save_filename = \
            Mock(return_value=(_w("/bar/bar.csv"), filter2))
        widget.save_file_as()
        self.assertEqual(widget.filename, _w("/bar/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/bar"))
        self.assertEqual(widget.filter, filter2)
        self.assertIn("bar.csv", widget.bt_save.text())
        widget._update_messages.assert_called()
        widget.save_file.assert_called()

        widget.get_save_filename = Mock(return_value=("", filter2))
        widget.save_file_as()
        self.assertEqual(widget.filename, _w("/bar/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/bar"))
        self.assertEqual(widget.filter, filter2)
        self.assertIn("bar.csv", widget.bt_save.text())
        widget._update_messages.assert_called()
        widget.save_file.assert_called()

    def test_save_file_calls_save_as(self):
        widget = self.widget
        widget.save_file_as = Mock()

        self.send_signal(widget.Inputs.data, self.iris)

        widget.filename = ""
        widget.save_file()
        widget.save_file_as.assert_called()
        widget.save_file_as.reset_mock()

        widget.filename = "bar.csv"
        widget.save_file()
        widget.save_file_as.assert_not_called()

    def test_save_file_checks_can_save(self):
        widget = self.widget
        widget.get_save_filename = Mock(return_value=("", 0))

        widget.save_file()
        widget.writer.write.assert_not_called()

        widget.filename = "foo"
        widget.save_file()
        widget.writer.write.assert_not_called()

        widget.filename = ""
        self.send_signal(widget.Inputs.data, self.iris)
        widget.save_file()
        widget.writer.write.assert_not_called()

        widget.filename = "foo"
        widget.save_file()
        widget.writer.write.assert_called()
        widget.writer.reset_mock()

        self.iris.X = sp.csr_matrix(self.iris.X)
        widget.save_file()
        widget.writer.write.assert_not_called()

        widget.writer.SUPPORT_SPARSE_DATA = True
        widget.save_file()
        widget.writer.write.assert_called()

    def test_save_file_write_errors(self):
        widget = self.widget
        datasig = widget.Inputs.data

        widget.auto_save = True
        widget.filename = _w("bar/foo")

        widget.writer.write.side_effect = IOError
        self.send_signal(datasig, self.iris)
        self.assertTrue(widget.Error.general_error.is_shown())

        widget.writer.write.side_effect = None
        self.send_signal(datasig, self.iris)
        self.assertFalse(widget.Error.general_error.is_shown())

        widget.writer.write.side_effect = IOError
        self.send_signal(datasig, self.iris)
        self.assertTrue(widget.Error.general_error.is_shown())

        widget.writer.write.side_effect = None
        self.send_signal(datasig, None)
        self.assertFalse(widget.Error.general_error.is_shown())

        widget.writer.write.side_effect = ValueError
        self.assertRaises(ValueError, self.send_signal, datasig, self.iris)

    def test_save_file_write(self):
        widget = self.widget
        datasig = widget.Inputs.data

        widget.auto_save = True

        widget.filename = _w("bar/foo.csv")
        widget.add_type_annotations = True
        self.send_signal(datasig, self.iris)
        widget.writer.write.assert_called_with(
            _w("bar/foo.csv"), self.iris, True)

    def test_file_name_label(self):
        widget = self.widget

        widget.filename = ""
        widget._update_messages()
        self.assertFalse(widget.Error.no_file_name.is_shown())

        widget.auto_save = True
        widget._update_messages()
        self.assertTrue(widget.Error.no_file_name.is_shown())

        widget.filename = _w("/foo/bar/baz.csv")
        widget._update_messages()
        self.assertFalse(widget.Error.no_file_name.is_shown())

    def test_sparse_error(self):
        widget = self.widget
        err = widget.Error.unsupported_sparse

        widget.writer = ExcelReader
        widget.filename = "foo.xlsx"
        widget.data = self.iris

        widget._update_messages()
        self.assertFalse(err.is_shown())

        widget.data.X = sp.csr_matrix(widget.data.X)
        widget._update_messages()
        self.assertTrue(err.is_shown())

        widget.writer = PickleReader
        widget._update_messages()
        self.assertFalse(err.is_shown())

        widget.writer = ExcelReader
        widget._update_messages()
        self.assertTrue(err.is_shown())

        widget.data = None
        widget._update_messages()
        self.assertFalse(err.is_shown())

    def test_ignored_annotation_warning(self):
        widget = self.widget

        widget.writer = ExcelReader
        widget.add_type_annotations = True
        widget._update_messages()
        self.assertFalse(widget.Warning.type_annotation_ignored.is_shown())

        widget.filename = "test.xlsx"
        widget._update_messages()
        self.assertFalse(widget.Warning.type_annotation_ignored.is_shown())

        self.send_signal(widget.Inputs.data, self.iris)
        widget._update_messages()
        self.assertTrue(widget.Warning.type_annotation_ignored.is_shown())

        widget.add_type_annotations = False
        widget._update_messages()
        self.assertFalse(widget.Warning.type_annotation_ignored.is_shown())

        widget.add_type_annotations = True
        widget._update_messages()
        self.assertTrue(widget.Warning.type_annotation_ignored.is_shown())

        widget.writer = TabReader
        widget._update_messages()
        self.assertFalse(widget.Warning.type_annotation_ignored.is_shown())

    def test_send_report(self):
        widget = self.widget

        widget.report_items = Mock()
        for writer in widget.filters.values():
            widget.writer = writer
            for widget.add_type_annotations in (False, True):
                widget.filename = f"foo.{writer.EXTENSIONS[0]}"
                widget.send_report()
                items = dict(widget.report_items.call_args[0][0])
                msg = f"for {writer}, annotations={widget.add_type_annotations}"
                self.assertEqual(items["File name"], widget.filename, msg=msg)
                if writer.OPTIONAL_TYPE_ANNOTATIONS:
                    self.assertEqual(
                        items["Type annotations"],
                        ["No", "Yes"][widget.add_type_annotations], msg=msg)
                else:
                    self.assertFalse(items["Type annotations"], msg=msg)

    def test_migration_to_version_2(self):
        const_settings = {
            'add_type_annotations': True, 'auto_save': False,
            'controlAreaVisible': True, 'last_dir': '/home/joe/Desktop',
            '__version__': 1}

        # No compression, Tab-separated values
        settings = {**const_settings,
                    'compress': False, 'compression': 'gzip (.gz)',
                    'filetype': 'Tab-separated values (.tab)'}
        OWSave.migrate_settings(settings)
        self.assertEqual(
            settings,
            {**const_settings,
             "filter": "Tab-separated values (*.tab)"})

        # Compression; ignore compression format (.xz is no longer supported)
        settings = {**const_settings,
                    'compress': True, 'compression': 'lzma (.xz)',
                    'filetype': 'Tab-separated values (.tab)'}
        OWSave.migrate_settings(settings)
        self.assertEqual(
            settings,
            {**const_settings,
             "filter": "Compressed Tab-separated values (*.tab.gz)"})

        # No compression, Excel
        settings = {**const_settings,
                    'compress': False, 'compression': 'lzma (.xz)',
                    'filetype': 'Microsoft Excel spreadsheet (.xlsx)'}
        OWSave.migrate_settings(settings)
        self.assertEqual(
            settings,
            {**const_settings,
             "filter": "Microsoft Excel spreadsheet (*.xlsx)"})

        # Excel with compression - compression must be ignored
        settings = {**const_settings,
                    'compress': True, 'compression': 'lzma (.xz)',
                    'filetype': 'Microsoft Excel spreadsheet (.xlsx)'}
        OWSave.migrate_settings(settings)
        self.assertEqual(
            settings,
            {**const_settings,
             "filter": "Microsoft Excel spreadsheet (*.xlsx)"})

        # Missing filetype (is this possible?)
        settings = {**const_settings,
                    'compress': True, 'compression': 'lzma (.xz)'}
        OWSave.migrate_settings(settings)
        self.assertTrue(settings["filter"] in OWSave.filters)

        # Unsupported file format (is this possible?)
        settings = {**const_settings,
                    'compress': True, 'compression': 'lzma (.xz)',
                    'filetype': 'Bar file (.bar)'}
        OWSave.migrate_settings(settings)
        self.assertTrue(settings["filter"] in OWSave.filters)



class TestFunctionalOWSave(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSave)  # type: OWSave
        self.iris = Table("iris")

    def test_save_uncompressed(self):
        widget = self.widget
        widget.auto_save = False

        spiris = Table("iris")
        spiris.X = sp.csr_matrix(spiris.X)

        for selected_filter, writer in widget.filters.items():
            widget.write = writer
            ext = writer.EXTENSIONS[0]
            with named_file("", suffix=ext) as filename:
                widget.get_save_filename = Mock(
                    return_value=(filename, selected_filter))

                self.send_signal(widget.Inputs.data, self.iris)
                widget.save_file_as()
                self.assertEqual(len(Table(filename)), 150)

                if writer.SUPPORT_SPARSE_DATA:
                    self.send_signal(widget.Inputs.data, spiris)
                    widget.save_file()
                    self.assertEqual(len(Table(filename)), 150)


class TestOWSaveUtils(unittest.TestCase):
    def test_replace_extension(self):
        replace = OWSave._replace_extension
        fname = "/bing.bada.boom/foo.bar.baz"
        self.assertEqual(replace(fname, ".baz"), fname)
        self.assertEqual(replace(fname, ".bar.baz"), fname)
        self.assertEqual(replace(fname, ".txt"), "/bing.bada.boom/foo.txt")

        fname = "foo.bar.baz"
        self.assertEqual(replace(fname, ".baz"), fname)
        self.assertEqual(replace(fname, ".bar.baz"), fname)
        self.assertEqual(replace(fname, ".txt"), "foo.txt")
        self.assertEqual(replace(fname, ".bar.txt"), "foo.bar.txt")

        fname = "/bing.bada.boom/foo"
        self.assertEqual(replace(fname, ".baz"), fname + ".baz")
        self.assertEqual(replace(fname, ".bar.baz"), fname + ".bar.baz")

    def test_extension_from_filter(self):
        self.assertEqual(
            OWSave._extension_from_filter("Description (*.ext)"), ".ext")
        self.assertEqual(
            OWSave._extension_from_filter("Description (*.foo.ba)"), ".foo.ba")
        self.assertEqual(
            OWSave._extension_from_filter("Description (.ext)"), ".ext")
        self.assertEqual(
            OWSave._extension_from_filter("Description (.foo.bar)"), ".foo.bar")


if __name__ == "__main__":
    unittest.main()

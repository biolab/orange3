# pylint: disable=missing-docstring, protected-access, unsubscriptable-object
import unittest
from unittest.mock import patch, Mock
import os
import sys

import scipy.sparse as sp
from AnyQt.QtWidgets import QFileDialog

from Orange.data import Table
from Orange.data.io import TabReader, PickleReader, ExcelReader, FileFormat
from Orange.tests import named_file
from Orange.widgets.data.owsave import OWSave, OWSaveBase
from Orange.widgets.utils.save.tests.test_owsavebase import \
    SaveWidgetsTestBaseMixin
from Orange.widgets.tests.base import WidgetTest, open_widget_classes


# Yay, MS Windows!
# This is not the proper general way to do it, but it's simplest and sufficient
# Short name is suitable for the function's purpose
def _w(s):  # pylint: disable=invalid-name
    return s.replace("/", os.sep)


class MockFormat(FileFormat):
    EXTENSIONS = ('.mock',)
    DESCRIPTION = "Mock file format"

    @staticmethod
    def write_file(filename, data):
        pass


class OWSaveTestBase(WidgetTest, SaveWidgetsTestBaseMixin):
    def setUp(self):
        with open_widget_classes():
            class OWSaveMockWriter(OWSave):
                writer = Mock()
                writer.EXTENSIONS = [".csv"]
                writer.SUPPORT_COMPRESSED = True
                writer.SUPPORT_SPARSE_DATA = False
                writer.OPTIONAL_TYPE_ANNOTATIONS = False

        self.widget = self.create_widget(OWSaveMockWriter)  # type: OWSave
        self.iris = Table("iris")


class TestOWSave(OWSaveTestBase):
    def test_dataset(self):
        widget = self.widget
        widget.auto_save = True
        savefile = widget.save_file = Mock()

        datasig = widget.Inputs.data
        self.send_signal(datasig, self.iris)
        savefile.reset_mock()

        widget.filename = "foo.tab"
        widget.writer = TabReader
        widget.auto_save = False
        self.send_signal(datasig, self.iris)
        savefile.assert_not_called()

        widget.auto_save = True
        self.send_signal(datasig, self.iris)
        savefile.assert_called()

    def test_initial_start_dir(self):
        widget = self.widget
        self.assertEqual(widget.initial_start_dir(),
                         _w(os.path.expanduser("~/")))

        with patch("os.path.exists", return_value=True):
            widget.filename = _w("/usr/foo/bar.csv")
            self.assertEqual(widget.initial_start_dir(), widget.filename)

            widget.filename = ""
            widget.last_dir = _w("/usr/bar")
            self.assertEqual(widget.initial_start_dir(), _w("/usr/bar/"))

            widget.last_dir = _w("/usr/bar")
            self.send_signal(widget.Inputs.data, self.iris)
            self.assertEqual(widget.initial_start_dir(),
                             _w("/usr/bar/iris.csv"))

            widget.last_dir = ""
            self.assertEqual(widget.initial_start_dir(),
                             os.path.expanduser(_w("~/iris.csv")))

    @patch("Orange.widgets.utils.save.owsavebase.QFileDialog.getSaveFileName")
    def test_save_file_sets_name(self, _filedialog):
        widget = self.widget
        filters = iter(widget.get_filters())
        filter1 = next(filters)
        filter2 = next(filters)

        widget.filename = _w("/usr/foo/bar.csv")
        widget.last_dir = _w("/usr/foo/")
        widget.filter = filter1

        widget.update_messages = Mock()
        widget.do_save = Mock()
        widget.data = Mock()

        widget.get_save_filename = Mock(return_value=("", filter2))
        widget.save_file_as()
        self.assertEqual(widget.filename, _w("/usr/foo/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/usr/foo/"))
        self.assertEqual(widget.filter, filter1)
        widget.update_messages.assert_not_called()
        widget.do_save.assert_not_called()

        widget.get_save_filename = \
            Mock(return_value=(_w("/bar/bar.csv"), filter2))
        widget.save_file_as()
        self.assertEqual(widget.filename, _w("/bar/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/bar"))
        self.assertEqual(widget.filter, filter2)
        self.assertIn("bar.csv", widget.bt_save.text())
        widget.update_messages.assert_called()
        widget.do_save.assert_called()
        widget.do_save.reset_mock()

        widget.get_save_filename = Mock(return_value=("", filter2))
        widget.save_file_as()
        self.assertEqual(widget.filename, _w("/bar/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/bar"))
        self.assertEqual(widget.filter, filter2)
        self.assertIn("bar.csv", widget.bt_save.text())
        widget.update_messages.assert_called()
        widget.do_save.assert_not_called()

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

        with self.iris.unlocked():
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
        widget.update_messages()
        self.assertFalse(widget.Error.no_file_name.is_shown())

        widget.auto_save = True
        widget.update_messages()
        self.assertTrue(widget.Error.no_file_name.is_shown())

        widget.filename = _w("/foo/bar/baz.csv")
        widget.update_messages()
        self.assertFalse(widget.Error.no_file_name.is_shown())

    def test_sparse_error(self):
        widget = self.widget
        err = widget.Error.unsupported_sparse

        widget.writer = ExcelReader
        widget.filename = "foo.xlsx"
        widget.data = self.iris

        widget.update_messages()
        self.assertFalse(err.is_shown())

        with self.iris.unlocked():
            widget.data.X = sp.csr_matrix(widget.data.X)
        widget.update_messages()
        self.assertTrue(err.is_shown())

        widget.writer = PickleReader
        widget.update_messages()
        self.assertFalse(err.is_shown())

        widget.writer = ExcelReader
        widget.update_messages()
        self.assertTrue(err.is_shown())

        widget.data = None
        widget.update_messages()
        self.assertFalse(err.is_shown())

    def test_valid_filters_for_sparse(self):
        widget = self.widget

        widget.data = None
        self.assertEqual(widget.get_filters(), widget.valid_filters())

        widget.data = self.iris
        self.assertEqual(widget.get_filters(), widget.valid_filters())

        with self.iris.unlocked():
            widget.data.X = sp.csr_matrix(widget.data.X)
        valid = widget.valid_filters()
        self.assertNotEqual(widget.get_filters(), {})
        # false positive, pylint: disable=no-member
        self.assertTrue(all(v.SUPPORT_SPARSE_DATA for v in valid.values()))

    def test_valid_default_filter(self):
        widget = self.widget
        for widget.filter, writer in widget.get_filters().items():
            if not writer.SUPPORT_SPARSE_DATA:
                break

        widget.data = None
        self.assertIs(widget.filter, widget.default_valid_filter())

        widget.data = self.iris
        self.assertIs(widget.filter, widget.default_valid_filter())

        with self.iris.unlocked():
            widget.data.X = sp.csr_matrix(widget.data.X)
        self.assertTrue(
            widget.get_filters()[widget.default_valid_filter()]
            .SUPPORT_SPARSE_DATA)

    def test_add_on_writers(self):
        # test adding file formats after registering the widget
        self.assertIn(MockFormat, self.widget.valid_filters().values())
        # this test doesn't call it - test_save_uncompressed does

    def test_send_report(self):
        widget = self.widget

        widget.report_items = Mock()
        for writer in widget.get_filters().values():
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
        self.assertTrue(settings["filter"] in OWSave.get_filters())

        # Unsupported file format
        settings = {**const_settings,
                    'compress': True, 'compression': 'lzma (.xz)',
                    'filetype': 'Bar file (.bar)'}
        OWSave.migrate_settings(settings)
        self.assertTrue(settings["filter"] in OWSave.get_filters())

    def test_migration_to_version_3(self):
        settings = {"add_type_annotations": True,
                    "stored_name": "zoo.xlsx",
                    "__version__": 2}
        widget = self.create_widget(OWSave, stored_settings=settings)
        self.assertFalse(widget.add_type_annotations)

        settings = {"add_type_annotations": True,
                    "stored_name": "zoo.tab",
                    "__version__": 2}
        widget = self.create_widget(OWSave, stored_settings=settings)
        self.assertTrue(widget.add_type_annotations)

        settings = {"add_type_annotations": False,
                    "stored_name": "zoo.xlsx",
                    "__version__": 2}
        widget = self.create_widget(OWSave, stored_settings=settings)
        self.assertFalse(widget.add_type_annotations)

        settings = {"add_type_annotations": False,
                    "stored_name": "zoo.tab",
                    "__version__": 2}
        widget = self.create_widget(OWSave, stored_settings=settings)
        self.assertFalse(widget.add_type_annotations)


class TestFunctionalOWSave(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSave)  # type: OWSave
        self.iris = Table("iris")

    def test_save_uncompressed(self):
        widget = self.widget
        widget.auto_save = False

        spiris = Table("iris")
        with spiris.unlocked():
            spiris.X = sp.csr_matrix(spiris.X)

        for selected_filter, writer in widget.get_filters().items():
            widget.write = writer
            ext = writer.EXTENSIONS[0]
            with named_file("", suffix=ext) as filename:
                widget.get_save_filename = Mock(
                    return_value=(filename, selected_filter))

                self.send_signal(widget.Inputs.data, self.iris)
                widget.save_file_as()
                if hasattr(writer, "read"):
                    self.assertEqual(len(writer(filename).read()), 150)

                if writer.SUPPORT_SPARSE_DATA:
                    self.send_signal(widget.Inputs.data, spiris)
                    widget.save_file()
                    if hasattr(writer, "read"):
                        self.assertEqual(len(writer(filename).read()), 150)

    def test_unsupported_file_format(self):
        widget = self.create_widget(
            OWSave,
            stored_settings=dict(
                filter="Unsupported filter (*.foo)", stored_name="test.foo",
                __version__=2)
        )
        filters = widget.get_filters()
        def_filter = filters[widget.default_filter()]

        iris = Table("iris")
        self.send_signal(widget.Inputs.data, iris)

        # With unsupported format from settings, the widget should indicate
        # an error
        with patch.object(def_filter, "write"):
            widget.save_file()
            self.assertTrue(widget.Error.unsupported_format.is_shown())
            def_filter.write.assert_not_called()

        # Without file name, if `filter` is set to unsupported format,
        # initial_start_dir should choose the default filter
        widget.stored_name = ""
        widget.initial_start_dir()
        self.assertIs(filters[widget.filter], def_filter)
        self.assertIs(widget.writer, def_filter)


@unittest.skipUnless(sys.platform == "linux", "Tests for dialog on Linux")
class TestOWSaveLinuxDialog(OWSaveTestBase):
    def test_get_save_filename_linux(self):
        widget = self.widget
        widget.initial_start_dir = lambda: "baz"
        widget.get_filters = lambda: dict.fromkeys("abc")
        widget.filter = "b"
        dlg = widget.SaveFileDialog = Mock()  # pylint: disable=invalid-name
        instance = dlg.return_value
        instance.selectedFiles.return_value = ["foo"]
        instance.selectedNameFilter.return_value = "bar"
        self.assertEqual(widget.get_save_filename(), ("foo", "bar"))
        self.assertEqual(dlg.call_args[0][3], "baz")
        self.assertEqual(dlg.call_args[0][4], "a;;b;;c")
        instance.selectNameFilter.assert_called_with("b")

        instance.exec.return_value = QFileDialog.Rejected
        self.assertEqual(widget.get_save_filename(), ("", ""))

    @patch.object(OWSaveBase, "filters", OWSave.get_filters())
    def test_save_file_dialog_enforces_extension_linux(self):
        dialog = OWSave.SaveFileDialog(
            OWSave, None, "Save File", "foo.bar",
            "Bar files (*.tab);;Low files (*.csv)")

        dialog.selectNameFilter("Low files (*.csv)")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/foo.csv"))

        dialog.selectFile("high.bar")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.bar.csv"))

        dialog.selectNameFilter("Bar files (*.tab)")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.bar.tab"))

        dialog.selectFile("middle.pkl")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/middle.tab"))

        dialog.filterSelected.emit("Low files (*.csv)")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/middle.csv"))

        dialog.selectFile("high.tab.gz")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.csv"))

        dialog.selectFile("high.tab.gz.tab.tab.gz")
        self.assertTrue(dialog.selectedFiles()[0].endswith("/high.csv"))

    def test_save_file_dialog_uses_valid_filters_linux(self):
        widget = self.widget
        widget.valid_filters = lambda: ["a (*.a)", "b (*.b)"]
        widget.default_valid_filter = lambda: "a (*.a)"
        dlg = widget.SaveFileDialog = Mock()  # pylint: disable=invalid-name
        instance = dlg.return_value
        instance.exec.return_value = dlg.Rejected = QFileDialog.Rejected
        widget.get_save_filename()
        self.assertEqual(dlg.call_args[0][4], "a (*.a);;b (*.b)")
        instance.selectNameFilter.assert_called_with("a (*.a)")


@unittest.skipUnless(sys.platform in ("darwin", "win32"),
                     "Test for native dialog on Windows and macOS")
class TestOWSaveDarwinDialog(OWSaveTestBase):  # pragma: no cover
    if sys.platform == "darwin":
        @staticmethod
        def remove_star(filt):
            return filt.replace(" (*.", " (.")
    else:
        @staticmethod
        def remove_star(filt):
            return filt

    @patch("Orange.widgets.utils.save.owsavebase.QFileDialog")
    def test_get_save_filename_darwin(self, dlg):
        widget = self.widget
        widget.initial_start_dir = lambda: "baz"
        widget.get_filters = \
            lambda: dict.fromkeys(("aa (*.a)", "bb (*.b)", "cc (*.c)"))
        widget.filter = "bb (*.b)"
        instance = dlg.return_value
        instance.exec.return_value = dlg.Accepted = QFileDialog.Accepted
        instance.selectedFiles.return_value = ["foo"]
        instance.selectedNameFilter.return_value = self.remove_star("aa (*.a)")
        self.assertEqual(widget.get_save_filename(), ("foo.a", "aa (*.a)"))
        self.assertEqual(dlg.call_args[0][2], "baz")
        self.assertEqual(
            dlg.call_args[0][3],
            self.remove_star("aa (*.a);;bb (*.b);;cc (*.c)"))
        instance.selectNameFilter.assert_called_with(
            self.remove_star("bb (*.b)"))

        instance.exec.return_value = dlg.Rejected = QFileDialog.Rejected
        self.assertEqual(widget.get_save_filename(), ("", ""))

    @patch("Orange.widgets.utils.save.owsavebase.QFileDialog")
    def test_save_file_dialog_enforces_extension_darwin(self, dlg):
        widget = self.widget
        filter1 = ""  # prevent pylint warning 'undefined-loop-variable'
        for filter1 in widget.get_filters():
            if OWSaveBase._extension_from_filter(filter1) == ".tab":
                break
        for filter2 in widget.get_filters():
            if OWSaveBase._extension_from_filter(filter2) == ".csv.gz":
                break

        widget.filter = filter1
        instance = dlg.return_value
        instance.exec.return_value = QFileDialog.Accepted

        instance.selectedNameFilter.return_value = self.remove_star(filter1)
        instance.selectedFiles.return_value = ["foo"]
        self.assertEqual(widget.get_save_filename()[0], "foo.tab")
        instance.selectedFiles.return_value = ["foo.pkl"]
        self.assertEqual(widget.get_save_filename()[0], "foo.tab")
        instance.selectedFiles.return_value = ["foo.tab.gz"]
        self.assertEqual(widget.get_save_filename()[0], "foo.tab")
        instance.selectedFiles.return_value = ["foo.csv.gz"]
        self.assertEqual(widget.get_save_filename()[0], "foo.tab")
        instance.selectedFiles.return_value = ["foo.bar"]
        self.assertEqual(widget.get_save_filename()[0], "foo.bar.tab")

        instance.selectedNameFilter.return_value = self.remove_star(filter2)
        instance.selectedFiles.return_value = ["foo"]
        self.assertEqual(widget.get_save_filename()[0], "foo.csv.gz")
        instance.selectedFiles.return_value = ["foo.pkl"]
        self.assertEqual(widget.get_save_filename()[0], "foo.csv.gz")
        instance.selectedFiles.return_value = ["foo.tab.gz"]
        self.assertEqual(widget.get_save_filename()[0], "foo.csv.gz")
        instance.selectedFiles.return_value = ["foo.csv.gz"]
        self.assertEqual(widget.get_save_filename()[0], "foo.csv.gz")
        instance.selectedFiles.return_value = ["foo.bar"]
        self.assertEqual(widget.get_save_filename()[0], "foo.bar.csv.gz")

    @patch("Orange.widgets.utils.save.owsavebase.QFileDialog")
    @patch("os.path.exists", new=lambda x: x == "old.tab")
    @patch("Orange.widgets.utils.save.owsavebase.QMessageBox")
    def test_save_file_dialog_asks_for_overwrite_darwin(self, msgbox, dlg):
        def selected_files():
            nonlocal attempts
            attempts += 1
            return [["old.tab", "new.tab"][attempts]]

        widget = self.widget
        widget.initial_start_dir = lambda: "baz"
        filter1 = ""  # prevent pylint warning 'undefined-loop-variable'
        for filter1 in widget.get_filters():
            if OWSaveBase._extension_from_filter(filter1) == ".tab":
                break

        widget.filter = filter1
        instance = dlg.return_value
        instance.exec.return_value = QFileDialog.Accepted
        instance.selectedFiles = selected_files
        instance.selectedNameFilter.return_value = self.remove_star(filter1)

        attempts = -1
        msgbox.question.return_value = msgbox.Yes = 1
        self.assertEqual(widget.get_save_filename()[0], "old.tab")

        attempts = -1
        msgbox.question.return_value = msgbox.No = 0
        self.assertEqual(widget.get_save_filename()[0], "new.tab")

    @patch("Orange.widgets.utils.save.owsavebase.QFileDialog")
    def test_save_file_dialog_uses_valid_filters_darwin(self, dlg):
        widget = self.widget
        widget.valid_filters = lambda: ["aa (*.a)", "bb (*.b)"]
        widget.default_valid_filter = lambda: "aa (*.a)"
        instance = dlg.return_value
        instance.exec.return_value = dlg.Rejected = QFileDialog.Rejected
        widget.get_save_filename()
        self.assertEqual(
            dlg.call_args[0][3], self.remove_star("aa (*.a);;bb (*.b)"))
        instance.selectNameFilter.assert_called_with(
            self.remove_star("aa (*.a)"))


if __name__ == "__main__":
    unittest.main()

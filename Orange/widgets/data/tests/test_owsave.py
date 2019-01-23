# pylint: disable=missing-docstring, protected-access
import unittest
from unittest.mock import patch, Mock
import os

from AnyQt.QtWidgets import QDialog

import scipy.sparse as sp

from Orange.data import Table
from Orange.data.io import CSVReader, TabReader, PickleReader, ExcelReader
from Orange.tests import named_file
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.data.owsave import OWSave


FILE_TYPES = [("{} ({})".format(w.DESCRIPTION, w.EXTENSIONS[0]), w)
              for w in (TabReader, CSVReader, PickleReader, ExcelReader)]


# Yay, MS Windows!
# This is not the proper general way to do it, but it's simplest and sufficient
# Short name is suitable for the functions, purpose
def _w(s):  # pylint: disable=invalid-name
    return s.replace("/", os.sep)


class TestOWSave(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSave)  # type: OWSave
        self.iris = Table("iris")

    def test_dataset(self):
        widget = self.widget
        insum = widget.info.set_input_summary = Mock()
        savefile = widget.save_file = Mock()

        datasig = widget.Inputs.data
        self.send_signal(datasig, self.iris)
        self.assertEqual(insum.call_args[0][0], "150")
        self.assertFalse(widget.bt_save.isEnabled())
        insum.reset_mock()
        savefile.reset_mock()

        widget.filename = "foo.tab"
        widget.writer = TabReader
        widget.auto_save = False
        self.send_signal(datasig, self.iris)
        self.assertEqual(insum.call_args[0][0], "150")
        self.assertTrue(widget.bt_save.isEnabled())
        savefile.assert_not_called()

        widget.auto_save = True
        self.send_signal(datasig, self.iris)
        self.assertEqual(insum.call_args[0][0], "150")
        self.assertTrue(widget.bt_save.isEnabled())
        savefile.assert_called()

        self.send_signal(datasig, None)
        insum.assert_called_with(widget.info.NoInput)
        self.assertFalse(widget.bt_save.isEnabled())

    @patch("Orange.widgets.data.owsave.QFileDialog")
    def test_set_file_name_start_dir(self, filedialog):
        widget = self.widget

        dlginst = filedialog.return_value
        dlginst.exec.return_value = dlginst.Rejected = QDialog.Rejected

        widget.filename = _w("/usr/foo/bar.csv")
        widget.filter = FILE_TYPES[1][0]
        widget.set_file_name()
        self.assertEqual(filedialog.call_args[0][2], widget.filename)

        widget.filename = ""
        widget.last_dir = _w("/usr/bar")
        widget.set_file_name()
        self.assertEqual(filedialog.call_args[0][2], _w("/usr/bar/"))

        self.send_signal(widget.Inputs.data, self.iris)
        widget.last_dir = _w("/usr/bar")
        widget.set_file_name()
        self.assertEqual(filedialog.call_args[0][2], _w("/usr/bar/iris.csv"))

        widget.last_dir = ""
        widget.set_file_name()
        self.assertEqual(filedialog.call_args[0][2],
                         os.path.expanduser(_w("~/iris.csv")))

    @patch("Orange.widgets.data.owsave.QFileDialog")
    def test_set_file_name(self, filedialog):
        widget = self.widget
        widget.filename = _w("/usr/foo/bar.csv")
        widget.last_dir = _w("/usr/foo/")
        widget.writer = widget.writers[1]
        widget.filter = FILE_TYPES[1][0]

        widget._update_controls = Mock()

        dlginst = filedialog.return_value
        dlginst.selectedFiles.return_value = [_w("/bar/baz.csv")]
        dlginst.selectedNameFilter.return_value = FILE_TYPES[0][0]

        dlginst.exec.return_value = dlginst.Rejected = QDialog.Rejected
        widget.set_file_name()
        self.assertEqual(widget.filename, _w("/usr/foo/bar.csv"))
        self.assertEqual(widget.last_dir, _w("/usr/foo/"))
        self.assertEqual(widget.filter, FILE_TYPES[1][0])
        self.assertIs(widget.writer, widget.writers[1])
        widget._update_controls.assert_not_called()

        dlginst.exec.return_value = dlginst.Accepted = QDialog.Accepted
        widget.set_file_name()
        self.assertEqual(widget.filename, _w("/bar/baz.csv"))
        self.assertEqual(widget.last_dir, _w("/bar"))
        self.assertEqual(widget.filter, FILE_TYPES[0][0])
        self.assertIs(widget.writer, widget.writers[0])
        widget._update_controls.assert_called()

    def set_mock_writer(self):
        widget = self.widget
        writer = widget.writer = Mock()
        writer.write = Mock()
        writer.SUPPORT_COMPRESSED = True
        writer.SUPPORT_SPARSE_DATA = False
        writer.OPTIONAL_TYPE_ANNOTATIONS = False

    def test_save_file_check_can_save(self):
        widget = self.widget
        self.set_mock_writer()

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
        self.set_mock_writer()

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

        self.set_mock_writer()
        widget.auto_save = True

        widget.filename = _w("bar/foo.csv")
        widget.compress = False
        widget.add_type_annotations = True
        self.send_signal(datasig, self.iris)
        widget.writer.write.assert_called_with(
            _w("bar/foo.csv"), self.iris, True)

        widget.compress = True
        self.send_signal(datasig, self.iris)
        widget.writer.write.assert_called_with(
            _w("bar/foo.csv.gz"), self.iris, True)

        widget.writer.SUPPORT_COMPRESSED = False
        self.send_signal(datasig, self.iris)
        widget.writer.write.assert_called_with(
            _w("bar/foo.csv"), self.iris, True)

    def test_file_label(self):
        widget = self.widget

        widget.filename = ""
        widget._update_controls()
        self.assertTrue(widget.lb_filename.isHidden())
        self.assertTrue(widget.Warning.no_file_name.is_shown())

        widget.filename = _w("/foo/bar/baz.csv")
        widget._update_controls()
        self.assertFalse(widget.lb_filename.isHidden())
        self.assertIn(
            widget.lb_filename.text(), _w("Save to: /foo/bar/baz.csv"))
        self.assertFalse(widget.Warning.no_file_name.is_shown())

        widget.filename = os.path.expanduser(_w("~/baz/bar/foo.csv"))
        widget._update_controls()
        self.assertFalse(widget.lb_filename.isHidden())
        self.assertEqual(
            widget.lb_filename.text(), _w("Save to: baz/bar/foo.csv"))

    def test_annotation_checkbox(self):
        widget = self.widget
        for _, widget.writer in FILE_TYPES:
            widget.filename = f"foo.{widget.writer.EXTENSIONS[0]}"
            widget._update_controls()
            self.assertIsNot(widget.controls.add_type_annotations.isHidden(),
                             widget.writer.OPTIONAL_TYPE_ANNOTATIONS,
                             msg=f"for {widget.writer}")
            self.assertIsNot(widget.controls.compress.isHidden(),
                             widget.writer.SUPPORT_COMPRESSED,
                             msg=f"for {widget.writer}")

        widget.writer = TabReader
        widget.filename = ""
        self.widget._update_controls()
        self.assertFalse(widget.controls.add_type_annotations.isVisible())
        self.assertFalse(widget.controls.compress.isVisible())

    def test_sparse_error(self):
        widget = self.widget
        err = widget.Error.unsupported_sparse

        widget.writer = ExcelReader
        widget.filename = "foo.xlsx"
        widget.data = self.iris

        widget._update_controls()
        self.assertFalse(err.is_shown())

        widget.data.X = sp.csr_matrix(widget.data.X)
        widget._update_controls()
        self.assertTrue(err.is_shown())

        widget.writer = PickleReader
        widget._update_controls()
        self.assertFalse(err.is_shown())

        widget.writer = ExcelReader
        widget._update_controls()
        self.assertTrue(err.is_shown())

        widget.data = None
        widget._update_controls()
        self.assertFalse(err.is_shown())

    def test_send_report(self):
        widget = self.widget

        widget.report_items = Mock()
        for _, writer in FILE_TYPES:
            widget.writer = writer
            for widget.compress in (False, True):
                for widget.add_type_annotations in (False, True):
                    widget.filename = f"foo.{writer.EXTENSIONS[0]}"
                    widget.send_report()
                    items = dict(widget.report_items.call_args[0][0])
                    msg = f"for {widget.writer}, " \
                        f"annotations={widget.add_type_annotations}, " \
                        f"compress={widget.compress}"
                    self.assertEqual(items["File name"],
                                     f"foo.{writer.EXTENSIONS[0]}", msg=msg)
                    if writer.SUPPORT_COMPRESSED:
                        self.assertEqual(
                            items["Compression"],
                            ["No", "Yes"][widget.compress],
                            msg=msg)
                    else:
                        self.assertFalse(items["Compression"], msg=msg)
                    if writer.OPTIONAL_TYPE_ANNOTATIONS:
                        self.assertEqual(
                            items["Type annotations"],
                            ["No", "Yes"][widget.add_type_annotations],
                            msg=msg)
                    else:
                        self.assertFalse(items["Type annotations"], msg=msg)


class TestFunctionalOWSave(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSave)  # type: OWSave
        self.iris = Table("iris")

    @patch("Orange.widgets.data.owsave.QFileDialog")
    def test_save_uncompressed(self, filedialog):
        widget = self.widget
        widget.auto_save = False

        dlg = filedialog.return_value
        dlg.exec.return_value = dlg.Accepted = QDialog.Accepted
        dlg = filedialog.return_value

        spiris = Table("iris")
        spiris.X = sp.csr_matrix(spiris.X)

        for dlg.selectedNameFilter.return_value, writer in FILE_TYPES:
            widget.write = writer
            ext = writer.EXTENSIONS[0]
            with named_file("", suffix=ext) as filename:
                dlg.selectedFiles.return_value = [filename]

                self.send_signal(widget.Inputs.data, self.iris)
                widget.bt_set_file.click()
                widget.bt_save.click()
                self.assertEqual(len(Table(filename)), 150)

                if writer.SUPPORT_SPARSE_DATA:
                    self.send_signal(widget.Inputs.data, spiris)
                    widget.bt_set_file.click()
                    widget.bt_save.click()
                    self.assertEqual(len(Table(filename)), 150)

            if writer.SUPPORT_COMPRESSED:
                with named_file("", suffix=ext + ".gz") as filename:
                    widget.compress = True
                    dlg.selectedFiles.return_value = [filename[:-3]]
                    self.send_signal(widget.Inputs.data, self.iris)
                    widget.bt_set_file.click()
                    widget.bt_save.click()
                    self.assertEqual(len(Table(filename)), 150)
                    widget.compress = False


if __name__ == "__main__":
    unittest.main()

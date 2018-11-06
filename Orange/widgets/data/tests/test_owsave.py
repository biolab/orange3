# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest.mock import patch, Mock
import itertools

from Orange.data import Table
from Orange.data.io import Compression, FileFormat, TabReader, CSVReader, PickleReader
from Orange.tests import named_file
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.filedialogs import format_filter
from Orange.widgets.data.owsave import OWSave

FILE_TYPES = [
    ("{} ({})".format(w.DESCRIPTION, w.EXTENSIONS[0]),
     w.EXTENSIONS[0],
     w.SUPPORT_SPARSE_DATA)
    for w in (TabReader, CSVReader, PickleReader)
]

COMPRESSIONS = [
    ("gzip ({})".format(Compression.GZIP), Compression.GZIP),
    ("bzip2 ({})".format(Compression.BZIP2), Compression.BZIP2),
    ("lzma ({})".format(Compression.XZ), Compression.XZ),
]


class AddedFormat(FileFormat):
    EXTENSIONS = ('.234',)
    DESCRIPTION = "Test if a dialog format works after reading OWSave"

    def write_file(self):
        pass


class TestOWSave(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWSave)  # type: OWSave

    def test_writer(self):
        compressions = [self.widget.controls.compression.itemText(i) for i in
                        range(self.widget.controls.compression.count())]
        types = [self.widget.controls.filetype.itemText(i)
                 for i in range(self.widget.controls.filetype.count())]
        for t, c, d in itertools.product(types, compressions, [True, False]):
            self.widget.compression = c
            self.widget.compress = d
            self.widget.filetype = t
            self.widget.update_extension()
            self.assertEqual(len(self.widget.get_writer_selected().EXTENSIONS), 1)

    def test_ordinary_save(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))

        for ext, suffix, _ in FILE_TYPES:
            self.widget.filetype = ext
            self.widget.update_extension()
            writer = self.widget.get_writer_selected()
            with named_file("", suffix=suffix) as filename:
                def choose_file(a, b, c, d, e, fn=filename, w=writer):
                    return fn, format_filter(w)

                with patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName", choose_file):
                    self.widget.save_file_as()
                self.assertEqual(len(Table(filename)), 150)

    def test_compression(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))

        self.widget.compress = True
        for type, compression in itertools.product([[x, ext] for x, ext, _ in FILE_TYPES],
                                                   COMPRESSIONS):
            self.widget.filetype = type[0]
            self.widget.compression = compression[0]
            self.widget.update_extension()
            writer = self.widget.get_writer_selected()
            with named_file("",
                            suffix=type[1] + compression[1]) as filename:
                def choose_file(a, b, c, d, e, fn=filename, w=writer):
                    return fn, format_filter(w)

                with patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName", choose_file):
                    self.widget.save_file_as()
                self.assertEqual(len(Table(filename)), 150)

    def test_format_combo(self):
        widget = self.widget
        filetype = widget.controls.filetype

        widget.save_file = Mock()

        data = Table("iris")
        sparse_data = Table("iris")
        sparse_data.is_sparse = Mock(return_value=True)

        self.send_signal(widget.Inputs.data, data)
        n_nonsparse = filetype.count()

        self.send_signal(widget.Inputs.data, sparse_data)
        n_sparse = filetype.count()
        self.assertGreater(n_nonsparse, n_sparse)

        self.send_signal(widget.Inputs.data, sparse_data)
        self.assertEqual(filetype.count(), n_sparse)

        self.send_signal(widget.Inputs.data, data)
        self.assertEqual(filetype.count(), n_nonsparse)

        self.send_signal(widget.Inputs.data, None)
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual(filetype.count(), n_nonsparse)

        self.send_signal(widget.Inputs.data, None)
        self.send_signal(widget.Inputs.data, sparse_data)
        self.assertEqual(filetype.count(), n_sparse)

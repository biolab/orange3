import os.path
import pathlib

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from Orange.data.table import Table
from Orange.data.io import Compression, FileFormat
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils import filedialogs
from Orange.widgets.widget import Input

FILE_TYPES = [
    ("Tab-delimited (.tab)", ".tab", False),
    ("Comma-seperated values (.csv)", ".csv", False),
    ("Pickle (.pkl)", ".pkl", True),
]

COMPRESSIONS = [
    ("gzip ({})".format(Compression.GZIP), Compression.GZIP),
    ("bzip2 ({})".format(Compression.BZIP2), Compression.BZIP2),
    ("lzma ({})".format(Compression.XZ), Compression.XZ),
]


class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = ["data", "save"]

    class Inputs:
        data = Input("Data", Table)

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")
    auto_save = Setting(False)
    filetype = Setting(FILE_TYPES[0][0])
    compression = Setting(COMPRESSIONS[0][0])
    compress = Setting(False)

    def get_writer_selected(self):
        writer = FileFormat.get_reader(self.type_ext)
        try:
            writer.EXTENSIONS = [writer.EXTENSIONS[writer.EXTENSIONS.index(self.type_ext + self.compression_ext)]]
            return writer
        except ValueError:
            self.Error.not_supported_extension()
            return

    @classmethod
    def remove_extensions(cls, filename):
        if not filename:
            return None
        for ext in pathlib.PurePosixPath(filename).suffixes:
            filename = filename.replace(ext, '')
        return filename

    class Error(widget.OWWidget.Error):
        not_supported_extension = widget.Msg("Selected extension is not supported.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.filename = ""
        self.basename = ""
        self.type_ext = ""
        self.compression_ext = ""
        self.writer = None

        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            rowWrapPolicy=QFormLayout.WrapLongRows,
            verticalSpacing=10,
        )

        box = gui.vBox(self.controlArea, "Format")

        gui.comboBox(
            box, self, "filetype",
            callback=self._update_text,
            items=[item for item, _, _ in FILE_TYPES],
            sendSelectedValue=True,
        )
        form.addRow("File type", self.controls.filetype, )

        gui.comboBox(
            box, self, "compression",
            callback=self._update_text,
            items=[item for item, _ in COMPRESSIONS],
            sendSelectedValue=True,
        )
        gui.checkBox(
            box, self, "compress", label="Use compression",
            callback=self._update_text,
        )

        form.addRow(self.controls.compress, self.controls.compression)

        box.layout().addLayout(form)

        self.save = gui.auto_commit(
            self.controlArea, self, "auto_save", "Save", box=False,
            commit=self.save_file, callback=self.adjust_label,
            disabled=True, addSpace=True
        )
        self.save_as = gui.button(
            self.controlArea, self, "Save As...",
            callback=self.save_file_as, disabled=True
        )
        self.save_as.setMinimumWidth(220)
        self.adjustSize()

    def adjust_label(self):
        if self.filename:
            # filename = os.path.split(self.filename)[1]
            text = "Auto save as '{}'" if self.auto_save else "Save as '{}'"
            self.save.button.setText(text.format(self.basename + self.type_ext + self.compression_ext))

    @Inputs.data
    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.save_as.setDisabled(data is None)
        if data is not None:
            self.save_file()

        self.controls.filetype.clear()
        if self.data.is_sparse():
            self.controls.filetype.insertItems(0, [item for item, _, supports_sparse in FILE_TYPES if supports_sparse])
        else:
            self.controls.filetype.insertItems(0, [item for item, _, _ in FILE_TYPES])

    def save_file_as(self):
        file_name = self.remove_extensions(self.filename) or os.path.join(self.last_dir or os.path.expanduser("~"),
                                                                          getattr(self.data, 'name', ''))
        self._update_extension()
        writer = self.get_writer_selected()
        if not writer:
            return

        filename, writer, _ = filedialogs.open_filename_dialog_save(
            file_name, '', [writer],
        )
        if not filename:
            return

        self.filename = filename
        self.writer = writer
        self.last_dir = os.path.split(self.filename)[0]
        self.basename = os.path.basename(self.remove_extensions(filename))
        self.unconditional_save_file()
        self.adjust_label()

    def save_file(self):
        if self.data is None:
            return
        if not self.filename:
            self.save_file_as()
        else:
            try:
                self.writer.write(os.path.join(self.last_dir, self.basename + self.type_ext + self.compression_ext),
                                  self.data)
            except Exception as err_value:
                self.error(str(err_value))
            else:
                self.error()

    def _update_extension(self):
        self.type_ext = [ext for name, ext, _ in FILE_TYPES if name == self.filetype][0]
        self.compression_ext = dict(COMPRESSIONS)[self.compression] if self.compress else ''

    def _update_text(self):
        self._update_extension()
        self.adjust_label()


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    table = Table("iris")
    ow = OWSave()
    ow.show()

    """
    from scipy import sparse
    table.X = sparse.csr_matrix(table.X)
    """

    table.is_sparse()
    ow.dataset(table)
    a.exec()
    ow.saveSettings()

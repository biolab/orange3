import os.path
import pathlib

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from Orange.data.table import Table
from Orange.data.io import Compression, FileFormat, TabReader, CSVReader, PickleReader
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils import filedialogs
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input

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


class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Error(widget.OWWidget.Error):
        unsupported_extension = widget.Msg("Selected extension is not supported.")

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")
    auto_save = Setting(False)
    filetype = Setting(FILE_TYPES[0][0])
    compression = Setting(COMPRESSIONS[0][0])
    compress = Setting(False)

    def __init__(self):
        super().__init__()
        self.data = None
        self.filename = ""
        self.basename = ""
        self.type_ext = ""
        self.compress_ext = ""

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

    def get_writer_selected(self):
        writer = FileFormat.get_reader(self.type_ext)

        ext = self.type_ext + self.compress_ext
        if ext not in writer.EXTENSIONS:
            self.Error.unsupported_extension()
            return None
        writer.EXTENSIONS = [ext]
        return writer

    @classmethod
    def remove_extensions(cls, filename):
        if not filename:
            return None
        for ext in pathlib.PurePosixPath(filename).suffixes:
            filename = filename.replace(ext, '')
        return filename

    def adjust_label(self):
        if self.filename:
            text = "Auto save as '{}'" if self.auto_save else "Save as '{}'"
            self.save.button.setText(
                text.format(self.basename + self.type_ext + self.compress_ext))

    @Inputs.data
    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.save_as.setDisabled(data is None)
        if data is None:
            return

        items = [item for item, _, supports_sparse in FILE_TYPES
                 if supports_sparse or not data.is_sparse()]
        if items != [self.controls.filetype.itemText(i) for i in
                     range(self.controls.filetype.count())]:
            self.controls.filetype.clear()
            self.controls.filetype.insertItems(0, items)
            self.update_extension()

        self.save_file()

    def save_file_as(self):
        file_name = self.remove_extensions(self.filename) or os.path.join(
            self.last_dir or os.path.expanduser("~"),
            getattr(self.data, 'name', ''))
        self.update_extension()
        writer = self.get_writer_selected()
        if not writer:
            return

        filename, writer, _ = filedialogs.open_filename_dialog_save(
            file_name, '', [writer],
        )
        if not filename:
            return

        self.filename = filename
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
                self.get_writer_selected().write(
                    os.path.join(
                        self.last_dir,
                        self.basename + self.type_ext + self.compress_ext),
                    self.data)
            except Exception as err_value:
                self.error(str(err_value))
            else:
                self.error()

    def update_extension(self):
        self.type_ext = [ext for name, ext, _ in FILE_TYPES if name == self.filetype][0]
        self.compress_ext = dict(COMPRESSIONS)[self.compression] if self.compress else ''

    def _update_text(self):
        self.update_extension()
        self.adjust_label()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSave).run(Table("iris"))

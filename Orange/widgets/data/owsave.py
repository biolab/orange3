import os.path
import pathlib

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from Orange.data.table import Table
from Orange.data.io import FileFormat
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils import filedialogs
from Orange.widgets.widget import Input

FILE_TYPES_NAMES = ["Tab-delimited (.tab)", "Comma-seperated values (.csv)", "Pickle (.pkl)"]
FILE_TYPES = {
    "Tab-delimited (.tab)": ".tab",
    "Comma-seperated values (.csv)": ".csv",
    "Pickle (.pkl)": ".pkl",
}
COMPRESSIONS_NAMES = ["gzip (.gz)", "bbzip2 (.bz2)", "Izma (.xz)"]
COMPRESSIONS = {
    "gzip (.gz)": ".gz",
    "bbzip2 (.bz2)": ".bz2",
    "Izma (.xz)": ".xz",
}


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
    filetype = Setting(FILE_TYPES_NAMES[0])
    compression = Setting(COMPRESSIONS_NAMES[0])
    compress = Setting(False)

    @classmethod
    def get_writers(cls, sparse):
        return [f for f in FileFormat.formats
                if getattr(f, 'write_file', None) and getattr(f, "EXTENSIONS", None)
                and (not sparse or getattr(f, 'SUPPORT_SPARSE_DATA', False))]

    def get_writer_selected(self):
        type = FILE_TYPES[self.filetype]
        compression = COMPRESSIONS[self.compression] if self.compress else ''
        writer = FileFormat.get_reader(type)
        writer.EXTENSIONS = [writer.EXTENSIONS[writer.EXTENSIONS.index(type + compression)]]
        return writer

    @classmethod
    def remove_extensions(cls, filename):
        if not filename:
            return None
        for ext in pathlib.PurePosixPath(filename).suffixes:
            filename = filename.replace(ext, '')
        return filename

    def __init__(self):
        super().__init__()
        self.data = None
        self.filename = ""
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
            callback=None,
            items=FILE_TYPES_NAMES,
            sendSelectedValue=True,
        )
        form.addRow("File type", self.controls.filetype, )

        gui.comboBox(
            box, self, "compression",
            callback=None,
            items=COMPRESSIONS_NAMES,
            sendSelectedValue=True,
        )
        gui.checkBox(box, self, "compress", label="Use compression", )

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
            filename = os.path.split(self.filename)[1]
            text = "Auto save as '{}'" if self.auto_save else "Save as '{}'"
            self.save.button.setText(text.format(filename))

    @Inputs.data
    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.save_as.setDisabled(data is None)
        if data is not None:
            self.save_file()

    def save_file_as(self):
        file_name = self.remove_extensions(self.filename) or \
                    os.path.join(self.last_dir or os.path.expanduser("~"),
                                 getattr(self.data, 'name', ''))


        filename, writer, _ = filedialogs.open_filename_dialog_save(
            file_name, '', [self.get_writer_selected()],
        )
        if not filename:
            return
        self.filename = filename
        self.writer = writer
        self.unconditional_save_file()
        self.last_dir = os.path.split(self.filename)[0]
        self.adjust_label()

    def save_file(self):
        if self.data is None:
            return
        if not self.filename:
            self.save_file_as()
        else:
            try:
                self.writer.write(self.filename, self.data)
            except Exception as err_value:
                self.error(str(err_value))
            else:
                self.error()


    def get_writer_selected_for_testing(self):
        type = FILE_TYPES[self.controls.filetype.currentText()]
        compression = COMPRESSIONS[self.controls.compress.currentText()] if self.compress  else ''
        writer = FileFormat.get_reader(type)
        writer.EXTENSIONS = [writer.EXTENSIONS[writer.EXTENSIONS.index(type + compression)]]
        return writer

if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    table = Table("iris")
    ow = OWSave()
    ow.show()
    ow.dataset(table)
    a.exec()
    ow.saveSettings()

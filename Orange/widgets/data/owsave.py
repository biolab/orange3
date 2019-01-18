import os.path

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QBrush
from AnyQt.QtWidgets import QGridLayout

from Orange.data.table import Table
from Orange.data.io import TabReader, CSVReader, PickleReader, ExcelReader
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils import filedialogs
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input


FILE_WRITERS = (TabReader, CSVReader, PickleReader, ExcelReader)
FILE_FORMATS = [f"{w.DESCRIPTION} ({w.EXTENSIONS[0]})" for w in FILE_WRITERS]


class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Error(widget.OWWidget.Error):
        unsupported_sparse = \
            widget.Msg("This file format does not support sparse data.")
        general_error = widget.Msg("{}")

    want_main_area = False
    resizing_enabled = False

    filetype = Setting(0)
    last_dir = Setting("")
    compression = Setting(False)
    add_type_annotations = Setting(True)
    auto_save = Setting(False)

    def __init__(self):
        super().__init__()
        self.data = None
        self.basename = ""
        self.dirty = False

        box = gui.vBox(self.controlArea, "Format")
        gui.comboBox(
            box, self, "filetype", items=FILE_FORMATS,
            callback=self._set_dirty, addSpace=2)
        box2 = gui.indentedBox(box, sep=10)
        gui.checkBox(
            box2, self, "compression", "Compress file (as gzip)",
            callback=self._set_dirty, addSpace=2)
        gui.checkBox(
            box2, self, "add_type_annotations", label="Add type annotations",
            callback=self._set_dirty)

        grid = QGridLayout()
        gui.widgetBox(self.controlArea, box=True, orientation=grid)
        self.save = gui.button(None, self, "Save", callback=self.save_file)
        self.save_as = gui.button(
            None, self, "Save as ...", callback=self.save_file_as)
        grid.addWidget(self.save, 0, 0)
        grid.addWidget(self.save_as, 0, 1)
        grid.setRowMinimumHeight(1, 8)
        grid.addWidget(
            gui.checkBox(
                None, self, "auto_save", "Autosave when receiving new data"),
            2, 0, 2, 2)
        self.adjustSize()
        self._update_controls()

    @Inputs.data
    def dataset(self, data):
        self.Error.clear()
        self.data = data
        self.dirty = data is not None
        self._update_combo_colors()
        if self.auto_save and self.basename:
            self.save_file()
        self._update_controls()

    def save_file_as(self):
        start_path = self.basename \
            or os.path.join(self.last_dir or os.path.expanduser("~"),
                            getattr(self.data, 'name', ''))
        filename, _, _ = filedialogs.open_filename_dialog_save(
            start_path, '', [FILE_WRITERS[self.filetype]])
        if not filename:
            return

        self.last_dir, filename = os.path.split(filename)
        self.basename, _ = os.path.splitext(filename)
        self.save_file()
        self._update_controls()

    def save_file(self):
        writer = FILE_WRITERS[self.filetype]
        if self.data is None \
                or self.data.is_sparse() and not writer.SUPPORT_SPARSE_DATA:
            return
        if not self.basename:
            self.save_file_as()
            return
        try:
            filename = os.path.join(self.last_dir, self._fullname)
            writer.write(filename, self.data, self.add_type_annotations)
        except Exception as err_value:
            self.Error.general_error(str(err_value))
        else:
            self.Error.general_error.clear()
            self.dirty = False
            self._update_controls()

    @property
    def _fullname(self):
        writer = FILE_WRITERS[self.filetype]
        return self.basename \
            + writer.EXTENSIONS[0] \
            + ".gzip" * (self.compression and writer.SUPPORT_COMPRESSED)

    def _update_combo_colors(self):
        sparse_data = self.data is not None and self.data.is_sparse()
        combo = self.controls.filetype
        brushes = QBrush(Qt.gray), QBrush(Qt.black)
        for i, writer in enumerate(FILE_WRITERS):
            combo.setItemData(
                i, brushes[not sparse_data or writer.SUPPORT_SPARSE_DATA],
                Qt.TextColorRole)

    def _set_dirty(self):
        self.dirty = True
        self._update_controls()

    def _update_controls(self):
        ctrl = self.controls
        writer = FILE_WRITERS[self.filetype]

        self.Error.unsupported_sparse(
            shown=self.data is not None
                  and self.data.is_sparse()
                  and not writer.SUPPORT_SPARSE_DATA)
        ctrl.add_type_annotations.setEnabled(writer.OPTIONAL_TYPE_ANNOTATIONS)
        ctrl.compression.setEnabled(writer.SUPPORT_COMPRESSED)

        self.save.setText(
            f"Save as '{self._fullname}'" if self.basename else "Save")
        enable_save = self.data is not None \
            and (not self.data.is_sparse() or writer.SUPPORT_SPARSE_DATA)
        self.save.setVisible(self.dirty and bool(self.basename))
        self.save.setEnabled(enable_save)
        self.save_as.setEnabled(enable_save)
        ctrl.auto_save.setEnabled(enable_save and bool(self.basename))

    @classmethod
    def migrate_settings(cls, settings, version=0):
        return
        if version < 2:
            for i, (compr, _) in enumerate(COMPRESSIONS):
                if compr == settings["compression"]:
                    settings["compression"] = i
                    break
            else:
                settings["compression"] = 0
            settings["filetype"] = FILE_FORMATS.index(settings["filetype"])


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSave).run(Table("iris"))

import os.path

from AnyQt.QtWidgets import QFileDialog, QGridLayout, QWidget

from Orange.data.table import Table
from Orange.data.io import TabReader, CSVReader, PickleReader, ExcelReader
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input

class FileDialog(QFileDialog):
    def changeEvent(self, e):
        print(e)
        super().selectFile(e)

class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = []

    settings_version = 2

    writers = [TabReader, CSVReader, PickleReader, ExcelReader]
    filters = [f"{w.DESCRIPTION} (*.*)" for w in writers]
    filt_ext = {filter: w.EXTENSIONS[0] for filter, w in zip(filters, writers)}
    userhome = os.path.expanduser(f"~{os.sep}")

    class Inputs:
        data = Input("Data", Table)

    class Error(widget.OWWidget.Error):
        unsupported_sparse = widget.Msg("Use .pkl format for sparse data.")
        no_file_name = widget.Msg("File name is not set.")
        general_error = widget.Msg("{}")

    class Warning(widget.OWWidget.Warning):
        ignored_flag = widget.Msg("{} ignored for this format.")

    want_main_area = False
    resizing_enabled = False

    compress: bool
    add_type_annotations: bool

    last_dir = Setting("")
    filter = Setting(filters[0])
    compress = Setting(False)
    add_type_annotations = Setting(True)
    auto_save = Setting(False)

    def __init__(self):
        super().__init__()
        self.data = None
        self.filename = ""
        self.writer = self.writers[0]

        grid = QGridLayout()
        gui.widgetBox(self.controlArea, box=True, orientation=grid)
        grid.setSpacing(8)
        self.bt_save = gui.button(None, self, "Save", callback=self.save_file)
        grid.addWidget(self.bt_save, 0, 0)
        grid.addWidget(
            gui.button(None, self, "Save as ...", callback=self.save_file_as),
            0, 1)
        grid.addWidget(
            gui.checkBox(None, self, "auto_save",
                         "Autosave when receiving new data",
                         callback=self._update_controls),
            1, 0, 1, 2)
        grid.addWidget(QWidget(), 2, 0, 1, 2)

        grid.addWidget(
            gui.checkBox(
                None, self, "add_type_annotations",
                "Save with type annotations", callback=self._update_controls),
            3, 0, 1, 2)
        grid.addWidget(
            gui.checkBox(
                None, self, "compress", "Compress file (gzip)",
                callback=self._update_controls),
            4, 0, 1, 2)

        self.adjustSize()
        self._update_controls()

    @Inputs.data
    def dataset(self, data):
        self.Error.clear()
        self.data = data

        self._update_controls()
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(len(self.data)),
                f"Data set {self.data.name or '(no name)'} "
                f"with {len(self.data)} instances")

        if self.auto_save and self.filename:
            self.save_file()

    def save_file_as(self):
        if self.filename:
            start_dir = self.filename
        else:
            data_name = getattr(self.data, 'name', '')
            if data_name:
                data_name += self.filt_ext[self.filter]
            start_dir = os.path.join(self.last_dir or self.userhome, data_name)

        dlg = FileDialog(None, "Set File", start_dir, ";;".join(self.filters))
        dlg.setLabelText(dlg.Accept, "Select")
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.setSupportedSchemes(["file"])
        dlg.selectNameFilter(self.filter)
        dlg.setOption(QFileDialog.HideNameFilterDetails)
        dlg.currentChanged.connect(print)
        if dlg.exec() == dlg.Rejected:
            return

        filename = dlg.selectedFiles()[0]
        selected_filter = dlg.selectedNameFilter()

#        filename, selected_filter = QFileDialog.getSaveFileName(
 #           self, "Save data", start_dir, ";;".join(self.filters), self.filter,
  #          QFileDialog.HideNameFilterDetails)
        if not filename:
            return

        self.filename = filename
        self.last_dir = os.path.split(filename)[0]
        self.filter = selected_filter
        self.writer = self.writers[self.filters.index(self.filter)]
        self._update_controls()
        self.save_file()

    def save_file(self):
        if not self.filename:
            self.save_file_as()
            return
        self.Error.general_error.clear()
        if not self._can_save():
            return
        try:
            self.writer.write(
                self._fullname(), self.data, self.add_type_annotations)
        except IOError as err_value:
            self.Error.general_error(str(err_value))

    def _fullname(self):
        return self.filename \
               + ".gz" * self.writer.SUPPORT_COMPRESSED * self.compress

    def _update_controls(self):
        if self.filename:
            self.bt_save.setText(
                f"Save as {os.path.split(self._fullname())[1]}")
        else:
            self.bt_save.setText("Save")
        self.Error.no_file_name(shown=not self.filename and self.auto_save)

        self.Error.unsupported_sparse(
            shown=self.data is not None and self.data.is_sparse()
            and self.filename and not self.writer.SUPPORT_SPARSE_DATA)

        if self.data is None or not self.filename:
            self.Warning.ignored_flag.clear()
        else:
            no_compress = self.compress \
                          and not self.writer.SUPPORT_COMPRESSED
            no_anotation = self.add_type_annotations \
                           and not self.writer.OPTIONAL_TYPE_ANNOTATIONS
            ignored = [
                "",
                "Compression flag is",
                "Type annotation flag is",
                "Compression and type annotation flags are"
            ][no_compress + 2 * no_anotation]
            self.Warning.ignored_flag(ignored, shown=bool(ignored))

    def _can_save(self):
        return not (
            self.data is None
            or not self.filename
            or self.data.is_sparse() and not self.writer.SUPPORT_SPARSE_DATA
        )

    def send_report(self):
        self.report_data_brief(self.data)
        writer = self.writer
        noyes = ["No", "Yes"]
        self.report_items((
            ("File name", self.filename or "not set"),
            ("Format", writer.DESCRIPTION),
            ("Compression", writer.SUPPORT_COMPRESSED and noyes[self.compress]),
            ("Type annotations",
             writer.OPTIONAL_TYPE_ANNOTATIONS
             and noyes[self.add_type_annotations])
        ))

    @classmethod
    def migrate_settings(cls, settings, version=0):
        settings.filter = next(iter(cls.filt_ext))
#        if version < 2:
 #           settings["filter"] = settings.pop("filetype")


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSave).run(Table("iris"))

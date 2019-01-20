import os.path

from AnyQt.QtWidgets import QFileDialog
from AnyQt.QtCore import Qt

from Orange.data.table import Table
from Orange.data.io import TabReader, CSVReader, PickleReader, ExcelReader
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input


class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = []

    settings_version = 2

    writers = [TabReader, CSVReader, PickleReader, ExcelReader]
    filters = [f"{w.DESCRIPTION} ({w.EXTENSIONS[0]})" for w in writers]
    filt_ext = {filter: w.EXTENSIONS[0] for filter, w in zip(filters, writers)}
    userhome = os.path.expanduser(f"~{os.sep}")

    class Inputs:
        data = Input("Data", Table)

    class Error(widget.OWWidget.Error):
        # This message is short to (almost) fit into the widget's width
        unsupported_sparse = widget.Msg("Format can't store sparse data.")
        general_error = widget.Msg("{}")

    class Warning(widget.OWWidget.Warning):
        no_file_name = widget.Msg("Set the file name.")
        general_error = widget.Msg("{}")

    want_main_area = False
    resizing_enabled = False

    compress: bool
    add_type_annotations: bool

    filetype = Setting(0)
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

        box = gui.vBox(self.controlArea, True)
        box.layout().setSpacing(8)
        self.lb_filename = gui.widgetLabel(box)
        gui.checkBox(
            box, self, "add_type_annotations", "Save with type annotations")
        gui.checkBox(
            box, self, "compress", "Compress file (gzip)")
        self.bt_set_file = gui.button(
            None, self, "Set File Name", callback=self.set_file_name)
        box.layout().addWidget(self.bt_set_file, Qt.AlignRight)

        box = gui.vBox(self.controlArea, box=True)
        box.layout().setSpacing(8)
        gui.checkBox(
            box, self, "auto_save", "Autosave when receiving new data")
        self.bt_save = gui.button(box, self, "Save", callback=self.save_file)
        self.adjustSize()
        self._update_controls()

    @Inputs.data
    def dataset(self, data):
        self.Error.clear()
        self.data = data
        if self.auto_save and self.filename:
            self.save_file()

        self._update_controls()
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(len(self.data)),
                f"Data set {self.data.name or '(no name)'} "
                f"with {len(self.data)} instances")

    def set_file_name(self):
        if self.filename:
            start_dir = self.filename
        else:
            data_name = getattr(self.data, 'name', '')
            if data_name:
                data_name += self.filt_ext[self.filter]
            start_dir = os.path.join(self.last_dir or self.userhome, data_name)

        dlg = QFileDialog(None, "Set File", start_dir, ";;".join(self.filters))
        dlg.setLabelText(dlg.Accept, "Select")
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.setSupportedSchemes(["file"])
        dlg.selectNameFilter(self.filter)
        if dlg.exec() == dlg.Rejected:
            return

        self.filename = dlg.selectedFiles()[0]
        self.last_dir = os.path.split(self.filename)[0]
        self.filter = dlg.selectedNameFilter()
        self.writer = self.writers[self.filters.index(self.filter)]
        self._update_controls()

    def save_file(self):
        self.Error.general_error.clear()
        if not self._can_save():
            return
        name = self.filename \
            + ".gz" * self.writer.SUPPORT_COMPRESSED * self.compress
        try:
            self.writer.write(name, self.data, self.add_type_annotations)
        except IOError as err_value:
            self.Error.general_error(str(err_value))

    def _update_controls(self):
        has_file = bool(self.filename)
        self.lb_filename.setVisible(has_file)
        self.Warning.no_file_name(shown=not has_file)
        if self.filename:
            name = self.filename
            if name.startswith(self.userhome):
                name = name[len(self.userhome):]
            self.lb_filename.setText(f"Save to: {name}")

        self.controls.add_type_annotations.setVisible(
            has_file and self.writer.OPTIONAL_TYPE_ANNOTATIONS)
        self.controls.compress.setVisible(
            has_file and self.writer.SUPPORT_COMPRESSED)

        self.Error.unsupported_sparse(
            shown=self.data is not None and self.data.is_sparse()
            and self.filename
            and not self.writer.SUPPORT_SPARSE_DATA)
        self.bt_save.setEnabled(self._can_save())

    def _can_save(self):
        return self.data is not None \
            and bool(self.filename) \
            and (not self.data.is_sparse() or self.writer.SUPPORT_SPARSE_DATA)

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
        if version < 2:
            settings["filter"] = settings.pop("filetype")


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSave).run(Table("iris"))

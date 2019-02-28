import os.path
import sys
import re

from AnyQt.QtWidgets import QFileDialog, QGridLayout, QMessageBox

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
    filters = {
        **{f"{w.DESCRIPTION} (*{w.EXTENSIONS[0]})": w
           for w in writers},
        **{f"Compressed {w.DESCRIPTION} (*{w.EXTENSIONS[0]}.gz)": w
           for w in writers if w.SUPPORT_COMPRESSED}
    }
    userhome = os.path.expanduser(f"~{os.sep}")

    class Inputs:
        data = Input("Data", Table)

    class Error(widget.OWWidget.Error):
        unsupported_sparse = widget.Msg("Use Pickle format for sparse data.")
        no_file_name = widget.Msg("File name is not set.")
        general_error = widget.Msg("{}")

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")
    filter = Setting(next(iter(filters)))
    filename = Setting("", schema_only=True)
    add_type_annotations = Setting(True)
    auto_save = Setting(False)

    def __init__(self):
        super().__init__()
        self.data = None

        grid = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=grid)
        grid.addWidget(
            gui.checkBox(
                None, self, "add_type_annotations",
                "Add type annotations to header",
                tooltip=
                "Some formats (Tab-delimited, Comma-separated) can include \n"
                "additional information about variables types in header rows.",
                callback=self._update_messages),
            0, 0, 1, 2)
        grid.setRowMinimumHeight(1, 8)
        grid.addWidget(
            gui.checkBox(
                None, self, "auto_save", "Autosave when receiving new data",
                callback=self._update_messages),
            2, 0, 1, 2)
        grid.setRowMinimumHeight(3, 8)
        self.bt_save = gui.button(None, self, "Save", callback=self.save_file)
        grid.addWidget(self.bt_save, 4, 0)
        grid.addWidget(
            gui.button(None, self, "Save as ...", callback=self.save_file_as),
            4, 1)

        self.adjustSize()
        self._update_messages()

    @property
    def writer(self):
        return self.filters[self.filter]

    @Inputs.data
    def dataset(self, data):
        self.Error.clear()
        self.data = data
        self._update_status()
        self._update_messages()
        if self.auto_save and self.filename:
            self.save_file()

    def save_file(self):
        if not self.filename:
            self.save_file_as()
            return

        self.Error.general_error.clear()
        if self.data is None \
                or not self.filename \
                or (self.data.is_sparse()
                        and not self.writer.SUPPORT_SPARSE_DATA):
            return
        try:
            self.writer.write(
                self.filename, self.data, self.add_type_annotations)
        except IOError as err_value:
            self.Error.general_error(str(err_value))

    def save_file_as(self):
        filename, selected_filter = self.get_save_filename()
        if not filename:
            return
        self.filename = filename
        self.filter = selected_filter
        self.last_dir = os.path.split(self.filename)[0]
        self.bt_save.setText(f"Save as {os.path.split(filename)[1]}")
        self._update_messages()
        self.save_file()

    def _update_messages(self):
        self.Error.no_file_name(
            shown=not self.filename and self.auto_save)
        self.Error.unsupported_sparse(
            shown=self.data is not None and self.data.is_sparse()
            and self.filename and not self.writer.SUPPORT_SPARSE_DATA)

    def _update_status(self):
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(len(self.data)),
                f"Data set {self.data.name or '(no name)'} "
                f"with {len(self.data)} instances")

    def send_report(self):
        self.report_data_brief(self.data)
        writer = self.writer
        noyes = ["No", "Yes"]
        self.report_items((
            ("File name", self.filename or "not set"),
            ("Format", writer.DESCRIPTION),
            ("Type annotations",
             writer.OPTIONAL_TYPE_ANNOTATIONS
             and noyes[self.add_type_annotations])
        ))

    @classmethod
    def migrate_settings(cls, settings, version=0):
        def migrate_to_version_2():
            # Set the default; change later if possible
            settings.pop("compression", None)
            settings["filter"] = next(iter(cls.filters))
            filetype = settings.pop("filetype", None)
            if filetype is None:
                return

            ext = cls._extension_from_filter(filetype)
            if settings.pop("compress", False):
                for afilter in cls.filters:
                    if ext + ".gz" in afilter:
                        settings["filter"] = afilter
                        return
                # If not found, uncompressed may have been erroneously set
                # for a writer that didn't support if (such as .xlsx), so
                # fall through to uncompressed
            for afilter in cls.filters:
                if ext in afilter:
                    settings["filter"] = afilter
                    return

        if version < 2:
            migrate_to_version_2()

    def _initial_start_dir(self):
        if self.filename and os.path.exists(os.path.split(self.filename)[0]):
            return self.filename
        else:
            data_name = getattr(self.data, 'name', '')
            if data_name:
                data_name += self.writer.EXTENSIONS[0]
            return os.path.join(self.last_dir or self.userhome, data_name)

    @staticmethod
    def _replace_extension(filename, extension):
        known_extensions = map(OWSave._extension_from_filter, OWSave.filters)
        for known_ext in sorted(known_extensions, key=len, reverse=True):
            if filename.endswith(known_ext):
                filename = filename[:-len(known_ext)]
                break
        return filename + extension

    @staticmethod
    def _extension_from_filter(selected_filter):
        return re.search(r".*\(\*?(\..*)\)$", selected_filter).group(1)

    # As of Qt 5.9, QFileDialog.setDefaultSuffix does not support double
    # suffixes, not even in non-native dialogs. We handle each OS separately.
    if sys.platform == "darwin":
        # On macOS, is double suffixes are passed to the dialog, they are
        # appended multiple times even if already present (QTBUG-44227).
        # The only known workaround with native dialog is to use suffix *.*.
        # We add the correct suffix after closing the dialog and only then check
        # if the file exists and ask whether to override.
        # It is a bit confusing that the user does not see the final name in the
        # dialog, but I see no better solution.
        def get_save_filename(self):  # pragma: no cover
            def no_suffix(filt):
                return filt.split("(")[0] + "(*.*)"

            mac_filters = {no_suffix(f): f for f in self.filters}
            filename = self._initial_start_dir()
            while True:
                dlg = QFileDialog(
                    None, "Save File", filename, ";;".join(mac_filters))
                dlg.setAcceptMode(dlg.AcceptSave)
                dlg.selectNameFilter(no_suffix(self.filter))
                dlg.setOption(QFileDialog.HideNameFilterDetails)
                dlg.setOption(QFileDialog.DontConfirmOverwrite)
                if dlg.exec() == QFileDialog.Rejected:
                    return "", ""
                filename = dlg.selectedFiles()[0]
                selected_filter = mac_filters[dlg.selectedNameFilter()]
                filename = self._replace_extension(
                    filename, self._extension_from_filter(selected_filter))
                if not os.path.exists(filename) or QMessageBox.question(
                        self, "Overwrite file?",
                        f"File {os.path.split(filename)[1]} already exists.\n"
                        "Overwrite?") == QMessageBox.Yes:
                    return filename, selected_filter

    elif sys.platform == "win32":
        # TODO: This is not tested!!!
        # Windows native dialog may work correctly; if not, we may do the same
        # as for macOS?
        def get_save_filename(self):  # pragma: no cover
            return QFileDialog.getSaveFileName(
                self, "Save File", self._initial_start_dir(),
                ";;".join(self.filters), self.filter)

    else:  # Linux and any unknown platforms
        # Qt does not use a native dialog on Linux, so we can connect to
        # filterSelected and to overload selectFile to change the extension
        # while the dialog is open.
        # For unknown platforms (which?), we also use the non-native dialog to
        # be sure we know what happens.
        class SaveFileDialog(QFileDialog):
            # pylint: disable=protected-access
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.setAcceptMode(QFileDialog.AcceptSave)
                self.setOption(QFileDialog.DontUseNativeDialog)
                self.filterSelected.connect(self.updateDefaultExtension)

            def selectNameFilter(self, selected_filter):
                super().selectNameFilter(selected_filter)
                self.updateDefaultExtension(selected_filter)

            def updateDefaultExtension(self, selected_filter):
                self.suffix = OWSave._extension_from_filter(selected_filter)
                files = self.selectedFiles()
                if files and not os.path.isdir(files[0]):
                    self.selectFile(files[0].split(".")[0])

            def selectFile(self, filename):
                filename = OWSave._replace_extension(filename, self.suffix)
                super().selectFile(filename)

        def get_save_filename(self):
            dlg = self.SaveFileDialog(
                None, "Save File", self._initial_start_dir(),
                ";;".join(self.filters))
            dlg.selectNameFilter(self.filter)
            if dlg.exec() == QFileDialog.Rejected:
                return "", ""
            else:
                return dlg.selectedFiles()[0], dlg.selectedNameFilter()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSave).run(Table("iris"))

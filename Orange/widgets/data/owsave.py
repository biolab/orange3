import os.path

from Orange.data.table import Table
from Orange.data.io import \
    TabReader, CSVReader, PickleReader, ExcelReader, XlsReader, FileFormat
from Orange.widgets import gui, widget
from Orange.widgets.widget import Input
from Orange.widgets.settings import Setting
from Orange.widgets.utils.save.owsavebase import OWSaveBase
from Orange.widgets.utils.widgetpreview import WidgetPreview


_userhome = os.path.expanduser(f"~{os.sep}")


class OWSave(OWSaveBase):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = ["export"]

    settings_version = 3

    class Inputs:
        data = Input("Data", Table)

    class Error(OWSaveBase.Error):
        unsupported_sparse = widget.Msg("Use Pickle format for sparse data.")

    add_type_annotations = Setting(True)

    builtin_order = [TabReader, CSVReader, PickleReader, ExcelReader, XlsReader]

    def __init__(self):
        super().__init__(2)
        self.grid.addWidget(
            gui.checkBox(
                None, self, "add_type_annotations",
                "Add type annotations to header",
                tooltip=
                "Some formats (Tab-delimited, Comma-separated) can include \n"
                "additional information about variables types in header rows.",
                callback=self.update_messages),
            0, 0, 1, 2)
        self.grid.setRowMinimumHeight(1, 8)
        self.adjustSize()

    @classmethod
    def get_filters(cls):
        writers = [format for format in FileFormat.formats
                   if getattr(format, 'write_file', None)
                   and getattr(format, "EXTENSIONS", None)]
        writers.sort(key=lambda writer: cls.builtin_order.index(writer)
                     if writer in cls.builtin_order else 99)

        return {
            **{f"{w.DESCRIPTION} (*{w.EXTENSIONS[0]})": w
               for w in writers},
            **{f"Compressed {w.DESCRIPTION} (*{w.EXTENSIONS[0]}.gz)": w
               for w in writers if w.SUPPORT_COMPRESSED}
        }

    @Inputs.data
    def dataset(self, data):
        self.data = data
        self.on_new_input()

    def do_save(self):
        if self.writer is None:
            super().do_save()  # This will do nothing but indicate an error
            return
        if self.data.is_sparse() and not self.writer.SUPPORT_SPARSE_DATA:
            return
        self.writer.write(self.filename, self.data, self.add_type_annotations)

    def update_messages(self):
        super().update_messages()
        self.Error.unsupported_sparse(
            shown=self.data is not None and self.data.is_sparse()
            and self.filename
            and self.writer is not None and not self.writer.SUPPORT_SPARSE_DATA)

    def send_report(self):
        self.report_data_brief(self.data)
        writer = self.writer
        noyes = ["No", "Yes"]
        self.report_items((
            ("File name", self.filename or "not set"),
            ("Format", writer and writer.DESCRIPTION),
            ("Type annotations",
             writer and writer.OPTIONAL_TYPE_ANNOTATIONS
             and noyes[self.add_type_annotations])
        ))

    @classmethod
    def migrate_settings(cls, settings, version=0):
        def migrate_to_version_2():
            # Set the default; change later if possible
            settings.pop("compression", None)
            settings["filter"] = next(iter(cls.get_filters()))
            filetype = settings.pop("filetype", None)
            if filetype is None:
                return

            ext = cls._extension_from_filter(filetype)
            if settings.pop("compress", False):
                for afilter in cls.get_filters():
                    if ext + ".gz" in afilter:
                        settings["filter"] = afilter
                        return
                # If not found, uncompressed may have been erroneously set
                # for a writer that didn't support if (such as .xlsx), so
                # fall through to uncompressed
            for afilter in cls.get_filters():
                if ext in afilter:
                    settings["filter"] = afilter
                    return

        if version < 2:
            migrate_to_version_2()

        if version < 3:
            if settings.get("add_type_annotations") and \
                    settings.get("stored_name") and \
                    os.path.splitext(settings["stored_name"])[1] == ".xlsx":
                settings["add_type_annotations"] = False

    def initial_start_dir(self):
        if self.filename and os.path.exists(os.path.split(self.filename)[0]):
            return self.filename
        else:
            data_name = getattr(self.data, 'name', '')
            if data_name:
                if self.writer is None:
                    self.filter = self.default_filter()
                assert self.writer is not None
                data_name += self.writer.EXTENSIONS[0]
            return os.path.join(self.last_dir or _userhome, data_name)

    def valid_filters(self):
        if self.data is None or not self.data.is_sparse():
            return self.get_filters()
        else:
            return {filt: writer for filt, writer in self.get_filters().items()
                    if writer.SUPPORT_SPARSE_DATA}

    def default_valid_filter(self):
        valid = self.valid_filters()
        if self.data is None or not self.data.is_sparse() \
                or (self.filter in valid
                        and valid[self.filter].SUPPORT_SPARSE_DATA):
            return self.filter
        return next(iter(valid))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSave).run(Table("iris"))

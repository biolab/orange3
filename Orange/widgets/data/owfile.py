import os
import logging
from warnings import catch_warnings
from urllib.parse import urlparse

import numpy as np
from AnyQt.QtWidgets import \
    QStyle, QComboBox, QMessageBox, QGridLayout, QLabel, \
    QLineEdit, QSizePolicy as Policy
from AnyQt.QtCore import Qt, QTimer, QSize

from Orange.canvas.gui.utils import OSX_NSURL_toLocalFile
from Orange.data.table import Table, get_sample_datasets_dir
from Orange.data.io import FileFormat, UrlReader, class_from_qualified_name
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler, SettingProvider
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin, \
    open_filename_dialog
from Orange.widgets.widget import Output

# Backward compatibility: class RecentPath used to be defined in this module,
# and it is used in saved (pickled) settings. It must be imported into the
# module's namespace so that old saved settings still work
from Orange.widgets.utils.filedialogs import RecentPath


log = logging.getLogger(__name__)


def add_origin(examples, filename):
    """
    Adds attribute with file location to each string variable
    Used for relative filenames stored in string variables (e.g. pictures)
    TODO: we should consider a cleaner solution (special variable type, ...)
    """
    if not filename:
        return
    vars = examples.domain.variables + examples.domain.metas
    strings = [var for var in vars if var.is_string]
    dir_name, _ = os.path.split(filename)
    for var in strings:
        if "type" in var.attributes and "origin" not in var.attributes:
            var.attributes["origin"] = dir_name


class NamedURLModel(PyListModel):
    def __init__(self, mapping):
        self.mapping = mapping
        super().__init__()

    def data(self, index, role):
        data = super().data(index, role)
        if role == Qt.DisplayRole:
            return self.mapping.get(data, data)
        return data

    def add_name(self, url, name):
        self.mapping[url] = name
        self.modelReset.emit()


class LineEditSelectOnFocus(QLineEdit):
    def focusInEvent(self, event):
        super().focusInEvent(event)
        # If selectAll is called directly, placing the cursor unselects the text
        QTimer.singleShot(0, self.selectAll)


class OWFile(widget.OWWidget, RecentPathsWComboMixin):
    name = "File"
    id = "orange.widgets.data.file"
    description = "Read data from an input file or network " \
                  "and send a data table to the output."
    icon = "icons/File.svg"
    priority = 10
    category = "Data"
    keywords = ["data", "file", "load", "read"]

    class Outputs:
        data = Output("Data", Table, doc="Attribute-valued dataset read from the input file.")

    want_main_area = False

    SEARCH_PATHS = [("sample-datasets", get_sample_datasets_dir())]
    SIZE_LIMIT = 1e7
    LOCAL_FILE, URL = range(2)

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    # Overload RecentPathsWidgetMixin.recent_paths to set defaults
    recent_paths = Setting([
        RecentPath("", "sample-datasets", "iris.tab"),
        RecentPath("", "sample-datasets", "titanic.tab"),
        RecentPath("", "sample-datasets", "housing.tab"),
        RecentPath("", "sample-datasets", "heart_disease.tab"),
    ])
    recent_urls = Setting([])
    source = Setting(LOCAL_FILE)
    xls_sheet = ContextSetting("")
    sheet_names = Setting({})
    url = Setting("")

    variables = ContextSetting([])

    domain_editor = SettingProvider(DomainEditor)

    class Warning(widget.OWWidget.Warning):
        file_too_big = widget.Msg("The file is too large to load automatically."
                                  " Press Reload to load.")
        load_warning = widget.Msg("Read warning:\n{}")

    class Error(widget.OWWidget.Error):
        file_not_found = widget.Msg("File not found.")
        missing_reader = widget.Msg("Missing reader.")
        sheet_error = widget.Msg("Error listing available sheets.")
        unknown = widget.Msg("Read error:\n{}")

    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.reader = None

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)
        vbox = gui.radioButtons(None, self, "source", box=True, addSpace=True,
                                callback=self.load_data, addToLayout=False)

        rb_button = gui.appendRadioButton(vbox, "File:", addToLayout=False)
        layout.addWidget(rb_button, 0, 0, Qt.AlignVCenter)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        box.layout().addWidget(self.file_combo)
        layout.addWidget(box, 0, 1)

        file_button = gui.button(
            None, self, '...', callback=self.browse_file, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.load_data, autoDefault=False)
        reload_button.setIcon(self.style().standardIcon(
            QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 3)

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_combo = gui.comboBox(None, self, "xls_sheet",
                                        callback=self.select_sheet,
                                        sendSelectedValue=True,)
        self.sheet_combo.setSizePolicy(
            Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_label = QLabel()
        self.sheet_label.setText('Sheet')
        self.sheet_label.setSizePolicy(
            Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(
            self.sheet_label, Qt.AlignLeft)
        self.sheet_box.layout().addWidget(
            self.sheet_combo, Qt.AlignVCenter)
        layout.addWidget(self.sheet_box, 2, 1)
        self.sheet_box.hide()

        rb_button = gui.appendRadioButton(vbox, "URL:", addToLayout=False)
        layout.addWidget(rb_button, 3, 0, Qt.AlignVCenter)

        self.url_combo = url_combo = QComboBox()
        url_model = NamedURLModel(self.sheet_names)
        url_model.wrap(self.recent_urls)
        url_combo.setLineEdit(LineEditSelectOnFocus())
        url_combo.setModel(url_model)
        url_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        url_combo.setEditable(True)
        url_combo.setInsertPolicy(url_combo.InsertAtTop)
        url_edit = url_combo.lineEdit()
        l, t, r, b = url_edit.getTextMargins()
        url_edit.setTextMargins(l + 5, t, r, b)
        layout.addWidget(url_combo, 3, 1, 3, 3)
        url_combo.activated.connect(self._url_set)

        box = gui.vBox(self.controlArea, "Info")
        self.info = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, '')

        box = gui.widgetBox(self.controlArea, "Columns (Double click to edit)")
        self.domain_editor = DomainEditor(self)
        self.editor_model = self.domain_editor.model()
        box.layout().addWidget(self.domain_editor)

        box = gui.hBox(self.controlArea)
        gui.button(
            box, self, "Browse documentation datasets",
            callback=lambda: self.browse_file(True), autoDefault=False)
        gui.rubber(box)

        self.apply_button = gui.button(
            box, self, "Apply", callback=self.apply_domain_edit)
        self.apply_button.setEnabled(False)
        self.apply_button.setFixedWidth(170)
        self.editor_model.dataChanged.connect(
            lambda: self.apply_button.setEnabled(True))

        self.set_file_list()
        # Must not call open_file from within __init__. open_file
        # explicitly re-enters the event loop (by a progress bar)

        self.setAcceptDrops(True)

        if self.source == self.LOCAL_FILE:
            last_path = self.last_path()
            if last_path and os.path.exists(last_path) and \
                    os.path.getsize(last_path) > self.SIZE_LIMIT:
                self.Warning.file_too_big()
                return

        QTimer.singleShot(0, self.load_data)

    def sizeHint(self):
        return QSize(600, 550)

    def select_file(self, n):
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.source = self.LOCAL_FILE
            self.load_data()
            self.set_file_list()

    def select_sheet(self):
        self.recent_paths[0].sheet = self.sheet_combo.currentText()
        self.load_data()

    def _url_set(self):
        url = self.url_combo.currentText()
        pos = self.recent_urls.index(url)
        url = url.strip()

        if not urlparse(url).scheme:
            url = 'http://' + url
            self.url_combo.setItemText(pos, url)
            self.recent_urls[pos] = url

        self.source = self.URL
        self.load_data()

    def browse_file(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation datasets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        readers = [f for f in FileFormat.formats
                   if getattr(f, 'read', None) and getattr(f, "EXTENSIONS", None)]
        filename, reader, _ = open_filename_dialog(start_file, None, readers)
        if not filename:
            return
        self.add_path(filename)
        if reader is not None:
            self.recent_paths[0].file_format = reader.qualified_name()

        self.source = self.LOCAL_FILE
        self.load_data()

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        # We need to catch any exception type since anything can happen in
        # file readers
        self.closeContext()
        self.domain_editor.set_domain(None)
        self.apply_button.setEnabled(False)
        self.clear_messages()
        self.set_file_list()

        error = self._try_load()
        if error:
            error()
            self.data = None
            self.sheet_box.hide()
            self.Outputs.data.send(None)
            self.info.setText("No data.")

    def _try_load(self):
        # pylint: disable=broad-except
        if self.last_path() and not os.path.exists(self.last_path()):
            return self.Error.file_not_found

        try:
            self.reader = self._get_reader()
            assert self.reader is not None
        except Exception:
            return self.Error.missing_reader

        try:
            self._update_sheet_combo()
        except Exception:
            return self.Error.sheet_error

        with catch_warnings(record=True) as warnings:
            try:
                data = self.reader.read()
            except Exception as ex:
                log.exception(ex)
                return lambda x=ex: self.Error.unknown(str(x))
            if warnings:
                self.Warning.load_warning(warnings[-1].message.args[0])

        self.info.setText(self._describe(data))

        self.loaded_file = self.last_path()
        add_origin(data, self.loaded_file)
        self.data = data
        self.openContext(data.domain)
        self.apply_domain_edit()  # sends data

    def _get_reader(self):
        """

        Returns
        -------
        FileFormat
        """
        if self.source == self.LOCAL_FILE:
            path = self.last_path()
            if self.recent_paths and self.recent_paths[0].file_format:
                qname = self.recent_paths[0].file_format
                reader_class = class_from_qualified_name(qname)
                reader = reader_class(path)
            else:
                reader = FileFormat.get_reader(path)
            if self.recent_paths and self.recent_paths[0].sheet:
                reader.select_sheet(self.recent_paths[0].sheet)
            return reader
        elif self.source == self.URL:
            url = self.url_combo.currentText().strip()
            if url:
                return UrlReader(url)

    def _update_sheet_combo(self):
        if len(self.reader.sheets) < 2:
            self.sheet_box.hide()
            self.reader.select_sheet(None)
            return

        self.sheet_combo.clear()
        self.sheet_combo.addItems(self.reader.sheets)
        self._select_active_sheet()
        self.sheet_box.show()

    def _select_active_sheet(self):
        if self.reader.sheet:
            try:
                idx = self.reader.sheets.index(self.reader.sheet)
                self.sheet_combo.setCurrentIndex(idx)
            except ValueError:
                # Requested sheet does not exist in this file
                self.reader.select_sheet(None)
        else:
            self.sheet_combo.setCurrentIndex(0)

    def _describe(self, table):
        domain = table.domain
        text = ""

        attrs = getattr(table, "attributes", {})
        descs = [attrs[desc]
                 for desc in ("Name", "Description") if desc in attrs]
        if len(descs) == 2:
            descs[0] = "<b>{}</b>".format(descs[0])
        if descs:
            text += "<p>{}</p>".format("<br/>".join(descs))

        text += "<p>{} instance(s), {} feature(s), {} meta attribute(s)".\
            format(len(table), len(domain.attributes), len(domain.metas))
        if domain.has_continuous_class:
            text += "<br/>Regression; numerical class."
        elif domain.has_discrete_class:
            text += "<br/>Classification; categorical class with {} values.".\
                format(len(domain.class_var.values))
        elif table.domain.class_vars:
            text += "<br/>Multi-target; {} target variables.".format(
                len(table.domain.class_vars))
        else:
            text += "<br/>Data has no target variable."
        text += "</p>"

        if 'Timestamp' in table.domain:
            # Google Forms uses this header to timestamp responses
            text += '<p>First entry: {}<br/>Last entry: {}</p>'.format(
                table[0, 'Timestamp'], table[-1, 'Timestamp'])
        return text

    def storeSpecificSettings(self):
        self.current_context.modified_variables = self.variables[:]

    def retrieveSpecificSettings(self):
        if hasattr(self.current_context, "modified_variables"):
            self.variables[:] = self.current_context.modified_variables

    def apply_domain_edit(self):
        if self.data is None:
            table = None
        else:
            domain, cols = self.domain_editor.get_domain(self.data.domain, self.data)
            if not (domain.variables or domain.metas):
                table = None
            else:
                X, y, m = cols
                table = Table.from_numpy(domain, X, y, m, self.data.W)
                table.name = self.data.name
                table.ids = np.array(self.data.ids)
                table.attributes = getattr(self.data, 'attributes', {})

        self.Outputs.data.send(table)
        self.apply_button.setEnabled(False)

    def get_widget_name_extension(self):
        _, name = os.path.split(self.loaded_file)
        return os.path.splitext(name)[0]

    def send_report(self):
        def get_ext_name(filename):
            try:
                return FileFormat.names[os.path.splitext(filename)[1]]
            except KeyError:
                return "unknown"

        if self.data is None:
            self.report_paragraph("File", "No file.")
            return

        if self.source == self.LOCAL_FILE:
            home = os.path.expanduser("~")
            if self.loaded_file.startswith(home):
                # os.path.join does not like ~
                name = "~" + os.path.sep + \
                       self.loaded_file[len(home):].lstrip("/").lstrip("\\")
            else:
                name = self.loaded_file
            if self.sheet_combo.isVisible():
                name += " ({})".format(self.sheet_combo.currentText())
            self.report_items("File", [("File name", name),
                                       ("Format", get_ext_name(name))])
        else:
            self.report_items("Data", [("Resource", self.url),
                                       ("Format", get_ext_name(self.url))])

        self.report_data("Data", self.data)

    def dragEnterEvent(self, event):
        """Accept drops of valid file urls"""
        urls = event.mimeData().urls()
        if urls:
            try:
                FileFormat.get_reader(OSX_NSURL_toLocalFile(urls[0]) or
                                      urls[0].toLocalFile())
                event.acceptProposedAction()
            except IOError:
                pass

    def dropEvent(self, event):
        """Handle file drops"""
        urls = event.mimeData().urls()
        if urls:
            self.add_path(OSX_NSURL_toLocalFile(urls[0]) or
                          urls[0].toLocalFile())  # add first file
            self.source = self.LOCAL_FILE
            self.load_data()

if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    ow = OWFile()
    ow.show()
    a.exec_()
    ow.saveSettings()

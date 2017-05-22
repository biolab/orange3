import os
import logging
from itertools import chain, count
from warnings import catch_warnings

import numpy as np
from AnyQt.QtWidgets import \
    QStyle, QComboBox, QMessageBox, QFileDialog, QGridLayout, QLabel, \
    QLineEdit
from AnyQt.QtWidgets import QSizePolicy as Policy
from AnyQt.QtCore import Qt, QTimer, QSize

from Orange.canvas.gui.utils import OSX_NSURL_toLocalFile
from Orange.data import Domain, DiscreteVariable, StringVariable
from Orange.data.table import Table, get_sample_datasets_dir
from Orange.data.io import FileFormat, UrlReader
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextHandler, ContextSetting, \
    PerfectDomainContextHandler
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin

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
    dir_name, basename = os.path.split(filename)
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


class XlsContextHandler(ContextHandler):
    def new_context(self, filename, sheet):
        context = super().new_context()
        context.filename = filename
        return context

    # noinspection PyMethodOverriding
    def match(self, context, filename, sheets):
        context_sheet = context.values.get("xls_sheet")
        if context.filename == filename and context_sheet in sheets:
            return ContextHandler.PERFECT_MATCH
        if context_sheet in sheets:
            return 1
        return ContextHandler.NO_MATCH


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
    outputs = [widget.OutputSignal(
        "Data", Table,
        doc="Attribute-valued data set read from the input file.")]

    want_main_area = False

    SEARCH_PATHS = [("sample-datasets", get_sample_datasets_dir())]
    SIZE_LIMIT = 1e7
    LOCAL_FILE, URL = range(2)

    settingsHandler = PerfectDomainContextHandler()

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

    dlg_formats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    class Warning(widget.OWWidget.Warning):
        file_too_big = widget.Msg("The file is too large to load automatically."
                                  " Press Reload to load.")

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
                                        sendSelectedValue=True)
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
        domain_editor = DomainEditor(self.variables)
        self.editor_model = domain_editor.model()
        box.layout().addWidget(domain_editor)

        box = gui.hBox(self.controlArea)
        gui.button(
            box, self, "Browse documentation data sets",
            callback=lambda: self.browse_file(True), autoDefault=False)
        gui.rubber(box)
        box.layout().addWidget(self.report_button)
        self.report_button.setFixedWidth(170)

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

        if self.source == self.LOCAL_FILE and \
                        os.path.getsize(self.last_path()) > self.SIZE_LIMIT:
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
        self.source = self.URL
        self.load_data()

    def browse_file(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation data sets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Orange Data File', start_file, self.dlg_formats)
        if not filename:
            return
        self.loaded_file = filename
        self.add_path(filename)
        self.source = self.LOCAL_FILE
        self.load_data()

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        # We need to catch any exception type since anything can happen in
        # file readers
        # pylint: disable=broad-except
        self.editor_model.set_domain(None)
        self.apply_button.setEnabled(False)
        self.Warning.file_too_big.clear()

        error = None
        try:
            self.reader = self._get_reader()
            if self.reader is None:
                self.data = None
                self.send("Data", None)
                self.info.setText("No data.")
                self.sheet_box.hide()
                return
        except Exception as ex:
            error = ex

        if not error:
            self._update_sheet_combo()
            with catch_warnings(record=True) as warnings:
                try:
                    data = self.reader.read()
                except Exception as ex:
                    log.exception(ex)
                    error = ex
                self.warning(warnings[-1].message.args[0] if warnings else '')

        if error:
            self.data = None
            self.send("Data", None)
            self.info.setText("An error occurred:\n{}".format(error))
            self.editor_model.reset()
            self.sheet_box.hide()
            return

        self.info.setText(self._describe(data))

        add_origin(data, self.loaded_file or self.last_path())
        self.send("Data", data)
        self.editor_model.set_domain(data.domain)
        self.data = data

    def _get_reader(self):
        """

        Returns
        -------
        FileFormat
        """
        if self.source == self.LOCAL_FILE:
            reader = FileFormat.get_reader(self.last_path())
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
            text += "<br/>Classification; discrete class with {} values.".\
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
        attributes = []
        class_vars = []
        metas = []
        places = [attributes, class_vars, metas]
        X, y, m = [], [], []
        cols = [X, y, m]  # Xcols, Ycols, Mcols

        def is_missing(x):
            return str(x) in ("nan", "")

        for column, (name, tpe, place, vals, is_con), (orig_var, orig_plc) in \
            zip(count(), self.editor_model.variables,
                chain([(at, 0) for at in self.data.domain.attributes],
                      [(cl, 1) for cl in self.data.domain.class_vars],
                      [(mt, 2) for mt in self.data.domain.metas])):
            if place == 3:
                continue
            if orig_plc == 2:
                col_data = list(chain(*self.data[:, orig_var].metas))
            else:
                col_data = list(chain(*self.data[:, orig_var]))
            if name == orig_var.name and tpe == type(orig_var):
                var = orig_var
            elif tpe == DiscreteVariable:
                values = list(str(i) for i in set(col_data) if not is_missing(i))
                var = tpe(name, values)
                col_data = [np.nan if is_missing(x) else values.index(str(x))
                            for x in col_data]
            elif tpe == StringVariable and type(orig_var) == DiscreteVariable:
                var = tpe(name)
                col_data = [orig_var.repr_val(x) if not np.isnan(x) else ""
                            for x in col_data]
            else:
                var = tpe(name)
            places[place].append(var)
            cols[place].append(col_data)
        domain = Domain(attributes, class_vars, metas)
        X = np.array(X).T if len(X) else np.empty((len(self.data), 0))
        y = np.array(y).T if len(y) else None
        dtpe = object if any(isinstance(m, StringVariable)
                             for m in domain.metas) else float
        m = np.array(m, dtype=dtpe).T if len(m) else None
        table = Table.from_numpy(domain, X, y, m, self.data.W)
        self.send("Data", table)
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
                name = "~/" + \
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

import os
from warnings import catch_warnings
from xlrd import open_workbook
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy as Policy

from Orange.canvas.gui.utils import OSX_NSURL_toLocalFile
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin
from Orange.data.table import Table, get_sample_datasets_dir
from Orange.data.io import FileFormat, ExcelReader

# Backward compatibility: class RecentPath used to be defined in this module,
# and it is used in saved (pickled) settings. It must be imported into the
# module's namespace so that old saved settings still work
from Orange.widgets.utils.filedialogs import RecentPath


def add_origin(examples, filename):
    """Adds attribute with file location to each variable"""
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
        if role == QtCore.Qt.DisplayRole:
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


class OWFile(widget.OWWidget, RecentPathsWComboMixin):
    name = "File"
    id = "orange.widgets.data.file"
    description = "Read a data from an input file or network " \
                  "and send the data table to the output."
    icon = "icons/File.svg"
    priority = 10
    category = "Data"
    keywords = ["data", "file", "load", "read"]
    outputs = [widget.OutputSignal(
        "Data", Table,
        doc="Attribute-valued data set read from the input file.")]

    want_main_area = False
    resizing_enabled = False

    SEARCH_PATHS = [("sample-datasets", get_sample_datasets_dir())]

    LOCAL_FILE, URL = range(2)

    settingsHandler = XlsContextHandler()

    # Overload RecentPathsWidgetMixin.recent_paths to set defaults
    recent_paths = Setting([
        RecentPath("", "sample-datasets", "iris.tab"),
        RecentPath("", "sample-datasets", "titanic.tab"),
        RecentPath("", "sample-datasets", "housing.tab"),
    ])
    recent_urls = Setting([])
    source = Setting(LOCAL_FILE)
    xls_sheet = ContextSetting("")
    sheet_names = Setting({})
    url = Setting("")

    dlg_formats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""

        layout = QtGui.QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)
        vbox = gui.radioButtons(None, self, "source", box=True, addSpace=True,
                                callback=self.load_data, addToLayout=False)

        rb_button = gui.appendRadioButton(vbox, "File:", addToLayout=False)
        layout.addWidget(rb_button, 0, 0, QtCore.Qt.AlignVCenter)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        box.layout().addWidget(self.file_combo)
        layout.addWidget(box, 0, 1)

        file_button = gui.button(
            None, self, '...', callback=self.browse_file, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(
            QtGui.QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.reload, autoDefault=False)
        reload_button.setIcon(self.style().standardIcon(
            QtGui.QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 3)

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_combo = gui.comboBox(None, self, "xls_sheet",
                                        callback=self.load_data,
                                        sendSelectedValue=True)
        self.sheet_combo.setSizePolicy(
            Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_label = QtGui.QLabel()
        self.sheet_label.setText('Sheet')
        self.sheet_label.setSizePolicy(
            Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(
            self.sheet_label, QtCore.Qt.AlignLeft)
        self.sheet_box.layout().addWidget(
            self.sheet_combo, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.sheet_box, 2, 1)
        self.sheet_box.hide()

        rb_button = gui.appendRadioButton(vbox, "URL:", addToLayout=False)
        layout.addWidget(rb_button, 3, 0, QtCore.Qt.AlignVCenter)

        self.url_combo = url_combo = QtGui.QComboBox()
        url_model = NamedURLModel(self.sheet_names)
        url_model.wrap(self.recent_urls)
        url_combo.setModel(url_model)
        url_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        url_combo.setMaximumWidth(500)
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

        box = gui.hBox(self.controlArea)
        gui.button(
            box, self, "Browse documentation data sets",
            callback=lambda: self.browse_file(True), autoDefault=False)
        gui.rubber(box)
        box.layout().addWidget(self.report_button)
        self.report_button.setFixedWidth(170)

        # Set word wrap, so long warnings won't expand the widget
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(Policy.Ignored, Policy.MinimumExpanding)

        self.set_file_list()
        if self.last_path() is not None:
            self.fill_sheet_combo(self.last_path())
        # Must not call open_file from within __init__. open_file
        # explicitly re-enters the event loop (by a progress bar)
        QtCore.QTimer.singleShot(0, self.load_data)

        self.setAcceptDrops(True)

    def reload(self):
        if self.recent_paths:
            basename = self.file_combo.currentText()
            path = self.recent_paths[0]
            if basename in [path.relpath, path.basename]:
                self.source = self.LOCAL_FILE
                if self.is_multisheet_excel(path.abspath):
                    self.fill_sheet_combo(path.abspath)
                return self.load_data()
        self.select_file(len(self.recent_paths) + 1)

    def select_file(self, n):
        if n < len(self.recent_paths):
            super().select_file(n)
            self.fill_sheet_combo(self.last_path())
        # TODO: This is weird. Has it remained here from "Browse data sets"
        # or from when this combo was editable? A `n` this large can come from
        # `reload`, but ... how?!
        elif n:
            path = self.file_combo.currentText()
            if os.path.exists(path):
                self.add_path(path)
            else:
                self.info.setText('Data was not loaded:')
                self.warnings.setText("File {} does not exist".format(path))
                self.file_combo.removeItem(n)
                self.file_combo.lineEdit().setText(path)
                return

        if self.recent_paths:
            self.source = self.LOCAL_FILE
            self.load_data()
            self.set_file_list()

    def _url_set(self):
        self.source = self.URL
        self.load_data()

    def browse_file(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QtGui.QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation data sets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Orange Data File', start_file, self.dlg_formats)
        if not filename:
            return

        self.add_path(filename)
        self.source = self.LOCAL_FILE
        self.fill_sheet_combo(filename)
        self.load_data()

    def fill_sheet_combo(self, path):
        if os.path.exists(path) and self.is_multisheet_excel(path):
            self.closeContext()
            self.sheet_combo.clear()
            self.sheet_box.show()
            book = open_workbook(path)
            sheet_names = [str(book.sheet_by_index(i).name)
                           for i in range(book.nsheets)]
            self.sheet_combo.addItems(sheet_names)
            self.openContext(path, sheet_names)
        else:
            self.sheet_box.hide()

    @staticmethod
    def is_multisheet_excel(fn):
        return len(FileFormat.get_reader(fn).sheets) > 0

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        def load(method, fn):
            with catch_warnings(record=True) as warnings:
                data = method(fn)
                self.warning(
                    33, warnings[-1].message.args[0] if warnings else '')
            return data, fn

        def load_from_file():
            filename = self.last_path()
            if not filename:
                return None, ""

            if not os.path.exists(filename):
                filename = os.path.basename(filename)
                filename = FileFormat.locate(filename, "")
                self.information("Loading '{}' from the current directory."
                                 .format(filename))

            reader = FileFormat.get_reader(filename)
            if isinstance(reader, ExcelReader):
                reader.select_sheet(self.xls_sheet)
            try:
                return load(lambda x: reader.read(), filename)
            except Exception as exc:
                self.warnings.setText(str(exc))
                # Let us not remove from recent files: user may fix them
                raise

        def load_from_network():
            combo = self.url_combo
            model = combo.model()
            # combo.currentText does not work when the widget is initialized
            url = model.data(model.index(combo.currentIndex()),
                             QtCore.Qt.EditRole)
            if not url:
                return None, ""
            elif "://" not in url:
                url = "http://" + url
            try:
                data, url = load(Table.from_url, url)
            except:
                self.warnings.setText(
                    "URL '{}' does not contain valid data".format(url))
                # Don't remove from recent_urls:
                # resource may reappear, or the user mistyped it
                # and would like to retrieve it from history and fix it.
                raise
            combo.clearFocus()
            if "://docs.google.com/spreadsheets" in url:
                model.add_name(url, data.name)
                self.url = \
                    "{} from {}".format(data.name.replace("- Sheet1", ""), url)
                combo.lineEdit().setPlaceholderText(self.url)
                return data, data.name
            else:
                self.url = url
                return data, url

        self.warning()
        self.information()

        try:
            loader = [load_from_file, load_from_network][self.source]
            self.data, self.loaded_file = loader()
        except:
            self.info.setText("Data was not loaded:")
            self.data = None
            self.loaded_file = ""
            return
        else:
            self.warnings.setText("")

        data = self.data
        if data is None:
            self.send("Data", None)
            self.info.setText("No data loaded")
            return

        domain = data.domain
        text = "{} instance(s), {} feature(s), {} meta attribute(s)".format(
            len(data), len(domain.attributes), len(domain.metas))
        if domain.has_continuous_class:
            text += "\nRegression; numerical class."
        elif domain.has_discrete_class:
            text += "\nClassification; discrete class with {} values.".format(
                len(domain.class_var.values))
        elif data.domain.class_vars:
            text += "\nMulti-target; {} target variables.".format(
                len(data.domain.class_vars))
        else:
            text += "\nData has no target variable."
        if 'Timestamp' in data.domain:
            # Google Forms uses this header to timestamp responses
            text += '\n\nFirst entry: {}\nLast entry: {}'.format(
                data[0, 'Timestamp'], data[-1, 'Timestamp'])
        self.info.setText(text)

        add_origin(data, self.loaded_file)
        self.send("Data", data)

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
    a = QtGui.QApplication(sys.argv)
    ow = OWFile()
    ow.show()
    a.exec_()
    ow.saveSettings()

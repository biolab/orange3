import os
import sys
import urllib
from warnings import catch_warnings

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QCompleter, QLineEdit, QSizePolicy

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table, get_sample_datasets_dir
from Orange.data.io import FileFormat
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.widget import OutputSignal


def add_origin(examples, filename):
    """Adds attribute with file location to each variable"""
    vars = examples.domain.variables + examples.domain.metas
    strings = [var for var in vars if var.is_string]
    dir_name, basename = os.path.split(filename)
    for var in strings:
        if "type" in var.attributes and "origin" not in var.attributes:
            var.attributes["origin"] = dir_name


class RecentPath:
    abspath = ''
    prefix = None   #: Option[str]  # BASEDIR | SAMPLE-DATASETS | ...
    relpath = ''  #: Option[str]  # path relative to `prefix`
    title = ''    #: Option[str]  # title of filename (e.g. from URL)

    def __init__(self, abspath, prefix, relpath, title=''):
        if os.name == "nt":
            # always use a cross-platform pathname component separator
            abspath = abspath.replace(os.path.sep, "/")
            if relpath is not None:
                relpath = relpath.replace(os.path.sep, "/")
        self.abspath = abspath
        self.prefix = prefix
        self.relpath = relpath
        self.title = title

    def __eq__(self, other):
        return (self.abspath == other.abspath or
                (self.prefix is not None and self.relpath is not None and
                 self.prefix == other.prefix and
                 self.relpath == other.relpath))

    @staticmethod
    def create(path, searchpaths):
        """
        Create a RecentPath item inferring a suitable prefix name and relpath.

        Parameters
        ----------
        path : str
            File system path.
        searchpaths : List[Tuple[str, str]]
            A sequence of (NAME, prefix) pairs. The sequence is searched
            for a item such that prefix/relpath == abspath. The NAME is
            recorded in the `prefix` and relpath in `relpath`.
            (note: the first matching prefixed path is chosen).

        """
        def isprefixed(prefix, path):
            """
            Is `path` contained within the directory `prefix`.

            >>> isprefixed("/usr/local/", "/usr/local/shared")
            True
            """
            normalize = lambda path: os.path.normcase(os.path.normpath(path))
            prefix, path = normalize(prefix), normalize(path)
            if not prefix.endswith(os.path.sep):
                prefix = prefix + os.path.sep
            return os.path.commonprefix([prefix, path]) == prefix

        abspath = os.path.normpath(os.path.abspath(path))
        for prefix, base in searchpaths:
            if isprefixed(base, abspath):
                relpath = os.path.relpath(abspath, base)
                return RecentPath(abspath, prefix, relpath)

        return RecentPath(abspath, None, None)

    def search(self, searchpaths):
        """
        Return a file system path, substituting the variable paths if required

        If the self.abspath names an existing path it is returned. Else if
        the `self.prefix` and `self.relpath` are not `None` then the
        `searchpaths` sequence is searched for the matching prefix and
        if found and the {PATH}/self.relpath exists it is returned.

        If all fails return None.

        Parameters
        ----------
        searchpaths : List[Tuple[str, str]]
            A sequence of (NAME, prefixpath) pairs.

        """
        if os.path.exists(self.abspath):
            return os.path.normpath(self.abspath)

        for prefix, base in searchpaths:
            if self.prefix == prefix:
                path = os.path.join(base, self.relpath)
                if os.path.exists(path):
                    return os.path.normpath(path)
        else:
            return None

    def resolve(self, searchpaths):
        if self.prefix is None and os.path.exists(self.abspath):
            return self
        elif self.prefix is not None:
            for prefix, base in searchpaths:
                if self.prefix == prefix:
                    path = os.path.join(base, self.relpath)
                    if os.path.exists(path):
                        return RecentPath(
                            os.path.normpath(path), self.prefix, self.relpath)
        return None

    @property
    def value(self):
        return os.path.basename(self.abspath)

    @property
    def icon(self):
        provider = QtGui.QFileIconProvider()
        return provider.icon(QtGui.QFileIconProvider.Drive)

    @property
    def dirname(self):
        return os.path.dirname(self.abspath)

    def __repr__(self):
        return ("{0.__class__.__name__}(abspath={0.abspath!r}, "
                "prefix={0.prefix!r}, relpath={0.relpath!r}, "
                "title={0.title!r})").format(self)

    __str__ = __repr__


class OWFile(widget.OWWidget):
    name = "File"
    id = "orange.widgets.data.file"
    description = "Read a data from an input file or network" \
                  "and send the data table to the output."
    icon = "icons/File.svg"
    priority = 10
    category = "Data"
    keywords = ["data", "file", "load", "read"]
    outputs = [OutputSignal(
        "Data", Table,
        doc="Attribute-valued data set read from the input file.")]

    want_main_area = False
    resizing_enabled = False

    LOCAL_FILE, URL = range(2)

    #: List[RecentPath]
    recent_paths = Setting([])
    recent_urls = Setting([])
    source = Setting(LOCAL_FILE)
    url = Setting("")

    dlgFormats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    def __init__(self):
        super().__init__()
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.recent_urls = ["asdf", "qwer"]  # TODO Remove!
        self._relocate_recent_files()

        vbox = gui.radioButtons(
            self.controlArea, self, "source", box="Source", addSpace=True)
        box = gui.widgetBox(vbox, orientation="horizontal")
        gui.appendRadioButton(vbox, "File", insertInto=box)
        self.file_combo = QtGui.QComboBox(
            box, sizeAdjustPolicy=QtGui.QComboBox.AdjustToContents)
        self.file_combo.setMinimumWidth(250)
        box.layout().addWidget(self.file_combo)
        self.file_combo.activated[int].connect(self.select_file)
        button = gui.button(
            box, self, '...', callback=self.browse_file, autoDefault=False)
        button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        button = gui.button(
            box, self, "Reload", callback=self.reload, autoDefault=False)
        button.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


        box = gui.widgetBox(vbox, orientation="horizontal")
        gui.appendRadioButton(vbox, "URL", insertInto=box)
        self.le_url = le_url = QLineEdit(self.url)
        l, t, r, b = le_url.getTextMargins()
        le_url.setTextMargins(l + 5, t, r, b)
        le_url.returnPressed.connect(self.read_url)
        box.layout().addWidget(le_url)

        self.completer_model = PyListModel()
        self.completer_model.wrap(self.recent_urls)
        completer = QCompleter()
        completer.setModel(self.completer_model)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionRole(Qt.DisplayRole)
        le_url.setCompleter(completer)

        box = gui.widgetBox(self.controlArea, "Info")
        self.info = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, ' ')

        box = gui.widgetBox(self.controlArea, orientation="horizontal")
        gui.button(box, self, "Browse demo files",
                   callback=lambda: self.browse_file(True), autoDefault=False)
        gui.rubber(box)
        box.layout().addWidget(self.report_button)
        self.report_button.setMaximumWidth(250)


    #Set word wrap, so long warnings won't expand the widget
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.MinimumExpanding)

        self.set_file_list()
        # Must not call open_file from within __init__. open_file
        # explicitly re-enters the event loop (by a progress bar)
        QtCore.QTimer.singleShot(0, [self.reload, self.read_url][self.source])

    def _url_focused(self):
        self.source = self.URL

    def _relocate_recent_files(self):
        paths = [("sample-datasets", get_sample_datasets_dir())]
        basedir = self.workflowEnv().get("basedir", None)
        if basedir is not None:
            paths.append(("basedir", basedir))

        rec = []
        for recent in self.recent_paths:
            resolved = recent.resolve(paths)
            if resolved is not None:
                rec.append(RecentPath.create(resolved.abspath, paths))
            elif recent.search(paths) is not None:
                rec.append(RecentPath.create(recent.search(paths), paths))
        self.recent_paths = rec

    def set_file_list(self):
        self.file_combo.clear()

        if not self.recent_paths:
            self.file_combo.addItem("(none)")
            self.file_combo.model().item(0).setEnabled(False)
        else:
            for i, recent in enumerate(self.recent_paths):
                self.file_combo.addItem(recent.value)
                self.file_combo.model().item(i).setToolTip(recent.abspath)
        self.file_combo.addItem("Browse documentation data sets...")

    def reload(self):
        self.source = self.LOCAL_FILE
        if self.recent_paths:
            basename = self.file_combo.currentText()
            if (basename == self.recent_paths[0].relpath or
                basename == self.recent_paths[0].value):
                return self.load_data(self.recent_paths[0].abspath)
        self.select_file(len(self.recent_paths) + 1)

    def select_file(self, n):
        self.source = self.LOCAL_FILE
        if n < len(self.recent_paths):
            recent = self.recent_paths[n]
            del self.recent_paths[n]
            self.recent_paths.insert(0, recent)
        elif n:
            path = self.file_combo.currentText()
            if path == "Browse documentation data sets...":
                self.browse_file(True)
            elif os.path.exists(path):
                self._add_path(path)
            else:
                self.info.setText('Data was not loaded.')
                self.warnings.setText("File {} does not exist".format(path))
                self.file_combo.removeItem(n)
                self.file_combo.lineEdit().setText(path)
                return

        if len(self.recent_paths) > 0:
            self.load_data(self.recent_paths[0].abspath)
            self.set_file_list()

    def read_url(self):
        self.source = self.URL
        self.url = url = self.le_url.text()
        if not url:
            return
        try:
            self.completer_model.remove(url or self.url)
        except ValueError:
            pass
        self.completer_model.insert(0, url)
        self.load_data(url)

    def browse_file(self, in_demos=False):
        if in_demos:
            try:
                start_file = get_sample_datasets_dir()
            except AttributeError:
                start_file = ""
            if not start_file or not os.path.exists(start_file):
                widgets_dir = os.path.dirname(gui.__file__)
                orange_dir = os.path.dirname(widgets_dir)
                start_file = os.path.join(orange_dir, "doc", "datasets")
            if not start_file or not os.path.exists(start_file):
                d = os.getcwd()
                if os.path.basename(d) == "canvas":
                    d = os.path.dirname(d)
                start_file = os.path.join(os.path.dirname(d), "doc", "datasets")
            if not os.path.exists(start_file):
                QtGui.QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with example data sets")
                return
        else:
            if self.recent_paths and self.recent_paths[0].prefix != 'url-datasets':
                start_file = self.recent_paths[0].abspath
            else:
                start_file = os.path.expanduser("~/")

        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Orange Data File', start_file, self.dlgFormats)
        if not filename:
            return

        self._add_path(filename)
        self.set_file_list()
        self.load_data(self.recent_paths[0].abspath)

    def _add_path(self, filename):
        searchpaths = [("sample-datasets", get_sample_datasets_dir())]
        basedir = self.workflowEnv().get("basedir", None)
        if basedir is not None:
            searchpaths.append(("basedir", basedir))

        recent = RecentPath.create(filename, searchpaths)

        if recent in self.recent_paths:
            self.recent_paths.remove(recent)

        self.recent_paths.insert(0, recent)

    # Open a file, create data from it and send it over the data channel
    def load_data(self, fn):
        self.warning()
        self.information()
        fn_original = fn
        if self.source == self.LOCAL_FILE:
            if not os.path.exists(fn):
                dir_name, basename = os.path.split(fn)
                if os.path.exists(os.path.join(".", basename)):
                    fn = os.path.join(".", basename)
                    self.information("Loading '{}' from the current directory."
                                     .format(basename))
        else:
            if not fn:
                fn = "(none)"
            elif "://" not in fn:
                fn = "http://" + fn

        if fn == "(none)":
            self.send("Data", None)
            self.info.setText("No data loaded")
            self.warnings.setText("")
            return

        self.loaded_file = ""

        data = None
        progress = gui.ProgressBar(self, 3)
        progress.advance()
        try:
            with catch_warnings(record=True) as warnings:
                data = Table(fn)
            self.warning(33, warnings[-1].message.args[0] if warnings else '')
            self.loaded_file = fn
        except Exception as exc:
            self.info.setText('Data was not loaded:')
            if self.source == self.LOCAL_FILE:
                self.warnings.setText(str(exc))
                ind = self.file_combo.currentIndex()
                self.file_combo.removeItem(ind)
                if ind < len(self.recent_paths) and \
                        self.recent_paths[ind].abspath == fn_original:
                    del self.recent_paths[ind]
            else:
                self.warning.setText(
                    "URL '{}' does not contain valid data".format(fn))
                # Don't remove; resource may reappear, or the user mistyped it
                # and would like to retrieve it from history and fix it.
        finally:
            progress.finish()

        if data is None:
            self.data = None
        else:
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
            self.warnings.setText("")

            add_origin(data, fn)

            self.data = data
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
            self.report_items("File", [("File name", name),
                                       ("Format", get_ext_name(name))])
        else:
            self.report_items("Data", [("URL", self.url),
                                       ("Format", get_ext_name(self.url))])

        self.report_data("Data", self.data)

    def workflowEnvChanged(self, key, value, oldvalue):
        if key == "basedir":
            self._relocate_recent_files()
            self.set_file_list()


if __name__ == "__main__":
    a = QtGui.QApplication(sys.argv)
    ow = OWFile()
    ow.show()
    a.exec_()
    ow.saveSettings()

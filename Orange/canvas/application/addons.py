import sys
import os
import errno
import shlex
import subprocess
import itertools
import concurrent.futures

from collections import namedtuple, deque
from xml.sax.saxutils import escape
from distutils import version

import pkg_resources

try:
    import docutils.core
except ImportError:
    docutils = None

from PyQt4.QtGui import (
    QWidget, QDialog, QLabel, QLineEdit, QTreeView, QHeaderView,
    QTextBrowser, QTextOption, QDialogButtonBox, QProgressDialog,
    QVBoxLayout, QPalette, QStandardItemModel, QStandardItem,
    QSortFilterProxyModel, QItemSelectionModel, QStyle, QStyledItemDelegate,
    QStyleOptionViewItemV4, QApplication
)

from PyQt4.QtCore import (
    Qt, QObject, QMetaObject, QEvent, QSize, QTimer, QThread, Q_ARG
)
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from ..gui.utils import message_warning, message_information, \
                        message_critical as message_error
from ..help.manager import get_dist_meta, trim

OFFICIAL_ADDONS = [
    "Orange-Bioinformatics",
    "Orange3-DataFusion",
    "Orange3-Prototypes",
    "Orange3-Text",
    "Orange3-Network",
]

Installable = namedtuple(
    "Installable",
    ["name",
     "version",
     "summary",
     "description",
     "package_url",
     "release_urls"]
)

ReleaseUrl = namedtuple(
    "ReleaseUrl",
    ["filename",
     "url",
     "size",
     "python_version",
     "package_type"
     ]
)

Available = namedtuple(
    "Available",
    ["installable"]
)

Installed = namedtuple(
    "Installed",
    ["installable",
     "local"]
)


def is_updatable(item):
    if isinstance(item, Available):
        return False
    elif item.installable is None:
        return False
    else:
        inst, dist = item
        try:
            v1 = version.StrictVersion(dist.version)
            v2 = version.StrictVersion(inst.version)
        except ValueError:
            pass
        else:
            return v1 < v2

        return (version.LooseVersion(dist.version) <
                version.LooseVersion(inst.version))


class TristateCheckItemDelegate(QStyledItemDelegate):
    """
    A QStyledItemDelegate which properly toggles Qt.ItemIsTristate check
    state transitions on user interaction.
    """
    def editorEvent(self, event, model, option, index):
        flags = model.flags(index)
        if not flags & Qt.ItemIsUserCheckable or \
                not option.state & QStyle.State_Enabled or \
                not flags & Qt.ItemIsEnabled:
            return False

        checkstate = model.data(index, Qt.CheckStateRole)
        if checkstate is None:
            return False

        widget = option.widget
        style = widget.style() if widget else QApplication.style()
        if event.type() in {QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                            QEvent.MouseButtonDblClick}:
            pos = event.pos()
            opt = QStyleOptionViewItemV4(option)
            self.initStyleOption(opt, index)
            rect = style.subElementRect(
                QStyle.SE_ItemViewItemCheckIndicator, opt, widget)

            if event.button() != Qt.LeftButton or not rect.contains(pos):
                return False

            if event.type() in {QEvent.MouseButtonPress,
                                QEvent.MouseButtonDblClick}:
                return True

        elif event.type() == QEvent.KeyPress:
            if event.key() != Qt.Key_Space and event.key() != Qt.Key_Select:
                return False
        else:
            return False

        if model.flags(index) & Qt.ItemIsTristate:
            checkstate = (checkstate + 1) % 3
        else:
            checkstate = \
                Qt.Unchecked if checkstate == Qt.Checked else Qt.Checked

        return model.setData(index, checkstate, Qt.CheckStateRole)


class AddonManagerWidget(QWidget):

    statechanged = Signal()

    def __init__(self, parent=None, **kwargs):
        super(AddonManagerWidget, self).__init__(parent, **kwargs)

        self.setLayout(QVBoxLayout())

        self.__header = QLabel(
            wordWrap=True,
            textFormat=Qt.RichText
        )
        self.__search = QLineEdit(
            placeholderText=self.tr("Filter")
        )

        self.layout().addWidget(self.__search)

        self.__view = view = QTreeView(
            rootIsDecorated=False,
            editTriggers=QTreeView.NoEditTriggers,
            selectionMode=QTreeView.SingleSelection,
            alternatingRowColors=True
        )
        self.__view.setItemDelegateForColumn(0, TristateCheckItemDelegate())
        self.layout().addWidget(view)

        self.__model = model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["", "Name", "Version", "Action"])
        model.dataChanged.connect(self.__data_changed)
        proxy = QSortFilterProxyModel(
            filterKeyColumn=1,
            filterCaseSensitivity=Qt.CaseInsensitive
        )
        proxy.setSourceModel(model)
        self.__search.textChanged.connect(proxy.setFilterFixedString)

        view.setModel(proxy)
        view.selectionModel().selectionChanged.connect(
            self.__update_details
        )
        header = self.__view.header()
        header.setResizeMode(0, QHeaderView.Fixed)
        header.setResizeMode(2, QHeaderView.ResizeToContents)

        self.__details = QTextBrowser(
            frameShape=QTextBrowser.NoFrame,
            readOnly=True,
            lineWrapMode=QTextBrowser.WidgetWidth,
            openExternalLinks=True,
        )

        self.__details.setWordWrapMode(QTextOption.WordWrap)
        palette = QPalette(self.palette())
        palette.setColor(QPalette.Base, Qt.transparent)
        self.__details.setPalette(palette)
        self.layout().addWidget(self.__details)

    def set_items(self, items):
        self.__items = items
        model = self.__model
        model.clear()
        model.setHorizontalHeaderLabels(["", "Name", "Version", "Action"])

        for item in items:
            if isinstance(item, Installed):
                installed = True
                ins, dist = item
                name = dist.project_name
                summary = get_dist_meta(dist).get("Summary", "")
                version = ins.version if ins is not None else dist.version
            else:
                installed = False
                (ins,) = item
                dist = None
                name = ins.name
                summary = ins.summary
                version = ins.version

            updatable = is_updatable(item)

            item1 = QStandardItem()
            item1.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                           Qt.ItemIsUserCheckable |
                           (Qt.ItemIsTristate if updatable else 0))

            if installed and updatable:
                item1.setCheckState(Qt.PartiallyChecked)
            elif installed:
                item1.setCheckState(Qt.Checked)
            else:
                item1.setCheckState(Qt.Unchecked)

            item2 = QStandardItem(name)

            item2.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item2.setToolTip(summary)
            item2.setData(item, Qt.UserRole)

            if updatable:
                version = "{} < {}".format(dist.version, ins.version)

            item3 = QStandardItem(version)
            item3.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            item4 = QStandardItem()
            item4.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            model.appendRow([item1, item2, item3, item4])

        self.__view.resizeColumnToContents(0)
        self.__view.setColumnWidth(
            1, max(150, self.__view.sizeHintForColumn(1)))
        self.__view.setColumnWidth(
            2, max(150, self.__view.sizeHintForColumn(2)))

        if self.__items:
            self.__view.selectionModel().select(
                self.__view.model().index(0, 0),
                QItemSelectionModel.Select | QItemSelectionModel.Rows
            )

    def item_state(self):
        steps = []
        for i, item in enumerate(self.__items):
            modelitem = self.__model.item(i, 0)
            state = modelitem.checkState()
            if modelitem.flags() & Qt.ItemIsTristate and state == Qt.Checked:
                steps.append((Upgrade, item))
            elif isinstance(item, Available) and state == Qt.Checked:
                steps.append((Install, item))
            elif isinstance(item, Installed) and state == Qt.Unchecked:
                steps.append((Uninstall, item))

        return steps

    def __selected_row(self):
        indices = self.__view.selectedIndexes()
        if indices:
            proxy = self.__view.model()
            indices = [proxy.mapToSource(index) for index in indices]
            return indices[0].row()
        else:
            return -1

    def __data_changed(self, topleft, bottomright):
        rows = range(topleft.row(), bottomright.row() + 1)
        proxy = self.__view.model()
        map_to_source = proxy.mapToSource

        for i in rows:
            sourceind = map_to_source(proxy.index(i, 0))
            modelitem = self.__model.itemFromIndex(sourceind)
            actionitem = self.__model.item(modelitem.row(), 3)
            item = self.__items[modelitem.row()]

            state = modelitem.checkState()
            flags = modelitem.flags()

            if flags & Qt.ItemIsTristate and state == Qt.Checked:
                actionitem.setText("Update")
            elif isinstance(item, Available) and state == Qt.Checked:
                actionitem.setText("Install")
            elif isinstance(item, Installed) and state == Qt.Unchecked:
                actionitem.setText("Uninstall")
            else:
                actionitem.setText("")
        self.statechanged.emit()

    def __update_details(self):
        index = self.__selected_row()
        if index == -1:
            self.__details.setText("")
        else:
            item = self.__model.item(index, 1)
            item = item.data(Qt.UserRole)
            assert isinstance(item, (Installed, Available))
#             if isinstance(item, Available):
#                 self.__installed_label.setText("")
#                 self.__available_label.setText(str(item.available.version))
#             elif item.installable is not None:
#                 self.__installed_label.setText(str(item.local.version))
#                 self.__available_label.setText(str(item.available.version))
#             else:
#                 self.__installed_label.setText(str(item.local.version))
#                 self.__available_label.setText("")

            text = self._detailed_text(item)
            self.__details.setText(text)

    def _detailed_text(self, item):
        if isinstance(item, Installed):
            remote, dist = item
            if remote is None:
                description = get_dist_meta(dist).get("Description")
                description = description
            else:
                description = remote.description
        else:
            description = item[0].description

        if docutils is not None:
            try:
                html = docutils.core.publish_string(
                    trim(description),
                    writer_name="html",
                    settings_overrides={
                        "output-encoding": "utf-8",
#                         "embed-stylesheet": False,
#                         "stylesheet": [],
#                         "stylesheet_path": []
                    }
                ).decode("utf-8")

            except docutils.utils.SystemMessage:
                html = "<pre>{}<pre>".format(escape(description))
            except Exception:
                html = "<pre>{}<pre>".format(escape(description))
        else:
            html = "<pre>{}<pre>".format(escape(description))
        return html

    def sizeHint(self):
        return QSize(480, 420)


def method_queued(method, sig, conntype=Qt.QueuedConnection):
    name = method.__name__
    obj = method.__self__
    assert isinstance(obj, QObject)

    def call(*args):
        args = [Q_ARG(atype, arg) for atype, arg in zip(sig, args)]
        return QMetaObject.invokeMethod(obj, name, conntype, *args)

    return call


class AddonManagerDialog(QDialog):
    _packages = None

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.addonwidget = AddonManagerWidget()
        self.layout().addWidget(self.addonwidget)
        buttons = QDialogButtonBox(
            orientation=Qt.Horizontal,
            standardButtons=QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.__accepted)
        buttons.rejected.connect(self.reject)

        self.layout().addWidget(buttons)

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if AddonManagerDialog._packages is None:
            self._f_pypi_addons = self._executor.submit(list_pypi_addons)
        else:
            self._f_pypi_addons = concurrent.futures.Future()
            self._f_pypi_addons.set_result(AddonManagerDialog._packages)

        self._f_pypi_addons.add_done_callback(
            method_queued(self._set_packages, (object,))
        )

        self.__progress = QProgressDialog(
            self, Qt.Sheet,
            minimum=0, maximum=0,
            labelText=self.tr("Retrieving package list"),
            sizeGripEnabled=False,
            windowTitle="Progress"
        )

        self.__progress.rejected.connect(self.reject)
        self.__thread = None
        self.__installer = None

    @Slot(object)
    def _set_packages(self, f):
        if self.__progress.isVisible():
            self.__progress.close()

        try:
            packages = f.result()
        except (IOError, OSError) as err:
            message_warning(
                "Could not retrieve package list",
                title="Error",
                informative_text=str(err),
                parent=self
            )
            packages = []
        except Exception:
            raise
        else:
            AddonManagerDialog._packages = packages

        installed = list_installed_addons()
        dists = {dist.project_name: dist for dist in installed}
        packages = {pkg.name: pkg for pkg in packages}

        # For every pypi available distribution not listed by
        # list_installed_addons, check if it is actually already
        # installed.
        ws = pkg_resources.WorkingSet()
        for pkg_name in set(packages.keys()).difference(set(dists.keys())):
            try:
                d = ws.find(pkg_resources.Requirement.parse(pkg_name))
            except pkg_resources.VersionConflict:
                pass
            except ValueError:
                # Requirements.parse error ?
                pass
            else:
                if d is not None:
                    dists[d.project_name] = d

        project_names = unique(
            itertools.chain(packages.keys(), dists.keys())
        )

        items = []
        for name in project_names:
            if name in dists and name in packages:
                item = Installed(packages[name], dists[name])
            elif name in dists:
                item = Installed(None, dists[name])
            elif name in packages:
                item = Available(packages[name])
            else:
                assert False
            items.append(item)

        self.addonwidget.set_items(items)

    def showEvent(self, event):
        super().showEvent(event)

        if not self._f_pypi_addons.done():
            QTimer.singleShot(0, self.__progress.show)

    def done(self, retcode):
        super().done(retcode)
        self._f_pypi_addons.cancel()
        self._executor.shutdown(wait=False)
        if self.__thread is not None:
            self.__thread.quit()
            self.__thread.wait(1000)

    def closeEvent(self, event):
        super().closeEvent(event)
        self._f_pypi_addons.cancel()
        self._executor.shutdown(wait=False)

        if self.__thread is not None:
            self.__thread.quit()
            self.__thread.wait(1000)

    def __accepted(self):
        steps = self.addonwidget.item_state()

        if steps:
            # Move all uninstall steps to the front
            steps = sorted(
                steps, key=lambda step: 0 if step[0] == Uninstall else 1
            )
            self.__installer = Installer(steps=steps)
            self.__thread = QThread(self)
            self.__thread.start()

            self.__installer.moveToThread(self.__thread)
            self.__installer.finished.connect(self.__on_installer_finished)
            self.__installer.error.connect(self.__on_installer_error)
            self.__installer.installStatusChanged.connect(
                self.__progress.setLabelText)

            self.__progress.show()
            self.__progress.setLabelText("Installing")

            self.__installer.start()

        else:
            self.accept()

    def __on_installer_error(self, command, pkg, retcode, output):
        message_error(
            "An error occurred while running a subprocess", title="Error",
            informative_text="{} exited with non zero status.".format(command),
            details="".join(output),
            parent=self
        )
        self.reject()

    def __on_installer_finished(self):
        message_information(
            "Please restart the application for changes to take effect.",
            parent=self)
        self.accept()


def list_pypi_addons():
    """
    List add-ons available on pypi.
    """
    from ..config import ADDON_PYPI_SEARCH_SPEC
    import xmlrpc.client
    pypi = xmlrpc.client.ServerProxy("http://pypi.python.org/pypi")
    addons = pypi.search(ADDON_PYPI_SEARCH_SPEC)

    for addon in OFFICIAL_ADDONS:
        if not any(a for a in addons if a['name'] == addon):
            versions = pypi.package_releases(addon)
            if versions:
                addons.append({"name": addon, "version": max(versions)})

    multicall = xmlrpc.client.MultiCall(pypi)
    for addon in addons:
        name, version = addon["name"], addon["version"]
        multicall.release_data(name, version)
        multicall.release_urls(name, version)

    results = list(multicall())
    release_data = results[::2]
    release_urls = results[1::2]
    packages = []

    for release, urls in zip(release_data, release_urls):
        urls = [ReleaseUrl(url["filename"], url["url"],
                           url["size"], url["python_version"],
                           url["packagetype"])
                for url in urls]
        packages.append(
            Installable(release["name"], release["version"],
                        release["summary"], release["description"],
                        release["package_url"],
                        urls)
        )

    return packages


def list_installed_addons():
    from ..config import ADDON_ENTRY
    workingset = pkg_resources.WorkingSet(sys.path)
    return [ep.dist for ep in
            workingset.iter_entry_points(ADDON_ENTRY)]


def unique(iterable):
    seen = set()

    def observed(el):
        observed = el in seen
        seen.add(el)
        return observed

    return (el for el in iterable if not observed(el))


Install, Upgrade, Uninstall = 1, 2, 3


class Installer(QObject):
    installStatusChanged = Signal(str)
    started = Signal()
    finished = Signal()
    error = Signal(str, object, int, list)

    def __init__(self, parent=None, steps=[]):
        QObject.__init__(self, parent)
        self.__interupt = False
        self.__queue = deque(steps)

    def start(self):
        QTimer.singleShot(0, self._next)

    def interupt(self):
        self.__interupt = True

    def setStatusMessage(self, message):
        self.__statusMessage = message
        self.installStatusChanged.emit(message)

    @Slot()
    def _next(self):
        def fmt_cmd(cmd):
            return "python " + (" ".join(map(shlex.quote, cmd)))

        command, pkg = self.__queue.popleft()
        if command == Install:
            inst = pkg.installable
            self.setStatusMessage("Installing {}".format(inst.name))
            links = []

            cmd = ["-m", "pip", "install"] + links + [inst.name]
            process = python_process(cmd, bufsize=-1, universal_newlines=True)
            retcode, output = self.__subprocessrun(process)

            if retcode != 0:
                self.error.emit(fmt_cmd(cmd), pkg, retcode, output)
                return

        elif command == Upgrade:
            inst = pkg.installable
            self.setStatusMessage("Upgrading {}".format(inst.name))

            cmd = ["-m", "pip", "install", "--upgrade", "--no-deps", inst.name]
            process = python_process(cmd, bufsize=-1, universal_newlines=True)
            retcode, output = self.__subprocessrun(process)

            if retcode != 0:
                self.error.emit(fmt_cmd(cmd), pkg, retcode, output)
                return

            cmd = ["-m", "pip", "install", inst.name]
            process = python_process(cmd, bufsize=-1, universal_newlines=True)
            retcode, output = self.__subprocessrun(process)

            if retcode != 0:
                self.error.emit(fmt_cmd(cmd), pkg, retcode, output)
                return

        elif command == Uninstall:
            dist = pkg.local
            self.setStatusMessage("Uninstalling {}".format(dist.project_name))

            cmd = ["-m", "pip", "uninstall", "--yes", dist.project_name]
            process = python_process(cmd, bufsize=-1, universal_newlines=True)
            retcode, output = self.__subprocessrun(process)

            if retcode != 0:
                self.error.emit(fmt_cmd(cmd), pkg, retcode, output)
                return

        if self.__queue:
            QTimer.singleShot(0, self._next)
        else:
            self.finished.emit()

    def __subprocessrun(self, process):
        output = []
        while process.poll() is None:
            try:
                line = process.stdout.readline()
            except IOError as ex:
                if ex.errno != errno.EINTR:
                    raise
            else:
                output.append(line)
                print(line, end="")
        # Read remaining output if any
        line = process.stdout.read()
        if line:
            output.append(line)
            print(line, end="")

        return process.returncode, output


def pip_install(args, **kwargs):
    return python_process(["-m", "pip", "install"] + args, **kwargs)


def pip_uninstall(args, **kwargs):
    return python_process(["-m", "pip", "uninstall"] + args, **kwargs)


def python_process(args, script_name=None, cwd=None, env=None, **kwargs):
    """
    Run a `sys.executable` in a subprocess with `args`.
    """
    executable = sys.executable
    if os.name == "nt" and os.path.basename(executable) == "pythonw.exe":
        # Don't run the script with a 'gui' (detached) process.
        dirname = os.path.dirname(executable)
        executable = os.path.join(dirname, "python.exe")
        # by default a new console window would show up when executing the
        # script
        startupinfo = subprocess.STARTUPINFO()
        if hasattr(subprocess, "STARTF_USESHOWWINDOW"):
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        else:
            # This flag was missing in inital releases of 2.7
            startupinfo.dwFlags |= subprocess._subprocess.STARTF_USESHOWWINDOW

        kwargs["startupinfo"] = startupinfo

    if script_name is not None:
        script = script_name
    else:
        script = executable

    process = subprocess.Popen(
        [script] + args,
        executable=executable,
        cwd=cwd,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        **kwargs
    )

    return process

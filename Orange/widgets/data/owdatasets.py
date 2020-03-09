import logging
import numbers
import os
import traceback

from xml.sax.saxutils import escape
from concurrent.futures import ThreadPoolExecutor, Future

from types import SimpleNamespace
from typing import Optional, Dict, Tuple, List
from collections import namedtuple

from AnyQt.QtWidgets import (
    QLabel, QLineEdit, QTextBrowser, QSplitter, QTreeView,
    QStyleOptionViewItem, QStyledItemDelegate, QStyle, QApplication
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QBrush, QColor
from AnyQt.QtCore import (
    Qt, QSize, QObject, QThread, QModelIndex, QSortFilterProxyModel,
    QItemSelectionModel,
    pyqtSlot as Slot, pyqtSignal as Signal
)

from serverfiles import LocalFiles, ServerFiles, sizeformat

import Orange.data
from Orange.misc.environ import data_dir
from Orange.widgets import settings, gui
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import OWWidget, Msg


log = logging.getLogger(__name__)


def ensure_local(index_url, file_path, local_cache_path,
                 force=False, progress_advance=None):
    localfiles = LocalFiles(local_cache_path,
                            serverfiles=ServerFiles(server=index_url))
    if force:
        localfiles.download(*file_path, callback=progress_advance)
    return localfiles.localpath_download(*file_path, callback=progress_advance)


def format_exception(error):
    # type: (BaseException) -> str
    return "\n".join(traceback.format_exception_only(type(error), error))


class UniformHeightDelegate(QStyledItemDelegate):
    """
    Item delegate that always includes the icon size in the size hint.
    """
    def sizeHint(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> QSize
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        opt.features |= QStyleOptionViewItem.HasDecoration
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        sh = style.sizeFromContents(
            QStyle.CT_ItemViewItem, opt, QSize(), widget)
        return sh


class SizeDelegate(UniformHeightDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        value = index.data(Qt.DisplayRole)
        if isinstance(value, numbers.Integral):
            option.text = sizeformat(int(value))
            option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter


class NumericalDelegate(UniformHeightDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        data = index.data(Qt.DisplayRole)
        align = index.data(Qt.TextAlignmentRole)
        if align is None and isinstance(data, numbers.Number):
            # Right align if the model does not specify otherwise
            option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter


class UniformHeightIndicatorDelegate(
        UniformHeightDelegate, gui.IndicatorItemDelegate):
    pass


class Namespace(SimpleNamespace):
    def __init__(self, **kwargs):
        self.file_path = None
        self.prefix = None
        self.filename = None
        self.islocal = None
        self.outdated = None

        # tags from JSON info file
        self.title = None
        self.description = None
        self.instances = None
        self.variables = None
        self.target = None
        self.size = None
        self.source = None
        self.year = None
        self.references = []
        self.seealso = []
        self.tags = []

        super(Namespace, self).__init__(**kwargs)

        # if title missing, use filename
        if not self.title and self.filename:
            self.title = self.filename


class TreeViewWithReturn(QTreeView):
    returnPressed = Signal()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Return:
            self.returnPressed.emit()
        else:
            super().keyPressEvent(e)


class OWDataSets(OWWidget):
    name = "Datasets"
    description = "Load a dataset from an online repository"
    icon = "icons/DataSets.svg"
    priority = 20
    replaces = ["orangecontrib.prototypes.widgets.owdatasets.OWDataSets"]
    keywords = ["online"]

    want_control_area = False

    # The following constants can be overridden in a subclass
    # to reuse this widget for a different repository
    # Take care when refactoring! (used in e.g. single-cell)
    INDEX_URL = "https://datasets.biolab.si/"
    DATASET_DIR = "datasets"

    # override HEADER_SCHEMA to define new columns
    # if schema is changed override methods: self.assign_delegates and
    # self.create_model
    HEADER_SCHEMA = [
        ['islocal', {'label': ''}],
        ['title', {'label': 'Title'}],
        ['size', {'label': 'Size'}],
        ['instances', {'label': 'Instances'}],
        ['variables', {'label': 'Variables'}],
        ['target', {'label': 'Target'}],
        ['tags', {'label': 'Tags'}]
    ]  # type: List[str, dict]

    IndicatorBrushes = (QBrush(Qt.darkGray), QBrush(QColor(0, 192, 0)))

    class Error(OWWidget.Error):
        no_remote_datasets = Msg("Could not fetch dataset list")

    class Warning(OWWidget.Warning):
        only_local_datasets = Msg("Could not fetch datasets list, only local "
                                  "cached datasets are shown")

    class Outputs:
        data = Output("Data", Orange.data.Table)

    #: Selected dataset id
    selected_id = settings.Setting(None)   # type: Optional[str]

    #: main area splitter state
    splitter_state = settings.Setting(b'')  # type: bytes
    header_state = settings.Setting(b'')    # type: bytes

    def __init__(self):
        super().__init__()
        self.allinfo_local = {}
        self.allinfo_remote = {}

        self.local_cache_path = os.path.join(data_dir(), self.DATASET_DIR)
        # current_output does not equal selected_id when, for instance, the
        # data is still downloading
        self.current_output = None

        self._header_labels = [
            header['label'] for _, header in self.HEADER_SCHEMA]
        self._header_index = namedtuple(
            '_header_index', [info_tag for info_tag, _ in self.HEADER_SCHEMA])
        self.Header = self._header_index(
            *[index for index, _ in enumerate(self._header_labels)])

        self.__awaiting_state = None  # type: Optional[_FetchState]

        self.filterLineEdit = QLineEdit(
            textChanged=self.filter, placeholderText="Search for data set ..."
        )
        self.mainArea.layout().addWidget(self.filterLineEdit)

        self.splitter = QSplitter(orientation=Qt.Vertical)

        self.view = TreeViewWithReturn(
            sortingEnabled=True,
            selectionMode=QTreeView.SingleSelection,
            alternatingRowColors=True,
            rootIsDecorated=False,
            editTriggers=QTreeView.NoEditTriggers,
            uniformRowHeights=True,
            toolTip="Press Return or double-click to send"
        )
        # the method doesn't exists yet, pylint: disable=unnecessary-lambda
        self.view.doubleClicked.connect(self.commit)
        self.view.returnPressed.connect(self.commit)
        box = gui.widgetBox(self.splitter, "Description", addToLayout=False)
        self.descriptionlabel = QLabel(
            wordWrap=True,
            textFormat=Qt.RichText,
        )
        self.descriptionlabel = QTextBrowser(
            openExternalLinks=True,
            textInteractionFlags=(Qt.TextSelectableByMouse |
                                  Qt.LinksAccessibleByMouse)
        )
        self.descriptionlabel.setFrameStyle(QTextBrowser.NoFrame)
        # no (white) text background
        self.descriptionlabel.viewport().setAutoFillBackground(False)

        box.layout().addWidget(self.descriptionlabel)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(box)

        self.info.set_output_summary(self.info.NoOutput)

        self.splitter.setSizes([300, 200])
        self.splitter.splitterMoved.connect(
            lambda:
            setattr(self, "splitter_state", bytes(self.splitter.saveState()))
        )
        self.mainArea.layout().addWidget(self.splitter)

        proxy = QSortFilterProxyModel()
        proxy.setFilterKeyColumn(-1)
        proxy.setFilterCaseSensitivity(False)
        self.view.setModel(proxy)

        if self.splitter_state:
            self.splitter.restoreState(self.splitter_state)

        self.assign_delegates()

        self.setBlocking(True)
        self.setStatusMessage("Initializing")

        self._executor = ThreadPoolExecutor(max_workers=1)
        f = self._executor.submit(self.list_remote)
        w = FutureWatcher(f, parent=self)
        w.done.connect(self.__set_index)

    def assign_delegates(self):
        # NOTE: All columns must have size hinting delegates.
        # QTreeView queries only the columns displayed in the viewport so
        # the layout would be different depending in the horizontal scroll
        # position
        self.view.setItemDelegate(UniformHeightDelegate(self))
        self.view.setItemDelegateForColumn(
            self.Header.islocal,
            UniformHeightIndicatorDelegate(self, role=Qt.DisplayRole, indicatorSize=4)
        )
        self.view.setItemDelegateForColumn(
            self.Header.size,
            SizeDelegate(self)
        )
        self.view.setItemDelegateForColumn(
            self.Header.instances,
            NumericalDelegate(self)
        )
        self.view.setItemDelegateForColumn(
            self.Header.variables,
            NumericalDelegate(self)
        )
        self.view.resizeColumnToContents(self.Header.islocal)

    def _parse_info(self, file_path):
        if file_path in self.allinfo_remote:
            info = self.allinfo_remote[file_path]
        else:
            info = self.allinfo_local[file_path]

        islocal = file_path in self.allinfo_local
        isremote = file_path in self.allinfo_remote

        outdated = islocal and isremote and (
            self.allinfo_remote[file_path].get('version', '')
            != self.allinfo_local[file_path].get('version', '')
        )
        islocal &= not outdated

        prefix = os.path.join('', *file_path[:-1])
        filename = file_path[-1]

        return Namespace(file_path=file_path, prefix=prefix, filename=filename,
                         islocal=islocal, outdated=outdated, **info)

    def create_model(self):
        allkeys = set(self.allinfo_local) | set(self.allinfo_remote)
        allkeys = sorted(allkeys)

        model = QStandardItemModel(self)
        model.setHorizontalHeaderLabels(self._header_labels)

        current_index = -1
        for i, file_path in enumerate(allkeys):
            datainfo = self._parse_info(file_path)
            item1 = QStandardItem()
            item1.setData(" " if datainfo.islocal else "", Qt.DisplayRole)
            item1.setData(self.IndicatorBrushes[0], Qt.ForegroundRole)
            item1.setData(datainfo, Qt.UserRole)
            item2 = QStandardItem(datainfo.title)
            item3 = QStandardItem()
            item3.setData(datainfo.size, Qt.DisplayRole)
            item4 = QStandardItem()
            item4.setData(datainfo.instances, Qt.DisplayRole)
            item5 = QStandardItem()
            item5.setData(datainfo.variables, Qt.DisplayRole)
            item6 = QStandardItem()
            item6.setData(datainfo.target, Qt.DisplayRole)
            if datainfo.target:
                item6.setIcon(variable_icon(datainfo.target))
            item7 = QStandardItem()
            item7.setData(", ".join(datainfo.tags) if datainfo.tags else "",
                          Qt.DisplayRole)
            row = [item1, item2, item3, item4, item5, item6, item7]
            model.appendRow(row)

            if os.path.join(*file_path) == self.selected_id:
                current_index = i

        return model, current_index

    @Slot(object)
    def __set_index(self, f):
        # type: (Future) -> None
        # set results from `list_remote` query.
        assert QThread.currentThread() is self.thread()
        assert f.done()
        self.setBlocking(False)
        self.setStatusMessage("")
        self.allinfo_local = self.list_local()

        try:
            self.allinfo_remote = f.result()
        except Exception:  # anytying can happen, pylint: disable=broad-except
            log.exception("Error while fetching updated index")
            if not self.allinfo_local:
                self.Error.no_remote_datasets()
            else:
                self.Warning.only_local_datasets()
            self.allinfo_remote = {}

        model, current_index = self.create_model()

        self.view.model().setSourceModel(model)
        self.view.selectionModel().selectionChanged.connect(
            self.__on_selection
        )

        self.view.resizeColumnToContents(0)
        self.view.setColumnWidth(
            1, min(self.view.sizeHintForColumn(1),
                   self.view.fontMetrics().width("X" * 37)))

        header = self.view.header()
        header.restoreState(self.header_state)

        if current_index != -1:
            selmodel = self.view.selectionModel()
            selmodel.select(
                self.view.model().mapFromSource(model.index(current_index, 0)),
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            self.commit()

    def __update_cached_state(self):
        model = self.view.model().sourceModel()
        localinfo = self.list_local()
        assert isinstance(model, QStandardItemModel)
        allinfo = []
        for i in range(model.rowCount()):
            item = model.item(i, 0)
            info = item.data(Qt.UserRole)
            is_local = info.file_path in localinfo
            is_current = (is_local and
                          os.path.join(self.local_cache_path, *info.file_path)
                          == self.current_output)
            item.setData(" " * (is_local + is_current), Qt.DisplayRole)
            item.setData(self.IndicatorBrushes[is_current], Qt.ForegroundRole)
            allinfo.append(info)

    def selected_dataset(self):
        """
        Return the current selected dataset info or None if not selected

        Returns
        -------
        info : Optional[Namespace]
        """
        rows = self.view.selectionModel().selectedRows(0)
        assert 0 <= len(rows) <= 1
        current = rows[0] if rows else None  # type: Optional[QModelIndex]
        if current is not None:
            info = current.data(Qt.UserRole)
            assert isinstance(info, Namespace)
        else:
            info = None
        return info

    def filter(self):
        filter_string = self.filterLineEdit.text().strip()
        proxyModel = self.view.model()
        if proxyModel:
            proxyModel.setFilterFixedString(filter_string)

    def __on_selection(self):
        # Main datasets view selection has changed
        rows = self.view.selectionModel().selectedRows(0)
        assert 0 <= len(rows) <= 1
        current = rows[0] if rows else None  # type: Optional[QModelIndex]
        if current is not None:
            current = self.view.model().mapToSource(current)
            di = current.data(Qt.UserRole)
            text = description_html(di)
            self.descriptionlabel.setText(text)
            self.selected_id = os.path.join(di.prefix, di.filename)
        else:
            self.descriptionlabel.setText("")
            self.selected_id = None

    def commit(self):
        """
        Commit a dataset to the output immediately (if available locally) or
        schedule download background and an eventual send.

        During the download the widget is in blocking state
        (OWWidget.isBlocking)
        """
        di = self.selected_dataset()
        if di is not None:
            self.Error.clear()

            if self.__awaiting_state is not None:
                # disconnect from the __commit_complete
                self.__awaiting_state.watcher.done.disconnect(
                    self.__commit_complete)
                # .. and connect to update_cached_state
                # self.__awaiting_state.watcher.done.connect(
                #     self.__update_cached_state)
                # TODO: There are possible pending __progress_advance queued
                self.__awaiting_state.pb.advance.disconnect(
                    self.__progress_advance)
                self.progressBarFinished()
                self.__awaiting_state = None

            if not di.islocal:
                pr = progress()
                callback = lambda pr=pr: pr.advance.emit()
                pr.advance.connect(self.__progress_advance, Qt.QueuedConnection)

                self.progressBarInit()
                self.setStatusMessage("Fetching...")
                self.setBlocking(True)

                f = self._executor.submit(
                    ensure_local, self.INDEX_URL, di.file_path,
                    self.local_cache_path, force=di.outdated,
                    progress_advance=callback)
                w = FutureWatcher(f, parent=self)
                w.done.connect(self.__commit_complete)
                self.__awaiting_state = _FetchState(f, w, pr)
            else:
                self.setStatusMessage("")
                self.setBlocking(False)
                self.commit_cached(di.file_path)
        else:
            self.load_and_output(None)

    @Slot(object)
    def __commit_complete(self, f):
        # complete the commit operation after the required file has been
        # downloaded
        assert QThread.currentThread() is self.thread()
        assert self.__awaiting_state is not None
        assert self.__awaiting_state.future is f

        if self.isBlocking():
            self.progressBarFinished()
            self.setBlocking(False)
            self.setStatusMessage("")

        self.__awaiting_state = None

        try:
            path = f.result()
        # anything can happen here, pylint: disable=broad-except
        except Exception as ex:
            log.exception("Error:")
            self.error(format_exception(ex))
            path = None
        self.load_and_output(path)

    def commit_cached(self, file_path):
        path = LocalFiles(self.local_cache_path).localpath(*file_path)
        self.load_and_output(path)

    @Slot()
    def __progress_advance(self):
        assert QThread.currentThread() is self.thread()
        self.progressBarAdvance(1)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        if self.__awaiting_state is not None:
            self.__awaiting_state.watcher.done.disconnect(self.__commit_complete)
            self.__awaiting_state.pb.advance.disconnect(self.__progress_advance)
            self.__awaiting_state = None

    @staticmethod
    def sizeHint():
        return QSize(1100, 500)

    def closeEvent(self, event):
        self.splitter_state = bytes(self.splitter.saveState())
        self.header_state = bytes(self.view.header().saveState())
        super().closeEvent(event)

    def load_and_output(self, path):
        if path is None:
            self.Outputs.data.send(None)
            self.info.set_output_summary(self.info.NoOutput)
        else:
            data = self.load_data(path)
            self.Outputs.data.send(data)
            self.info.set_output_summary(len(data),
                                         format_summary_details(data))

        self.current_output = path
        self.__update_cached_state()

    @staticmethod
    def load_data(path):
        return Orange.data.Table(path)

    def list_remote(self):
        # type: () -> Dict[Tuple[str, ...], dict]
        client = ServerFiles(server=self.INDEX_URL)
        return client.allinfo()

    def list_local(self):
        # type: () -> Dict[Tuple[str, ...], dict]
        return LocalFiles(self.local_cache_path).allinfo()


class FutureWatcher(QObject):
    done = Signal(object)
    _p_done_notify = Signal(object)

    def __init__(self, future, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__future = future
        self._p_done_notify.connect(self.__on_done, Qt.QueuedConnection)
        future.add_done_callback(self._p_done_notify.emit)

    @Slot(object)
    def __on_done(self, f):
        assert f is self.__future
        self.done.emit(self.__future)


class progress(QObject):
    advance = Signal()


class _FetchState:
    def __init__(self, future, watcher, pb):
        self.future = future
        self.watcher = watcher
        self.pb = pb


def variable_icon(name):
    if name == "categorical":
        return gui.attributeIconDict[Orange.data.DiscreteVariable("x")]
    elif name == "numeric":  # ??
        return gui.attributeIconDict[Orange.data.ContinuousVariable("x")]
    else:
        return gui.attributeIconDict[-1]


def make_html_list(items):
    if items is None:
        return ''
    style = '"margin: 5px; text-indent: -40px; margin-left: 40px;"'

    def format_item(i):
        return '<p style={}><small>{}</small></p>'.format(style, i)

    return '\n'.join([format_item(i) for i in items])


def description_html(datainfo):
    # type: (Namespace) -> str
    """
    Summarize a data info as a html fragment.
    """
    html = []
    year = " ({})".format(str(datainfo.year)) if datainfo.year else ""
    source = ", from {}".format(datainfo.source) if datainfo.source else ""

    html.append("<b>{}</b>{}{}".format(escape(datainfo.title), year, source))
    html.append("<p>{}</p>".format(datainfo.description))
    seealso = make_html_list(datainfo.seealso)
    if seealso:
        html.append("<small><b>See Also</b>\n" + seealso + "</small>")
    refs = make_html_list(datainfo.references)
    if refs:
        html.append("<small><b>References</b>\n" + refs + "</small>")
    return "\n".join(html)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataSets).run()

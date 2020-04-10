# -*- coding: utf-8 -*-
"""
CSV File Import Widget
----------------------

"""
import sys
import os
import csv
import enum
import io
import traceback
import warnings
import logging
import weakref
import json

import gzip
import lzma
import bz2
import zipfile

from xml.sax.saxutils import escape
from functools import singledispatch
from concurrent import futures
from contextlib import ExitStack

import typing
from typing import (
    List, Tuple, Dict, Optional, Any, Callable, Iterable, Hashable,
    Union, AnyStr, BinaryIO
)

from PyQt5.QtCore import (
    Qt, QFileInfo, QTimer, QSettings, QObject, QSize, QMimeDatabase, QMimeType
)
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPalette
from PyQt5.QtWidgets import (
    QLabel, QComboBox, QPushButton, QDialog, QDialogButtonBox, QGridLayout,
    QVBoxLayout, QSizePolicy, QStyle, QFileIconProvider, QFileDialog,
    QApplication, QMessageBox, QTextBrowser
)
from PyQt5.QtCore import pyqtSlot as Slot, pyqtSignal as Signal

import numpy as np
import pandas.errors
import pandas as pd

from pandas.api import types as pdtypes

import Orange.data

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import textimport, concurrent as qconcurrent
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.settings import (
    QSettings_readArray, QSettings_writeArray
)
from Orange.widgets.utils.state_summary import format_summary_details


if typing.TYPE_CHECKING:
    # pylint: disable=invalid-name
    T = typing.TypeVar("T")
    K = typing.TypeVar("K")

__all__ = ["OWCSVFileImport"]

_log = logging.getLogger(__name__)

ColumnType = textimport.ColumnType
RowSpec = textimport.RowSpec


def enum_lookup(enumtype, name):
    # type: (typing.Type[T], str) -> Optional[T]
    """
    Return an value from `enumtype` by its symbolic name or None if not found.
    """
    try:
        return enumtype[name]
    except LookupError:
        return None


def dialect_eq(lhs, rhs):
    # type: (csv.Dialect, csv.Dialect) -> bool
    """Compare 2 `csv.Dialect` instances for equality."""
    return (lhs.delimiter == rhs.delimiter and
            lhs.quotechar == rhs.quotechar and
            lhs.doublequote == rhs.doublequote and
            lhs.escapechar == rhs.escapechar and
            lhs.quoting == rhs.quoting and
            lhs.skipinitialspace == rhs.skipinitialspace)


class Options:
    """
    Stored options for loading CSV-like file.

    Arguments
    ---------
    encoding : str
        A encoding to use for reading.
    dialect : csv.Dialect
        A csv.Dialect instance.
    columntypes: Iterable[Tuple[range, ColumnType]]
        A list of column type ranges specifying the types for columns.
        Need not list all columns. Columns not listed are assumed to have auto
        type inference.
    rowspec : Iterable[Tuple[range, RowSpec]]
         A list of row spec ranges.
    decimal_separator : str
        Decimal separator - a single character string; default: `"."`
    group_separator : str
        Thousands group separator - empty or a single character string;
        default: empty string
    """
    RowSpec = RowSpec
    ColumnType = ColumnType

    def __init__(self, encoding='utf-8', dialect=csv.excel(), columntypes=[],
                 rowspec=((range(0, 1), RowSpec.Header),),
                 decimal_separator=".", group_separator=""):
        # type: (str, csv.Dialect, List[Tuple[range, ColumnType]], ...) -> None
        self.encoding = encoding
        self.dialect = dialect
        self.columntypes = list(columntypes)
        self.rowspec = list(rowspec)  # type: List[Tuple[range, Options.RowSpec]]
        self.decimal_separator = decimal_separator
        self.group_separator = group_separator

    def __eq__(self, other):
        # type: (Options) -> bool
        """
        Compare this instance to `other` for equality.
        """
        if isinstance(other, Options):
            return (dialect_eq(self.dialect, other.dialect) and
                    self.encoding == other.encoding and
                    self.columntypes == other.columntypes and
                    self.rowspec == other.rowspec and
                    self.group_separator == other.group_separator and
                    self.decimal_separator == other.decimal_separator)
        else:
            return NotImplemented

    def __repr__(self):
        class_, args = self.__reduce__()
        return "{}{!r}".format(class_.__name__, args)
    __str__ = __repr__

    def __reduce__(self):
        return type(self), (self.encoding, self.dialect,
                            self.columntypes, self.rowspec)

    def as_dict(self):
        # type: () -> Dict[str, Any]
        """
        Return return Option parameters as plain types suitable for
        serialization (e.g JSON serializable).
        """
        return {
            "encoding": self.encoding,
            "delimiter": self.dialect.delimiter,
            "quotechar": self.dialect.quotechar,
            "doublequote": self.dialect.doublequote,
            "skipinitialspace": self.dialect.skipinitialspace,
            "quoting": self.dialect.quoting,
            "columntypes": Options.spec_as_encodable(self.columntypes),
            "rowspec": Options.spec_as_encodable(self.rowspec),
            "decimal_separator": self.decimal_separator,
            "group_separator": self.group_separator,
        }

    @staticmethod
    def from_dict(mapping):
        # type: (Dict[str, Any]) -> Options
        """
        Reconstruct a `Options` from a plain dictionary (see :func:`as_dict`).
        """
        encoding = mapping["encoding"]
        delimiter = mapping["delimiter"]
        quotechar = mapping["quotechar"]
        doublequote = mapping["doublequote"]
        quoting = mapping["quoting"]
        skipinitialspace = mapping["skipinitialspace"]

        dialect = textimport.Dialect(
            delimiter, quotechar, None, doublequote, skipinitialspace,
            quoting=quoting)

        colspec = mapping["columntypes"]
        rowspec = mapping["rowspec"]
        colspec = Options.spec_from_encodable(colspec, ColumnType)
        rowspec = Options.spec_from_encodable(rowspec, RowSpec)
        decimal = mapping.get("decimal_separator", ".")
        group = mapping.get("group_separator", "")

        return Options(encoding, dialect, colspec, rowspec,
                       decimal_separator=decimal,
                       group_separator=group)

    @staticmethod
    def spec_as_encodable(spec):
        # type: (List[Tuple[range, enum.Enum]]) -> List[Dict[str, Any]]
        return [{"start": r.start, "stop": r.stop, "value": value.name}
                for r, value in spec]

    @staticmethod
    def spec_from_encodable(spec, enumtype):
        # type: (List[Dict[str, Any]], typing.Type[T]) -> List[Tuple[range, T]]
        r = []
        for v in spec:
            try:
                start, stop, name = v["start"], v["stop"], v["value"]
            except (KeyError, ValueError):
                pass
            else:
                r.append((range(start, stop), enum_lookup(enumtype, name)))
        return r


class CSVImportDialog(QDialog):
    """
    A dialog for selecting CSV file import options.
    """
    def __init__(self, parent=None, flags=Qt.Dialog, **kwargs):
        super().__init__(parent, flags, **kwargs)
        self.setLayout(QVBoxLayout())

        self._options = None
        self._path = None
        # Finalizer for opened file handle (in _update_preview)
        self.__finalizer = None  # type: Optional[Callable[[], None]]
        self._optionswidget = textimport.CSVImportWidget()
        self._optionswidget.previewReadErrorOccurred.connect(
            self.__on_preview_error
        )
        self._optionswidget.previewModelReset.connect(
            self.__on_preview_reset
        )
        self._buttons = buttons = QDialogButtonBox(
            orientation=Qt.Horizontal,
            standardButtons=(QDialogButtonBox.Ok | QDialogButtonBox.Cancel |
                             QDialogButtonBox.Reset |
                             QDialogButtonBox.RestoreDefaults),
            objectName="dialog-button-box",
        )
        # TODO: Help button
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        b = buttons.button(QDialogButtonBox.Reset)
        b.clicked.connect(self.reset)
        b = buttons.button(QDialogButtonBox.RestoreDefaults)
        b.clicked.connect(self.restoreDefaults)
        self.layout().addWidget(self._optionswidget)
        self.layout().addWidget(buttons)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._overlay = OverlayWidget(self)
        self._overlay.setWidget(self._optionswidget.dataview)
        self._overlay.setLayout(QVBoxLayout())
        self._overlay.layout().addWidget(QLabel(wordWrap=True))
        self._overlay.hide()

    def setOptions(self, options):
        # type: (Options) -> None
        self._options = options
        self._optionswidget.setEncoding(options.encoding)
        self._optionswidget.setDialect(options.dialect)
        self._optionswidget.setNumbersFormat(
            options.group_separator, options.decimal_separator)
        self._optionswidget.setColumnTypeRanges(options.columntypes)
        self._optionswidget.setRowStates(
            {i: v for r, v in options.rowspec for i in r}
        )

    def options(self):
        # type: () -> Options
        rowspec_ = self._optionswidget.rowStates()
        rowspec = [(range(i, i + 1), v) for i, v in rowspec_.items()]
        numformat = self._optionswidget.numbersFormat()
        return Options(
            encoding=self._optionswidget.encoding(),
            dialect=self._optionswidget.dialect(),
            columntypes=self._optionswidget.columnTypeRanges(),
            rowspec=rowspec,
            decimal_separator=numformat["decimal"],
            group_separator=numformat["group"],
        )

    def setPath(self, path):
        """
        Set the preview path.
        """
        if self._path != path:
            self._path = path
            self.__update_preview()

    def path(self):
        """Return the preview path"""
        return self._path

    def reset(self):
        """
        Reset the options to the state previously set with `setOptions`
        effectively undoing any user modifications since then.
        """
        self.setOptions(self._options)

    def restoreDefaults(self):
        """
        Restore the options to default state.
        """
        # preserve `_options` if set by clients (for `reset`).
        opts = self._options
        self.setOptions(Options("utf-8", csv.excel()))
        self._options = opts

    def __update_preview(self):
        if not self._path:
            return
        try:
            f = _open(self._path, "rb")
        except OSError as err:
            traceback.print_exc(file=sys.stderr)
            fmt = "".join(traceback.format_exception_only(type(err), err))
            self.__set_error(fmt)
        else:
            self.__clear_error()
            self._optionswidget.setSampleContents(f)
            closeexisting = self.__finalizer
            if closeexisting is not None:
                self.destroyed.disconnect(closeexisting)
                closeexisting()
            self.__finalizer = weakref.finalize(self, f.close)
            self.destroyed.connect(self.__finalizer)

    def __set_error(self, text, format=Qt.PlainText):
        self._optionswidget.setEnabled(False)
        label = self._overlay.findChild(QLabel)  # type: QLabel
        label.setText(text)
        label.setTextFormat(format)
        self._overlay.show()
        self._overlay.raise_()
        dialog_button_box_set_enabled(self._buttons, False)

    def __clear_error(self):
        if self._overlay.isVisibleTo(self):
            self._overlay.hide()
            self._optionswidget.setEnabled(True)

    # Enable/disable the accept buttons on the most egregious errors.
    def __on_preview_error(self):
        b = self._buttons.button(QDialogButtonBox.Ok)
        b.setEnabled(False)

    def __on_preview_reset(self):
        b = self._buttons.button(QDialogButtonBox.Ok)
        b.setEnabled(True)


def dialog_button_box_set_enabled(buttonbox, enabled):
    # type: (QDialogButtonBox, bool) -> None
    """
    Disable/enable buttons in a QDialogButtonBox based on their role.

    All buttons except the ones with RejectRole or HelpRole are disabled.
    """
    stashname = "__p_dialog_button_box_set_enabled"
    for b in buttonbox.buttons():
        role = buttonbox.buttonRole(b)
        if not enabled:
            if b.property(stashname) is None:
                b.setProperty(stashname, b.isEnabledTo(buttonbox))
            b.setEnabled(
                role == QDialogButtonBox.RejectRole or
                role == QDialogButtonBox.HelpRole
            )
        else:
            stashed_state = b.property(stashname)
            if isinstance(stashed_state, bool):
                state = stashed_state
                b.setProperty(stashname, None)
            else:
                state = True
            b.setEnabled(state)


class ImportItem(QStandardItem):
    """
    An item representing a file path and associated load options
    """
    PathRole = Qt.UserRole + 12
    OptionsRole = Qt.UserRole + 14

    def path(self):
        # type: () -> str
        path = self.data(ImportItem.PathRole)
        return path if isinstance(path, str) else ""

    def setPath(self, path):
        # type: (str) -> None
        self.setData(path, ImportItem.PathRole)

    def options(self):
        # type: () -> Optional[Options]
        options = self.data(ImportItem.OptionsRole)
        return options if isinstance(options, Options) else None

    def setOptions(self, options):
        # type: (Options) -> None
        self.setData(options, ImportItem.OptionsRole)

    @classmethod
    def fromPath(cls, path):
        # type: (str) -> ImportItem
        """
        Create a `ImportItem` from a local file system path.
        """
        iconprovider = QFileIconProvider()
        basename = os.path.basename(path)
        item = cls()
        item.setText(basename)
        item.setToolTip(path)
        finfo = QFileInfo(path)
        if finfo.exists():
            item.setIcon(iconprovider.icon(finfo))
        else:
            item.setIcon(iconprovider.icon(QFileIconProvider.File))

        item.setData(path, ImportItem.PathRole)
        if not os.path.isfile(path):
            item.setEnabled(False)
            item.setToolTip(item.toolTip() + " (missing from filesystem)")
        return item


def qname(type_):
    # type: (type) -> str
    """
    Return the fully qualified name for a `type_`.
    """

    return "{0.__module__}.{0.__qualname__}".format(type_)


class OWCSVFileImport(widget.OWWidget):
    name = "CSV File Import"
    description = "Import a data table from a CSV formatted file."
    icon = "icons/CSVFile.svg"
    priority = 11
    category = "Data"
    keywords = ["file", "load", "read", "open", "csv"]

    outputs = [
        widget.OutputSignal(
            name="Data",
            type=Orange.data.Table,
            doc="Loaded data set."),
        widget.OutputSignal(
            name="Data Frame",
            type=pd.DataFrame,
            doc=""
        )
    ]

    class Error(widget.OWWidget.Error):
        error = widget.Msg(
            "Unexpected error"
        )
        encoding_error = widget.Msg(
            "Encoding error\n"
            "The file might be encoded in an unsupported encoding or it "
            "might be binary"
        )

    #: Paths and options of files accessed in a 'session'
    _session_items = settings.Setting(
        [], schema_only=True)  # type: List[Tuple[str, dict]]

    #: Saved dialog state (last directory and selected filter)
    dialog_state = settings.Setting({
        "directory": "",
        "filter": ""
    })  # type: Dict[str, str]
    MaxHistorySize = 50

    want_main_area = False
    buttons_area_orientation = None
    resizing_enabled = False

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.__committimer = QTimer(self, singleShot=True)
        self.__committimer.timeout.connect(self.commit)

        self.__executor = qconcurrent.ThreadExecutor()
        self.__watcher = None  # type: Optional[qconcurrent.FutureWatcher]

        self.controlArea.layout().setSpacing(-1)  # reset spacing
        grid = QGridLayout()
        grid.addWidget(QLabel("File:", self), 0, 0, 1, 1)

        self.import_items_model = QStandardItemModel(self)
        self.recent_combo = QComboBox(
            self, objectName="recent-combo", toolTip="Recent files.",
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=16,
        )
        self.recent_combo.setModel(self.import_items_model)
        self.recent_combo.activated.connect(self.activate_recent)
        self.recent_combo.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.browse_button = QPushButton(
            "…", icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
            toolTip="Browse filesystem", autoDefault=False,
        )
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.browse_button.clicked.connect(self.browse)
        grid.addWidget(self.recent_combo, 0, 1, 1, 1)
        grid.addWidget(self.browse_button, 0, 2, 1, 1)
        self.controlArea.layout().addLayout(grid)

        ###########
        # Info text
        ###########
        box = gui.widgetBox(self.controlArea, "Info", addSpace=False)
        self.summary_text = QTextBrowser(
            verticalScrollBarPolicy=Qt.ScrollBarAsNeeded,
            readOnly=True,
        )
        self.summary_text.viewport().setBackgroundRole(QPalette.NoRole)
        self.summary_text.setFrameStyle(QTextBrowser.NoFrame)
        self.summary_text.setMinimumHeight(self.fontMetrics().ascent() * 2 + 4)
        self.summary_text.viewport().setAutoFillBackground(False)
        box.layout().addWidget(self.summary_text)

        self.info.set_output_summary(self.info.NoOutput)

        button_box = QDialogButtonBox(
            orientation=Qt.Horizontal,
            standardButtons=QDialogButtonBox.Cancel | QDialogButtonBox.Retry
        )
        self.load_button = b = button_box.button(QDialogButtonBox.Retry)
        b.setText("Load")
        b.clicked.connect(self.__committimer.start)
        b.setEnabled(False)
        b.setDefault(True)

        self.cancel_button = b = button_box.button(QDialogButtonBox.Cancel)
        b.clicked.connect(self.cancel)
        b.setEnabled(False)
        b.setAutoDefault(False)

        self.import_options_button = QPushButton(
            "Import Options…", enabled=False, autoDefault=False,
            clicked=self._activate_import_dialog
        )

        def update_buttons(cbindex):
            self.import_options_button.setEnabled(cbindex != -1)
            self.load_button.setEnabled(cbindex != -1)
        self.recent_combo.currentIndexChanged.connect(update_buttons)

        button_box.addButton(
            self.import_options_button, QDialogButtonBox.ActionRole
        )
        button_box.setStyleSheet(
            "button-layout: {:d};".format(QDialogButtonBox.MacLayout)
        )
        self.controlArea.layout().addWidget(button_box)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)

        self._restoreState()
        item = self.current_item()
        if item is not None:
            self.set_selected_file(item.path(), item.options())

    @Slot(int)
    def activate_recent(self, index):
        """
        Activate an item from the recent list.
        """
        if 0 <= index < self.import_items_model.rowCount():
            item = self.import_items_model.item(index)
            assert item is not None
            path = item.data(ImportItem.PathRole)
            opts = item.data(ImportItem.OptionsRole)
            if not isinstance(opts, Options):
                opts = None
            self.set_selected_file(path, opts)
        else:
            self.recent_combo.setCurrentIndex(-1)

    @Slot()
    def browse(self):
        """
        Open a file dialog and select a user specified file.
        """
        formats = [
            "Text - comma separated (*.csv, *)",
            "Text - tab separated (*.tsv, *)",
            "Text - all files (*)"
        ]

        dlg = QFileDialog(
            self, windowTitle="Open Data File",
            acceptMode=QFileDialog.AcceptOpen,
            fileMode=QFileDialog.ExistingFile
        )
        dlg.setNameFilters(formats)
        state = self.dialog_state
        lastdir = state.get("directory", "")
        lastfilter = state.get("filter", "")

        if lastdir and os.path.isdir(lastdir):
            dlg.setDirectory(lastdir)
        if lastfilter:
            dlg.selectNameFilter(lastfilter)

        status = dlg.exec_()
        dlg.deleteLater()
        if status == QFileDialog.Accepted:
            self.dialog_state["directory"] = dlg.directory().absolutePath()
            self.dialog_state["filter"] = dlg.selectedNameFilter()

            selected_filter = dlg.selectedNameFilter()
            path = dlg.selectedFiles()[0]
            # pre-flight check; try to determine the nature of the file
            mtype = _mime_type_for_path(path)
            if not mtype.inherits("text/plain"):
                mb = QMessageBox(
                    parent=self,
                    windowTitle="",
                    icon=QMessageBox.Question,
                    text="The '{basename}' may be a binary file.\n"
                         "Are you sure you want to continue?".format(
                             basename=os.path.basename(path)),
                    standardButtons=QMessageBox.Cancel | QMessageBox.Yes
                )
                mb.setWindowModality(Qt.WindowModal)
                if mb.exec() == QMessageBox.Cancel:
                    return

            # initialize dialect based on selected extension
            if selected_filter in formats[:-1]:
                filter_idx = formats.index(selected_filter)
                if filter_idx == 0:
                    dialect = csv.excel()
                elif filter_idx == 1:
                    dialect = csv.excel_tab()
                else:
                    dialect = csv.excel_tab()
                header = True
            else:
                try:
                    dialect, header = sniff_csv_with_path(path)
                except Exception:  # pylint: disable=broad-except
                    dialect, header = csv.excel(), True

            options = None
            # Search for path in history.
            # If found use the stored params to initialize the import dialog
            items = self.itemsFromSettings()
            idx = index_where(items, lambda t: samepath(t[0], path))
            if idx is not None:
                _, options_ = items[idx]
                if options_ is not None:
                    options = options_

            if options is None:
                if not header:
                    rowspec = []
                else:
                    rowspec = [(range(0, 1), RowSpec.Header)]
                options = Options(
                    encoding="utf-8", dialect=dialect, rowspec=rowspec)

            dlg = CSVImportDialog(
                self, windowTitle="Import Options", sizeGripEnabled=True)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setPath(path)
            dlg.setOptions(options)
            status = dlg.exec_()
            dlg.deleteLater()
            if status == QDialog.Accepted:
                self.set_selected_file(path, dlg.options())

    def current_item(self):
        # type: () -> Optional[ImportItem]
        """
        Return the current selected item (file) or None if there is no
        current item.
        """
        idx = self.recent_combo.currentIndex()
        if idx == -1:
            return None

        item = self.recent_combo.model().item(idx)  # type: QStandardItem
        if isinstance(item, ImportItem):
            return item
        else:
            return None

    def _activate_import_dialog(self):
        """Activate the Import Options dialog for the  current item."""
        item = self.current_item()
        assert item is not None
        dlg = CSVImportDialog(
            self, windowTitle="Import Options", sizeGripEnabled=True,
        )
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        settings = QSettings()
        qualname = qname(type(self))
        settings.beginGroup(qualname)
        size = settings.value("size", QSize(), type=QSize)  # type: QSize
        if size.isValid():
            dlg.resize(size)

        path = item.data(ImportItem.PathRole)
        options = item.data(ImportItem.OptionsRole)
        dlg.setPath(path)  # Set path before options so column types can
        if isinstance(options, Options):
            dlg.setOptions(options)

        def update():
            newoptions = dlg.options()
            item.setData(newoptions, ImportItem.OptionsRole)
            # update the stored item
            self._add_recent(path, newoptions)
            if newoptions != options:
                self._invalidate()
        dlg.accepted.connect(update)

        def store_size():
            settings.setValue("size", dlg.size())
        dlg.finished.connect(store_size)
        dlg.show()

    def set_selected_file(self, filename, options=None):
        """
        Set the current selected filename path.
        """
        self._add_recent(filename, options)
        self._invalidate()

    #: Saved options for a filename
    SCHEMA = {
        "path": str,  # Local filesystem path
        "options": str,  # json encoded 'Options'
    }

    @classmethod
    def _local_settings(cls):
        # type: () -> QSettings
        """Return a QSettings instance with local persistent settings."""
        filename = "{}.ini".format(qname(cls))
        fname = os.path.join(settings.widget_settings_dir(), filename)
        return QSettings(fname, QSettings.IniFormat)

    def _add_recent(self, filename, options=None):
        # type: (str, Optional[Options]) -> None
        """
        Add filename to the list of recent files.
        """
        model = self.import_items_model
        index = index_where(
            (model.index(i, 0).data(ImportItem.PathRole)
             for i in range(model.rowCount())),
            lambda path: isinstance(path, str) and samepath(path, filename)
        )
        if index is not None:
            item, *_ = model.takeRow(index)
        else:
            item = ImportItem.fromPath(filename)

        model.insertRow(0, item)

        if options is not None:
            item.setOptions(options)

        self.recent_combo.setCurrentIndex(0)
        # store items to local persistent settings
        s = self._local_settings()
        arr = QSettings_readArray(s, "recent", OWCSVFileImport.SCHEMA)
        item = {"path": filename}
        if options is not None:
            item["options"] = json.dumps(options.as_dict())

        arr = [item for item in arr if item.get("path") != filename]
        arr.append(item)
        QSettings_writeArray(s, "recent", arr)

        # update workflow session items
        items = self._session_items[:]
        idx = index_where(items, lambda t: samepath(t[0], filename))
        if idx is not None:
            del items[idx]
        items.insert(0, (filename, options.as_dict()))
        self._session_items = items[:OWCSVFileImport.MaxHistorySize]

    def _invalidate(self):
        # Invalidate the current output and schedule a new commit call.
        # (NOTE: The widget enters a blocking state)
        self.__committimer.start()
        if self.__watcher is not None:
            self.__cancel_task()
        self.setBlocking(True)

    def commit(self):
        """
        Commit the current state and submit the load task for execution.

        Note
        ----
        Any existing pending task is canceled.
        """
        self.__committimer.stop()
        if self.__watcher is not None:
            self.__cancel_task()
        self.error()

        item = self.current_item()
        if item is None:
            return
        path = item.data(ImportItem.PathRole)
        opts = item.data(ImportItem.OptionsRole)
        if not isinstance(opts, Options):
            return

        task = state = TaskState()
        state.future = ...
        state.watcher = qconcurrent.FutureWatcher()
        state.progressChanged.connect(
            self.__set_read_progress, Qt.DirectConnection)

        def progress_(i, j):
            task.emitProgressChangedOrCancel(i, j)

        task.future = self.__executor.submit(
            clear_stack_on_cancel(load_csv),
            path, opts, progress_,
        )
        task.watcher.setFuture(task.future)
        w = task.watcher
        w.done.connect(self.__handle_result)
        w.progress = state
        self.__watcher = w
        self.__set_running_state()

    @Slot('qint64', 'qint64')
    def __set_read_progress(self, read, count):
        if count > 0:
            self.progressBarSet(100 * read / count)

    def __cancel_task(self):
        # Cancel and dispose of the current task
        assert self.__watcher is not None
        w = self.__watcher
        w.future().cancel()
        w.progress.cancel = True
        w.done.disconnect(self.__handle_result)
        w.progress.progressChanged.disconnect(self.__set_read_progress)
        w.progress.deleteLater()
        # wait until completion
        futures.wait([w.future()])
        self.__watcher = None

    def cancel(self):
        """
        Cancel current pending or executing task.
        """
        if self.__watcher is not None:
            self.__cancel_task()
            self.__clear_running_state()
            self.setStatusMessage("Cancelled")
            self.summary_text.setText(
                "<div>Cancelled<br/><small>Press 'Reload' to try again</small></div>"
            )

    def __set_running_state(self):
        self.progressBarInit()
        self.setBlocking(True)
        self.setStatusMessage("Running")
        self.cancel_button.setEnabled(True)
        self.load_button.setText("Restart")
        path = self.current_item().path()
        self.Error.clear()
        self.summary_text.setText(
            "<div>Loading: <i>{}</i><br/>".format(prettyfypath(path))
        )

    def __clear_running_state(self, ):
        self.progressBarFinished()
        self.setStatusMessage("")
        self.setBlocking(False)
        self.cancel_button.setEnabled(False)
        self.load_button.setText("Reload")

    def __set_error_state(self, err):
        self.Error.clear()
        if isinstance(err, UnicodeDecodeError):
            self.Error.encoding_error(exc_info=err)
        else:
            self.Error.error(exc_info=err)

        path = self.current_item().path()
        basename = os.path.basename(path)
        if isinstance(err, UnicodeDecodeError):
            text = (
                "<div><i>{basename}</i> was not loaded due to a text encoding "
                "error. The file might be saved in an unknown or invalid "
                "encoding, or it might be a binary file.</div>"
            ).format(
                basename=escape(basename)
            )
        else:
            text = (
                "<div><i>{basename}</i> was not loaded due to an error:"
                "<p style='white-space: pre;'>{err}</p>"
            ).format(
                basename=escape(basename),
                err="".join(traceback.format_exception_only(type(err), err))
            )
        self.summary_text.setText(text)

    def __clear_error_state(self):
        self.Error.error.clear()
        self.summary_text.setText("")

    def onDeleteWidget(self):
        """Reimplemented."""
        if self.__watcher is not None:
            self.__cancel_task()
            self.__executor.shutdown()
        super().onDeleteWidget()

    @Slot(object)
    def __handle_result(self, f):
        # type: (qconcurrent.Future[pd.DataFrame]) -> None
        assert f.done()
        assert f is self.__watcher.future()
        self.__watcher = None
        self.__clear_running_state()

        try:
            df = f.result()
            assert isinstance(df, pd.DataFrame)
        except pandas.errors.EmptyDataError:
            df = pd.DataFrame({})
        except Exception as e:  # pylint: disable=broad-except
            self.__set_error_state(e)
            df = None
        else:
            self.__clear_error_state()

        if df is not None:
            table = pandas_to_table(df)
        else:
            table = None
        self.send("Data Frame", df)
        self.send('Data', table)
        self._update_status_messages(table)

    def _update_status_messages(self, data):
        summary = len(data) if data else self.info.NoOutput
        details = format_summary_details(data) if data else ""
        self.info.set_output_summary(summary, details)
        if data is None:
            return

        def pluralize(seq):
            return "s" if len(seq) != 1 else ""

        summary = ("{n_instances} row{plural_1}, "
                   "{n_features} feature{plural_2}, "
                   "{n_meta} meta{plural_3}").format(
                       n_instances=len(data), plural_1=pluralize(data),
                       n_features=len(data.domain.attributes),
                       plural_2=pluralize(data.domain.attributes),
                       n_meta=len(data.domain.metas),
                       plural_3=pluralize(data.domain.metas))
        self.summary_text.setText(summary)

    def itemsFromSettings(self):
        # type: () -> List[Tuple[str, Options]]
        """
        Return items from local history.
        """
        s = self._local_settings()
        items_ = QSettings_readArray(s, "recent", OWCSVFileImport.SCHEMA)
        items = []  # type: List[Tuple[str, Options]]
        for item in items_:
            path = item.get("path", "")
            if not path:
                continue
            opts_json = item.get("options", "")
            try:
                opts = Options.from_dict(json.loads(opts_json))
            except (csv.Error, LookupError, TypeError, json.JSONDecodeError):
                _log.error("Could not reconstruct options for '%s'", path,
                           exc_info=True)
            else:
                items.append((path, opts))
        return items[::-1]

    def _restoreState(self):
        # Restore the state. Merge session (workflow) items with the
        # local history.
        model = self.import_items_model
        # local history
        items = self.itemsFromSettings()
        # stored session items
        sitems = []
        for p, m in self._session_items:
            try:
                item_ = (p, Options.from_dict(m))
            except (csv.Error, LookupError):
                # Is it better to fail then to lose a item slot?
                _log.error("Failed to restore '%s'", p, exc_info=True)
            else:
                sitems.append(item_)

        items = sitems + items
        items = unique(items, key=lambda t: pathnormalize(t[0]))

        curr = self.recent_combo.currentIndex()
        if curr != -1:
            currentpath = self.recent_combo.currentData(ImportItem.PathRole)
        else:
            currentpath = None
        for path, options in items:
            item = ImportItem.fromPath(path)
            item.setOptions(options)
            model.appendRow(item)

        if currentpath is not None:
            idx = self.recent_combo.findData(currentpath, ImportItem.PathRole)
            if idx != -1:
                self.recent_combo.setCurrentIndex(idx)


@singledispatch
def sniff_csv(file, samplesize=2 ** 20):
    sniffer = csv.Sniffer()
    sample = file.read(samplesize)
    dialect = sniffer.sniff(sample)
    dialect = textimport.Dialect(
        dialect.delimiter, dialect.quotechar,
        dialect.escapechar, dialect.doublequote,
        dialect.skipinitialspace, dialect.quoting
    )
    has_header = sniffer.has_header(sample)
    return dialect, has_header


@sniff_csv.register(str)
@sniff_csv.register(bytes)
def sniff_csv_with_path(path, encoding="utf-8", samplesize=2 ** 20):
    with _open(path, "rt", encoding=encoding) as f:
        return sniff_csv(f, samplesize)


def _open(path, mode, encoding=None):
    # type: (str, str, Optional[str]) -> typing.IO[Any]
    """
    Open a local file `path` for reading. The file may be gzip, bz2 or zip
    compressed.

    If a zip archive then a single archive member is expected.

    Parameters
    ----------
    path : str
        File system path
    mode : str
        'r', 'rb' or 'rt'
    encoding : Optional[str]
        Optional text encoding, for opening in text mode.

    Returns
    -------
    stream: io.BaseIO
        A stream opened for reading.
    """
    if mode not in {'r', 'rb', 'rt'}:
        raise ValueError('r')
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".gz":
        return gzip.open(path, mode, encoding=encoding)
    elif ext == ".bz2":
        return bz2.open(path, mode, encoding=encoding)
    elif ext == ".xz":
        return lzma.open(path, mode, encoding=encoding)
    elif ext == ".zip":
        arh = zipfile.ZipFile(path, 'r')
        filelist = arh.infolist()
        if len(filelist) == 1:
            filename = filelist[0]
            zinfo = arh.getinfo(filename)
            f = arh.open(zinfo.filename, 'r')
            if 't' in mode:
                f = io.TextIOWrapper(f, encoding=encoding)
            return f
        else:
            raise ValueError("Expected a single file in the archive.")
    else:
        return open(path, mode, encoding=encoding)


compression_types = [
    "application/gzip", "application/zip",
    "application/x-xz", "application/x-bzip",
    # application/x-lz4
]


def _mime_type_for_path(path):
    # type: (str) -> QMimeType
    """
    Return the mime type of the file on a local filesystem.

    In case the path is a compressed file return the mime type of its contents

    Parameters
    ----------
    path : str
        Local filesystem path

    Returns
    -------
    mimetype: QMimeType
    """
    db = QMimeDatabase()
    mtype = db.mimeTypeForFile(path, QMimeDatabase.MatchDefault)
    if any(mtype.inherits(t) for t in compression_types):
        # peek contents
        try:
            with _open(path, "rb") as f:
                sample = f.read(4096)
        except Exception:  # pylint: disable=broad-except
            sample = b''
        mtype = db.mimeTypeForData(sample)
    return mtype


NA_DEFAULT = ["", "?", ".", "~", "nan", "NAN", "NaN", "N/A", "n/a", "NA"]

NA_VALUES = {
    ColumnType.Numeric: NA_DEFAULT,
    ColumnType.Categorical: NA_DEFAULT,
    ColumnType.Time: NA_DEFAULT + ["NaT", "NAT"],
    ColumnType.Text: [],
    ColumnType.Auto: NA_DEFAULT,
}


def load_csv(path, opts, progress_callback=None):
    # type: (Union[AnyStr, BinaryIO], Options, ...) -> pd.DataFrame
    def dtype(coltype):
        # type: (ColumnType) -> Optional[str]
        if coltype == ColumnType.Numeric:
            return "float"
        elif coltype == ColumnType.Categorical:
            return "category"
        elif coltype == ColumnType.Time:
            return "object"
        elif coltype == ColumnType.Text:
            return "object"
        elif coltype == ColumnType.Skip:
            return None
        elif coltype == ColumnType.Auto:
            return None
        else:
            raise TypeError

    def expand(ranges):
        # type: (Iterable[Tuple[range, T]]) -> Iterable[Tuple[int, T]]
        return ((i, x) for r, x in ranges for i in r)

    dtypes = {i: dtype(c) for i, c in expand(opts.columntypes)}
    dtypes = {i: dtp for i, dtp in dtypes.items()
              if dtp is not None and dtp != ColumnType.Auto}

    columns_ignored = {i for i, c in expand(opts.columntypes)
                       if c == ColumnType.Skip}
    dtcols = {i for i, c in expand(opts.columntypes)
              if c == ColumnType.Time}
    parse_dates = sorted(dtcols)
    na_values = {i: NA_VALUES.get(c, NA_DEFAULT)
                 for i, c in expand(opts.columntypes)}
    if not parse_dates:
        parse_dates = False

    # fixup header indices to account for skipped rows (header row indices
    # pick rows after skiprows)

    hspec = sorted(opts.rowspec, key=lambda t: t[0].start)
    header_ranges = []
    nskiped = 0
    for range_, state in hspec:
        if state == RowSpec.Skipped:
            nskiped += len(range_)
        elif state == RowSpec.Header:
            header_ranges.append(
                range(range_.start - nskiped, range_.stop - nskiped)
            )
    headers = [i for r in header_ranges for i in r]
    skiprows = [row for r, st in hspec if st == RowSpec.Skipped for row in r]

    if not headers:
        header = None
        prefix = "X."

    elif len(headers) == 1:
        header = headers[0]
        prefix = None
    else:
        header = headers
        prefix = None

    if not skiprows:
        skiprows = None

    numbers_format_kwds = {}

    if opts.decimal_separator != ".":
        numbers_format_kwds["decimal"] = opts.decimal_separator

    if opts.group_separator != "":
        numbers_format_kwds["thousands"] = opts.group_separator

    with ExitStack() as stack:
        if isinstance(path, (str, bytes)):
            f = stack.enter_context(_open(path, 'rb'))
        elif isinstance(path, (io.RawIOBase, io.BufferedIOBase)) or \
                hasattr(path, "read"):
            f = path
        else:
            raise TypeError()
        file = TextReadWrapper(
            f, encoding=opts.encoding,
            progress_callback=progress_callback)
        stack.callback(file.detach)
        df = pd.read_csv(
            file, sep=opts.dialect.delimiter, dialect=opts.dialect,
            skipinitialspace=opts.dialect.skipinitialspace,
            header=header, skiprows=skiprows,
            dtype=dtypes, parse_dates=parse_dates, prefix=prefix,
            na_values=na_values, keep_default_na=False,
            float_precision="round_trip",
            **numbers_format_kwds
        )
        if columns_ignored:
            # TODO: use 'usecols' parameter in `read_csv` call to
            # avoid loading/parsing the columns
            df.drop(
                columns=[df.columns[i] for i in columns_ignored
                         if i < len(df.columns)],
                inplace=True
            )
        return df


def clear_stack_on_cancel(f):
    """
    A decorator that catches the TaskState.UserCancelException exception
    and clears the exception's traceback to remove local references.

    Parameters
    ----------
    f : callable

    Returns
    -------
    wrapped : callable
    """
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TaskState.UserCancelException as e:
            # TODO: Is this enough to allow immediate gc of the stack?
            # How does it chain across cython code?
            # Maybe just return None.
            e = e.with_traceback(None)
            e.__context__ = None
            e.__cause__ = None
            raise e
        except BaseException as e:
            traceback.clear_frames(e.__traceback__)
            raise

    return wrapper


class TaskState(QObject):
    class UserCancelException(BaseException):
        """User interrupt exception."""

    #: Signal emitted with the current read progress. First value is the current
    #: progress state, second value is the total progress to complete
    #: (-1 if unknown)
    progressChanged = Signal('qint64', 'qint64')
    __progressChanged = Signal('qint64', 'qint64')
    #: Was cancel requested.
    cancel = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # route the signal via this object's queue
        self.__progressChanged.connect(
            self.progressChanged, Qt.QueuedConnection)

    def emitProgressChangedOrCancel(self, current, total):
        # type: (int, int) -> None
        """
        Emit the progressChanged signal with `current` and `total`.
        """
        if self.cancel:
            raise TaskState.UserCancelException()
        else:
            self.__progressChanged.emit(current, total)


class TextReadWrapper(io.TextIOWrapper):
    """
    TextIOWrapper reporting the read progress.

    Assuming a single forward read pass.
    """

    #: A `Callable[[int, int], []]` called when the file position is
    #: advanced by read; called with current byte position and total
    #: file size.
    progress_callback = ...  # type: Callable[[int, int], None]

    def __init__(self, buffer, *args, progress_callback=None, **kwargs):
        super().__init__(buffer, *args, **kwargs)
        if progress_callback is None:
            def progress_callback(i, j):  # pylint: disable=unused-argument
                pass
        self.progress_callback = progress_callback
        try:
            self.__size = os.fstat(buffer.fileno()).st_size
        except OSError:
            self.__size = -1

    def read(self, size=-1):
        s = super().read(size)
        # try to go around any gzip/bz2/lzma wrappers to the base
        # raw file (does not work for zipfile.ZipExtFile; should
        # dispatch on buffer type)
        try:
            fd = self.buffer.fileno()
        except (AttributeError, io.UnsupportedOperation):
            pos = -1
        else:
            try:
                pos = os.lseek(fd, 0, os.SEEK_CUR)
            except OSError:
                pos = -1

        self.progress_callback(pos, self.__size)
        return s


def index_where(iterable, pred):
    # type: (Iterable[T], Callable[[T], bool]) -> Optional[int]
    """
    Return the (first) index of el in `iterable` where `pred(el)` returns True.

    If no element matches return `None`.
    """
    for i, el in enumerate(iterable):
        if pred(el):
            return i
    return None


def unique(iterable, key=None):
    # type: (Iterable[T], Optional[Callable[[T], Hashable]]) -> Iterable[T]
    """
    Return an iterator over unique elements of `iterable`.

    If `key` is supplied it is used as a substitute for determining
    'uniqueness' of elements.

    Parameters
    ----------
    iterable : Iterable[T]
    key : Callable[[T], Hashable]

    Returns
    -------
    unique : Iterable[T]
    """
    seen = set()
    if key is None:
        key = lambda t: t
    for el in iterable:
        el_k = key(el)
        if el_k not in seen:
            seen.add(el_k)
            yield el


def samepath(p1, p2):
    # type: (str, str) -> bool
    """
    Return True if the paths `p1` and `p2` match after case and path
    normalization.
    """
    return pathnormalize(p1) == pathnormalize(p2)


def pathnormalize(p):
    """
    Normalize a path (apply both path and case normalization.
    """
    return os.path.normcase(os.path.normpath(p))


def prettyfypath(path):
    """
    Return the path with the $HOME prefix shortened to '~/' if applicable.

    Example
    -------
    >>> prettyfypath("/home/user/file.dat")
    '~/file.dat'
    """
    home = os.path.expanduser("~/")
    home_n = pathnormalize(home)
    path_n = pathnormalize(path)
    if path_n.startswith(home_n):
        path = os.path.join("~", os.path.relpath(path, home))
    return path


def pandas_to_table(df):
    # type: (pd.DataFrame) -> Orange.data.Table
    """
    Convert a pandas.DataFrame to a Orange.data.Table instance.
    """
    index = df.index
    if not isinstance(index, pd.RangeIndex):
        df = df.reset_index()

    columns = []  # type: List[Tuple[Orange.data.Variable, np.ndarray]]

    for header, series in df.items():  # type: (Any, pd.Series)
        if pdtypes.is_categorical(series):
            coldata = series.values  # type: pd.Categorical
            categories = [str(c) for c in coldata.categories]
            var = Orange.data.DiscreteVariable.make(
                str(header), values=categories, ordered=coldata.ordered
            )
            # Remap the coldata into the var.values order/set
            coldata = pd.Categorical(
                coldata, categories=var.values, ordered=coldata.ordered
            )
            codes = coldata.codes
            assert np.issubdtype(codes.dtype, np.integer)
            orangecol = np.array(codes, dtype=np.float)
            orangecol[codes < 0] = np.nan
        elif pdtypes.is_datetime64_any_dtype(series):
            # Check that this converts tz local to UTC
            series = series.astype(np.dtype("M8[ns]"))
            coldata = series.values  # type: np.ndarray
            assert coldata.dtype == "M8[ns]"
            mask = np.isnat(coldata)
            orangecol = coldata.astype(np.int64) / 10 ** 9
            orangecol[mask] = np.nan
            var = Orange.data.TimeVariable.make(str(header))
            var.have_date = var.have_time = 1
        elif pdtypes.is_object_dtype(series):
            coldata = series.fillna('').values
            assert isinstance(coldata, np.ndarray)
            orangecol = coldata
            var = Orange.data.StringVariable.make(str(header))
        elif pdtypes.is_integer_dtype(series):
            coldata = series.values
            var = Orange.data.ContinuousVariable.make(str(header))
            var.number_of_decimals = 0
            orangecol = coldata.astype(np.float64)
        elif pdtypes.is_numeric_dtype(series):
            orangecol = series.values.astype(np.float64)
            var = Orange.data.ContinuousVariable.make(str(header))
        else:
            warnings.warn(
                "Column '{}' with dtype: {} skipped."
                .format(header, series.dtype),
                UserWarning
            )
            continue
        columns.append((var, orangecol))

    cols_x = [(var, col) for var, col in columns if var.is_primitive()]
    cols_m = [(var, col) for var, col in columns if not var.is_primitive()]

    variables = [v for v, _ in cols_x]
    if cols_x:
        X = np.column_stack([a for _, a in cols_x])
    else:
        X = np.empty((df.shape[0], 0), dtype=np.float)
    metas = [v for v, _ in cols_m]
    if cols_m:
        M = np.column_stack([a for _, a in cols_m])
    else:
        M = None

    domain = Orange.data.Domain(variables, metas=metas)
    return Orange.data.Table.from_numpy(domain, X, None, M)


def main(argv=None):  # pragma: no cover
    app = QApplication(argv or [])
    w = OWCSVFileImport()
    w.show()
    w.raise_()
    app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))

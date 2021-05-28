# -*- coding: utf-8 -*-
"""
CSV File Import Widget
----------------------

"""
import sys
import types
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
from contextlib import ExitStack

import typing
from typing import (
    List, Tuple, Dict, Optional, Any, Callable, Iterable,
    Union, AnyStr, BinaryIO, Set, Type, Mapping, Sequence, NamedTuple
)

from AnyQt.QtCore import (
    Qt, QFileInfo, QTimer, QSettings, QObject, QSize, QMimeDatabase, QMimeType
)
from AnyQt.QtGui import (
    QStandardItem, QStandardItemModel, QPalette, QColor, QIcon
)
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QDialog, QDialogButtonBox, QGridLayout,
    QVBoxLayout, QSizePolicy, QStyle, QFileIconProvider, QFileDialog,
    QApplication, QMessageBox, QTextBrowser, QMenu
)
from AnyQt.QtCore import pyqtSlot as Slot, pyqtSignal as Signal

import numpy as np
import pandas.errors
import pandas as pd

from pandas.api import types as pdtypes

import Orange.data
from Orange.misc.collections import natural_sorted

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.concurrent import PyOwned
from Orange.widgets.utils import (
    textimport, concurrent as qconcurrent, unique_everseen, enum_get, qname
)
from Orange.widgets.utils.combobox import ItemStyledComboBox
from Orange.widgets.utils.pathutils import (
    PathItem, VarPath, AbsPath, samepath, prettyfypath, isprefixed,
)
from Orange.widgets.utils.overlay import OverlayWidget
from Orange.widgets.utils.settings import (
    QSettings_readArray, QSettings_writeArray
)

if typing.TYPE_CHECKING:
    # pylint: disable=invalid-name
    T = typing.TypeVar("T")
    K = typing.TypeVar("K")
    E = typing.TypeVar("E", bound=enum.Enum)

__all__ = ["OWCSVFileImport"]

_log = logging.getLogger(__name__)

ColumnType = textimport.ColumnType
RowSpec = textimport.RowSpec


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

    def __init__(self, encoding='utf-8', dialect=csv.excel(),
                 columntypes: Iterable[Tuple[range, 'ColumnType']] = (),
                 rowspec=((range(0, 1), RowSpec.Header),),
                 decimal_separator=".", group_separator="") -> None:
        self.encoding = encoding
        self.dialect = dialect
        self.columntypes = list(columntypes)  # type: List[Tuple[range, ColumnType]]
        self.rowspec = list(rowspec)  # type: List[Tuple[range, RowSpec]]
        self.decimal_separator = decimal_separator
        self.group_separator = group_separator

    def __eq__(self, other):
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
        # type: (Iterable[Tuple[range, enum.Enum]]) -> List[Dict[str, Any]]
        return [{"start": r.start, "stop": r.stop, "value": value.name}
                for r, value in spec]

    @staticmethod
    def spec_from_encodable(spec, enumtype):
        # type: (Iterable[Dict[str, Any]], Type[E]) -> List[Tuple[range, E]]
        r = []
        for v in spec:
            try:
                start, stop, name = v["start"], v["stop"], v["value"]
            except (KeyError, ValueError):
                pass
            else:
                r.append((range(start, stop), enum_get(enumtype, name, None)))
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


def icon_for_path(path: str) -> QIcon:
    iconprovider = QFileIconProvider()
    finfo = QFileInfo(path)
    if finfo.exists():
        return iconprovider.icon(finfo)
    else:
        return iconprovider.icon(QFileIconProvider.File)


class VarPathItem(QStandardItem):
    PathRole = Qt.UserRole + 4502
    VarPathRole = PathRole + 1

    def path(self) -> str:
        """Return the resolved path or '' if unresolved or missing"""
        path = self.data(VarPathItem.PathRole)
        return path if isinstance(path, str) else ""

    def setPath(self, path: str) -> None:
        """Set absolute path."""
        self.setData(PathItem.AbsPath(path), VarPathItem.VarPathRole)

    def varPath(self) -> Optional[PathItem]:
        vpath = self.data(VarPathItem.VarPathRole)
        return vpath if isinstance(vpath, PathItem) else None

    def setVarPath(self, vpath: PathItem) -> None:
        """Set variable path item."""
        self.setData(vpath, VarPathItem.VarPathRole)

    def resolve(self, vpath: PathItem) -> Optional[str]:
        """
        Resolve `vpath` item. This implementation dispatches to parent model's
        (:func:`VarPathItemModel.resolve`)
        """
        model = self.model()
        if isinstance(model, VarPathItemModel):
            return model.resolve(vpath)
        else:
            return vpath.resolve({})

    def data(self, role=Qt.UserRole + 1) -> Any:
        if role == Qt.DisplayRole:
            value = super().data(role)
            if value is not None:
                return value
            vpath = self.varPath()
            if isinstance(vpath, PathItem.AbsPath):
                return os.path.basename(vpath.path)
            elif isinstance(vpath, PathItem.VarPath):
                return os.path.basename(vpath.relpath)
            else:
                return None
        elif role == Qt.DecorationRole:
            return icon_for_path(self.path())
        elif role == VarPathItem.PathRole:
            vpath = self.data(VarPathItem.VarPathRole)
            if isinstance(vpath, PathItem.AbsPath):
                return vpath.path
            elif isinstance(vpath, VarPath):
                path = self.resolve(vpath)
                if path is not None:
                    return path
            return super().data(role)
        elif role == Qt.ToolTipRole:
            vpath = self.data(VarPathItem.VarPathRole)
            if isinstance(vpath, VarPath.AbsPath):
                return vpath.path
            elif isinstance(vpath, VarPath):
                text = f"${{{vpath.name}}}/{vpath.relpath}"
                p = self.resolve(vpath)
                if p is None or not os.path.exists(p):
                    text += " (missing)"
                return text
        elif role == Qt.ForegroundRole:
            vpath = self.data(VarPathItem.VarPathRole)
            if isinstance(vpath, PathItem):
                p = self.resolve(vpath)
                if p is None or not os.path.exists(p):
                    return QColor(Qt.red)
        return super().data(role)


class ImportItem(VarPathItem):
    """
    An item representing a file path and associated load options
    """
    OptionsRole = Qt.UserRole + 14
    IsSessionItemRole = Qt.UserRole + 15

    def options(self) -> Optional[Options]:
        options = self.data(ImportItem.OptionsRole)
        return options if isinstance(options, Options) else None

    def setOptions(self, options: Options) -> None:
        self.setData(options, ImportItem.OptionsRole)

    def setIsSessionItem(self, issession: bool) -> None:
        self.setData(issession, ImportItem.IsSessionItemRole)

    def isSessionItem(self) -> bool:
        return bool(self.data(ImportItem.IsSessionItemRole))

    @classmethod
    def fromPath(cls, path: Union[str, PathItem]) -> 'ImportItem':
        """
        Create a `ImportItem` from a local file system path.
        """
        if isinstance(path, str):
            path = PathItem.AbsPath(path)
        if isinstance(path, PathItem.VarPath):
            basename = os.path.basename(path.relpath)
            text = f"${{{path.name}}}/{path.relpath}"
        elif isinstance(path, PathItem.AbsPath):
            basename = os.path.basename(path.path)
            text = path.path
        else:
            raise TypeError

        item = cls()
        item.setText(basename)
        item.setToolTip(text)
        item.setData(path, ImportItem.VarPathRole)
        return item


class VarPathItemModel(QStandardItemModel):
    def __init__(self, *args, replacementEnv=types.MappingProxyType({}),
                 **kwargs):
        self.__replacements = types.MappingProxyType(dict(replacementEnv))
        super().__init__(*args, **kwargs)

    def setReplacementEnv(self, env: Mapping[str, str]) -> None:
        self.__replacements = types.MappingProxyType(dict(env))
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1)
        )

    def replacementEnv(self) -> Mapping[str, str]:
        return self.__replacements

    def resolve(self, vpath: PathItem) -> Optional[str]:
        return vpath.resolve(self.replacementEnv())


def move_item_to_index(model: QStandardItemModel, item: QStandardItem, index: int):
    if item.row() == index:
        return
    assert item.model() is model
    [item_] = model.takeRow(item.row())
    assert item_ is item
    model.insertRow(index, [item])


class FileFormat(NamedTuple):
    mime_type: str
    name: str
    globs: Sequence[str]


FileFormats = [
    FileFormat("text/csv", "Text - comma separated", ("*.csv", "*")),
    FileFormat("text/tab-separated-values", "Text - tab separated", ("*.tsv", "*")),
    FileFormat("text/plain", "Text - all files", ("*.txt", "*")),
]


class FileDialog(QFileDialog):
    __formats: Sequence[FileFormat] = ()

    @staticmethod
    def filterStr(f: FileFormat) -> str:
        return f"{f.name} ({', '.join(f.globs)})"

    def setFileFormats(self, formats: Sequence[FileFormat]):
        filters = [FileDialog.filterStr(f) for f in formats]
        self.__formats = tuple(formats)
        self.setNameFilters(filters)

    def fileFormats(self) -> Sequence[FileFormat]:
        return self.__formats

    def selectedFileFormat(self) -> FileFormat:
        filter_ = self.selectedNameFilter()
        index = index_where(
            self.__formats, lambda f: FileDialog.filterStr(f) == filter_
        )
        return self.__formats[index]


def default_options_for_mime_type(
        path: str, mime_type: str
) -> Options:
    defaults = {
        "text/csv": (csv.excel(), True),
        "text/tab-separated-values": (csv.excel_tab(), True)
    }
    dialect, header, encoding = csv.excel(), True, "utf-8"
    delimiters = None
    try_encodings = ["utf-8", "utf-16", "iso8859-1"]
    if mime_type in defaults:
        dialect, header = defaults[mime_type]
        delimiters = [dialect.delimiter]

    for encoding_ in try_encodings:
        try:
            dialect, header = sniff_csv_with_path(
                path, encoding=encoding_, delimiters=delimiters)
            encoding = encoding_
        except (OSError, UnicodeError, csv.Error):
            pass
        else:
            break
    if header:
        rowspec = [(range(0, 1), RowSpec.Header)]
    else:
        rowspec = []
    return Options(dialect=dialect, encoding=encoding, rowspec=rowspec)


class OWCSVFileImport(widget.OWWidget):
    name = "CSV File Import"
    description = "Import a data table from a CSV formatted file."
    icon = "icons/CSVFile.svg"
    priority = 11
    category = "Data"
    keywords = ["file", "load", "read", "open", "csv"]

    class Outputs:
        data = widget.Output(
            name="Data",
            type=Orange.data.Table,
            doc="Loaded data set.")
        data_frame = widget.Output(
            name="Data Frame",
            type=pd.DataFrame,
            doc="",
            auto_summary=False
        )

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

    _session_items_v2 = settings.Setting(
        [], schema_only=True)  # type: List[Tuple[Dict[str, str], dict]]
    #: Saved dialog state (last directory and selected filter)
    dialog_state = settings.Setting({
        "directory": "",
        "filter": ""
    })  # type: Dict[str, str]

    # we added column type guessing to this widget, which breaks compatibility
    # with older saved workflows, where types not guessed differently, when
    # compatibility_mode=True widget have older guessing behaviour
    settings_version = 3
    compatibility_mode = settings.Setting(False, schema_only=True)

    MaxHistorySize = 50

    want_main_area = False
    buttons_area_orientation = None
    resizing_enabled = False

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.settingsAboutToBePacked.connect(self._saveState)

        self.__committimer = QTimer(self, singleShot=True)
        self.__committimer.timeout.connect(self.commit)

        self.__executor = qconcurrent.ThreadExecutor()
        self.__watcher = None  # type: Optional[qconcurrent.FutureWatcher]

        self.controlArea.layout().setSpacing(-1)  # reset spacing
        grid = QGridLayout()
        grid.addWidget(QLabel("File:", self), 0, 0, 1, 1)

        self.import_items_model = VarPathItemModel(self)
        self.import_items_model.setReplacementEnv(self._replacements())
        self.recent_combo = ItemStyledComboBox(
            self, objectName="recent-combo", toolTip="Recent files.",
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=16, placeholderText="Recent files…"
        )
        self.recent_combo.setModel(self.import_items_model)
        self.recent_combo.activated.connect(self.activate_recent)
        self.recent_combo.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.browse_button = QPushButton(
            "…", icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
            toolTip="Browse filesystem", autoDefault=False,
        )
        # A button drop down menu with selection of explicit workflow dir
        # relative import. This is only enabled when 'basedir' workflow env
        # is set. XXX: Always use menu, disable Import relative... action?
        self.browse_menu = menu = QMenu(self.browse_button)
        ac = menu.addAction("Import any file…")
        ac.triggered.connect(self.browse)

        ac = menu.addAction("Import relative to workflow file…")
        ac.setToolTip("Import a file within the workflow file directory")
        ac.triggered.connect(lambda: self.browse_relative("basedir"))

        if "basedir" in self._replacements():
            self.browse_button.setMenu(menu)

        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.browse_button.clicked.connect(self.browse)
        grid.addWidget(self.recent_combo, 0, 1, 1, 1)
        grid.addWidget(self.browse_button, 0, 2, 1, 1)
        self.controlArea.layout().addLayout(grid)

        ###########
        # Info text
        ###########
        box = gui.widgetBox(self.controlArea, "Info")
        self.summary_text = QTextBrowser(
            verticalScrollBarPolicy=Qt.ScrollBarAsNeeded,
            readOnly=True,
        )
        self.summary_text.viewport().setBackgroundRole(QPalette.NoRole)
        self.summary_text.setFrameStyle(QTextBrowser.NoFrame)
        self.summary_text.setMinimumHeight(self.fontMetrics().ascent() * 2 + 4)
        self.summary_text.viewport().setAutoFillBackground(False)
        box.layout().addWidget(self.summary_text)

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
            self._invalidate()

    def workflowEnvChanged(self, key, value, oldvalue):
        super().workflowEnvChanged(key, value, oldvalue)
        if key == "basedir":
            self.browse_button.setMenu(self.browse_menu)
            self.import_items_model.setReplacementEnv(self._replacements())

    @Slot(int)
    def activate_recent(self, index):
        """
        Activate an item from the recent list.
        """
        model = self.import_items_model
        cb = self.recent_combo
        if 0 <= index < model.rowCount():
            item = model.item(index)
            assert isinstance(item, ImportItem)
            path = item.path()
            item.setData(True, ImportItem.IsSessionItemRole)
            move_item_to_index(model, item, 0)
            if not os.path.exists(path):
                self._browse_for_missing(
                    item, onfinished=lambda status: self._invalidate()
                )
            else:
                cb.setCurrentIndex(0)
                self._invalidate()
        else:
            self.recent_combo.setCurrentIndex(-1)

    def _browse_for_missing(
            self, item: ImportItem, *, onfinished: Optional[Callable[[int], Any]] = None):
        dlg = self._browse_dialog()
        model = self.import_items_model

        if onfinished is None:
            onfinished = lambda status: None

        vpath = item.varPath()
        prefixpath = None
        if isinstance(vpath, PathItem.VarPath):
            prefixpath = self._replacements().get(vpath.name)
        if prefixpath is not None:
            dlg.setDirectory(prefixpath)
        dlg.setAttribute(Qt.WA_DeleteOnClose)

        def accepted():
            path = dlg.selectedFiles()[0]
            if isinstance(vpath, VarPath) and not isprefixed(prefixpath, path):
                mb = self._path_must_be_relative_mb(prefixpath)
                mb.show()
                mb.finished.connect(lambda _: onfinished(QDialog.Rejected))
                return

            # pre-flight check; try to determine the nature of the file
            mtype = _mime_type_for_path(path)
            if not mtype.inherits("text/plain"):
                mb = self._might_be_binary_mb(path)
                if mb.exec() == QMessageBox.Cancel:
                    if onfinished:
                        onfinished(QDialog.Rejected)
                    return

            if isinstance(vpath, VarPath):
                vpath_ = VarPath(vpath.name, os.path.relpath(path, prefixpath))
            else:
                vpath_ = AbsPath(path)
            item.setVarPath(vpath_)
            if item.row() != 0:
                move_item_to_index(model, item, 0)
            item.setData(True, ImportItem.IsSessionItemRole)
            self.set_selected_file(path, item.options())
            self._note_recent(path, item.options())
            onfinished(QDialog.Accepted)

        dlg.accepted.connect(accepted)
        dlg.open()

    def _browse_dialog(self):
        dlg = FileDialog(
            self, windowTitle=self.tr("Open Data File"),
            acceptMode=QFileDialog.AcceptOpen,
            fileMode=QFileDialog.ExistingFile
        )

        dlg.setFileFormats(FileFormats)
        state = self.dialog_state
        lastdir = state.get("directory", "")
        lastfilter = state.get("filter", "")
        if lastdir and os.path.isdir(lastdir):
            dlg.setDirectory(lastdir)
        if lastfilter:
            dlg.selectNameFilter(lastfilter)

        def store_state():
            state["directory"] = dlg.directory().absolutePath()
            state["filter"] = dlg.selectedNameFilter()
        dlg.accepted.connect(store_state)
        return dlg

    def _might_be_binary_mb(self, path) -> QMessageBox:
        mb = QMessageBox(
            parent=self,
            windowTitle=self.tr(""),
            icon=QMessageBox.Question,
            text=self.tr("The '{basename}' may be a binary file.\n"
                         "Are you sure you want to continue?").format(
                             basename=os.path.basename(path)),
            standardButtons=QMessageBox.Cancel | QMessageBox.Yes
        )
        mb.setWindowModality(Qt.WindowModal)
        return mb

    def _path_must_be_relative_mb(self, prefix: str) -> QMessageBox:
        mb = QMessageBox(
            parent=self, windowTitle=self.tr("Invalid path"),
            icon=QMessageBox.Warning,
            text=self.tr("Selected path is not within '{prefix}'").format(
                prefix=prefix
            ),
        )
        mb.setAttribute(Qt.WA_DeleteOnClose)
        return mb

    @Slot(str)
    def browse_relative(self, prefixname):
        path = self._replacements().get(prefixname)
        self.browse(prefixname=prefixname, directory=path)

    @Slot()
    def browse(self, prefixname=None, directory=None):
        """
        Open a file dialog and select a user specified file.
        """
        dlg = self._browse_dialog()
        if directory is not None:
            dlg.setDirectory(directory)

        status = dlg.exec()
        dlg.deleteLater()
        if status == QFileDialog.Accepted:
            selected_filter = dlg.selectedFileFormat()
            path = dlg.selectedFiles()[0]
            if prefixname:
                _prefixpath = self._replacements().get(prefixname, "")
                if not isprefixed(_prefixpath, path):
                    mb = self._path_must_be_relative_mb(_prefixpath)
                    mb.show()
                    return
                varpath = VarPath(prefixname, os.path.relpath(path, _prefixpath))
            else:
                varpath = PathItem.AbsPath(path)

            # pre-flight check; try to determine the nature of the file
            mtype = _mime_type_for_path(path)
            if not mtype.inherits("text/plain"):
                mb = self._might_be_binary_mb(path)
                if mb.exec() == QMessageBox.Cancel:
                    return
            # initialize options based on selected format
            options = default_options_for_mime_type(
                path, selected_filter.mime_type,
            )
            # Search for path in history.
            # If found use the stored params to initialize the import dialog
            items = self.itemsFromSettings()
            idx = index_where(items, lambda t: samepath(t[0], path))
            if idx is not None:
                _, options_ = items[idx]
                if options_ is not None:
                    options = options_
            dlg = CSVImportDialog(
                self, windowTitle="Import Options", sizeGripEnabled=True)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setPath(path)
            dlg.setOptions(options)
            status = dlg.exec()
            dlg.deleteLater()
            if status == QDialog.Accepted:
                self.set_selected_file(path, dlg.options())
                self.current_item().setVarPath(varpath)

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
        settings = self._local_settings()
        settings.beginGroup(qname(type(dlg)))
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
            # update local recent paths list
            self._note_recent(path, newoptions)
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

        # item.setData(VarPath(filename), ImportItem.VarPathRole)
        item.setData(True, ImportItem.IsSessionItemRole)
        model.insertRow(0, item)

        if options is not None:
            item.setOptions(options)

        self.recent_combo.setCurrentIndex(0)

        if not os.path.exists(filename):
            return
        self._note_recent(filename, options)

    def _note_recent(self, filename, options):
        # store item to local persistent settings
        s = self._local_settings()
        arr = QSettings_readArray(s, "recent", OWCSVFileImport.SCHEMA)
        item = {"path": filename}
        if options is not None:
            item["options"] = json.dumps(options.as_dict())
        arr = [item for item in arr if not samepath(item.get("path"), filename)]
        arr.append(item)
        QSettings_writeArray(s, "recent", arr)

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
        path = item.path()
        opts = item.options()
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
            path, opts, progress_, self.compatibility_mode
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
            filename = self.current_item().path()
            table.name = os.path.splitext(os.path.split(filename)[-1])[0]
        else:
            table = None
        self.Outputs.data_frame.send(df)
        self.Outputs.data.send(table)
        self._update_status_messages(table)

    def _update_status_messages(self, data):
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

    def _replacements(self) -> Mapping[str, str]:
        replacements = []
        basedir = self.workflowEnv().get("basedir", None)
        if basedir is not None:
            replacements += [('basedir', basedir)]
        return dict(replacements)

    def _saveState(self):
        session_items = []
        model = self.import_items_model
        for item in map(model.item, range(model.rowCount())):
            if isinstance(item, ImportItem) and item.data(ImportItem.IsSessionItemRole):
                vp = item.data(VarPathItem.VarPathRole)
                session_items.append((vp.as_dict(), item.options().as_dict()))
        self._session_items_v2 = session_items

    def _restoreState(self):
        # Restore the state. Merge session (workflow) items with the
        # local history.
        model = self.import_items_model
        model.setReplacementEnv(self._replacements())

        # local history
        items = self.itemsFromSettings()
        # stored session items
        sitems = []
        # replacements = self._replacements()
        for p, m in self._session_items_v2:
            try:
                p, m = (PathItem.from_dict(p), Options.from_dict(m))
            except (csv.Error, LookupError, ValueError):
                _log.error("Failed to restore '%s'", p, exc_info=True)
            else:
                sitems.append((p, m, True))

        items = sitems + [(PathItem.AbsPath(p), m, False) for p, m in items]
        items = unique_everseen(items, key=lambda t: t[0])
        curr = self.recent_combo.currentIndex()
        if curr != -1:
            currentpath = self.recent_combo.currentData(ImportItem.PathRole)
        else:
            currentpath = None

        for path, options, is_session in items:
            item = ImportItem.fromPath(path)
            item.setOptions(options)
            item.setData(is_session, ImportItem.IsSessionItemRole)
            model.appendRow(item)

        if currentpath:
            idx = self.recent_combo.findData(currentpath, ImportItem.PathRole)
        elif model.data(model.index(0, 0), ImportItem.IsSessionItemRole):
            # restore last (current) session item
            idx = 0
        else:
            idx = -1
        self.recent_combo.setCurrentIndex(idx)

    @classmethod
    def migrate_settings(cls, settings, version):
        if not version or version < 2:
            settings["compatibility_mode"] = True

        if version is not None and version < 3:
            items_ = settings.pop("_session_items", [])
            items_v2 = [(PathItem.AbsPath(p).as_dict(), m) for p, m in items_]
            settings["_session_items_v2"] = items_v2


@singledispatch
def sniff_csv(file, samplesize=2 ** 20, delimiters=None):
    sniffer = csv.Sniffer()
    sample = file.read(samplesize)
    dialect = sniffer.sniff(sample, delimiters=delimiters)
    dialect = textimport.Dialect(
        dialect.delimiter, dialect.quotechar,
        dialect.escapechar, dialect.doublequote,
        dialect.skipinitialspace, dialect.quoting
    )
    has_header = HeaderSniffer(dialect).has_header(sample)
    return dialect, has_header


class HeaderSniffer(csv.Sniffer):
    def __init__(self, dialect: csv.Dialect):
        super().__init__()
        self.dialect = dialect

    def sniff(self, *_args, **_kwargs):  # pylint: disable=signature-differs
        # return fixed constant dialect, has_header sniffs dialect itself,
        # so it can't detect headers for a predefined dialect
        return self.dialect


@sniff_csv.register(str)
@sniff_csv.register(bytes)
def sniff_csv_with_path(path, encoding="utf-8", samplesize=2 ** 20, delimiters=None):
    with _open(path, "rt", encoding=encoding) as f:
        return sniff_csv(f, samplesize, delimiters)


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
            f = arh.open(filelist[0], 'r')
            # patch the f.close to also close the main archive file
            f_close = f.close

            def close_():
                f_close()
                arh.close()
            f.close = close_
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


def load_csv(path, opts, progress_callback=None, compatibility_mode=False):
    # type: (Union[AnyStr, BinaryIO], Options, Optional[Callable[[int, int], None]], bool) -> pd.DataFrame
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

    if numbers_format_kwds:
        # float_precision = "round_trip" cannot handle non c-locale decimal and
        # thousands sep (https://github.com/pandas-dev/pandas/issues/35365).
        # Fallback to 'high'.
        numbers_format_kwds["float_precision"] = "high"
    else:
        numbers_format_kwds["float_precision"] = "round_trip"

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
            **numbers_format_kwds
        )

        # for older workflows avoid guessing type guessing
        if not compatibility_mode:
            df = guess_types(df, dtypes, columns_ignored)

        if columns_ignored:
            # TODO: use 'usecols' parameter in `read_csv` call to
            # avoid loading/parsing the columns
            df.drop(
                columns=[df.columns[i] for i in columns_ignored
                         if i < len(df.columns)],
                inplace=True
            )
        return df


def guess_types(
        df: pd.DataFrame, dtypes: Dict[int, str], columns_ignored: Set[int]
) -> pd.DataFrame:
    """
    Guess data type for variables according to values.

    Parameters
    ----------
    df
        Data frame
    dtypes
        The dictionary with data types set by user. We will guess values only
        for columns that does not have data type defined.
    columns_ignored
        List with indices of ignored columns. Ignored columns are skipped.

    Returns
    -------
    A data frame with changed dtypes according to the strategy.
    """
    for i, col in enumerate(df):
        # only when automatic is set in widget dialog
        if dtypes.get(i, None) is None and i not in columns_ignored:
            df[col] = guess_data_type(df[col])
    return df


def guess_data_type(col: pd.Series) -> pd.Series:
    """
    Guess column types. Logic is same than in guess_data_type from io_utils
    module. This function only change the dtype of the column such that later
    correct Orange.data.variable is used.
    Logic:
    - if can converted to date-time (ISO) -> TimeVariable
    - if numeric (only numbers)
        - only values {0, 1} or {1, 2} -> DiscreteVariable
        - else -> ContinuousVariable
    - if not numbers:
        - num_unique_values < len(data) ** 0.7 and < 100 -> DiscreteVariable
        - else -> StringVariable

    Parameters
    ----------
    col
        Data column

    Returns
    -------
    Data column with correct dtype
    """
    def parse_dates(s):
        """
        This is an extremely fast approach to datetime parsing.
        For large data, the same dates are often repeated. Rather than
        re-parse these, we store all unique dates, parse them, and
        use a lookup to convert all dates.
        """
        try:
            dates = {date: pd.to_datetime(date) for date in s.unique()}
        except ValueError:
            return None
        return s.map(dates)

    if pdtypes.is_numeric_dtype(col):
        unique_values = col.unique()
        if len(unique_values) <= 2 and (
                len(np.setdiff1d(unique_values, [0, 1])) == 0
                or len(np.setdiff1d(unique_values, [1, 2])) == 0):
            return col.astype("category")
    else:  # object
        # try parse as date - if None not a date
        parsed_col = parse_dates(col)
        if parsed_col is not None:
            return parsed_col
        unique_values = col.unique()
        if len(unique_values) < 100 and len(unique_values) < len(col)**0.7:
            return col.astype("category")
    return col


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


class TaskState(QObject, PyOwned):
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
        if pdtypes.is_categorical_dtype(series):
            coldata = series.values  # type: pd.Categorical
            categories = natural_sorted(str(c) for c in coldata.categories)
            var = Orange.data.DiscreteVariable.make(
                str(header), values=categories
            )
            # Remap the coldata into the var.values order/set
            coldata = pd.Categorical(
                coldata.astype("str"), categories=var.values
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
    app.exec()
    w.saveSettings()
    w.onDeleteWidget()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))

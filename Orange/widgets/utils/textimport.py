"""
Utility widgets/helper for reading csv like files.

Contents
--------

* CSVOptionsWidget
  Edit options for interpreting a csv file

* CSVImportWidget
  Read and preview part of the file

* TablePreviewModel
  An QAbstractTableModel feeding data from a csv.reader like rows iterator
  implementing lazy iterative loading (`QAbstractItemModel.fetchMore`)

"""
# TODO: Consider a wizard-like interface:
#   * 1. Select encoding, delimiter, ... (preview is all text)
#   * 2. Define column types (preview is parsed and rendered type appropriate)

import sys
import io
import enum
import codecs
import csv
import traceback
import itertools

from functools import singledispatch
from collections import defaultdict
from types import MappingProxyType

import typing
from typing import (
    List, Tuple, Dict, Iterator, Optional, Any, Union, Callable, Mapping
)

from AnyQt.QtCore import (
    Qt, QSize, QPoint, QRect, QRectF, QRegularExpression, QAbstractTableModel,
    QModelIndex, QItemSelectionModel, QTextBoundaryFinder, QTimer, QEvent
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtGui import (
    QRegularExpressionValidator, QColor, QBrush, QPalette, QHelpEvent,
    QStandardItemModel, QStandardItem, QIcon, QIconEngine, QPainter, QPixmap,
    QFont
)
from AnyQt.QtWidgets import (
    QWidget, QComboBox, QFormLayout, QHBoxLayout, QVBoxLayout, QLineEdit,
    QHeaderView, QFrame, QTableView, QMenu, QLabel, QAction, QActionGroup,
    QStyleOptionFrame, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
    QApplication, QAbstractItemView, QToolTip, QStyleOption
)

from Orange.widgets.utils import encodings
from Orange.widgets.utils.overlay import OverlayWidget


__all__ = ["ColumnType", "RowSpec", "CSVOptionsWidget", "CSVImportWidget"]

if typing.TYPE_CHECKING:
    # pylint: disable=invalid-name
    _A = typing.TypeVar("_A")
    _B = typing.TypeVar("_B")


class StampIconEngine(QIconEngine):
    def __init__(self, char, brush):
        # type: (str, Union[QBrush, QColor]) -> None
        super().__init__()
        self.__char = char
        self.__brush = QBrush(brush)

    def paint(self, painter, rect, mode, state):
        # type: (QPainter, QRect, QIcon.Mode, QIcon.State) -> None
        size = rect.size()
        if size.isNull():
            return  # pragma: no cover
        dpr = 1.0
        try:
            dpr = painter.device().devicePixelRatioF()
        except AttributeError:
            pass
        if dpr != 1.0:
            size = size * dpr
        painter.drawPixmap(rect, self.pixmap(size, mode, state))

    def pixmap(self, size, mode, state):
        # type: (QSize, QIcon.Mode, QIcon.State) -> QPixmap
        pm = QPixmap(size)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing |
            QPainter.SmoothPixmapTransform
        )
        size = size.width()
        color = self.__brush.color()
        painter.setPen(color)
        painter.setBrush(color)
        margin = 1 + size // 16
        text_margin = size // 20
        rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        painter.drawRoundedRect(rect, 30.0, 30.0, Qt.RelativeSize)
        painter.setPen(Qt.white)

        font = painter.font()  # type: QFont
        font.setPixelSize(size - 2 * margin - 2 * text_margin)
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(rect, Qt.AlignCenter, self.__char)
        painter.end()

        style = QApplication.style()
        if style is not None:
            opt = QStyleOption()
            opt.palette = QApplication.palette()
            pm = style.generatedIconPixmap(mode, pm, opt)
        return pm


class Dialect(csv.Dialect):
    def __init__(self, delimiter, quotechar, escapechar, doublequote,
                 skipinitialspace, quoting=csv.QUOTE_MINIMAL):
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.escapechar = escapechar
        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace
        self.quoting = quoting
        self.lineterminator = "\r\n"
        super().__init__()

    def __repr__(self):
        _, args, *_ = self.__reduce__()
        args = ", ".join(map("{!r}".format, args))
        return "Dialect(" + args + ")"

    def __reduce__(self):
        return type(self), (self.delimiter, self.quotechar, self.escapechar,
                            self.doublequote, self.skipinitialspace,
                            self.quoting)


class ColumnType(enum.Enum):
    """
    Possible column types
    """
    # Skip column
    Skip = "Skip"
    # Autodetect column type
    Auto = "Auto"
    # Numeric (real) column
    Numeric = "Numeric"
    # Categorical column
    Categorical = "Categorical"
    # Text column
    Text = "Text"
    # Date time column
    Time = "Time"


class LineEdit(QLineEdit):
    """
    A line edit widget with a `minimumContentsLength` property.

    Similar to QComboBox.minimumContentsLength
    """
    # These constants are taken from Qt's sources for QLineEdit
    _verticalMargin = 1
    _horizontalMargin = 2

    def __init__(self, *args, minimumContentsLength=0, **kwargs):
        self.__minimumContentsLength = minimumContentsLength
        super().__init__(*args, **kwargs)

    def setMinimumContentsLength(self, characters):
        # type: (int) -> None
        """
        Set the minimum number of characters that should fit into the
        line edit (used for size hinting).
        """
        if self.__minimumContentsLength != characters:
            self.__minimumContentsLength = characters
            self.updateGeometry()

    def minimumContentsLength(self):
        # type: () -> int
        """
        Return the minimum number of characters that should fit into the
        line edit.
        """
        return self.__minimumContentsLength

    def sizeHint(self):
        # type: () -> QSize
        """Reimplemented."""
        # Most of this is taken from QLineEdit's sources, the difference
        # is only in the use of minimumContentsLength instead of a
        # hardcoded constant.
        self.ensurePolished()
        fm = self.fontMetrics()
        textmargins = self.textMargins()
        contentsmargins = self.contentsMargins()

        h = (max(fm.height(), 14) + 2 * self._verticalMargin +
             textmargins.top() + textmargins.bottom() +
             contentsmargins.top() + contentsmargins.bottom())

        nchar = self.__minimumContentsLength
        if nchar <= 0:
            nchar = 17

        w = (fm.horizontalAdvance("X") * nchar + 2 * self._horizontalMargin +
             textmargins.left() + textmargins.right() +
             contentsmargins.left() + contentsmargins.right())

        opt = QStyleOptionFrame()
        self.initStyleOption(opt)
        size = self.style().sizeFromContents(
            QStyle.CT_LineEdit, opt, QSize(w, h), self
        )
        return size

    def minimumSizeHint(self):
        # type: () -> QSize
        """Reimplemented."""
        if self.__minimumContentsLength > 0:
            return self.sizeHint()
        else:
            return super(LineEdit, self).sizeHint()


class TextEditCombo(QComboBox):
    def text(self):
        # type: () -> str
        """
        Return the current text.
        """
        return self.itemText(self.currentIndex())

    def setText(self, text):
        # type: (str) -> None
        """
        Set `text` as the current text (adding it to the model if necessary).
        """
        idx = self.findData(text, Qt.EditRole, Qt.MatchExactly)
        if idx != -1:
            self.setCurrentIndex(idx)
        else:
            self.addItem(text)
            self.setCurrentIndex(self.count() - 1)


class CSVOptionsWidget(QWidget):
    """
    A widget presenting common CSV options.
    """
    DelimiterTab, DelimiterComma, DelimiterSemicolon, DelimiterSpace = range(4)
    DelimiterOther = DelimiterSpace + 2  # note DelimiterSpace + 1 is reserved

    PresetDelimiters = [
        ("Tab", "\t"),
        ("Comma", ","),
        ("Semicolon", ";"),
        ("Space", " "),
    ]

    #: Signal emitted when the format (dialect) changes
    optionsChanged = Signal()
    #: Signal emitted when the format (dialect) is edited by the user
    optionsEdited = Signal()

    def __init__(self, *args, **kwargs):
        self._delimiter_idx = 0
        self._delimiter = ","
        self._delimiter_custom = "|"
        self._quotechar = "\""
        self._encoding = kwargs.pop("selectedEncoding", "utf-8")

        super().__init__(*args, **kwargs)

        # Dialect options form
        form = QFormLayout()
        self.encoding_cb = QComboBox(
            objectName="encoding-combo-box",
            toolTip="Select file text encoding",
        )
        self.__set_visible_codecs(encodings.list_selected_encodings())
        self.encoding_cb.activated.connect(self.__on_encoding_activated)

        self.delimiter_cb = QComboBox(
            objectName="delimiter-combo-box",
            toolTip="Select cell delimiter character."
        )
        self.delimiter_cb.addItems(
            [name for name, _ in CSVOptionsWidget.PresetDelimiters]
        )
        self.delimiter_cb.insertSeparator(self.delimiter_cb.count())
        self.delimiter_cb.addItem("Other")

        self.delimiter_cb.setCurrentIndex(self._delimiter_idx)
        self.delimiter_cb.activated.connect(self.__on_delimiter_idx_activated)

        validator = QRegularExpressionValidator(QRegularExpression("."))
        self.delimiteredit = LineEdit(
            self._delimiter_custom,
            enabled=self._delimiter_idx == CSVOptionsWidget.DelimiterOther,
            minimumContentsLength=2,
            objectName="custom-delimiter-edit"
        )
        self.delimiteredit.setValidator(validator)
        self.delimiteredit.editingFinished.connect(self.__on_delimiter_edited)

        delimlayout = QHBoxLayout()
        delimlayout.setContentsMargins(0, 0, 0, 0)
        delimlayout.addWidget(self.delimiter_cb)
        delimlayout.addWidget(self.delimiteredit)
        self.quoteedit = TextEditCombo(
            editable=True, minimumContentsLength=1,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            objectName="quote-edit-combo-box"
        )
        self.quoteedit.addItems(["\"", "'"])
        self.quoteedit.setValidator(validator)
        self.quoteedit.setText(self._quotechar)
        self.quoteedit.activated.connect(self.__on_quotechar_edited)

        quotelayout = QHBoxLayout()
        quotelayout.setContentsMargins(0, 0, 0, 0)
        quotelayout.addWidget(self.quoteedit)

        form.addRow("Encoding", self.encoding_cb)
        form.addRow(QFrame(self, frameShape=QFrame.HLine))
        form.addRow("Cell delimiter", delimlayout)
        form.addRow("Quote character", self.quoteedit)
        self.setLayout(form)

    def dialect(self):
        # type: () -> csv.Dialect
        """
        Return the current state as a `csv.Dialect` instance.
        """
        if self._delimiter_idx == CSVOptionsWidget.DelimiterOther:
            delimiter = self._delimiter_custom
        elif 0 <= self._delimiter_idx < len(CSVOptionsWidget.PresetDelimiters):
            _, delimiter = CSVOptionsWidget.PresetDelimiters[self._delimiter_idx]
        else:
            assert False

        quotechar = self.quoteedit.text() or None
        skipinitialspace = True
        escapechar = None
        quoting = csv.QUOTE_MINIMAL if quotechar is not None else csv.QUOTE_NONE
        return Dialect(delimiter, quotechar, escapechar,
                       doublequote=True, skipinitialspace=skipinitialspace,
                       quoting=quoting)

    def setDialect(self, dialect):
        # type: (csv.Dialect) -> None
        """
        Set the current state to match dialect instance.
        """
        changed = False
        delimiter = dialect.delimiter
        presets = [d for _, d in CSVOptionsWidget.PresetDelimiters]
        try:
            index = presets.index(delimiter)
        except ValueError:
            index = CSVOptionsWidget.DelimiterOther
            if self._delimiter_custom != delimiter:
                self._delimiter_custom = delimiter
                changed = True

        if self._delimiter_idx != index:
            self._delimiter_idx = index
            self.delimiter_cb.setCurrentIndex(index)
            self.delimiteredit.setText(delimiter)
            changed = True
        if self._quotechar != dialect.quotechar:
            self._quotechar = dialect.quotechar
            self.quoteedit.setText(dialect.quotechar or '')
            changed = True

        if changed:
            self.optionsChanged.emit()

    def setSelectedEncoding(self, encoding):
        # type: (str) -> None
        """
        Set the current selected encoding.

        Parameters
        ----------
        encoding : str
            Encoding name such that `codecs.lookup` finds it.
        """
        co = codecs.lookup(encoding)
        cb = self.encoding_cb
        index = cb.findData(co.name, Qt.UserRole)

        if index == -1:
            # insert the encoding before the separator.
            sepidx = cb.findData("separator", Qt.AccessibleDescriptionRole)
            if sepidx == -1:
                sepidx = cb.count()
            cb.insertItem(sepidx, encodings.display_name(co.name),
                          userData=co.name)
            index = sepidx
            assert cb.itemData(index, Qt.UserRole) == co.name
            changed = True
            self._encoding = encoding
        else:
            changed = index != self.encoding_cb.currentIndex()
            self._encoding = encoding

        self.encoding_cb.setCurrentIndex(index)

        if changed:
            self.optionsChanged.emit()

    def encoding(self):
        # type: () -> str
        """
        Return the current selected encoding.
        """
        index = self.encoding_cb.currentIndex()
        if index >= 0:
            data = self.encoding_cb.itemData(index, Qt.UserRole)
            if isinstance(data, str):
                return data
        return "latin-1"

    def __on_encoding_activated(self, idx):
        current = self._encoding
        data = self.encoding_cb.itemData(idx, Qt.UserRole)
        if data is ...:
            # restore the previous item
            idx = self.encoding_cb.findData(current, Qt.UserRole)
            self.encoding_cb.setCurrentIndex(idx)
            self.__show_encodings_widget()
        elif isinstance(data, str):
            assert codecs.lookup(data)
            self._encoding = data
            self.optionsEdited.emit()
            self.optionsChanged.emit()

    def __show_encodings_widget(self):
        """
        Show the encodings widget for selection
        """
        # If tool window is already shown just raise it
        w = self.findChild(
            encodings.SelectEncodingsWidget,
            "-encoding-selection-tool-window"
        )
        if w is not None and w.isVisible():  # pragma: no coverage
            w.raise_()
            w.activateWindow()
            return

        w = encodings.SelectEncodingsWidget(
            self, Qt.Tool,
            windowTitle="Customize Encodings List",
            objectName="-encoding-selection-tool-window"
        )
        w.setAttribute(Qt.WA_DeleteOnClose)
        model = w.model()
        model.dataChanged.connect(
            lambda: self.__set_visible_codecs(w.selectedEncodings())
        )
        w.show()

    def __set_visible_codecs(self, codecs):
        # type: (List[str]) -> None
        # Set the list of current visible/selectable codecs in the encoding_cb
        if not codecs:
            # never clear all items from the drop down
            codecs = ["ascii"]
        cb = self.encoding_cb
        current = self._encoding
        cb.clear()

        for c in codecs:
            cb.addItem(encodings.display_name(c), userData=c)
        cb.insertSeparator(cb.count())
        cb.addItem("Customize Encodings List...", userData=...)
        idx = cb.findData(current, Qt.UserRole)
        if idx != -1:
            cb.setCurrentIndex(idx)
        elif codecs:
            cb.setCurrentIndex(0)
            self._encoding = codecs[0]
            self.__on_encoding_activated(0)
        else:
            cb.setCurrentIndex(-1)
            self._encoding = ""

    def __on_delimiter_idx_activated(self, index):
        presets = CSVOptionsWidget.PresetDelimiters
        if 0 <= index < CSVOptionsWidget.DelimiterOther:
            self.delimiteredit.setText(presets[index][1])
            self.delimiteredit.setEnabled(False)
        else:
            self.delimiteredit.setText(self._delimiter_custom)
            self.delimiteredit.setEnabled(True)

        if self._delimiter_idx != index:
            self._delimiter_idx = index
            self.optionsChanged.emit()
            self.optionsEdited.emit()

    def __on_delimiter_edited(self):
        delimiter = self.delimiteredit.text()
        if self._delimiter_custom != delimiter:
            self._delimiter_custom = delimiter
            self.optionsChanged.emit()
            self.optionsEdited.emit()

    def __on_quotechar_edited(self):
        quotechar = self.quoteedit.text()
        if self._quotechar != quotechar:
            self._quotechar = quotechar
            self.optionsChanged.emit()
            self.optionsEdited.emit()


class Item(QStandardItem):
    """
    A QStandardItem subclass using a python dict as a backing store.

    Note
    ----
    Unlike QStandardItem, this class does not map `Qt.DisplayRole` and
    `Qt.EditRole` to the same value. Also, accessing or setting via
    `model.itemData` `model.setItemData` and will not work.
    """
    def __init__(self, data=MappingProxyType({})):
        # type: (Mapping[Qt.ItemDataRole, Any]) -> None
        super().__init__()
        self.__data = dict(data)

    def clone(self):
        # type: () -> Item
        """Reimplemented from QStandardItem"""
        return Item(self.__data)

    def setData(self, value, role=Qt.UserRole+1):
        # type: (Any, Qt.ItemDataRole) -> None
        """Reimplemented from QStandardItem"""
        self.__data[role] = value
        model = self.model()
        if model is not None:
            midx = model.indexFromItem(self)
            model.dataChanged.emit(midx, midx, [role])

    def data(self, role=Qt.UserRole+1):
        # type: (Qt.ItemDataRole) -> Any
        """Reimplemented from QStandardItem"""
        if role == Qt.EditRole and role not in self.__data:
            role = Qt.DisplayRole
        elif role == Qt.DisplayRole and role not in self.__data:
            role = Qt.EditRole
        return self.__data.get(role, None)

    def setItemData(self, data):
        # type: (Dict[Qt.ItemDataRole, Any]) -> bool
        roles = list(data.keys())
        self.__data.update(data)
        m = self.model()  #
        if m is not None:
            midx = m.indexFromItem(self)
            m.dataChanged.emit(midx, midx, roles)
        return True

    def itemData(self):
        return self.__data.copy()


class CSVImportWidget(QWidget):
    """
    CSV import widget with a live table preview
    """
    #: Signal emitted on any format option change.
    optionsChanged = Signal()
    #: Signal emitted when a user changes format options.
    optionsEdited = Signal()
    #: Signal emitted when a user changes type affiliation for a column
    columnTypesChanged = Signal()
    #: Signal emitted when the preview content parsing ends with an error.
    #: Note: this does not include errors in cell content interpretation.
    previewReadErrorOccurred = Signal(str)
    #: Signal emitted when the preview model is reset. This is either because
    #: of `setPreviewContents` or a options change.
    previewModelReset = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__previewmodel = None  # type: Optional[TablePreviewModel]
        self.__textwrapper = None  # type: Optional[io.TextIOWrapper]
        self.__sample = None
        self.__buffer = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.optionswidget = CSVOptionsWidget()
        self.optionswidget.optionsChanged.connect(self.optionsChanged)
        self.optionswidget.optionsEdited.connect(self.optionsEdited)

        self.dataview = TablePreview(
            selectionBehavior=QTableView.SelectColumns,
            tabKeyNavigation=False,
        )
        self.dataview.setContextMenuPolicy(Qt.CustomContextMenu)
        self.dataview.customContextMenuRequested.connect(
            self.__dataview_context_menu
        )
        header = self.dataview.horizontalHeader()  # type: QHeaderView
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(
            self.__hheader_context_menu
        )

        header = self.dataview.verticalHeader()
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(
            self.__vheader_context_menu
        )
        style = self.style()
        opt = self.dataview.viewOptions()
        opt.text = "X"
        opt.features |= QStyleOptionViewItem.HasDisplay
        csize = style.sizeFromContents(
            QStyle.CT_ItemViewItem, opt, QSize(18, 18), self.dataview
        )
        header.ensurePolished()
        header.setDefaultSectionSize(max(csize.height(),
                                         header.minimumSectionSize()))
        layout.addWidget(self.optionswidget)
        form = self.optionswidget.layout()
        assert isinstance(form, QFormLayout)
        number_sep_layout = QHBoxLayout()
        self.grouping_sep_edit_cb = TextEditCombo(
            editable=True, objectName="grouping-separator-combo-box",
            toolTip="Thousands group separator",
            minimumContentsLength=1,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        items = [
            {Qt.DisplayRole: "None", Qt.EditRole: "",
             Qt.ToolTipRole: "No separator"},
            {Qt.DisplayRole: ".", Qt.EditRole: "."},
            {Qt.DisplayRole: ",", Qt.EditRole: ","},
            {Qt.DisplayRole: "Space", Qt.EditRole: " "},
            {Qt.DisplayRole: "'", Qt.EditRole: "'"},
        ]
        m = QStandardItemModel(self)
        m.invisibleRootItem().appendRows([Item(data) for data in items])
        self.grouping_sep_edit_cb.setModel(m)
        # TODO: Treat all space (THIN SPACE, NO-BREAK SPACE, ...) the same?
        # For instance Apple's Numbers exports csv with \N{NO-BREAK SPACE}.
        # Maybe just use unicodedata.normalize('NFKC', ...) as a converter?
        # For now only allow a limited set
        self.grouping_sep_edit_cb.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"(\.|,| |')?"), self)
        )
        self.grouping_sep_edit_cb.activated[str].connect(
            self.__group_sep_activated)

        self.decimal_sep_edit_cb = TextEditCombo(
            editable=True, objectName="decimal-separator-combo-box",
            toolTip="Decimal separator",
            minimumContentsLength=1,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.decimal_sep_edit_cb.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"(\.|,)"), self))
        self.decimal_sep_edit_cb.addItems([".", ","])
        self.decimal_sep_edit_cb.activated[str].connect(
            self.__decimal_sep_activated)

        number_sep_layout.addWidget(QLabel("Grouping:"))
        number_sep_layout.addWidget(self.grouping_sep_edit_cb)
        number_sep_layout.addWidget(QLabel("Decimal:"))
        number_sep_layout.addWidget(self.decimal_sep_edit_cb)
        number_sep_layout.addStretch(10)
        form.addRow("Number separators:", number_sep_layout)
        self.column_type_edit_cb = QComboBox(
            enabled=False, objectName="column-type-edit-combo-box"
        )
        self.column_type_edit_cb.activated.connect(
            self.__on_column_type_edit_activated
        )
        types = [
            {Qt.DisplayRole: "Auto",
             Qt.ToolTipRole: "The type will be determined automatically based "
                             "on column contents.",
             Qt.UserRole: ColumnType.Auto},
            {Qt.DisplayRole: "Numeric", Qt.UserRole: ColumnType.Numeric},
            {Qt.DisplayRole: "Categorical",
             Qt.UserRole: ColumnType.Categorical},
            {Qt.DisplayRole: "Text", Qt.UserRole: ColumnType.Text},
            {Qt.DisplayRole: "Datetime", Qt.UserRole: ColumnType.Time},
            {Qt.AccessibleDescriptionRole: "separator"},
            {Qt.DisplayRole: "Ignore",
             Qt.UserRole: ColumnType.Skip,
             Qt.ToolTipRole: "The column will not be loaded"}
        ]
        typemodel = QStandardItemModel(self)
        for itemdata in types:
            item = Item(itemdata)
            if itemdata.get(Qt.AccessibleDescriptionRole) == "separator":
                item.setFlags(Qt.NoItemFlags)
            typemodel.appendRow(item)

        self.column_type_edit_cb.setModel(typemodel)
        self.column_type_edit_cb.setCurrentIndex(-1)

        form.addRow(QFrame(frameShape=QFrame.HLine))
        form.addRow("Column type", self.column_type_edit_cb)
        layout.addWidget(self.dataview)
        # Overlay error message widget in the bottom left corner of the data
        # view
        self.__overlay = overlay = OverlayWidget(
            parent=self.dataview.viewport(),
            alignment=Qt.AlignBottom | Qt.AlignLeft,
            objectName="-error-overlay",
            visible=False,
        )
        overlay.setLayout(QVBoxLayout(margin=0))
        self.__error_label = label = QLabel(objectName="-error-text-label")
        overlay.layout().addWidget(label)
        overlay.setWidget(self.dataview.viewport())

        self.setLayout(layout)

        self.__timer = QTimer(self, singleShot=True)
        self.__timer.timeout.connect(self.__resetPreview)
        self.optionswidget.optionsChanged.connect(self.__timer.start)

    def setDialect(self, dialect):
        # type: (csv.Dialect) -> None
        """
        Set the current state to match dialect instance.
        """
        self.optionswidget.setDialect(dialect)

    def dialect(self):
        # type: () -> csv.Dialect
        """
        Return the current dialect.
        """
        return self.optionswidget.dialect()

    def setEncoding(self, encoding):
        # type: (str) -> None
        """Set the current text encoding."""
        self.optionswidget.setSelectedEncoding(encoding)

    def encoding(self):
        # type: () -> str
        """Return the curent text encoding."""
        return self.optionswidget.encoding()

    def setNumbersFormat(self, groupsep, decimalsep):
        changed = False

        if groupsep != self.grouping_sep_edit_cb.text():
            self.grouping_sep_edit_cb.setText(groupsep)
            changed = True

        if decimalsep != self.grouping_sep_edit_cb.text():
            self.decimal_sep_edit_cb.setText(decimalsep)
            changed = True

        if changed:
            self.__update_numbers_format()
            self.optionsChanged.emit()

    def numbersFormat(self):
        group = self.grouping_sep_edit_cb.text()
        decimal = self.decimal_sep_edit_cb.text()
        return {"group": group, "decimal": decimal}

    def __decimal_sep_activated(self, sep):
        group_sep = self.grouping_sep_edit_cb.text()
        preferred_replace = {".": ",", ",": "."}
        if sep == group_sep and sep in preferred_replace:
            self.grouping_sep_edit_cb.setText(preferred_replace[sep])
        elif sep == group_sep:
            cb = self.grouping_sep_edit_cb
            cb.setCurrentIndex((cb.currentIndex() + 1) % cb.count())

        self.__update_numbers_format()
        self.optionsEdited.emit()
        self.optionsChanged.emit()

    def __group_sep_activated(self, sep):
        decimal_sep = self.decimal_sep_edit_cb.text()
        preferred_replace = {".": ",", ",": "."}
        if sep == decimal_sep and sep in preferred_replace:
            self.decimal_sep_edit_cb.setText(preferred_replace[sep])
        elif sep == decimal_sep:
            cb = self.decimal_sep_edit_cb
            cb.setCurrentIndex((cb.currentIndex() + 1) % cb.count())
        self.__update_numbers_format()

        self.optionsEdited.emit()
        self.optionsChanged.emit()

    def __update_numbers_format(self):
        groupsep = self.grouping_sep_edit_cb.text()
        decimalsep = self.decimal_sep_edit_cb.text()

        model = self.__previewmodel
        if model is None:
            return
        parser = number_parser(groupsep, decimalsep)

        # update the delegates
        view = self.dataview
        for i in range(model.columnCount()):
            coltype = model.headerData(
                i, Qt.Horizontal, TablePreviewModel.ColumnTypeRole)
            if coltype == ColumnType.Numeric:
                delegate = ColumnValidateItemDelegate(view, converter=parser)
                view.setItemDelegateForColumn(i, delegate)

    def columnTypes(self):
        # type: () -> Dict[int, ColumnType]
        """
        Return the current column type annotations.

        Returns
        -------
        mapping : Dict[int, Optional[ColumnType]]
            Mapping from column indices column types.
        """
        # types = dict.fromkeys(range(model.rowCount()), ColumnType.Auto)
        types = {}
        types.update(self.__columnTypes())
        return types

    def setColumnTypes(self, types):
        # type: (Dict[int, Optional[ColumnType]]) -> None
        """
        Set column type annotations.

        Parameters
        ----------
        types : Dict[int, Optional[ColumnType]]
            Mapping from column indices to column types, `None` indicates
            default (unspecified type, will be inferred)
        """
        # This depends on encoding/dialect. Force preview model update.
        if self.__timer.isActive():
            self.__resetPreview()
        self.__setColumnTypes(types)

    def setStateForRow(self, row, state):
        # type: (int, TablePreview.RowSpec) -> None
        """
        Set the state for row.
        """
        if self.__timer.isActive():
            self.__resetPreview()

        model = self.__previewmodel
        if model is None:
            return
        rowcount = model.rowCount()
        while row > rowcount - 1 and model.canFetchMore():
            model.fetchMore()
            if model.rowCount() == rowcount:
                break
            rowcount = model.rowCount()

        model.setHeaderData(
            row, Qt.Vertical, state, TablePreviewModel.RowStateRole)
        self.dataview.setRowHints({row: state})

    def stateForRow(self, row):
        # type: (int) -> Optional[TablePreview.RowSpec]
        """
        Return the state for row.
        """
        model = self.__previewmodel
        if model is not None:
            return model.headerData(
                row, Qt.Vertical, TablePreviewModel.RowStateRole)
        else:
            return None

    def rowStates(self):
        # type: () -> (Dict[int, RowSpec])
        """
        Return states for all rows with non None state
        """
        return self.__rowStates()

    def setRowStates(self, rowstate):
        # type: (Dict[int, RowSpec]) -> None
        """
        Set the state for rows.

        Note
        ----
        States for all rows not passed in rowstate is reset to `None`.
        """
        if self.__timer.isActive():
            self.__resetPreview()
        model = self.__previewmodel
        if model is None:
            return

        currstate = self.rowStates()
        newstate = dict.fromkeys(currstate.keys(), None)
        newstate.update(rowstate)
        for row, state in newstate.items():
            self.setStateForRow(row, state)

    def __rowStates(self):
        model = self.__previewmodel
        items = (
            (row, model.headerData(row, Qt.Vertical,
                                   TablePreviewModel.RowStateRole))
            for row in range(model.rowCount())
        )
        return {row: state for row, state in items if state is not None}

    def setSampleContents(self, stream):
        # type: (io.BinaryIO) -> None
        """
        Set a binary file-like stream for displaying sample content.

        The stream will be read as needed when the data view is scrolled.

        Note
        ----
        If the stream is not seekable, its contents will be cached in memory.
        If and existing stream is already set it is NOT closed. The caller
        is responsible for managing its lifetime.
        """
        self.__sample = stream
        self.__buffer = io.BytesIO()
        self.__resetPreview()

    def __resetPreview(self):
        # Reset the preview model and view
        self.__timer.stop()
        colstate = {}
        rowstate = {}
        if self.__previewmodel is not None:
            # store the column/row specs
            colstate = self.__columnTypes()
            rowstate = self.__rowStates()
            self.__previewmodel.errorOccurred.disconnect(self.__set_error)
            self.__previewmodel.deleteLater()
            self.__previewmodel = None

        if self.__textwrapper is not None:
            self.__textwrapper.detach()
            self.__textwrapper = None

        self.__set_error("")
        self.previewModelReset.emit()

        if self.__sample is None:
            return

        self.__previewmodel = TablePreviewModel(self)
        self.__previewmodel.errorOccurred.connect(self.__set_error)
        try:
            seekable = self.__sample.seekable()
        except AttributeError:
            seekable = False

        if seekable:
            # Might be better to always use buffer? (compressed streams are
            # seekable but slower)
            base = self.__sample
            base.seek(0)
        else:
            self.__buffer.seek(0)
            base = CachedBytesIOWrapper(self.__sample, self.__buffer)

        wrapper = io.TextIOWrapper(
            base, encoding=self.encoding(),
            # use surrogate escape to validate/detect encoding errors in
            # delegates
            errors="surrogateescape"
        )
        rows = csv.reader(
            wrapper, dialect=self.dialect()
        )

        self.__textwrapper = wrapper
        self.__previewmodel.setPreviewStream(rows)
        if self.__previewmodel.canFetchMore():
            # TODO: Fetch until the same number of rows as at method entry?
            self.__previewmodel.fetchMore()

        self.dataview.setModel(self.__previewmodel)
        self.dataview.selectionModel().selectionChanged.connect(
            self.__update_column_type_edit, Qt.UniqueConnection
        )
        if self.__previewmodel.columnCount() == len(colstate):
            self.__setColumnTypes(colstate)
        for row, state in rowstate.items():
            self.__previewmodel.setHeaderData(row, Qt.Vertical, state,
                                              TablePreviewModel.RowStateRole)
        self.dataview.setRowHints(rowstate)

    @Slot()
    def __update_column_type_edit(self):
        # Update the 'Column type' edit control based on current column
        # selection
        smodel = self.dataview.selectionModel()
        model = self.dataview.model()
        cb = self.column_type_edit_cb
        columns = smodel.selectedColumns(0)
        types = {model.headerData(c.column(), Qt.Horizontal,
                                  TablePreviewModel.ColumnTypeRole)
                 for c in columns}

        types = {ColumnType.Auto if t is None else t for t in types}
        if len(types) == 0:
            # no selection, disabled
            cb.setCurrentIndex(-1)
            cb.setEnabled(False)
        elif len(types) == 1:
            idx = cb.findData(types.pop(), Qt.UserRole)
            cb.setCurrentIndex(idx)
            cb.setEnabled(True)
        else:
            cb.setCurrentIndex(-1)
            cb.setEnabled(True)

    def __on_column_type_edit_activated(self, idx):
        # Column type set via the combo box.
        coltype = self.column_type_edit_cb.itemData(idx, Qt.UserRole)
        smodel = self.dataview.selectionModel()
        columns = smodel.selectedColumns(0)
        columns = [c.column() for c in columns]
        self.__setColumnType(columns, coltype)

    def __dataview_context_menu(self, pos):
        pos = self.dataview.viewport().mapToGlobal(pos)
        cols = self.dataview.selectionModel().selectedColumns(0)
        cols = [idx.column() for idx in cols]
        self.__run_type_columns_menu(pos, cols)

    def __hheader_context_menu(self, pos):
        pos = self.dataview.horizontalHeader().mapToGlobal(pos)
        cols = self.dataview.selectionModel().selectedColumns(0)
        cols = [idx.column() for idx in cols]
        self.__run_type_columns_menu(pos, cols)

    def __vheader_context_menu(self, pos):
        header = self.dataview.verticalHeader()  # type: QHeaderView
        index = header.logicalIndexAt(pos)
        pos = header.mapToGlobal(pos)
        model = header.model()  # type: QAbstractTableModel

        RowStateRole = TablePreviewModel.RowStateRole
        state = model.headerData(index, Qt.Vertical, RowStateRole)
        m = QMenu(header)
        skip_action = m.addAction("Skip")
        skip_action.setCheckable(True)
        skip_action.setChecked(state == TablePreview.Skipped)
        m.addSection("")
        mark_header = m.addAction("Header")
        mark_header.setCheckable(True)
        mark_header.setChecked(state == TablePreview.Header)

        def update_row_state(action):
            # type: (QAction) -> None
            state = None
            if action is mark_header:
                state = TablePreview.Header if action.isChecked() else None
            elif action is skip_action:
                state = TablePreview.Skipped if action.isChecked() else None
            model.setHeaderData(index, Qt.Vertical, state, RowStateRole)
            self.dataview.setRowHints({index: state})

        m.triggered.connect(update_row_state)
        m.popup(pos)

    def __run_type_columns_menu(self, pos, columns):
        # type: (QPoint, List[int]) -> None
        # Open a QMenu at pos for setting column types for column indices list
        # `columns`
        model = self.__previewmodel
        if model is None:
            return
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose)
        coltypes = {model.headerData(
                        i, Qt.Horizontal, TablePreviewModel.ColumnTypeRole)
                    for i in columns}
        coltypes = {ColumnType.Auto if t is None else t for t in coltypes}
        if len(coltypes) == 1:
            current = coltypes.pop()
        else:
            current = None
        cb = self.column_type_edit_cb
        g = QActionGroup(menu)
        current_action = None
        # 'Copy' the column types model into a menu
        for i in range(cb.count()):
            if cb.itemData(i, Qt.AccessibleDescriptionRole) == "separator":
                menu.addSeparator()
                continue

            ac = menu.addAction(cb.itemIcon(i), cb.itemText(i))
            ac.setData(cb.itemData(i, Qt.UserRole))
            ac.setCheckable(True)
            if ac.data() == current:
                ac.setChecked(True)
                current_action = ac
            g.addAction(ac)

        def update_types(action):
            newtype = action.data()
            self.__setColumnType(columns, newtype)

        menu.triggered.connect(update_types)
        menu.triggered.connect(self.__update_column_type_edit)
        menu.popup(pos, current_action)

    def __setColumnType(self, columns, coltype):
        # type: (List[int], ColumnType) -> None
        view = self.dataview
        model = view.model()  # type: QAbstractTableModel
        numbersformat = self.numbersFormat()
        numberconverter = number_parser(
            numbersformat["group"], numbersformat["decimal"])
        if coltype == ColumnType.Numeric:
            delegate = ColumnValidateItemDelegate(self.dataview,
                                                  converter=numberconverter)
        elif coltype == ColumnType.Text:
            delegate = ColumnValidateItemDelegate(self.dataview,
                                                  converter=str.strip)
        elif coltype == ColumnType.Time:
            delegate = ColumnValidateItemDelegate(self.dataview,
                                                  converter=parse_datetime)
        elif coltype == ColumnType.Skip:
            delegate = SkipItemDelegate(self.dataview)
        else:
            delegate = None

        changed = False
        for i in columns:
            current = model.headerData(
                i, Qt.Horizontal, TablePreviewModel.ColumnTypeRole)
            changed = changed or current != coltype
            model.setHeaderData(
                i, Qt.Horizontal, coltype, TablePreviewModel.ColumnTypeRole
            )
            model.setHeaderData(
                i, Qt.Horizontal, icon_for_column_type(coltype),
                Qt.DecorationRole
            )
            self.dataview.setItemDelegateForColumn(i, delegate)

        if changed:
            self.__update_column_type_edit()
            self.columnTypesChanged.emit()

    def __setColumnTypes(self, coltypes):
        # type: (Dict[int, ColumnType]) -> None
        def mapping_invert(mapping):
            # type: (Dict[_A, _B]) -> Dict[_B, List[_A]]
            m = defaultdict(list)
            for key, val in mapping.items():
                m[val].append(key)
            return m
        model = self.__previewmodel
        if model is None:
            return
        coltypes_ = dict.fromkeys(range(model.columnCount()), ColumnType.Auto)
        coltypes_.update(coltypes)
        for coltype, cols in mapping_invert(coltypes_).items():
            self.__setColumnType(cols, coltype)

    def __columnTypes(self):
        # type: () -> Dict[int, ColumnType]
        model = self.__previewmodel
        if model is None:
            return {}
        res = {
            i: model.headerData(i, Qt.Horizontal,
                                TablePreviewModel.ColumnTypeRole)
            for i in range(model.columnCount())
        }
        return {i: val for i, val in res.items()
                if val is not None and val != ColumnType.Auto}

    def columnTypeRanges(self):
        # type: () -> List[Tuple[range, ColumnType]]
        """
        Return the column type specs as column ranges.

        Returns
        -------
        coltypes : List[Tuple[range, ColumnType]]
            A list of `(range, coltype)` tuples where `range` are ranges
            with step 1 and coltype a ColumnType. The ranges are sorted
            in ascending order.

        Note
        ----
        Unlike `columnTypes` this method does not omit ColumnTypes.Auto
        entries.
        """
        model = self.__previewmodel
        if model is None:
            return []
        res = dict.fromkeys(range(model.columnCount()), ColumnType.Auto)
        res.update(self.__columnTypes())
        types = sorted(res.items())
        res = []

        # Group by increasing indices (with step 1) and coltype
        def groupkey(item, __incseq=iter(itertools.count())):
            index, val = item
            return index - next(__incseq), val

        for (_, key), items in itertools.groupby(types, key=groupkey):
            items = list(items)
            start = items[0][0]
            last = items[-1][0]
            res.append((range(start, last + 1), key))
        return res

    def setColumnTypeRanges(self, ranges):
        # type: (List[Tuple[range, ColumnType]]) -> None
        """
        Set column type specs for ranges.

        Parameters
        ----------
        ranges : List[Tuple[range, ColumnType]]
            For every `(range, coltype)` tuple set the corresponding coltype.
        """
        self.setColumnTypes({i: coltype for r, coltype in ranges for i in r})

    def __set_error(self, errorstr):
        # type: (str) -> None
        if not errorstr:
            self.__overlay.hide()
            self.__error_label.setText("")
        else:
            self.__overlay.show()
            self.__error_label.setText(errorstr)
            self.previewReadErrorOccurred.emit(errorstr)


class CachedBytesIOWrapper(io.BufferedIOBase):
    """
    Read and cache data from `base`. When cache is not empty prepend data from
    the cache before switching to base

    Base needs to implement `read` method, cache must be read/write and
    seekable.

    Utility wrapper to implement restartable reads for streams that are not
    seekable.
    """
    def __init__(self, base, cache):
        # type: (io.BinaryIO, io.BytesIO) -> None
        super().__init__()
        self.__base = base
        self.__cache = cache

    def detach(self):
        base = self.__base
        self.__base = None
        return base

    def read(self, size=-1):
        # type: (Optional[int]) -> bytes
        base, cache = self.__base, self.__cache
        if size is None or size < 0:
            b1 = cache.read()
            b2 = base.read()
            cache.write(b2)
            return b1 + b2
        else:
            if cache.tell() < len(cache.getbuffer()):
                b1 = cache.read(size)
                if len(b1) < size:
                    assert len(cache.getbuffer()) == cache.tell()
                    b2 = base.read(size - len(b1))
                    cache.write(b2)
                    assert len(cache.getbuffer()) == cache.tell()
                    b = b1 + b2
                else:
                    b = b1
            else:
                b = base.read(size)
                cache.write(b)
                assert len(cache.getbuffer()) == cache.tell()
            return b

    def read1(self, size=-1):
        # Does not exactly conform to spec, but necessary for io.TextIOWrapper
        return self.read(size)

    def readable(self):
        return True

    def writable(self):
        return False


class RowSpec(enum.IntEnum):
    """Row spec flags"""
    #: Header row
    Header = 1
    #: Row is skipped
    Skipped = 2


class TablePreview(QTableView):
    RowSpec = RowSpec
    Header, Skipped = RowSpec

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setItemDelegate(PreviewItemDelegate(self))

    def rowsInserted(self, parent, start, end):
        # type: (QModelIndex, int, int) -> None
        super().rowsInserted(parent, start, end)
        behavior = self.selectionBehavior()
        if behavior & (QTableView.SelectColumns | QTableView.SelectRows):
            # extend the selection to the new rows
            smodel = self.selectionModel()
            selection = smodel.selection()
            command = QItemSelectionModel.Select
            if behavior & QTableView.SelectRows:
                command |= QItemSelectionModel.Rows
            if behavior & QTableView.SelectColumns:
                command |= QItemSelectionModel.Columns
            smodel.select(selection, command)

    def setRowHints(self, hints):
        # type: (Dict[int, TablePreview.RowSpec]) -> None
        for row, hint in hints.items():
            current = self.itemDelegateForRow(row)
            if current is not None:
                current.deleteLater()
            if hint == TablePreview.Header:
                delegate = HeaderItemDelegate(self)
            elif hint == TablePreview.Skipped:
                delegate = SkipItemDelegate(self)
            else:
                delegate = None
            self.setItemDelegateForRow(row, delegate)

    def sizeHint(self):
        sh = super().sizeHint()  # type: QSize
        hh = self.horizontalHeader()  # type: QHeaderView
        vh = self.verticalHeader()  # type: QHeaderView
        hsection = hh.defaultSectionSize()
        vsection = vh.defaultSectionSize()
        return sh.expandedTo(QSize(8 * hsection, 20 * vsection))


def is_surrogate_escaped(text: str) -> bool:
    """Does `text` contain any surrogate escape characters."""
    return any("\udc80" <= c <= "\udcff" for c in text)


class PreviewItemDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        if len(option.text) > 500:
            # Shorten long text (long text layout takes too long)
            f = QTextBoundaryFinder(QTextBoundaryFinder.Grapheme, option.text)
            f.setPosition(500)
            i = f.toNextBoundary()
            if i != -1:
                option.text = option.text[:i] + "..."
        model = index.model()
        coltype = model.headerData(index.column(), Qt.Horizontal,
                                   TablePreviewModel.ColumnTypeRole)
        if coltype == ColumnType.Numeric or coltype == ColumnType.Time:
            option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter

        if not self.validate(option.text):
            option.palette.setBrush(
                QPalette.All, QPalette.Text, QBrush(Qt.red, Qt.SolidPattern)
            )
            option.palette.setBrush(
                QPalette.All, QPalette.HighlightedText,
                QBrush(Qt.red, Qt.SolidPattern)
            )

    def validate(self, value: str) -> bool:  # pylint: disable=no-self-use
        return not is_surrogate_escaped(value)

    def helpEvent(self, event, view, option, index):
        # type: (QHelpEvent, QAbstractItemView, QStyleOptionViewItem, QModelIndex) -> bool
        if event.type() == QEvent.ToolTip:
            ttip = index.data(Qt.ToolTipRole)
            if ttip is None:
                ttip = index.data(Qt.DisplayRole)
                ttip = self.displayText(ttip, option.locale)
                QToolTip.showText(event.globalPos(), ttip, view)
                return True
        return super().helpEvent(event, view, option, index)


class HeaderItemDelegate(PreviewItemDelegate):
    """
    Paint the items with an alternate color scheme
    """
    NoFeatures = 0
    AutoDecorate = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__features = HeaderItemDelegate.NoFeatures

    def features(self):
        return self.__features

    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        palette = option.palette
        shadow = palette.color(QPalette.Foreground)  # type: QColor
        if shadow.isValid():
            shadow.setAlphaF(0.1)
            option.backgroundBrush = QBrush(shadow, Qt.SolidPattern)
        option.displayAlignment = Qt.AlignCenter
        model = index.model()
        if option.icon.isNull() and \
                self.__features & HeaderItemDelegate.AutoDecorate:
            ctype = model.headerData(index.column(), Qt.Horizontal,
                                     TablePreviewModel.ColumnTypeRole)
            option.icon = icon_for_column_type(ctype)
        if not option.icon.isNull():
            option.features |= QStyleOptionViewItem.HasDecoration


def icon_for_column_type(coltype):
    # type: (ColumnType) -> QIcon
    if coltype == ColumnType.Numeric:
        icon = QIcon(StampIconEngine("N", QColor("red")))
    elif coltype == ColumnType.Categorical:
        icon = QIcon(StampIconEngine("C", QColor("green")))
    elif coltype == ColumnType.Text:
        icon = QIcon(StampIconEngine("S", QColor("black")))
    elif coltype == ColumnType.Time:
        icon = QIcon(StampIconEngine("T", QColor("deepskyblue")))
    else:
        icon = QIcon()
    return icon


class SkipItemDelegate(PreviewItemDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        color = QColor(Qt.red)
        palette = option.palette  # type: QPalette
        base = palette.color(QPalette.Base)
        if base.isValid() and base.value() > 127:
            # blend on 'light' base, not on dark (low contrast)
            color.setAlphaF(0.2)
        option.backgroundBrush = QBrush(color, Qt.DiagCrossPattern)


class ColumnValidateItemDelegate(PreviewItemDelegate):
    def __init__(self, *args, converter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.converter = converter or float

    def validate(self, value):
        if value in {"NA", "Na", "na", "n/a", "N/A", "?", "", "."}:
            return True
        try:
            self.converter(value)
        except ValueError:
            return False
        else:
            return super().validate(value)


def number_parser(groupsep, decimalsep):
    # type: (str, str) -> Callable[[str], float]
    if groupsep == "" and decimalsep == ".":
        return float
    elif groupsep == "":
        return lambda value: float(value.replace(decimalsep, "."))
    elif decimalsep != groupsep and decimalsep != "" and groupsep != "":
        table = {ord(groupsep): None, ord(decimalsep): ord(".")}
        return lambda value: float(value.translate(table))
    else:
        return float


class TablePreviewModel(QAbstractTableModel):
    """
    Lazy populated table preview model.

    The model reads rows on demand from an 'rows' iterable when requested
    (via fetchMore).
    Additionally the client can set column/row header data.
    """
    ColumnTypeRole = Qt.UserRole + 11
    RowStateRole = Qt.UserRole + 12

    #: Signal emitted when an error occurs while iterating over the preview
    #: stream.
    errorOccurred = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__rowCount = self.__colCount = 0
        self.__rows = []
        self.__canFetchMore = False
        self.__error = None
        self.__iter = None
        # extra header data for use by setHeaderData
        self.__headerData = {
            Qt.Horizontal: defaultdict(dict),
            Qt.Vertical: defaultdict(dict),
        }

    def setPreviewStream(self, stream):
        # type: (Iterator[List[str]]) -> None
        """
        Set an iterator over the rows.

        The iterator will be advanced on demand by `fetchMore`, while storing
        the returned values. Previous stream and its cached data is discarded.
        """
        self.beginResetModel()
        self.__iter = stream
        self.__rows = []
        self.__rowCount = self.__colCount = 0
        self.__canFetchMore = True
        self.__error = None
        self.endResetModel()

    def canFetchMore(self, parent=QModelIndex()):
        """Reimplemented."""
        if not parent.isValid():
            return self.__canFetchMore
        else:
            return False

    def fetchMore(self, parent=QModelIndex()):
        """Reimplemented."""
        if not parent.isValid():
            error = self.__error
            if self.__rowCount == 0:
                newrows = self.__tryFetchRows(20)
            else:
                newrows = self.__tryFetchRows(5)

            if newrows:
                extent = len(newrows), max(len(row) for row in newrows)
                rows, cols = self.__rowCount, self.__colCount

                self.beginInsertRows(QModelIndex(), rows, rows + extent[0] - 1)
                self.__rows.extend(newrows)
                self.__rowCount += extent[0]
                self.endInsertRows()

                if cols < extent[1]:
                    newColCount = max(cols, extent[1])
                    self.beginInsertColumns(QModelIndex(), cols, newColCount - 1)
                    self.__colCount = newColCount
                    self.endInsertColumns()

            # Emit error after inserting the final rows
            if self.__error is not None and self.__error != error:
                self.errorOccurred.emit(self.__error)

    def __tryFetchRows(self, n=10):
        # type: (int) -> List[List[str]]
        """
        Fetch and return a maximum of `n` rows from the source preview stream.
        """
        rows = []
        for _ in range(n):
            try:
                row = next(self.__iter)
            except StopIteration:
                self.__canFetchMore = False
                break
            except Exception as err:  # pylint: disable=broad-except
                print("".join(traceback.format_exception(*sys.exc_info())),
                      file=sys.stderr)
                self.__error = format_exception(err)
                self.__canFetchMore = False
                break
            else:
                rows.append(row)
        return rows

    def rowCount(self, parent=QModelIndex()):
        # type: (QModelIndex) -> int
        """Reimplemented."""
        return 0 if parent.isValid() else self.__rowCount

    def columnCount(self, parent=QModelIndex()):
        # type: (QModelIndex) -> int
        """Reimplemented."""
        return 0 if parent.isValid() else self.__colCount

    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, int) -> Any
        """Reimplemented."""
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        assert self.__rowCount == len(self.__rows)
        if not 0 <= row < self.__rowCount:
            return None
        row = self.__rows[row]

        if not 0 <= col < len(row):
            return None

        value = row[col]
        if role == Qt.DisplayRole:
            return value
        elif role == TablePreviewModel.ColumnTypeRole:
            return self.__headerData[Qt.Horizontal][index.column()].get(role)
        else:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # type: (int, Qt.Orientation, int) -> Any
        """Reimplemented."""
        if role == Qt.DisplayRole:
            return section + 1
        else:
            return self.__headerData[orientation][section].get(role)

    def setHeaderData(self, section, orientation, value, role=Qt.EditRole):
        # type: (int, Qt.Orientation, Any, Qt.ItemDataRole) -> bool
        """Reimplemented."""
        current = self.__headerData[orientation][section].get(role, None)
        if current != value:
            if value is None:
                del self.__headerData[orientation][section][role]
            else:
                self.__headerData[orientation][section][role] = value
            self.headerDataChanged.emit(orientation, section, section)
        return True

    def updateHeaderData(self, orientation, values):
        # type: (Qt.Orientation, Dict[int, Dict[Qt.ItemDataRole, Any]]) -> None
        """
        Update/set multiple header sections/roles at once.

        Parameters
        ----------
        orientation : Qt.Orientation
        values : Dict[int, Dict[Qt.ItemDataRole, Any]]
            A mapping of section indices to mapping of role to values.
            e.g. `{1: {Qt.DisplayRole: "A"}}` sets the display text to "A"
        """
        data = self.__headerData[orientation]
        if orientation == Qt.Horizontal:
            length = self.__colCount
        else:
            length = self.__rowCount
        sections = []
        for section, itemdata in values.items():
            if 0 <= section < length:
                data[section].update(itemdata)
                sections.append(section)
        if not sections:
            return

        first = min(sections)
        last = max(sections)

        self.headerDataChanged.emit(orientation, first, last)

    def flags(self, index):
        # type: (QModelIndex) -> Qt.ItemFlags
        """Reimplemented."""
        # pylint: disable=unused-argument,no-self-use
        return Qt.ItemFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def errorString(self):
        # type: () -> Optional[str]
        """
        Return the error string or None if no error occurred.
        """
        return self.__error


@singledispatch
def format_exception(err):
    return "".join(traceback.format_exception_only(type(err), err)).rstrip()


@format_exception.register(csv.Error)
def format_exception_csv(err):
    return "CSV parsing error: " + str(err)


_to_datetime = None


def parse_datetime(text):
    global _to_datetime
    if _to_datetime is None:
        from pandas import to_datetime as _to_datetime
    return _to_datetime(text)


TEST_DATA = b"""\
 ,A,B,C,D
1,a,1,1,
2,b,2,2,
3,c,3,3,
4,d,4,4,,\
"""


def main(argv=None):  # pragma: no cover
    app = QApplication(argv or [])
    argv = app.arguments()
    w = CSVImportWidget()
    w.show()
    w.raise_()

    if len(argv) > 1:
        path = argv[1]
        f = open(path, "rb")
    else:
        f = io.BytesIO(TEST_DATA)
    try:
        w.setSampleContents(f)
        app.exec()
    finally:
        f.close()


if __name__ == "__main__":  # pragma: no cover
    csv.field_size_limit(4 * 2 ** 20)
    main(sys.argv)

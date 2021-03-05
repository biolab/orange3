import io
import csv

from AnyQt.QtCore import Signal, QItemSelectionModel, Qt, QSize, QEvent, \
    QByteArray, QMimeData
from AnyQt.QtGui import QMouseEvent
from AnyQt.QtWidgets import QTableView, QStyleOptionViewItem, QStyle

from .headerview import HeaderView


def table_view_compact(view: QTableView) -> None:
    """
    Give the view a more compact default vertical header section size.
    """
    vheader = view.verticalHeader()
    option = view.viewOptions()
    option.text = "X"
    option.features |= QStyleOptionViewItem.HasDisplay
    size = view.style().sizeFromContents(
        QStyle.CT_ItemViewItem, option,
        QSize(20, 20), view
    )
    vheader.ensurePolished()
    vheader.setDefaultSectionSize(
        max(size.height(), vheader.minimumSectionSize())
    )


class TableView(QTableView):
    """
    A QTableView subclass that is more suited for displaying large data models.
    """
    #: Signal emitted when selection finished. It is not emitted during
    #: mouse drag selection updates.
    selectionFinished = Signal()

    __mouseDown = False
    __selectionDidChange = False

    def __init__(self, *args, **kwargs,):
        kwargs.setdefault("horizontalScrollMode", QTableView.ScrollPerPixel)
        kwargs.setdefault("verticalScrollMode", QTableView.ScrollPerPixel)
        super().__init__(*args, **kwargs)
        hheader = HeaderView(Qt.Horizontal, self, highlightSections=True)
        vheader = HeaderView(Qt.Vertical, self, highlightSections=True)
        hheader.setSectionsClickable(True)
        vheader.setSectionsClickable(True)
        self.setHorizontalHeader(hheader)
        self.setVerticalHeader(vheader)
        table_view_compact(self)

    def setSelectionModel(self, selectionModel: QItemSelectionModel) -> None:
        """Reimplemented from QTableView"""
        sm = self.selectionModel()
        if sm is not None:
            sm.selectionChanged.disconnect(self.__on_selectionChanged)
        super().setSelectionModel(selectionModel)
        if selectionModel is not None:
            selectionModel.selectionChanged.connect(self.__on_selectionChanged)

    def __on_selectionChanged(self):
        if self.__mouseDown:
            self.__selectionDidChange = True
        else:
            self.selectionFinished.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.__mouseDown = event.button() == Qt.LeftButton
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if self.__mouseDown and event.button() == Qt.LeftButton:
            self.__mouseDown = False
        if self.__selectionDidChange:
            self.__selectionDidChange = False
            self.selectionFinished.emit()

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in (QEvent.StyleChange, QEvent.FontChange):
            table_view_compact(self)
        super().changeEvent(event)


def table_selection_to_mime_data(table):
    """Copy the current selection in a QTableView to the clipboard.
    """
    lines = table_selection_to_list(table)

    as_csv = lines_to_csv_string(lines, dialect="excel").encode("utf-8")
    as_tsv = lines_to_csv_string(lines, dialect="excel-tab").encode("utf-8")

    mime = QMimeData()
    mime.setData("text/csv", QByteArray(as_csv))
    mime.setData("text/tab-separated-values", QByteArray(as_tsv))
    mime.setData("text/plain", QByteArray(as_tsv))
    return mime


def lines_to_csv_string(lines, dialect="excel"):
    stream = io.StringIO()
    writer = csv.writer(stream, dialect=dialect)
    writer.writerows(lines)
    return stream.getvalue()


def table_selection_to_list(table):
    model = table.model()
    indexes = table.selectedIndexes()

    rows = sorted(set(index.row() for index in indexes))
    columns = sorted(set(index.column() for index in indexes))

    lines = []
    for row in rows:
        line = []
        for col in columns:
            val = model.index(row, col).data(Qt.DisplayRole)
            # TODO: use style item delegate displayText?
            line.append("" if val is None else str(val))
        lines.append(line)

    return lines

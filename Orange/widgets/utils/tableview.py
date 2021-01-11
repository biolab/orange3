from AnyQt.QtCore import Signal, QItemSelectionModel, Qt, QSize, QEvent
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

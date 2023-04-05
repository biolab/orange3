import sys
from itertools import chain, starmap
from typing import Sequence, Tuple, cast, Optional

import numpy as np

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QSize, QAbstractProxyModel, QItemSelection,
    QItemSelectionModel, QItemSelectionRange, QAbstractItemModel
)
from AnyQt.QtGui import QPainter
from AnyQt.QtWidgets import (
    QStyle, QWidget, QStyleOptionHeader, QAbstractButton
)

import Orange.data
import Orange.data.sql.table

from Orange.widgets.data.utils.models import RichTableModel
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.itemselectionmodel import (
    BlockSelectionModel, selection_blocks, ranges
)
from Orange.widgets.utils.tableview import TableView


class DataTableView(TableView):
    """
    A TableView with settable corner text.
    """
    class __CornerPainter(QObject):
        def drawCorner(self, button: QWidget):
            opt = QStyleOptionHeader()
            view = self.parent()
            assert isinstance(view, DataTableView)
            header = view.horizontalHeader()
            opt.initFrom(header)
            state = QStyle.State_None
            if button.isEnabled():
                state |= QStyle.State_Enabled
            if button.isActiveWindow():
                state |= QStyle.State_Active
            if button.isDown():
                state |= QStyle.State_Sunken
            opt.state = state
            opt.rect = button.rect()
            opt.text = button.text()
            opt.position = QStyleOptionHeader.OnlyOneSection
            style = header.style()
            painter = QPainter(button)
            style.drawControl(QStyle.CE_Header, opt, painter, header)

        def eventFilter(self, receiver: QObject, event: QEvent) -> bool:
            if event.type() == QEvent.Paint:
                self.drawCorner(receiver)
                return True
            return super().eventFilter(receiver, event)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cornerText = ""
        self.__cornerButton = btn = self.findChild(QAbstractButton)
        self.__cornerButtonFilter = DataTableView.__CornerPainter(self)
        btn.installEventFilter(self.__cornerButtonFilter)
        btn.clicked.disconnect(self.selectAll)
        btn.clicked.connect(self.cornerButtonClicked)
        if sys.platform == "darwin":
            btn.setAttribute(Qt.WA_MacSmallSize)

    def setCornerText(self, text: str) -> None:
        """Set the corner text."""
        self.__cornerButton.setText(text)
        self.__cornerText = text
        self.__cornerButton.update()
        assert self.__cornerButton is self.findChild(QAbstractButton)
        opt = QStyleOptionHeader()
        opt.initFrom(self.__cornerButton)
        opt.text = text
        s = self.__cornerButton.style().sizeFromContents(
            QStyle.CT_HeaderSection, opt, QSize(), self.__cornerButton
        )
        if s.isValid():
            self.verticalHeader().setMinimumWidth(s.width())

    def cornerText(self):
        """Return the corner text."""
        return self.__cornerText

    def cornerButtonClicked(self):
        model = self.model()
        selection = self.selectionModel()
        selection = selection.selection()
        if len(selection) == 1:
            srange = selection[0]
            if srange.top() == 0 and srange.left() == 0 \
                    and srange.right() == model.columnCount() - 1 \
                    and srange.bottom() == model.rowCount() - 1:
                self.clearSelection()
            else:
                self.selectAll()
        else:
            self.selectAll()


def source_model(model: QAbstractItemModel) -> Optional[QAbstractItemModel]:
    while isinstance(model, QAbstractProxyModel):
        model = model.sourceModel()
    return model


def is_table_sortable(table):
    if isinstance(table, Orange.data.sql.table.SqlTable):
        return False
    elif isinstance(table, Orange.data.Table):
        return True
    else:
        return False


class RichTableView(DataTableView):
    """
    The preferred table view for RichTableModel.

    Handles the display of variable's labels keys in top left corner.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        header = self.horizontalHeader()
        header.setSortIndicator(-1, Qt.AscendingOrder)

    def setModel(self, model: QAbstractItemModel):
        current = self.model()
        if current is not None:
            current.headerDataChanged.disconnect(self.__headerDataChanged)
        super().setModel(model)
        if model is not None:
            model.headerDataChanged.connect(self.__headerDataChanged)
            self.__headerDataChanged(Qt.Horizontal)
            select_rows = self.selectionBehavior() == TableView.SelectRows
            sel_model = BlockSelectionModel(model, selectBlocks=not select_rows)
            self.setSelectionModel(sel_model)

            sortable = self.isModelSortable(model)
            self.setSortingEnabled(sortable)
            header = self.horizontalHeader()
            header.setSectionsClickable(sortable)
            header.setSortIndicatorShown(sortable)

    def isModelSortable(self, model: QAbstractItemModel) -> bool:
        """
        Should the `model` be sortable via the view header click.

        This predicate is called when a model is set on the view and
        enables/disables the model sorting and header section sort indicators.
        """
        model = source_model(model)
        if isinstance(model, TableModel):
            table = model.source
            return is_table_sortable(table)
        return False

    def __headerDataChanged(
            self,
            orientation: Qt.Orientation,
    ) -> None:
        if orientation == Qt.Horizontal:
            model = self.model()
            model = source_model(model)
            if isinstance(model, RichTableModel) and \
                    model.richHeaderFlags() & RichTableModel.Labels:
                items = model.headerData(
                    0, Qt.Horizontal, RichTableModel.LabelsItemsRole
                )
                text = "\n"
                text += "\n".join(key for key, _ in items)
            else:
                text = ""
            self.setCornerText(text)

    def setBlockSelection(
            self, rows: Sequence[int], columns: Sequence[int]
    ) -> None:
        """
        Set the block row and column selection.

        Note
        ----
        The `rows` indices refer to the underlying TableModel's rows.

        Parameters
        ----------
        rows: Sequence[int]
            The rows to select.
        columns: Sequence[int]
            The columns to select.

        See Also
        --------
        blockSelection()
        """
        model = self.model()
        if model is None:
            return
        sel_model = self.selectionModel()
        assert isinstance(sel_model, BlockSelectionModel)
        if not rows or not columns or model.rowCount() <= rows[-1] or \
                model.columnCount() <= columns[-1]:
            # selection out of range for the model
            rows = columns = []
        proxy_chain = []
        while isinstance(model, QAbstractProxyModel):
            proxy_chain.append(model)
            model = model.sourceModel()
        assert isinstance(model, TableModel)

        rows = model.mapFromSourceRows(rows)

        selection = QItemSelection()
        rowranges = list(ranges(rows))
        colranges = list(ranges(columns))

        for rowstart, rowend in rowranges:
            for colstart, colend in colranges:
                selection.append(
                    QItemSelectionRange(
                        model.index(rowstart, colstart),
                        model.index(rowend - 1, colend - 1)
                    )
                )
        for proxy in proxy_chain[::-1]:
            selection = proxy.mapSelectionFromSource(selection)
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def blockSelection(self) -> Tuple[Sequence[int], Sequence[int]]:
        """
        Return the current selected rows and columns.

        Note
        ----
        The `rows` indices refer to the underlying TableModel's rows.
        """
        model = self.model()
        if model is None:
            return [], []
        sel_model = self.selectionModel()
        selection = sel_model.selection()

        # map through the proxies into input table.
        while isinstance(model, QAbstractProxyModel):
            selection = model.mapSelectionToSource(selection)
            model = model.sourceModel()

        assert isinstance(sel_model, BlockSelectionModel)
        assert isinstance(model, TableModel)

        row_spans, col_spans = selection_blocks(selection)
        rows = list(chain.from_iterable(starmap(range, row_spans)))
        cols = list(chain.from_iterable(starmap(range, col_spans)))
        rows = np.array(rows, dtype=np.intp)
        # map the rows through the applied sorting (if any)
        rows = model.mapToSourceRows(rows)
        rows = cast(Sequence[int], rows.tolist())
        return rows, cols

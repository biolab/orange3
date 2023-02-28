from typing import NamedTuple, List, Optional, Dict, Any

import numpy as np

from AnyQt.QtWidgets import QTableView, QHeaderView
from AnyQt.QtGui import QColor, QBrush, QFont
from AnyQt.QtCore import Qt, QAbstractTableModel

from Orange.misc import DistMatrix
from Orange.widgets import gui
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.itemdelegates import FixedFormatNumericColumnDelegate


class LabelData(NamedTuple):
    labels: Optional[List[str]] = None
    colors: Optional[np.ndarray] = None


class DistMatrixModel(QAbstractTableModel):
    _brushes = np.array([QBrush(QColor.fromHsv(120, int(i / 255 * 170), 255))
                         for i in range(256)])

    _diverging_brushes = np.array([
        QBrush(col) for col in
        colorpalettes.ContinuousPalettes['diverging_tritanopic_cwr_75_98_c20'
        ].qcolors])

    def __init__(self):
        super().__init__()
        self.distances: Optional[DistMatrix] = None
        self.colors: Optional[np.ndarray] = None
        self.brushes: Optional[np.ndarray] = None
        self.__header_data: Dict[Any, Optional[LabelData]] = {
            Qt.Horizontal: LabelData(),
            Qt.Vertical: LabelData()}
        self.__zero_diag: bool = True
        self.__span: Optional[float] = None

        self.__header_font = QFont()
        self.__header_font.setBold(True)

    def set_data(self, distances):
        self.beginResetModel()
        self.distances = distances
        self.__header_data = dict.fromkeys(self.__header_data, LabelData())
        if distances is None:
            self.__span = self.colors = self.brushes = None
            return
        minc = min(0, np.min(distances))
        maxc = np.max(distances)
        if minc < 0:
            self.__span = max(-minc, maxc)
            self.brushes = self._diverging_brushes
            self.colors = 127 + (distances / self.__span * 128).astype(int)
        else:
            self.__span = maxc
            self.brushes = self._brushes
            self.colors = (distances / self.__span * 255).astype(int)

        self.__zero_diag = \
            distances.is_symmetric() and np.allclose(np.diag(distances), 0)
        self.endResetModel()

    def set_labels(self, orientation, labels: Optional[List[str]],
                   colors: Optional[np.ndarray] = None):
        self.__header_data[orientation] = LabelData(labels, colors)
        rc, cc = self.rowCount() - 1, self.columnCount() - 1
        self.headerDataChanged.emit(
            orientation, 0, rc if orientation == Qt.Vertical else cc)
        self.dataChanged.emit(self.index(0, 0), self.index(rc, cc))

    def rowCount(self, parent=None):
        if parent and parent.isValid() or self.distances is None:
            return 0
        return self.distances.shape[0]

    def columnCount(self, parent=None):
        if parent and parent.isValid() or self.distances is None:
            return 0
        return self.distances.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter | Qt.AlignVCenter
        if self.distances is None:
            return None

        row, col = index.row(), index.column()
        if role == Qt.DisplayRole and not (self.__zero_diag and row == col):
            return float(self.distances[row, col])
        if role == Qt.BackgroundRole:
            return self.brushes[self.colors[row, col]]
        if role == Qt.ForegroundRole:
            return QColor(Qt.black)  # the background is light-ish
        if role == FixedFormatNumericColumnDelegate.ColumnDataSpanRole:
            return 0., self.__span
        return None

    def headerData(self, ind, orientation, role):
        if role == Qt.FontRole:
            return self.__header_font

        __header_data = self.__header_data[orientation]
        if role == Qt.DisplayRole:
            if __header_data.labels is not None \
                    and ind < len(__header_data.labels):
                return __header_data.labels[ind]

        colors = self.__header_data[orientation].colors
        if colors is not None:
            color = colors[ind].lighter(150)
            if role == Qt.BackgroundRole:
                return QBrush(color)
            if role == Qt.ForegroundRole:
                return QColor(Qt.black if color.value() > 128 else Qt.white)
        return None


class DistMatrixView(gui.HScrollStepMixin, QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideNone)
        self.setEditTriggers(QTableView.NoEditTriggers)
        self.setItemDelegate(
            FixedFormatNumericColumnDelegate(
                roles=(Qt.DisplayRole, Qt.BackgroundRole, Qt.ForegroundRole,
                       Qt.TextAlignmentRole)))
        for header in (self.horizontalHeader(), self.verticalHeader()):
            header.setResizeContentsPrecision(1)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            header.setHighlightSections(True)
            header.setSectionsClickable(False)
        self.verticalHeader().setDefaultAlignment(
            Qt.AlignRight | Qt.AlignVCenter)

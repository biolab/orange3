import itertools
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui

from PyQt4.QtGui import QTableView, QBrush, QColor, QItemSelectionModel
from PyQt4.QtCore import Qt, SIGNAL, QAbstractTableModel, QModelIndex

import Orange.data
import Orange.misc

from Orange.widgets import widget, gui
from Orange.widgets.data.owtable import ranges
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.itemmodels import VariableListModel


class DistanceMatrixModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.distances = None
        self.fact = 70
        self.labels = None
        self.colors = None

    def set_data(self, distances):
        self.emit(SIGNAL("modelAboutToBeReset()"))
        self.distances = distances
        if distances is None:
            return
        span = distances.max()
        self.colors = (distances * (170 / span if span > 1e-10 else 0)
                       ).astype(np.int)
        self.emit(SIGNAL("modelReset()"))

    def set_labels(self, labels):
        if bool(self.labels) != bool(labels):
            self.emit(SIGNAL("modelAboutToBeReset()"))
            self.labels = labels
            self.emit(SIGNAL("modelReset()"))
        else:
            self.emit(SIGNAL("modelAboutToBeReset()"))
            self.labels = labels
            self.emit(SIGNAL("modelReset()"))

    def dimension(self, parent=None):
        if parent and parent.isValid() or self.distances is None or \
                self.distances is None:
            return 0
        return len(self.distances) + bool(self.labels)

    columnCount = rowCount = dimension

    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        if row == col or self.distances is None:
            return
        has_labels = bool(self.labels)
        row -= has_labels
        col -= has_labels
        if row == -1 or col == -1:
            if has_labels and role == Qt.DisplayRole:
                return self.labels[row + col + 1]
        elif role == Qt.DisplayRole:
            return "{:5f}".format(self.distances[row, col])
        elif role == Qt.BackgroundColorRole and self.colors is not None:
            return QBrush(QColor.fromHsv(120, self.colors[row, col], 255))


class SymmetricSelectionModel(QItemSelectionModel):
    def select(self, selection, flags):
        if isinstance(selection, QModelIndex):
            selection = QtGui.QItemSelection(selection, selection)

        model = self.model()
        indexes = self.selectedIndexes()
        selected = {ind.row() for ind in indexes}
        new_selection = QtGui.QItemSelection()
        if flags & QItemSelectionModel.Select:
            if flags & QItemSelectionModel.Clear:
                selected = set()
            indexes = selection.indexes()
            selected |= {ind.row() for ind in indexes} | \
                        {ind.column() for ind in indexes}
            regions = list(ranges(sorted(selected)))
            for r_start, r_end in regions:
                for c_start, c_end in regions:
                    top_left = model.index(r_start, c_start)
                    bottom_right = model.index(r_end - 1, c_end - 1)
                    new_selection.select(top_left, bottom_right)
        elif flags & QItemSelectionModel.Deselect:
            indexes = selection.indexes()

            def to_ranges(indices):
                return [range(*r) for r in ranges(indices)]

            regions = to_ranges(sorted(selected))
            de_regions = {ind.row() for ind in indexes}

            for row_range, col_range in \
                    itertools.product(regions, de_regions):
                new_selection.select(
                    model.index(row_range.start, col_range.start),
                    model.index(row_range.stop - 1, col_range.stop - 1))
                new_selection.select(
                    model.index(col_range.start, row_range.start),
                    model.index(col_range.stop - 1, row_range.stop - 1))

        QItemSelectionModel.select(self, new_selection, flags)

    def selected_items(self):
        has_labels = bool(self.model().labels)
        return list({ind.row() - has_labels for ind in self.selectedIndexes()})


class OWDistanceMatrix(widget.OWWidget):
    name = "Distance Matrix"
    description = "View distance matrix"
    icon = "icons/DistanceMatrix.svg"
    priority = 200

    inputs = [("Distances", Orange.misc.DistMatrix, "set_distances")]
    outputs = [("Distances", Orange.misc.DistMatrix),
               ("Table", Orange.data.Table)]

    annotation_idx = Setting(0)
    auto_commit = Setting(True)
    selection = ContextSetting([])

    def __init__(self):
        super().__init__()
        self.distances = None
        self.items = None

        box = gui.widgetBox(self.controlArea, "Annotations")
        self.annot_combo = gui.comboBox(
            box, self, "annotation_idx", callback=self._invalidate_annotations,
            contentsLength=12)
        self.annot_combo.setModel(VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selected Data", "Auto send is on")

        self.tablemodel = DistanceMatrixModel()
        view = self.tableview = QTableView(
            editTriggers=QTableView.NoEditTriggers)
        view.setModel(self.tablemodel)
        view.horizontalHeader().hide()
        view.verticalHeader().hide()
        selmodel = SymmetricSelectionModel(view.model(), view)
        selmodel.selectionChanged.connect(self.commit)
        view.setSelectionModel(selmodel)
        self.mainArea.layout().addWidget(view)

    def sizeHint(self):
        return QtCore.QSize(800, 500)

    def set_distances(self, distances):
        self.distances = distances
        self.tablemodel.set_data(self.distances)
        if distances is not None:
            self.set_items(distances.row_items, distances.axis)
        else:
            self.set_items(None)

    def _invalidate_annotations(self):
        if self.distances is not None:
            self._update_labels()

    def _update_labels(self, ):
        if self.annotation_idx == 0:
            labels = None
        elif self.annotation_idx == 1:
            labels = [str(i + 1) for i in range(self.distances.shape[0])]
        elif self.annot_combo.model()[self.annotation_idx] == "Attribute names":
            attr = self.distances.row_items.domain.attributes
            labels = [str(attr[i]) for i in range(self.distances.shape[0])]
        elif self.annotation_idx == 2 and \
                isinstance(self.items, widget.AttributeList):
            labels = [v.name for v in self.items]
        elif isinstance(self.items, Orange.data.Table):
            var = self.annot_combo.model()[self.annotation_idx]
            column, _ = self.items.get_column_view(var)
            labels = [var.repr_val(value) for value in column]
        self.tablemodel.set_labels(labels)

    def set_items(self, items, axis=1):
        self.items = items
        model = self.annot_combo.model()
        if items is None:
            model[:] = ["None", "Enumeration"]
        elif not axis:
            model[:] = ["None", "Enumeration", "Attribute names"]
            self.annotation_idx = 2
        elif isinstance(items, Orange.data.Table):
            model[:] = (["None", "Enumeration"] +
                        list(items.domain) + list(items.domain.metas))
        elif isinstance(items, list) and \
                all(isinstance(item, Orange.data.Variable) for item in items):
            model[:] = ["None", "Enumeration", "Name"]
        else:
            model[:] = ["None", "Enumeration"]
        self.annotation_idx = min(self.annotation_idx, len(model) - 1)

    def commit(self):
        if self.distances is None:
            self.send("Table", None)
            self.send("Distances", None)
            return

        inds = self.tableview.selectionModel().selected_items()
        dist = self.distances
        if isinstance(self.items, Orange.data.Table):
            items = self.items[inds]
        elif isinstance(self.items, list):
            items = [self.items[i] for i in inds]
        else:
            items = None
        table = items if isinstance(items, Orange.data.Table) else None
        self.send("Table", table)
        self.send("Distances", dist.submatrix(inds))

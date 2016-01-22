from math import isnan
import itertools
from contextlib import contextmanager

import numpy as np

from PyQt4.QtGui import QTableView, QColor, QItemSelectionModel, \
    QItemDelegate, QPen, QBrush, QItemSelection
from PyQt4.QtCore import Qt, SIGNAL, QAbstractTableModel, QModelIndex, QSize

from Orange.data import Table, Variable, ContinuousVariable, DiscreteVariable
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui
from Orange.widgets.data.owtable import ranges
from Orange.widgets.gui import OrangeUserRole
from Orange.widgets.settings import Setting, ContextSetting, ContextHandler
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from Orange.widgets.utils.itemmodels import VariableListModel


class DistanceMatrixModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.distances = None
        self.fact = 70
        self.labels = None
        self.colors = None
        self.variable = None
        self.values = None
        self.label_colors = None
        self.zero_diag = True
        self.has_labels = False

    @contextmanager
    def model_reset_signal(self):
        self.emit(SIGNAL("modelAboutToBeReset()"))
        yield
        self.emit(SIGNAL("modelReset()"))

    def set_data(self, distances):
        with self.model_reset_signal():
            self.distances = distances
            if distances is None:
                return
            span = distances.max()
            self.colors = \
                (distances * (170 / span if span > 1e-10 else 0)).astype(np.int)
            self.zero_diag = not self.distances.diagonal().any()

    def set_labels(self, labels, variable=None, values=None):
        with self.model_reset_signal():
            self.labels = labels
            self.has_labels = bool(self.labels)
            self.variable = variable
            self.values = values
            if isinstance(variable, ContinuousVariable):
                palette = ContinuousPaletteGenerator(*variable.colors)
                off, m = values.min(), values.max()
                fact = off != m and 1 / (m - off)
                self.label_colors = [
                    palette[x] if not isnan(x) else Qt.lightGray
                    for x in (values - off) * fact]
            else:
                self.label_colors = None

    def dimension(self, parent=None):
        if parent and parent.isValid() or self.distances is None or \
                self.distances is None:
            return 0
        return len(self.distances) + bool(self.labels)

    columnCount = rowCount = dimension

    def data(self, index, role=Qt.DisplayRole):
        def color_for_ind(ind, light=100):
            color = Qt.lightGray
            if self.variable is not None:
                if ind == -1:
                    color = Qt.white
                elif isinstance(self.variable, ContinuousVariable):
                    color = self.label_colors[ind].lighter(light)
                elif isinstance(self.variable, DiscreteVariable):
                    value = self.values[ind]
                    if not isnan(value):
                        color = QColor(*self.variable.colors[value])
            return QBrush(color)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter
        row, col = index.row(), index.column()
        row -= self.has_labels
        col -= self.has_labels
        if row == -1 or col == -1:
            ind = row + col + 1
            if role == Qt.DisplayRole:
                if not row == col == -1:
                    return self.labels[ind]
            if role == Qt.BackgroundColorRole:
                return color_for_ind(ind, 200)
            return
        if self.distances is None:
            return
        if row == col and self.zero_diag:
            if role == Qt.BackgroundColorRole and self.variable:
                return color_for_ind(row, 200)
            return
        if role == Qt.DisplayRole:
            return "{:.3f}".format(self.distances[row, col])
        if role == Qt.BackgroundColorRole and self.colors is not None:
            return QBrush(QColor.fromHsv(120, self.colors[row, col], 255))
        if role == TableBorderItem.BorderColorRole and self.variable:
            return color_for_ind(row), color_for_ind(col)


class TableBorderItem(QItemDelegate):
    BorderColorRole = next(OrangeUserRole)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        colors = index.data(self.BorderColorRole)
        vcolor, hcolor = colors or (None, None)
        if vcolor is not None or hcolor is not None:
            painter.save()
            x1, y1, x2, y2 = option.rect.getCoords()
            if vcolor is not None:
                painter.setPen(
                    QPen(QBrush(vcolor), 1, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(x1, y1, x1, y2)
            if hcolor is not None:
                painter.setPen(
                    QPen(QBrush(hcolor), 1, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(x1, y1, x2, y1)
            painter.restore()


class SymmetricSelectionModel(QItemSelectionModel):
    def select(self, selection, flags):
        if isinstance(selection, QModelIndex):
            selection = QItemSelection(selection, selection)

        model = self.model()
        indexes = self.selectedIndexes()
        selected = {ind.row() for ind in indexes}
        new_selection = QItemSelection()
        if flags & QItemSelectionModel.Select:
            if flags & QItemSelectionModel.Clear:
                selected = set()
            indexes = selection.indexes()
            selected |= {ind.row() for ind in indexes} | \
                        {ind.column() for ind in indexes}
            if self.model().has_labels and 0 in selected:
                selected.remove(0)
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
        """
        Return indices of selected items.

        Indices come from selectedIndexes, but decreased by 1 if labels
        are shown.
        """
        has_labels = self.model().has_labels
        return list({ind.row() - has_labels for ind in self.selectedIndexes()})

    def set_selected_items(self, inds):
        model = self.model()
        has_labels = model.has_labels
        selection = QItemSelection()
        for i in inds:
            i += has_labels
            selection.select(model.index(i, i), model.index(i, i))
        self.select(selection, QItemSelectionModel.Select)


class DistanceMatrixContextHandler(ContextHandler):
    @staticmethod
    def _var_names(annotations):
        return [a.name if isinstance(a, Variable) else a for a in annotations]

    def new_context(self, matrix, annotations):
        context = super().new_context()
        context.dim = matrix.shape[0]
        context.annotations = self._var_names(annotations)
        context.annotation_idx = 1
        context.selection = []
        return context

    def match(self, context, matrix, annotations):
        annotations = self._var_names(annotations)
        if context.dim != matrix.shape[0] or \
                context.annotation not in annotations:
            return 0
        return 1 + (context.annotations == annotations)

    def settings_from_widget(self, widget):
        context = widget.current_context
        context.annotation = widget.annot_combo.currentText()
        context.selection = widget.tableview.selectionModel().selected_items()

    def settings_to_widget(self, widget):
        context = widget.current_context
        widget.annotation_idx = context.annotations.index(context.annotation)
        widget.tableview.selectionModel().set_selected_items(context.selection)


class OWDistanceMatrix(widget.OWWidget):
    name = "Distance Matrix"
    description = "View distance matrix"
    icon = "icons/DistanceMatrix.svg"
    priority = 200

    inputs = [("Distances", DistMatrix, "set_distances")]
    outputs = [("Distances", DistMatrix),
               ("Table", Table)]

    settingsHandler = DistanceMatrixContextHandler()
    auto_commit = Setting(True)
    annotation_idx = ContextSetting(1)
    selection = ContextSetting([])

    want_control_area = False

    def __init__(self):
        super().__init__()
        self.distances = None
        self.items = None

        self.tablemodel = DistanceMatrixModel()
        view = self.tableview = QTableView(
            editTriggers=QTableView.NoEditTriggers)
        view.setItemDelegate(TableBorderItem())
        view.setModel(self.tablemodel)
        view.horizontalHeader().hide()
        view.verticalHeader().hide()
        selmodel = SymmetricSelectionModel(view.model(), view)
        selmodel.selectionChanged.connect(self.commit)
        view.setSelectionModel(selmodel)
        self.mainArea.layout().addWidget(view)

        settings_box = gui.widgetBox(self.mainArea, orientation="horizontal")

        self.annot_combo = gui.comboBox(
            settings_box, self, "annotation_idx", label="Labels: ",
            orientation="horizontal",
            callback=self._invalidate_annotations, contentsLength=12)
        self.annot_combo.setModel(VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]
        gui.rubber(settings_box)
        gui.auto_commit(
            settings_box, self, "auto_commit",
            "Send Selected Data", "Auto send is on", box=None)

    def sizeHint(self):
        return QSize(800, 500)

    def set_distances(self, distances):
        self.closeContext()
        self.distances = distances
        self.tablemodel.set_data(self.distances)
        self.selected_indices = []

        self.items = items = distances.row_items
        annotations = ["None", "Enumerate"]
        if items and not distances.axis:
            annotations.append("Attribute names")
        elif isinstance(items, list) and \
                 all(isinstance(item, Variable) for item in items):
            annotations.append("Name")
        elif isinstance(items, Table):
            annotations.extend(itertools.chain(items.domain, items.domain.metas))

        self.annot_combo.model()[:] = annotations
        self.annotation_idx = 1 + len(annotations) == 3

        self.openContext(distances, annotations)
        self._update_labels()
        self.tableview.resizeColumnsToContents()

    def _invalidate_annotations(self):
        if self.distances is not None:
            self._update_labels()

    def _update_labels(self):
        var = column = None
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
        elif isinstance(self.items, Table):
            var = self.annot_combo.model()[self.annotation_idx]
            column, _ = self.items.get_column_view(var)
            labels = [var.repr_val(value) for value in column]
        saved_selection = self.tableview.selectionModel().selected_items()
        self.tablemodel.set_labels(labels, var, column)
        self.tableview.resizeColumnsToContents()
        self.tableview.selectionModel().set_selected_items(saved_selection)

    def commit(self):
        if self.distances is None:
            self.send("Table", None)
            self.send("Distances", None)
            return

        inds = self.tableview.selectionModel().selected_items()
        dist = self.distances
        if isinstance(self.items, Table):
            items = self.items[inds]
        elif isinstance(self.items, list):
            items = [self.items[i] for i in inds]
        else:
            items = None
        table = items if isinstance(items, Table) else None
        self.send("Table", table)
        self.send("Distances", dist.submatrix(inds))

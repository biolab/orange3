from math import isnan
import itertools

import numpy as np

from AnyQt.QtWidgets import QTableView, QItemDelegate, QHeaderView
from AnyQt.QtGui import QColor, QPen, QBrush
from AnyQt.QtCore import Qt, QAbstractTableModel, QModelIndex, \
    QItemSelectionModel, QItemSelection, QSize

from Orange.data import Table, Variable, ContinuousVariable, DiscreteVariable
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui
from Orange.widgets.data.owtable import ranges
from Orange.widgets.gui import OrangeUserRole
from Orange.widgets.settings import Setting, ContextSetting, ContextHandler
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Output


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

    def set_data(self, distances):
        self.beginResetModel()
        self.distances = distances
        if distances is None:
            return
        span = distances.max()
        self.colors = \
            (distances * (170 / span if span > 1e-10 else 0)).astype(np.int)
        self.zero_diag = all(distances.diagonal() < 1e-6)
        self.endResetModel()

    def set_labels(self, labels, variable=None, values=None):
        self.beginResetModel()
        self.labels = labels
        self.variable = variable
        self.values = values
        if isinstance(variable, ContinuousVariable):
            palette = ContinuousPaletteGenerator(*variable.colors)
            off, m = values.min(), values.max()
            fact = off != m and 1 / (m - off)
            self.label_colors = [palette[x] if not isnan(x) else Qt.lightGray
                                 for x in (values - off) * fact]
        else:
            self.label_colors = None

        self.endResetModel()

    def dimension(self, parent=None):
        if parent and parent.isValid() or self.distances is None:
            return 0
        return len(self.distances)

    columnCount = rowCount = dimension

    def color_for_label(self, ind, light=100):
        color = Qt.lightGray
        if isinstance(self.variable, ContinuousVariable):
            color = self.label_colors[ind].lighter(light)
        elif isinstance(self.variable, DiscreteVariable):
            value = self.values[ind]
            if not isnan(value):
                color = QColor(*self.variable.colors[int(value)])
        return QBrush(color)

    def color_for_cell(self, row, col):
        return QBrush(QColor.fromHsv(120, self.colors[row, col], 255))

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight | Qt.AlignVCenter
        row, col = index.row(), index.column()
        if self.distances is None:
            return
        if role == TableBorderItem.BorderColorRole:
            return self.color_for_label(col), self.color_for_label(row)
        if row == col and self.zero_diag:
            if role == Qt.BackgroundColorRole and self.variable:
                return self.color_for_label(row, 200)
            return
        if role == Qt.DisplayRole:
            return "{:.3f}".format(self.distances[row, col])
        if role == Qt.BackgroundColorRole:
            return self.color_for_cell(row, col)

    def headerData(self, ind, orientation, role):
        if not self.labels:
            return
        if role == Qt.DisplayRole and ind < len(self.labels):
            return self.labels[ind]
        # On some systems, Qt doesn't respect the following role in the header
        if role == Qt.BackgroundRole:
            return self.color_for_label(ind, 200)


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
        indexes = selection.indexes()
        sel_inds = {ind.row() for ind in indexes} | \
                   {ind.column() for ind in indexes}
        if flags == QItemSelectionModel.ClearAndSelect:
            selected = set()
        else:
            selected = {ind.row() for ind in self.selectedIndexes()}
        if flags & QItemSelectionModel.Select:
            selected |= sel_inds
        elif flags & QItemSelectionModel.Deselect:
            selected -= sel_inds
        new_selection = QItemSelection()
        regions = list(ranges(sorted(selected)))
        for r_start, r_end in regions:
            for c_start, c_end in regions:
                top_left = model.index(r_start, c_start)
                bottom_right = model.index(r_end - 1, c_end - 1)
                new_selection.select(top_left, bottom_right)
        QItemSelectionModel.select(self, new_selection,
                                   QItemSelectionModel.ClearAndSelect)

    def selected_items(self):
        return list({ind.row() for ind in self.selectedIndexes()})

    def set_selected_items(self, inds):
        index = self.model().index
        selection = QItemSelection()
        for i in inds:
            selection.select(index(i, i), index(i, i))
        self.select(selection, QItemSelectionModel.ClearAndSelect)


class DistanceMatrixContextHandler(ContextHandler):
    @staticmethod
    def _var_names(annotations):
        return [a.name if isinstance(a, Variable) else a for a in annotations]

    def new_context(self, matrix, annotations):
        context = super().new_context()
        context.dim = matrix.shape[0]
        context.annotations = self._var_names(annotations)
        context.annotation = context.annotations[1]
        context.selection = []
        return context

    # noinspection PyMethodOverriding
    def match(self, context, matrix, annotations):
        annotations = self._var_names(annotations)
        if context.dim != matrix.shape[0] or \
                context.annotation not in annotations:
            return 0
        return 1 + (context.annotations == annotations)

    def settings_from_widget(self, widget, *args):
        context = widget.current_context
        if context is not None:
            context.annotation = widget.annot_combo.currentText()
            context.selection = widget.tableview.selectionModel().selected_items()

    def settings_to_widget(self, widget, *args):
        context = widget.current_context
        widget.annotation_idx = context.annotations.index(context.annotation)
        widget.tableview.selectionModel().set_selected_items(context.selection)


class OWDistanceMatrix(widget.OWWidget):
    name = "Distance Matrix"
    description = "View distance matrix."
    icon = "icons/DistanceMatrix.svg"
    priority = 200

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        distances = Output("Distances", DistMatrix, dynamic=False)
        table = Output("Table", Table)

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
        view = self.tableview = QTableView()
        view.setEditTriggers(QTableView.NoEditTriggers)
        view.setItemDelegate(TableBorderItem())
        view.setModel(self.tablemodel)
        view.setShowGrid(False)
        for header in (view.horizontalHeader(), view.verticalHeader()):
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            header.setHighlightSections(True)
            header.setSectionsClickable(False)
        view.verticalHeader().setDefaultAlignment(
            Qt.AlignRight | Qt.AlignVCenter)
        selmodel = SymmetricSelectionModel(view.model(), view)
        view.setSelectionModel(selmodel)
        view.setSelectionBehavior(QTableView.SelectItems)
        self.mainArea.layout().addWidget(view)

        settings_box = gui.hBox(self.mainArea)

        self.annot_combo = gui.comboBox(
            settings_box, self, "annotation_idx", label="Labels: ",
            orientation=Qt.Horizontal,
            callback=self._invalidate_annotations, contentsLength=12)
        self.annot_combo.setModel(VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]
        gui.rubber(settings_box)
        acb = gui.auto_commit(settings_box, self, "auto_commit",
                              "Send Selected", "Send Automatically", box=None)
        acb.setFixedWidth(200)
        # Signal must be connected after self.commit is redirected
        selmodel.selectionChanged.connect(self.commit)

    def sizeHint(self):
        return QSize(800, 500)

    @Inputs.distances
    def set_distances(self, distances):
        self.closeContext()
        self.distances = distances
        self.tablemodel.set_data(self.distances)
        self.selection = []
        self.tableview.selectionModel().set_selected_items([])

        self.items = items = distances is not None and distances.row_items
        annotations = ["None", "Enumerate"]
        self.annotation_idx = 1
        if items and not distances.axis:
            annotations.append("Attribute names")
            self.annotation_idx = 2
        elif isinstance(items, list) and \
                all(isinstance(item, Variable) for item in items):
            annotations.append("Name")
            self.annotation_idx = 2
        elif isinstance(items, Table):
            annotations.extend(
                itertools.chain(items.domain.variables, items.domain.metas))
            if items.domain.class_var:
                self.annotation_idx = 2 + len(items.domain.attributes)
        self.annot_combo.model()[:] = annotations

        if items:
            self.openContext(distances, annotations)
            self._update_labels()
            self.tableview.resizeColumnsToContents()
        self.commit()

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
            labels = [var.str_val(value) for value in column]
        saved_selection = self.tableview.selectionModel().selected_items()
        self.tablemodel.set_labels(labels, var, column)
        if labels:
            self.tableview.horizontalHeader().show()
            self.tableview.verticalHeader().show()
        else:
            self.tableview.horizontalHeader().hide()
            self.tableview.verticalHeader().hide()
        self.tableview.resizeColumnsToContents()
        self.tableview.selectionModel().set_selected_items(saved_selection)

    def commit(self):
        sub_table = sub_distances = None
        if self.distances is not None:
            inds = self.tableview.selectionModel().selected_items()
            if inds:
                sub_distances = self.distances.submatrix(inds)
                if self.distances.axis and isinstance(self.items, Table):
                    sub_table = self.items[inds]
        self.Outputs.distances.send(sub_distances)
        self.Outputs.table.send(sub_table)

    def send_report(self):
        if self.distances is None:
            return
        model = self.tablemodel
        dim = self.distances.shape[0]
        col_cell = model.color_for_cell

        def _rgb(brush):
            return "rgb({}, {}, {})".format(*brush.color().getRgb())
        if model.labels:
            col_label = model.color_for_label
            label_colors = [_rgb(col_label(i)) for i in range(dim)]
            self.report_raw('<table style="border-collapse:collapse">')
            self.report_raw("<tr><td></td>")
            self.report_raw("".join(
                    '<td style="background-color: {}">{}</td>'.format(*cv)
                    for cv in zip(label_colors, model.labels)))
            self.report_raw("</tr>")
            for i in range(dim):
                self.report_raw("<tr>")
                self.report_raw(
                    '<td style="background-color: {}">{}</td>'.
                    format(label_colors[i], model.labels[i]))
                self.report_raw(
                    "".join(
                        '<td style="background-color: {};'
                        'border-top:1px solid {}; border-left:1px solid {};">'
                        '{:.3f}</td>'.format(
                            _rgb(col_cell(i, j)),
                            label_colors[i], label_colors[j],
                            self.distances[i, j])
                        for j in range(dim)))
                self.report_raw("</tr>")
            self.report_raw("</table>")
        else:
            self.report_raw('<table>')
            for i in range(dim):
                self.report_raw(
                    "<tr>" +
                    "".join('<td style="background-color: {}">{:.3f}</td>'.
                            format(_rgb(col_cell(i, j)), self.distances[i, j])
                            for j in range(dim)) +
                    "</tr>")
            self.report_raw("</table>")

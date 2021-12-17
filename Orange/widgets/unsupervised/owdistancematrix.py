import itertools

import numpy as np

from AnyQt.QtWidgets import QTableView, QHeaderView
from AnyQt.QtGui import QColor, QPen, QBrush
from AnyQt.QtCore import Qt, QAbstractTableModel, QSize

from Orange.data import Table, Variable, StringVariable
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui
from Orange.widgets.gui import OrangeUserRole
from Orange.widgets.settings import Setting, ContextSetting, ContextHandler
from Orange.widgets.utils.itemdelegates import FixedFormatNumericColumnDelegate
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.itemselectionmodel import SymmetricSelectionModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
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
        self.span = None

    def set_data(self, distances):
        self.beginResetModel()
        self.distances = distances
        if distances is None:
            return
        self.span = span = float(distances.max())

        self.colors = \
            (distances * (170 / span if span > 1e-10 else 0)).astype(np.int)
        self.zero_diag = all(distances.diagonal() < 1e-6)
        self.endResetModel()

    def set_labels(self, labels, variable=None, values=None):
        self.labels = labels
        self.variable = variable
        self.values = values
        if self.values is not None and not isinstance(self.variable,
                                                      StringVariable):
            # Meta variables can be of type np.object
            if values.dtype is not float:
                values = values.astype(float)
            self.label_colors = variable.palette.values_to_qcolors(values)
        else:
            self.label_colors = None
        self.headerDataChanged.emit(Qt.Vertical, 0, self.rowCount() - 1)
        self.headerDataChanged.emit(Qt.Horizontal, 0, self.columnCount() - 1)
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1)
        )

    def dimension(self, parent=None):
        if parent and parent.isValid() or self.distances is None:
            return 0
        return len(self.distances)

    columnCount = rowCount = dimension

    def color_for_label(self, ind, light=100):
        if self.label_colors is None:
            return Qt.lightGray
        return QBrush(self.label_colors[ind].lighter(light))

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
        if role == FixedFormatNumericColumnDelegate.ColumnDataSpanRole:
            return 0., self.span
        if row == col and self.zero_diag:
            if role == Qt.BackgroundColorRole and self.variable:
                return self.color_for_label(row, 200)
            return
        if role == Qt.DisplayRole:
            return float(self.distances[row, col])
        if role == Qt.BackgroundColorRole:
            return self.color_for_cell(row, col)

    def headerData(self, ind, orientation, role):
        if not self.labels:
            return
        if role == Qt.DisplayRole and ind < len(self.labels):
            return self.labels[ind]
        # On some systems, Qt doesn't respect the following role in the header
        if role == Qt.BackgroundRole:
            return self.color_for_label(ind, 150)


class TableBorderItem(FixedFormatNumericColumnDelegate):
    BorderColorRole = next(OrangeUserRole)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        colors = self.cachedData(index, self.BorderColorRole)
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


class TableView(gui.HScrollStepMixin, QTableView):
    pass


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
            context.selection = widget.tableview.selectionModel().selectedItems()

    def settings_to_widget(self, widget, *args):
        context = widget.current_context
        widget.annotation_idx = context.annotations.index(context.annotation)
        widget.tableview.selectionModel().setSelectedItems(context.selection)


class OWDistanceMatrix(widget.OWWidget):
    name = "Distance Matrix"
    description = "View distance matrix."
    icon = "icons/DistanceMatrix.svg"
    priority = 200
    keywords = []

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        distances = Output("Distances", DistMatrix, dynamic=False)
        table = Output("Selected Data", Table, replaces=["Table"])

    settingsHandler = DistanceMatrixContextHandler()
    auto_commit = Setting(True)
    annotation_idx = ContextSetting(1)
    selection = ContextSetting([])

    want_control_area = True
    want_main_area = False

    def __init__(self):
        super().__init__()
        self.distances = None
        self.items = None

        self.tablemodel = DistanceMatrixModel()
        view = self.tableview = TableView()
        view.setWordWrap(False)
        view.setTextElideMode(Qt.ElideNone)
        view.setEditTriggers(QTableView.NoEditTriggers)
        view.setItemDelegate(TableBorderItem(roles=(Qt.DisplayRole, Qt.BackgroundRole)))
        view.setModel(self.tablemodel)
        view.setShowGrid(False)
        for header in (view.horizontalHeader(), view.verticalHeader()):
            header.setResizeContentsPrecision(1)
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            header.setHighlightSections(True)
            header.setSectionsClickable(False)
        view.verticalHeader().setDefaultAlignment(
            Qt.AlignRight | Qt.AlignVCenter)
        selmodel = SymmetricSelectionModel(view.model(), view)
        selmodel.selectionChanged.connect(self.commit.deferred)
        view.setSelectionModel(selmodel)
        view.setSelectionBehavior(QTableView.SelectItems)
        self.controlArea.layout().addWidget(view)

        self.annot_combo = gui.comboBox(
            self.buttonsArea, self, "annotation_idx", label="Labels: ",
            orientation=Qt.Horizontal,
            callback=self._invalidate_annotations, contentsLength=12)
        self.annot_combo.setModel(VariableListModel())
        self.annot_combo.model()[:] = ["None", "Enumeration"]
        gui.rubber(self.buttonsArea)
        acb = gui.auto_send(self.buttonsArea, self, "auto_commit", box=False)
        acb.setFixedWidth(200)

    def sizeHint(self):
        return QSize(800, 500)

    @Inputs.distances
    def set_distances(self, distances):
        self.closeContext()
        self.distances = distances
        self.tablemodel.set_data(self.distances)
        self.selection = []
        self.tableview.selectionModel().clear()

        self.items = items = distances is not None and distances.row_items
        annotations = ["None", "Enumerate"]
        pending_idx = 1
        if items and not distances.axis:
            annotations.append("Attribute names")
            pending_idx = 2
        elif isinstance(items, list) and \
                all(isinstance(item, Variable) for item in items):
            annotations.append("Name")
            pending_idx = 2
        elif isinstance(items, Table):
            annotations.extend(
                itertools.chain(items.domain.variables, items.domain.metas))
            if items.domain.class_var:
                pending_idx = 2 + len(items.domain.attributes)
        self.annot_combo.model()[:] = annotations
        self.annotation_idx = pending_idx

        if items:
            self.openContext(distances, annotations)
            self._update_labels()
            self.tableview.resizeColumnsToContents()
        self.commit.now()

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
        if labels:
            self.tableview.horizontalHeader().show()
            self.tableview.verticalHeader().show()
        else:
            self.tableview.horizontalHeader().hide()
            self.tableview.verticalHeader().hide()
        self.tablemodel.set_labels(labels, var, column)
        self.tableview.resizeColumnsToContents()

    @gui.deferred
    def commit(self):
        sub_table = sub_distances = None
        if self.distances is not None:
            inds = self.tableview.selectionModel().selectedItems()
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


if __name__ == "__main__":  # pragma: no cover
    import Orange.distance
    data = Orange.data.Table("iris")
    dist = Orange.distance.Euclidean(data)
    WidgetPreview(OWDistanceMatrix).run(dist)

import itertools
import logging
from functools import partial
from typing import NamedTuple, List, Optional, Dict, Any

import numpy as np

from AnyQt.QtWidgets import QTableView, QHeaderView
from AnyQt.QtGui import QColor, QBrush, QFont
from AnyQt.QtCore import Qt, QAbstractTableModel, QSize, QItemSelection, \
    QItemSelectionRange

from Orange.data import Table, Variable, StringVariable
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, ContextHandler
from Orange.widgets.utils.itemdelegates import FixedFormatNumericColumnDelegate
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.itemselectionmodel import SymmetricSelectionModel, \
    BlockSelectionModel, selection_blocks, ranges
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

log = logging.getLogger(__name__)


class LabelData(NamedTuple):
    labels: Optional[List[str]] = None
    colors: Optional[np.ndarray] = None


class DistanceMatrixModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.distances: Optional[DistMatrix] = None
        self.colors: Optional[np.ndarray] = None
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
            self.__span = self.colors = None
            return
        self.__span = span = np.max(distances)

        self.colors = \
            (distances * (170 / span if span > 1e-10 else 0)).astype(int)
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
            return QBrush(QColor.fromHsv(120, self.colors[row, col], 255))
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
            color = colors[ind].lighter(180)
            if role == Qt.BackgroundRole:
                return QBrush(color)
            if role == Qt.ForegroundRole:
                return QColor(Qt.black if color.value() > 128 else Qt.white)
        return None


class TableView(gui.HScrollStepMixin, QTableView):
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


class DistanceMatrixContextHandler(ContextHandler):
    @staticmethod
    def _var_names(annotations):
        return [a.name if isinstance(a, Variable) else a for a in annotations]

    def new_context(self, matrix, annotations):
        context = super().new_context()
        context.shape = matrix.shape
        context.symmetric = matrix.is_symmetric()
        context.annotations = self._var_names(annotations)
        context.annotation = context.annotations[1]
        context.selection = [] if context.symmetric else ([], [])
        return context

    # noinspection PyMethodOverriding
    def match(self, context, matrix, annotations):
        annotations = self._var_names(annotations)
        if context.shape != matrix.shape \
                or context.symmetric is not matrix.is_symmetric() \
                or context.annotation not in annotations:
            return 0
        return 1 + (context.annotations == annotations)

    def settings_from_widget(self, widget, *args):
        context = widget.current_context
        if context is not None:
            context.annotation = widget.annot_combo.currentText()
            context.selection, _ = widget._get_selection()

    def settings_to_widget(self, widget, *args):
        context = widget.current_context
        widget.annotation_idx = context.annotations.index(context.annotation)
        widget._set_selection(context.selection)


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
    settings_version = 2
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
        view.setModel(self.tablemodel)
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
        self.items = None

        annotations = ["None", "Enumerate"]
        view = self.tableview

        if distances is not None:
            pending_idx = 1

            if not distances.is_symmetric():
                seltype = BlockSelectionModel
                if distances.row_items is not None \
                        or distances.col_items is not None:
                    annotations.append("Labels")
                    pending_idx = 2
            else:
                seltype = SymmetricSelectionModel
                self.items = items = distances.row_items

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
                    pending_idx = annotations.index(self._choose_label(items))

            selmodel = seltype(view.model(), view)
            selmodel.selectionChanged.connect(self.commit.deferred)
            view.setSelectionModel(selmodel)
        else:
            pending_idx = 0
            view.selectionModel().clear()

        self.annot_combo.model()[:] = annotations
        self.annotation_idx = pending_idx
        if distances is not None:
            self.openContext(distances, annotations)
        self._update_labels()
        view.resizeColumnsToContents()
        self.commit.now()

    @staticmethod
    def _choose_label(data: Table):
        attr = max((attr for attr in data.domain.metas
                    if isinstance(attr, StringVariable)),
                   key=lambda x: len(set(data.get_column(x))),
                   default=None)
        return attr or data.domain.class_var or "Enumerate"

    def _invalidate_annotations(self):
        if self.distances is not None:
            self._update_labels()

    def _update_labels(self):
        def enumeration(n):
            return [str(i + 1) for i in range(n)]

        hor_labels = ver_labels = None
        colors = None

        if self.annotation_idx == 0:
            pass

        elif self.annotation_idx == 1:
            ver_labels, hor_labels = map(enumeration, self.distances.shape)

        elif self.annot_combo.model()[self.annotation_idx] == "Attribute names":
            attr = self.distances.row_items.domain.attributes
            ver_labels = hor_labels = [
                str(attr[i]) for i in range(self.distances.shape[0])]

        elif self.annot_combo.model()[self.annotation_idx] == "Labels":
            if self.distances.col_items is not None:
                hor_labels = [
                    str(x)
                    for x in self.distances.get_labels(self.distances.col_items)]
            else:
                hor_labels = enumeration(self.distances.shape[1])
            if self.distances.row_items is not None:
                ver_labels = [
                    str(x)
                    for x in self.distances.get_labels(self.distances.row_items)]
            else:
                ver_labels = enumeration(self.distances.shape[0])

        elif self.annotation_idx == 2 and \
                isinstance(self.items, widget.AttributeList):
            ver_labels = hor_labels = [v.name for v in self.items]

        elif isinstance(self.items, Table):
            var = self.annot_combo.model()[self.annotation_idx]
            column = self.items.get_column(var)
            if var.is_primitive():
                colors = var.palette.values_to_qcolors(column)
            ver_labels = hor_labels = [var.str_val(value) for value in column]

        for header, labels in ((self.tableview.horizontalHeader(), hor_labels),
                               (self.tableview.verticalHeader(), ver_labels)):
            self.tablemodel.set_labels(header.orientation(), labels, colors)
            if labels is None:
                header.hide()
            else:
                header.show()
        self.tableview.resizeColumnsToContents()

    @gui.deferred
    def commit(self):
        sub_table = sub_distances = None
        if self.distances is not None:
            inds, symmetric = self._get_selection()
            if symmetric:
                if inds:
                    sub_distances = self.distances.submatrix(inds)
                    if self.distances.axis and isinstance(self.items, Table):
                        sub_table = self.items[inds]
            elif all(inds):
                sub_distances = self.distances.submatrix(*inds)
        self.Outputs.distances.send(sub_distances)
        self.Outputs.table.send(sub_table)

    def _get_selection(self):
        selmodel = self.tableview.selectionModel()
        if isinstance(selmodel, SymmetricSelectionModel):
            return self.tableview.selectionModel().selectedItems(), True
        else:
            row_spans, col_spans = selection_blocks(selmodel.selection())
            rows = list(itertools.chain.from_iterable(
                itertools.starmap(range, row_spans)))
            cols = list(itertools.chain.from_iterable(
                itertools.starmap(range, col_spans)))
            return (rows, cols), False

    def _set_selection(self, selection):
        selmodel = self.tableview.selectionModel()
        if isinstance(selmodel, SymmetricSelectionModel):
            if not isinstance(selection, list):
                log.error("wrong data for symmetric selection")
                return None
            selmodel.setSelectedItems(selection)
        else:
            if not isinstance(selection, tuple) and len(selection) == 2:
                log.error("wrong data for asymmetric selection")
                return None
            rows, cols = selection
            selection = QItemSelection()
            rowranges = list(ranges(rows))
            colranges = list(ranges(cols))

            index = self.tablemodel.index
            for rowstart, rowend in rowranges:
                for colstart, colend in colranges:
                    selection.append(
                        QItemSelectionRange(
                            index(rowstart, colstart),
                            index(rowend - 1, colend - 1)
                        )
                    )
            selmodel.select(selection, selmodel.ClearAndSelect)

    def send_report(self):
        if self.distances is None:
            return
        model = self.tablemodel
        index = model.index
        ndec = self.tableview.itemDelegate().ndecimals
        header = model.headerData
        h, w = self.distances.shape

        hor_header = bool(header(0, Qt.Horizontal, Qt.DisplayRole))
        ver_header = bool(header(0, Qt.Vertical, Qt.DisplayRole))

        def cell(func, num):
            label, brush = (func(role)
                            for role in (Qt.DisplayRole, Qt.BackgroundRole))
            if brush:
                style = f' style="background-color: {brush.color().name()}"'
            else:
                style = ""
            label = "" if label is None else f"{label:.{ndec}f}" if num else label
            self.report_raw(f"<td {style}>{label}</td>\n")

        self.report_raw('<table style="border-collapse:collapse">')
        if hor_header:
            self.report_raw("<tr>")
            if ver_header:
                self.report_raw("<td></td>")
            for col in range(w):
                cell(partial(header, col, Qt.Horizontal), False)
            self.report_raw("</tr>")

        for row in range(h):
            self.report_raw("<tr>")
            if ver_header:
                cell(partial(header, row, Qt.Vertical), False)
            for col in range(w):
                cell(index(row, col).data, True)
            self.report_raw("</tr>")
        self.report_raw("</table>")

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            context.shape = (context.dim, context.dim)
            context.symmetric = True


if __name__ == "__main__":  # pragma: no cover
    import Orange.distance
    data = Orange.data.Table("zoo")
    dist = Orange.distance.Euclidean(data)
    # dist = DistMatrix([[1, 2, 3], [4, 5, 6]])
    # dist.row_items = DistMatrix._labels_to_tables(["aa", "bb"])
    # dist.col_items = DistMatrix._labels_to_tables(["cc", "dd", "ee"])
    WidgetPreview(OWDistanceMatrix).run(dist)

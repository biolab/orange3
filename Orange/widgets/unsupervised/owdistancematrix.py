import itertools
import logging
from functools import partial

from AnyQt.QtWidgets import QTableView
from AnyQt.QtCore import Qt, QSize, QItemSelection, QItemSelectionRange

from Orange.data import Table, Variable, StringVariable
from Orange.misc import DistMatrix
from Orange.widgets.utils.distmatrixmodel import \
    DistMatrixModel, DistMatrixView
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, ContextHandler
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.itemselectionmodel import SymmetricSelectionModel, \
    BlockSelectionModel, selection_blocks, ranges
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

log = logging.getLogger(__name__)


class DistanceMatrixContextHandler(ContextHandler):
    @staticmethod
    def _var_names(annotations):
        return [a.name if isinstance(a, Variable) else a for a in annotations]

    # pylint: disable=arguments-differ
    def new_context(self, matrix, annotations):
        context = super().new_context()
        context.shape = matrix.shape
        context.symmetric = matrix.is_symmetric()
        context.annotations = self._var_names(annotations)
        context.annotation = context.annotations[1]
        context.selection = [] if context.symmetric else ([], [])
        return context

    # pylint: disable=arguments-differ
    def match(self, context, matrix, annotations):
        annotations = self._var_names(annotations)
        if context.shape != matrix.shape \
                or context.symmetric is not matrix.is_symmetric() \
                or context.annotation not in annotations:
            return 0
        return 1 + (context.annotations == annotations)

    def settings_from_widget(self, widget, *args):
        # pylint: disable=protected-access
        context = widget.current_context
        if context is not None:
            context.annotation = widget.annot_combo.currentText()
            context.selection, _ = widget._get_selection()

    def settings_to_widget(self, widget, *args):
        # pylint: disable=protected-access
        context = widget.current_context
        widget.annotation_idx = context.annotations.index(context.annotation)
        widget._set_selection(context.selection)


class OWDistanceMatrix(widget.OWWidget):
    name = "Distance Matrix"
    description = "View distance matrix."
    icon = "icons/DistanceMatrix.svg"
    priority = 200
    keywords = "distance matrix"

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

        self.tablemodel = DistMatrixModel()
        view = self.tableview = DistMatrixView()
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
                return
            selmodel.setSelectedItems(selection)
        else:
            if not isinstance(selection, tuple) and len(selection) == 2:
                log.error("wrong data for asymmetric selection")
                return
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

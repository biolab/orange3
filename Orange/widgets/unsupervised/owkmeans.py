import math
import operator
import re
from itertools import chain

from AnyQt.QtWidgets import QGridLayout, QSizePolicy as Policy, QTableView, \
    QStyle
from AnyQt.QtGui import QIntValidator, QColor, QFontMetrics
from AnyQt.QtCore import Qt, QTimer, QAbstractTableModel, QModelIndex, QSize

from Orange.clustering import KMeans
from Orange.data import Table, Domain, DiscreteVariable
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input


class ClusterTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scores = {}
        self.offsets = self.spans = self.nplaces = []
        self.k_from = self.k_to = 0

    def rowCount(self, index=QModelIndex()):
        return 0 if index.isValid() else self.k_to - self.k_from + 1

    def columnCount(self, index=QModelIndex()):
        return 4

    def flags(self, index):
        if isinstance(self.scores[index.row() + self.k_from], str):
            return Qt.NoItemFlags
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_scores(self, scores, k_from, k_to):
        self.modelAboutToBeReset.emit()
        self.scores = scores
        self.k_from, self.k_to = k_from, k_to
        self.offsets = [0]
        self.spans = [1]
        self.nplaces = [3]
        for metrics in OWKMeans.SCORE_ATTRS[1:]:
            valid_scores = [
                getattr(km, metrics)
                for km in (scores[k] for k in range(self.k_from, self.k_to))
                if not isinstance(km, str)]
            min_score = min(valid_scores, default=0)
            max_score = max(valid_scores, default=0)
            self.offsets.append(min_score)
            self.spans.append((max_score - min_score) or 1)
            self.nplaces.append(
                min(5, int(abs(math.log(max(max_score, 1e-10)))) + 2))
        self.modelReset.emit()

    def data(self, index, role=Qt.DisplayRole):
        def common_data():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                return str(k)
            elif role == Qt.TextAlignmentRole:
                return Qt.AlignRight | Qt.AlignVCenter if metrics == -1 \
                    else Qt.AlignLeft

        def data_on_fail():
            if metrics >= 0 and (role == Qt.DisplayRole or role == Qt.EditRole):
                return "NA"
            elif role == Qt.ForegroundRole:
                return Qt.gray
            elif role == Qt.ToolTipRole:
                return self.scores[k]
            else:
                return common_data()

        def data_on_success():
            if role == Qt.ForegroundRole:
                return Qt.black
            if metrics >= 0:
                score = getattr(km, OWKMeans.SCORE_ATTRS[metrics])
                if role == Qt.DisplayRole or role == Qt.EditRole:
                    return "{:.{}f}".format(score, self.nplaces[metrics])
                elif role == gui.BarRatioRole:
                    p = 0.95 * (score - self.offsets[metrics]) / \
                        self.spans[metrics]
                    return p if metrics == 0 else 1 - p
            return common_data()

        k = index.row() + self.k_from
        metrics = index.column() - 1
        km = self.scores[k]
        return data_on_fail() if isinstance(km, str) else data_on_success()

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["k", "Silhouette", "Inter-cluster", "Inertia"][col]


class ClusterTableItemDelegate(gui.ColoredBarItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        painter.setPen(QColor(212, 212, 212))
        painter.drawLine(option.rect.bottomLeft(),
                         option.rect.bottomRight())
        painter.restore()
        super().paint(painter, option, index)


class TableViewWSizeHint(QTableView):
    def sizeHint(self):
        ncolumns = self.model().columnCount()
        return QSize(sum(self.columnWidth(i) + 1 for i in range(ncolumns)), 1)


class OWKMeans(widget.OWWidget):
    name = "k-Means"
    description = "k-Means clustering algorithm with silhouette-based " \
                  "quality estimation."
    icon = "icons/KMeans.svg"
    priority = 2100

    inputs = [("Data", Table, "set_data")]

    outputs = [("Annotated Data", Table, widget.Default),
               ("Centroids", Table)]

    class Error(widget.OWWidget.Error):
        failed = widget.Msg("Clustering failed\nError: {}")

    INIT_KMEANS, INIT_RANDOM = range(2)
    INIT_METHODS = "Initialize with KMeans++", "Random initialization"

    SCORE_ATTRS = "silhouette", "inter_cluster", "inertia"
    SCORING_METHODS = \
        ("Silhouette", "Inter-cluster distance", "Distance to centroids")

    resizing_enabled = False

    k = Setting(3)
    k_from = Setting(2)
    k_to = Setting(8)
    optimize_k = Setting(False)
    max_iterations = Setting(300)
    n_init = Setting(10)
    smart_init = Setting(INIT_KMEANS)
    scoring = Setting(0)
    append_cluster_ids = Setting(True)
    auto_run = Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None
        self.optimization_runs = {}
        self.last_selection = 0

        layout = QGridLayout()
        bg = gui.radioButtonsInBox(
            self.controlArea, self, "optimize_k", orientation=layout,
            box="Number of Clusters", callback=self.apply)
        layout.addWidget(
            gui.appendRadioButton(bg, "Fixed:", addToLayout=False), 1, 1)
        sb = gui.hBox(None, margin=0)
        gui.spin(
            sb, self, "k", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight, callback=self.update_k)
        gui.rubber(sb)
        layout.addWidget(sb, 1, 2)

        layout.addWidget(
            gui.appendRadioButton(bg, "From", addToLayout=False), 2, 1)
        ftobox = gui.hBox(None)
        ftobox.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(ftobox, 2, 2)
        gui.spin(
            ftobox, self, "k_from", minv=2, maxv=29,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_from)
        gui.widgetLabel(ftobox, "to")
        gui.spin(
            ftobox, self, "k_to", minv=3, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_to)
        gui.rubber(ftobox)

        box2 = gui.hBox(None)
        box2.layout().addSpacing(
            bg.style().pixelMetric(QStyle.PM_ExclusiveIndicatorWidth))
        gui.comboBox(
            box2, self, "scoring", label="Choose by", orientation=Qt.Horizontal,
            items=self.SCORING_METHODS + ("(Manual)", ), callback=self.apply)
        layout.addWidget(box2, 3, 1, 3, 2)

        box = gui.vBox(self.controlArea, "Initialization")
        gui.comboBox(
            box, self, "smart_init", items=self.INIT_METHODS,
            callback=self.invalidate)

        layout = QGridLayout()
        box2 = gui.widgetBox(box, orientation=layout)
        box2.setSizePolicy(Policy.Minimum, Policy.Minimum)
        layout.addWidget(gui.widgetLabel(None, "Re-runs: "), 0, 0, Qt.AlignLeft)
        sb = gui.hBox(None, margin=0)
        layout.addWidget(sb, 0, 1)
        gui.lineEdit(
            sb, self, "n_init", controlWidth=60,
            valueType=int, validator=QIntValidator(), callback=self.invalidate)
        layout.addWidget(
            gui.widgetLabel(None, "Maximal iterations: "), 1, 0, Qt.AlignLeft)
        sb = gui.hBox(None, margin=0)
        layout.addWidget(sb, 1, 1)
        gui.lineEdit(
            sb, self, "max_iterations", controlWidth=60, valueType=int,
            validator=QIntValidator(), callback=self.invalidate)

        self.apply_button = gui.auto_commit(
            self.buttonsArea, self, "auto_run", "Apply", box=None,
            commit=self.apply)
        gui.rubber(self.controlArea)

        self.table_model = ClusterTableModel(self)

        table = self.table_view = TableViewWSizeHint(self.mainArea)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSelectionMode(QTableView.SingleSelection)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.verticalHeader().hide()
        table.setItemDelegate(ClusterTableItemDelegate(self, color=Qt.cyan))
        table.setShowGrid(False)
        table.setModel(self.table_model)
        table.selectionModel().selectionChanged.connect(self.select_row)
        self.mainArea.layout().addWidget(table)

        metrics = QFontMetrics(table.font())
        table.setColumnWidth(0, metrics.width("9999"))
        metrics = QFontMetrics(table.horizontalHeader().font())
        max_width = max(metrics.width(title) for title in self.SCORING_METHODS)
        for i in range(1, 4):
            table.setColumnWidth(i, max_width)

    def adjustSize(self):
        self.ensurePolished()
        s = self.sizeHint()
        self.resize(s)

    def update_k(self):
        self.optimize_k = False
        self.apply()

    def update_from(self):
        self.k_to = max(self.k_from + 1, self.k_to)
        self.optimize_k = True
        self.apply()

    def update_to(self):
        self.k_from = min(self.k_from, self.k_to - 1)
        self.optimize_k = True
        self.apply()

    def check_data_size(self, n, msg_group):
        msg_group.add_message(
            "not_enough_data",
            "Too few ({}) unique data instances for {} clusters")
        data_ok = n <= len(self.data)
        msg_group.not_enough_data(len(self.data), n, shown=not data_ok)
        return data_ok

    def _compute_clustering(self, k):
        # False positives (Setting is not recognized as int)
        # pylint: disable=invalid-sequence-index
        try:
            self.optimization_runs[k] = KMeans(
                n_clusters=k,
                init=['random', 'k-means++'][self.smart_init],
                n_init=self.n_init,
                max_iter=self.max_iterations,
                compute_silhouette_score=True)(self.data)
        except BaseException as exc:
            self.optimization_runs[k] = str(exc)
            return False
        else:
            return True

    def run_optimization(self):
        # Disabling is needed since this function is not reentrant
        # Fast clicking on, say, "To: " causes multiple calls
        try:
            self.controlArea.setDisabled(True)
            if not self.check_data_size(self.k_from, self.Error):
                return
            self.check_data_size(self.k_to, self.Warning)
            k_to = min(self.k_to, len(self.data))
            needed_ks = [k for k in range(self.k_from, k_to + 1)
                         if k not in self.optimization_runs]
            if not needed_ks:
                return  # Skip showing progress bar
            with self.progressBar(len(needed_ks)) as progress:
                for k in needed_ks:
                    progress.advance()
                    self._compute_clustering(k)
            if all(isinstance(score, str)
                   for score in self.optimization_runs.values()):
                # Tooltip shows just the last error
                # pylint: disable=undefined-loop-variable
                self.Error.failed(self.optimization_runs[k])
                self.mainArea.hide()
        finally:
            self.controlArea.setDisabled(False)

    def cluster(self):
        if self.k in self.optimization_runs or \
                not self.check_data_size(self.k, self.Error):
            return
        try:
            self.controlArea.setDisabled(True)
            if not self._compute_clustering(self.k):
                self.Error.failed(self.optimization_runs[self.k])
        finally:
            self.controlArea.setDisabled(False)

    def apply(self):
        self.clear_messages()
        if self.data is not None:
            if self.optimize_k:
                self.run_optimization()
                self.mainArea.show()
                self.update_results()
            else:
                self.cluster()
                self.mainArea.hide()
        else:
            self.mainArea.hide()
        QTimer.singleShot(100, self.adjustSize)
        self.send_data()

    def invalidate(self, force_apply=False):
        self.Error.clear()
        self.Warning.clear()
        self.optimization_runs = {}
        if force_apply:
            self.unconditional_apply()
        else:
            self.apply()

    def select_best_score(self):
        scoring = self.scoring
        if scoring == len(self.SCORING_METHODS):
            self.table_view.selectRow(
                min(self.table_model.rowCount() - 1, self.last_selection))
            self.table_view.setFocus(Qt.OtherFocusReason)
            return
        best = best_row = None
        better = operator.gt if scoring == 0 else operator.lt
        metrics = self.SCORE_ATTRS[scoring]
        for row, k in enumerate(range(self.k_from, self.k_to + 1)):
            km = self.optimization_runs[k]
            if not isinstance(km, str):
                score = getattr(km, metrics)
                if best is None or better(score, best):
                    best = score
                    best_row = row
        row = self.selected_row()
        if row != best_row:
            self.table_view.clearSelection()
            if best_row is not None:
                self.table_view.selectRow(best_row)
            self.table_view.setFocus(Qt.OtherFocusReason)
        self.scoring = scoring  # changing selection has reset this to manual

    def update_results(self):
        self.table_model.set_scores(
            self.optimization_runs, self.k_from, self.k_to)
        self.select_best_score()
        self.table_view.resizeRowsToContents()

    def selected_row(self):
        indices = self.table_view.selectedIndexes()
        if indices:
            return indices[0].row()

    def select_row(self):
        self.scoring = len(self.SCORING_METHODS)
        self.send_data()

    def _get_var_name(self):
        domain = self.data.domain
        re_cluster = re.compile(r"Cluster \((\d+)\)")
        names = [var.name for var in chain(domain, domain.metas)]
        matches = (re_cluster.fullmatch(name) for name in names)
        matches = [m for m in matches if m]
        name = "Cluster"
        if matches or "Cluster" in names:
            last_num = max((int(m.group(1)) for m in matches), default=0)
            name += " ({})".format(last_num + 1)
        return name

    def send_data(self):
        if self.optimize_k:
            row = self.selected_row()
            self.last_selection = row or 0
            k = self.k_from + row if row is not None else None
        else:
            k = self.k
        km = self.optimization_runs.get(k)
        if not self.data or km is None or isinstance(km, str):
            self.send("Annotated Data", None)
            self.send("Centroids", None)
            return

        clust_var = DiscreteVariable(
            self._get_var_name(), values=["C%d" % (x + 1) for x in range(km.k)])
        clust_ids = km(self.data)
        domain = self.data.domain
        new_domain = Domain(
            domain.attributes, [clust_var], domain.metas + domain.class_vars)
        new_table = Table.from_table(new_domain, self.data)
        new_table.get_column_view(clust_var)[0][:] = clust_ids.X.ravel()

        centroids = Table(Domain(km.pre_domain.attributes), km.centroids)

        self.send("Annotated Data", new_table)
        self.send("Centroids", centroids)

    @check_sql_input
    def set_data(self, data):
        self.data = data
        self.invalidate(True)

    def send_report(self):
        # False positives (Setting is not recognized as int)
        # pylint: disable=invalid-sequence-index
        if self.optimize_k and self.selected_row() is not None:
            k_clusters = self.k_from + self.selected_row()
        else:
            k_clusters = self.k
        self.report_items((
            ("Number of clusters", k_clusters),
            ("Optimization", "{}, {} re-runs limited to {} steps".format(
                self.INIT_METHODS[self.smart_init].lower(),
                self.n_init, self.max_iterations))))
        if self.data:
            self.report_data("Data", self.data)
            if self.optimize_k:
                self.report_table(
                    "Scores for different numbers of clusters", self.table_view)


def main():
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWKMeans()
    d = Table("iris.tab")
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()

if __name__ == "__main__":
    main()

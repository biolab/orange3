import math
import operator

from AnyQt.QtWidgets import QGridLayout, QSizePolicy, QTableView
from AnyQt.QtGui import QIntValidator
from AnyQt.QtCore import Qt, QTimer, QAbstractTableModel, QModelIndex

from Orange.clustering import KMeans
from Orange.data import Table, Domain, DiscreteVariable
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input


class ClusterTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scores = []
        self.error = []
        self.from_k = 0
        self.col_title = ""
        self.min_score, self.score_span = 0, 1

    def rowCount(self, index=QModelIndex()):
        return 0 if index.isValid() or not self.scores else len(self.scores)

    def columnCount(self, index=QModelIndex()):
        # TODO: Show columns with all metrics
        return 2

    def flags(self, index):
        if self.error[index.row()]:
            return Qt.NoItemFlags
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_scores(self, scores, minimize, normalized, from_k):
        self.modelAboutToBeReset.emit()
        self.scores = scores
        self.from_k = from_k
        self.col_title = \
            "Score ({} is better)".format(["bigger", "smaller"][minimize])
        if normalized:
            self.min_score, self.score_span = 0, 1
            nplaces = 3
        else:
            valid_scores = [score
                            for score in scores if isinstance(score, float)]
            self.min_score = min(valid_scores, default=0)
            max_score = max(valid_scores, default=0)
            nplaces = min(5, int(abs(math.log(max(max_score, 1e-10)))) + 2)
            self.score_span = (max_score - self.min_score) or 1
        self.error = [score if isinstance(score, str) else None
                      for score in scores]
        format = "{{:.{}f}}".format(nplaces).format
        self.scores = [format(score)
                       if isinstance(score, float) else "clustering failed"
                       for score in scores]
        self.modelReset.emit()

    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        col = index.column()
        score = self.scores[row]
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return str(self.from_k + row) if col == 0 else score
        elif role == Qt.ForegroundRole:
            return [Qt.gray, Qt.black][not self.error[row]]
        elif role == Qt.TextAlignmentRole:
            return [Qt.AlignRight | Qt.AlignVCenter, Qt.AlignLeft][col]
        elif role == Qt.ToolTipRole:
            return self.error[row]
        elif role == gui.BarRatioRole and not self.error[row]:
            return 0.95 * (float(score) - self.min_score) / self.score_span

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and col < 2:
            return ["k", self.col_title][col]


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

    SILHOUETTE, INTERCLUSTER, DISTANCES = range(3)
    SCORING_METHODS = [
        ("Silhouette", lambda km: km.silhouette, False, True),
        ("Inter-cluster distance", lambda km: km.inter_cluster, True, False),
        ("Distance to centroids", lambda km: km.inertia, True, False)]

    resizing_enabled = False

    k = Setting(3)
    k_from = Setting(2)
    k_to = Setting(8)
    optimize_k = Setting(False)
    max_iterations = Setting(300)
    n_init = Setting(10)
    smart_init = Setting(INIT_KMEANS)
    scoring = Setting(SILHOUETTE)
    append_cluster_ids = Setting(True)
    auto_run = Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None
        self.optimization_runs = {}

        box = gui.vBox(self.controlArea, "Number of Clusters")
        layout = QGridLayout()
        bg = gui.radioButtonsInBox(
            box, self, "optimize_k", [], orientation=layout,
            callback=self.apply)
        layout.addWidget(
            gui.appendRadioButton(bg, "Fixed:", addToLayout=False), 1, 1)
        sb = gui.hBox(None, margin=0)
        gui.spin(
            sb, self, "k", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight, callback=self.update_k)
        gui.rubber(sb)
        layout.addWidget(sb, 1, 2)

        layout.addWidget(
            gui.appendRadioButton(
                bg, "Optimized from", addToLayout=False), 2, 1)
        ftobox = gui.hBox(None)
        ftobox.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(ftobox)
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

        layout.addWidget(
            gui.widgetLabel(None, "Scoring: "), 5, 1, Qt.AlignRight)
        layout.addWidget(
            gui.comboBox(
                None, self, "scoring", label="Scoring",
                items=list(zip(*self.SCORING_METHODS))[0], callback=self.apply),
            5, 2)

        box = gui.vBox(self.controlArea, "Initialization")
        gui.comboBox(
            box, self, "smart_init", items=self.INIT_METHODS,
            callback=self.invalidate)

        layout = QGridLayout()
        box2 = gui.widgetBox(box, orientation=layout)
        box2.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
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

        gui.separator(self.buttonsArea, 30)
        self.apply_button = gui.auto_commit(
            self.buttonsArea, self, "auto_run", "Apply", box=None,
            commit=self.apply)
        gui.rubber(self.controlArea)

        self.table_model = ClusterTableModel(self)

        table = self.table_view = QTableView(self.mainArea)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSelectionMode(QTableView.SingleSelection)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.verticalHeader().hide()
        table.setItemDelegateForColumn(
            1, gui.ColoredBarItemDelegate(self, color=Qt.cyan))
        table.setModel(self.table_model)
        table.selectionModel().selectionChanged.connect(self.send_data)
        table.setColumnWidth(0, 40)
        table.horizontalHeader().setStretchLastSection(True)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mainArea.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.table_view.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.mainArea.layout().addWidget(self.table_view)

    def adjustSize(self):
        self.ensurePolished()
        s = self.sizeHint()
        self.resize(s)

    def sizeHint(self):
        s = self.controlArea.sizeHint()
        if self.optimize_k and not self.mainArea.isHidden():
            s.setWidth(s.width() + self.mainArea.sizeHint().width() +
                       4 * self.childrenRect().x())
        return s

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

    def select_best_score(self, scores, minimize):
        best = best_row = None
        better = operator.lt if minimize else operator.gt
        for row, score in enumerate(scores):
            if not isinstance(score, str) \
                    and (best is None or better(score, best)):
                best = score
                best_row = row
        row = self.selected_row()
        if row != best_row:
            self.table_view.clearSelection()
            if best_row is not None:
                self.table_view.selectRow(best_row)
            self.table_view.setFocus(Qt.OtherFocusReason)

    def update_results(self):
        # False positives (Setting is not recognized as int)
        # pylint: disable=invalid-sequence-index
        _, scoring_method, minimize, normal = self.SCORING_METHODS[self.scoring]
        scores = [
            scoring_method(run) if not isinstance(run, str) else run
            for run in (self.optimization_runs[k]
                        for k in range(self.k_from, self.k_to + 1))]
        self.table_model.set_scores(scores, minimize, normal, self.k_from)
        self.select_best_score(scores, minimize)
        self.table_view.resizeRowsToContents()

    def selected_row(self):
        indices = self.table_view.selectedIndexes()
        if indices:
            return indices[0].row()

    def send_data(self):
        if self.optimize_k:
            row = self.selected_row()
            k = self.k_from + row if row is not None else None
        else:
            k = self.k
        km = self.optimization_runs.get(k)
        if not self.data or km is None or isinstance(km, str):
            self.send("Annotated Data", None)
            self.send("Centroids", None)
            return

        # TODO: add (n) if a column with this name is already in domain
        clust_var = DiscreteVariable(
            "Cluster", values=["C%d" % (x + 1) for x in range(km.k)])
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
        k_clusters = self.k
        if self.optimize_k and self.optimization_runs and \
                self.selected_row() is not None:
            k_clusters = self.k_from + self.selected_row()
        self.report_items((
            ("Number of clusters", k_clusters),
            ("Optimization",
             self.optimize_k != 0 and
             "{}, {} re-runs limited to {} steps".format(
                 self.INIT_METHODS[self.smart_init].lower(),
                 self.n_init, self.max_iterations))))
        if self.data:
            self.report_data("Data", self.data)
            if self.optimize_k:
                self.report_table(
                    "Scoring by {}".format(
                        self.SCORING_METHODS[self.scoring][0]),
                    self.table_view)


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

import re
from itertools import chain

from AnyQt.QtWidgets import QGridLayout, QSizePolicy as Policy, QTableView
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
        self.scores = []
        self.start_k = 0

    def rowCount(self, index=QModelIndex()):
        return 0 if index.isValid() else len(self.scores)

    def columnCount(self, index=QModelIndex()):
        return 2

    def flags(self, index):
        if isinstance(self.scores[index.row()], str):
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_scores(self, scores, start_k):
        self.modelAboutToBeReset.emit()
        self.scores = scores
        self.start_k = start_k
        self.modelReset.emit()

    def data(self, index, role=Qt.DisplayRole):
        column = index.column()
        score = self.scores[index.row()]
        valid = not isinstance(score, str)
        if role == Qt.DisplayRole:
            if column == 0:
                return str(self.start_k + index.row())
            else:
                return "{:.3f}".format(score) if valid else "NA"
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter | [Qt.AlignRight, Qt.AlignLeft][column]
        elif role == Qt.ForegroundRole:
            return [Qt.gray, Qt.black][valid]
        elif role == Qt.ToolTipRole and not valid:
            return score
        elif role == gui.BarRatioRole and column == 1 and valid:
            return 0.95 * score

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["k", "Silhouette"][col]


class ClusterTableItemDelegate(gui.ColoredBarItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        painter.setPen(QColor(212, 212, 212))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        painter.restore()
        super().paint(painter, option, index)


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

    resizing_enabled = False
    buttons_area_orientation = Qt.Vertical

    k = Setting(3)
    k_from = Setting(2)
    k_to = Setting(8)
    optimize_k = Setting(False)
    max_iterations = Setting(300)
    n_init = Setting(10)
    smart_init = Setting(INIT_KMEANS)
    auto_run = Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None
        self.clusterings = {}

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
        self.buttonsArea.layout().addWidget(self.report_button)
        gui.rubber(self.controlArea)

        self.table_model = ClusterTableModel(self)

        table = self.table_view = QTableView(self.mainArea)
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
        ## This is platform independent and looks good
        table.setMaximumWidth(metrics.width("9999" + 3 * "Silhouette"))
        table.horizontalHeader().setStretchLastSection(True)

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
            self.clusterings[k] = KMeans(
                n_clusters=k,
                init=['random', 'k-means++'][self.smart_init],
                n_init=self.n_init,
                max_iter=self.max_iterations,
                compute_silhouette_score=True)(self.data)
        except Exception as exc:
            self.clusterings[k] = str(exc)
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
                         if k not in self.clusterings]
            if not needed_ks:
                return  # Skip showing progress bar
            with self.progressBar(len(needed_ks)) as progress:
                for k in needed_ks:
                    progress.advance()
                    self._compute_clustering(k)
            if all(isinstance(score, str)
                   for score in self.clusterings.values()):
                # Tooltip shows just the last error
                # pylint: disable=undefined-loop-variable
                self.Error.failed(self.clusterings[k])
                self.mainArea.hide()
        finally:
            self.controlArea.setDisabled(False)

    def cluster(self):
        if self.k in self.clusterings or \
                not self.check_data_size(self.k, self.Error):
            return
        try:
            self.controlArea.setDisabled(True)
            if not self._compute_clustering(self.k):
                self.Error.failed(self.clusterings[self.k])
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
        self.clusterings = {}
        if force_apply:
            self.unconditional_apply()
        else:
            self.apply()

    def update_results(self):
        scores = [
            mk if isinstance(mk, str) else mk.silhouette for mk in (
                self.clusterings[k] for k in range(self.k_from, self.k_to + 1))]
        self.table_model.set_scores(scores, self.k_from)

        best_row = max(range(len(scores)), default=0,
                       key=lambda mk: 0 if isinstance(mk, str) else mk)
        self.table_view.selectRow(best_row)
        self.table_view.setFocus(Qt.OtherFocusReason)
        self.table_view.resizeRowsToContents()

    def selected_row(self):
        indices = self.table_view.selectedIndexes()
        if indices:
            return indices[0].row()

    def select_row(self):
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
            k = self.k_from + row if row is not None else None
        else:
            k = self.k
        km = self.clusterings.get(k)
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
                    "Silhouette scores for different numbers of clusters",
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

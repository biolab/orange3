import math

import numpy as np
from AnyQt.QtWidgets import QGridLayout, QSizePolicy, QTableView
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QIntValidator, QBrush
from AnyQt.QtCore import Qt, QTimer

from Orange.clustering import KMeans
from Orange.data import Table, Domain, DiscreteVariable
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input


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
    SCORING_METHODS = [("Silhouette", lambda km: km.silhouette, False, True),
                       ("Inter-cluster distance",
                        lambda km: km.inter_cluster, True, False),
                       ("Distance to centroids",
                        lambda km: km.inertia, True, False)]

    OUTPUT_CLASS, OUTPUT_ATTRIBUTE, OUTPUT_META = range(3)
    OUTPUT_METHODS = ("Class", "Feature", "Meta")

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
    place_cluster_ids = Setting(OUTPUT_CLASS)
    output_name = Setting("Cluster")
    auto_run = Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None
        self.km = None
        self.optimization_runs = []

        box = gui.vBox(self.controlArea, "Number of Clusters")
        layout = QGridLayout()
        self.n_clusters = bg = gui.radioButtonsInBox(
            box, self, "optimize_k", [], orientation=layout,
            callback=self.update)
        layout.addWidget(
            gui.appendRadioButton(bg, "Fixed:", addToLayout=False),
            1, 1)
        sb = gui.hBox(None, margin=0)
        self.fixedSpinBox = gui.spin(
            sb, self, "k", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight, callback=self.update_k)
        gui.rubber(sb)
        layout.addWidget(sb, 1, 2)

        layout.addWidget(
            gui.appendRadioButton(bg, "Optimized from", addToLayout=False), 2, 1)
        ftobox = gui.hBox(None)
        ftobox.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(ftobox)
        gui.spin(
            ftobox, self, "k_from", minv=2, maxv=29,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_from)
        gui.widgetLabel(ftobox, "to")
        self.fixedSpinBox = gui.spin(
            ftobox, self, "k_to", minv=3, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_to)
        gui.rubber(ftobox)

        layout.addWidget(gui.widgetLabel(None, "Scoring: "),
                         5, 1, Qt.AlignRight)
        layout.addWidget(
            gui.comboBox(
                None, self, "scoring", label="Scoring",
                items=list(zip(*self.SCORING_METHODS))[0],
                callback=self.update), 5, 2)

        box = gui.vBox(self.controlArea, "Initialization")
        gui.comboBox(
            box, self, "smart_init", items=self.INIT_METHODS,
            callback=self.update)

        layout = QGridLayout()
        box2 = gui.widgetBox(box, orientation=layout)
        box2.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        layout.addWidget(gui.widgetLabel(None, "Re-runs: "),
                         0, 0, Qt.AlignLeft)
        sb = gui.hBox(None, margin=0)
        layout.addWidget(sb, 0, 1)
        gui.lineEdit(
            sb, self, "n_init", controlWidth=60,
            valueType=int, validator=QIntValidator(),
            callback=self.update)
        layout.addWidget(gui.widgetLabel(None, "Maximal iterations: "),
                         1, 0, Qt.AlignLeft)
        sb = gui.hBox(None, margin=0)
        layout.addWidget(sb, 1, 1)
        gui.lineEdit(sb, self, "max_iterations",
                     controlWidth=60, valueType=int,
                     validator=QIntValidator(),
                     callback=self.update)

        box = gui.vBox(self.controlArea, "Output")
        gui.comboBox(box, self, "place_cluster_ids",
                     label="Append cluster ID as:", orientation=Qt.Horizontal,
                     callback=self.send_data, items=self.OUTPUT_METHODS)
        gui.lineEdit(box, self, "output_name",
                     label="Name:", orientation=Qt.Horizontal,
                     callback=self.send_data)

        gui.separator(self.buttonsArea, 30)
        self.apply_button = gui.auto_commit(
            self.buttonsArea, self, "auto_run", "Apply", box=None,
            commit=self.commit
        )
        gui.rubber(self.controlArea)

        self.table_model = QStandardItemModel(self)
        self.table_model.setHorizontalHeaderLabels(["k", "Score"])
        self.table_model.setColumnCount(2)

        self.table_box = gui.vBox(
            self.mainArea, "Optimization Report", addSpace=0)
        table = self.table_view = QTableView(self.table_box)
        table.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSelectionMode(QTableView.SingleSelection)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.verticalHeader().hide()
        self.bar_delegate = gui.ColoredBarItemDelegate(self, color=Qt.cyan)
        table.setItemDelegateForColumn(1, self.bar_delegate)
        table.setModel(self.table_model)
        table.selectionModel().selectionChanged.connect(
            self.table_item_selected)
        table.setColumnWidth(0, 40)
        table.setColumnWidth(1, 120)
        table.horizontalHeader().setStretchLastSection(True)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mainArea.setSizePolicy(QSizePolicy.Maximum,
                                    QSizePolicy.Preferred)
        self.table_box.setSizePolicy(QSizePolicy.Fixed,
                                     QSizePolicy.MinimumExpanding)
        self.table_view.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.MinimumExpanding)
        self.table_box.layout().addWidget(self.table_view)
        self.hide_show_opt_results()

    def adjustSize(self):
        self.ensurePolished()
        s = self.sizeHint()
        self.resize(s)

    def hide_show_opt_results(self):
        [self.mainArea.hide, self.mainArea.show][self.optimize_k]()
        QTimer.singleShot(100, self.adjustSize)

    def sizeHint(self):
        s = self.controlArea.sizeHint()
        if self.optimize_k and not self.mainArea.isHidden():
            s.setWidth(s.width() + self.mainArea.sizeHint().width() +
                       4 * self.childrenRect().x())
        return s

    def update_k(self):
        self.optimize_k = False
        self.update()

    def update_from(self):
        self.k_to = max(self.k_from + 1, self.k_to)
        self.optimize_k = True
        self.update()

    def update_to(self):
        self.k_from = min(self.k_from, self.k_to - 1)
        self.optimize_k = True
        self.update()

    def set_optimization(self):
        self.updateOptimizationGui()
        self.update()

    def check_data_size(self, n, msg_group):
        msg_group.add_message(
            "not_enough_data",
            "Too few ({}) unique data instances for {} clusters")
        if n > len(self.data):
            msg_group.not_enough_data(len(self.data), n)
            return False
        else:
            msg_group.not_enough_data.clear()
            return True

    def run_optimization(self):
        # Disabling is needed since this function is not reentrant
        # Fast clicking on, say, "To: " causes multiple calls
        try:
            self.controlArea.setDisabled(True)
            self.optimization_runs = []
            error = ""
            if not self.check_data_size(self.k_from, self.Error):
                return
            self.check_data_size(self.k_to, self.Warning)
            k_to = min(self.k_to, len(self.data))
            kmeans = KMeans(
                init=['random', 'k-means++'][self.smart_init],
                n_init=self.n_init, max_iter=self.max_iterations,
                compute_silhouette_score=self.scoring == self.SILHOUETTE)
            with self.progressBar(k_to - self.k_from + 1) as progress:
                for k in range(self.k_from, k_to + 1):
                    progress.advance()
                    kmeans.params["n_clusters"] = k
                    try:
                        self.optimization_runs.append((k, kmeans(self.data)))
                    except BaseException as exc:
                        error = str(exc)
                        self.optimization_runs.append((k, error))
            if all(isinstance(score, str)
                   for _, score in self.optimization_runs):
                self.Error.failed(error)  # Report just the last error
                self.optimization_runs = []
        finally:
            self.controlArea.setDisabled(False)
        self.show_results()
        self.send_data()

    def cluster(self):
        if not self.check_data_size(self.k, self.Error):
            return
        try:
            self.km = KMeans(
                n_clusters=self.k,
                init=['random', 'k-means++'][self.smart_init],
                n_init=self.n_init,
                max_iter=self.max_iterations)(self.data)
        except BaseException as exc:
            self.Error.failed(str(exc))
            self.km = None
        self.send_data()

    def run(self):
        self.clear_messages()
        if not self.data:
            return
        if self.optimize_k:
            self.run_optimization()
        else:
            self.cluster()

    def commit(self):
        self.run()

    def show_results(self):
        _, scoring_method, minimize, normal = self.SCORING_METHODS[self.scoring]
        k_scores = [(k,
                     scoring_method(run) if not isinstance(run, str) else run)
                    for k, run in self.optimization_runs]
        scores = [score for _, score in k_scores if not isinstance(score, str)]

        min_score, max_score = min(scores, default=0), max(scores, default=1)
        best_score = min_score if minimize else max_score
        if normal:
            min_score, max_score = 0, 1
            nplaces = 3
        else:
            nplaces = min(5, np.floor(abs(math.log(max(max_score, 1e-10)))) + 2)
        score_span = (max_score - min_score) or 1
        self.bar_delegate.scale = (min_score, max_score)
        self.bar_delegate.float_fmt = "%%.%if" % int(nplaces)

        model = self.table_model
        model.setRowCount(len(k_scores))
        no_selection = True
        for i, (k, score) in enumerate(k_scores):
            item0 = model.item(i, 0) or QStandardItem()
            item0.setData(k, Qt.DisplayRole)
            item0.setTextAlignment(Qt.AlignCenter)
            model.setItem(i, 0, item0)
            item = model.item(i, 1) or QStandardItem()
            if not isinstance(score, str):
                item.setData(score, Qt.DisplayRole)
                item.setData(None, Qt.ToolTipRole)
                bar_ratio = 0.95 * (score - min_score) / score_span
                item.setData(bar_ratio, gui.BarRatioRole)
                if no_selection and score == best_score:
                    self.table_view.selectRow(i)
                    no_selection = False
                color = Qt.black
                flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            else:
                item.setData("clustering failed", Qt.DisplayRole)
                item.setData(score, Qt.ToolTipRole)
                item.setData(None, gui.BarRatioRole)
                color = Qt.gray
                flags = Qt.NoItemFlags
            item0.setData(QBrush(color), Qt.ForegroundRole)
            item0.setFlags(flags)
            item.setData(QBrush(color), Qt.ForegroundRole)
            item.setFlags(flags)
            model.setItem(i, 1, item)
        self.table_view.resizeRowsToContents()

        self.table_view.show()
        if minimize:
            self.table_box.setTitle("Scoring (smaller is better)")
        else:
            self.table_box.setTitle("Scoring (bigger is better)")
        QTimer.singleShot(0, self.adjustSize)

    def update(self):
        self.hide_show_opt_results()
        self.run()

    def selected_row(self):
        indices = self.table_view.selectedIndexes()
        rows = {ind.row() for ind in indices}
        if len(rows) == 1:
            return rows.pop()

    def table_item_selected(self):
        row = self.selected_row()
        if row is not None:
            self.send_data()

    def send_data(self):
        if self.optimize_k:
            row = self.selected_row() if self.optimization_runs else None
            km = self.optimization_runs[row][1] if row is not None else None
        else:
            km = self.km
        if not self.data or not km:
            self.send("Annotated Data", None)
            self.send("Centroids", None)
            return

        clust_var = DiscreteVariable(
            self.output_name, values=["C%d" % (x + 1) for x in range(km.k)])
        clust_ids = km(self.data)
        domain = self.data.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        if self.place_cluster_ids == self.OUTPUT_CLASS:
            if classes:
                meta_attrs += classes
            classes = [clust_var]
        elif self.place_cluster_ids == self.OUTPUT_ATTRIBUTE:
            attributes += (clust_var, )
        else:
            meta_attrs += (clust_var, )

        domain = Domain(attributes, classes, meta_attrs)
        new_table = Table.from_table(domain, self.data)
        new_table.get_column_view(clust_var)[0][:] = clust_ids.X.ravel()

        centroids = Table(Domain(km.pre_domain.attributes), km.centroids)

        self.send("Annotated Data", new_table)
        self.send("Centroids", centroids)

    @check_sql_input
    def set_data(self, data):
        self.data = data
        if data is None:
            self.Error.clear()
            self.Warning.clear()
            self.table_model.setRowCount(0)
            self.send("Annotated Data", None)
            self.send("Centroids", None)
        else:
            self.data = data
            self.run()

    def send_report(self):
        k_clusters = self.k
        if self.optimize_k and self.optimization_runs and self.selected_row() is not None:
            k_clusters = self.optimization_runs[self.selected_row()][1].k
        self.report_items((
            ("Number of clusters", k_clusters),
            ("Optimization",
             self.optimize_k != 0 and
             "{}, {} re-runs limited to {} steps".format(
                 self.INIT_METHODS[self.smart_init].lower(),
                 self.n_init, self.max_iterations)),
            ("Cluster ID in output",
             self.append_cluster_ids and
             "'{}' (as {})".format(
                 self.output_name,
                 self.OUTPUT_METHODS[self.place_cluster_ids].lower()))
        ))
        if self.data:
            self.report_data("Data", self.data)
            if self.optimize_k:
                self.report_table(
                    "Scoring by {}".format(self.SCORING_METHODS[self.scoring][0]
                                           ),
                    self.table_view)


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWKMeans()
    d = Table("iris.tab")
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()


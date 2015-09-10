import math
import numpy as np

from PyQt4.QtGui import QGridLayout, QSizePolicy, \
    QTableView, QStandardItemModel, QStandardItem, QIntValidator
from PyQt4.QtCore import Qt, QTimer

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, DiscreteVariable
from Orange.clustering import KMeans


class OWKMeans(widget.OWWidget):
    name = "k-Means"
    description = "k-means clustering algorithm with silhouette-based " \
                  "quality estimation."
    icon = "icons/KMeans.svg"
    priority = 2100

    inputs = [("Data", Table, "set_data")]

    outputs = [("Annotated Data", Table, widget.Default),
               ("Centroids", Table)]

    INIT_KMEANS, INIT_RANDOM = range(2)
    INIT_METHODS = "Initialize with KMeans++", "Random initialization"

    SILHOUETTE, INTERCLUSTER, DISTANCES = range(3)
    SCORING_METHODS = [("Silhouette", lambda km: km.silhouette, False),
                       ("Inter-cluster distance",
                        lambda km: km.inter_cluster, True),
                       ("Distance to centroids",
                        lambda km: km.inertia, True)]

    OUTPUT_CLASS, OUTPUT_ATTRIBUTE, OUTPUT_META = range(3)
    OUTPUT_METHODS = ("Class", "Feature", "Meta")

    k = Setting(8)
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
    auto_run = Setting(False)

    def __init__(self):
        super().__init__()

        self.data = None
        self.km = None
        self.optimization_runs = []

        box = gui.widgetBox(self.controlArea, "Number of Clusters")
        layout = QGridLayout()
        bg = gui.radioButtonsInBox(
            box, self, "optimize_k", [], orientation=layout,
            callback=self.update)
        layout.addWidget(
            gui.appendRadioButton(bg, "Fixed", addToLayout=False),
            1, 1)
        sb = gui.widgetBox(None, margin=0, orientation="horizontal")
        self.fixedSpinBox = gui.spin(
            sb, self, "k", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight, callback=self.update_k)
        gui.rubber(sb)
        layout.addWidget(sb, 1, 2)

        layout.addWidget(
            gui.appendRadioButton(bg, "Optimized", addToLayout=False), 2, 1)
        ftobox = gui.widgetBox(None, orientation="horizontal")
        ftobox.layout().setMargin(0)
        layout.addWidget(ftobox)
        gui.spin(
            ftobox, self, "k_from", minv=2, maxv=29,
            controlWidth=60, alignment=Qt.AlignRight,
            callback=self.update_from)
        gui.widgetLabel(ftobox, "  To: ")
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

        box = gui.widgetBox(self.controlArea, "Initialization")
        gui.comboBox(
            box, self, "smart_init", items=self.INIT_METHODS,
            callback=self.update)

        layout = QGridLayout()
        box2 = gui.widgetBox(box, orientation=layout)
        box2.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        layout.addWidget(gui.widgetLabel(None, "Re-runs: "),
                         0, 0, Qt.AlignLeft)
        sb = gui.widgetBox(None, margin=0, orientation="horizontal")
        layout.addWidget(sb, 0, 1)
        gui.lineEdit(
            sb, self, "n_init", controlWidth=60,
            valueType=int, validator=QIntValidator(),
            callback=self.update)
        layout.addWidget(gui.widgetLabel(None, "Maximal iterations: "),
                         1, 0, Qt.AlignLeft)
        sb = gui.widgetBox(None, margin=0, orientation="horizontal")
        layout.addWidget(sb, 1, 1)
        gui.lineEdit(sb, self, "max_iterations",
                     controlWidth=60, valueType=int,
                     validator=QIntValidator(),
                     callback=self.update)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.comboBox(box, self, "place_cluster_ids",
                     label="Append cluster id as ", orientation="horizontal",
                     callback=self.send_data, items=self.OUTPUT_METHODS)
        gui.lineEdit(box, self, "output_name",
                     label="Name ", orientation="horizontal",
                     callback=self.send_data)

        gui.auto_commit(self.controlArea, self, "auto_run", "Run",
                        checkbox_label="Run after any change  ",
                        orientation="horizontal")
        gui.rubber(self.controlArea)

        self.table_model = QStandardItemModel(self)
        self.table_model.setHorizontalHeaderLabels(["k", "Score"])
        self.table_model.setColumnCount(2)

        self.table_box = gui.widgetBox(
            self.mainArea, "Optimization Report", addSpace=0)
        table = self.table_view = QTableView(self.table_box)
        table.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSelectionMode(QTableView.SingleSelection)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.verticalHeader().hide()
        table.setItemDelegateForColumn(1, gui.TableBarItem(self))
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
        s = self.leftWidgetPart.sizeHint()
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

    def check_data_size(self, n):
        if n > len(self.data):
            self.error("Not enough unique data instances ({}) for given "
                       "number of clusters ({}).".format(len(self.data), n))
            return False
        return True

    def run_optimization(self):
        # Disabling is needed since this function is not reentrant
        # Fast clicking on, say, "To: " causes multiple calls
        try:
            self.controlArea.setDisabled(True)
            self.optimization_runs = []
            if self.check_data_size(self.k_to):
                self.progressBarInit()
                progress_steps = self.k_to - self.k_from + 1
                self.optimization_runs = []
                kmeans = KMeans(
                    init=['random', 'k-means++'][self.smart_init],
                    n_init=self.n_init,
                    max_iter=self.max_iterations)
                for k in range(self.k_from, self.k_to + 1):
                    self.progressBarSet(100.0 * (k - self.k_from) /
                                        progress_steps)
                    kmeans.params["n_clusters"] = k
                    self.optimization_runs.append((k, kmeans(self.data)))
                self.progressBarFinished()
        finally:
            self.controlArea.setDisabled(False)
        self.show_results()
        self.send_data()


    def cluster(self):
        if not self.check_data_size(self.k):
            return
        self.km = KMeans(
            n_clusters=self.k,
            init=['random', 'k-means++'][self.smart_init],
            n_init=self.n_init,
            max_iter=self.max_iterations)(self.data)
        self.send_data()

    def run(self):
        self.error()
        if not self.data:
            return
        if self.optimize_k:
            self.run_optimization()
        else:
            self.cluster()

    commit = run

    def show_results(self):
        minimize = self.SCORING_METHODS[self.scoring][2]
        k_scores = [(k, self.SCORING_METHODS[self.scoring][1](run)) for
                    k, run in self.optimization_runs]
        scores = list(zip(*k_scores))[1]
        if minimize:
            best_score, worst_score = min(scores), max(scores)
        else:
            best_score, worst_score = max(scores), min(scores)

        best_run = scores.index(best_score)
        score_span = (best_score - worst_score) or 1
        max_score = max(scores)
        nplaces = min(5, int(abs(math.log(max(max_score, 1e-10)))) + 2)
        fmt = "{{:.{}}}".format(nplaces)
        model = self.table_model
        model.setRowCount(len(k_scores))
        for i, (k, score) in enumerate(k_scores):
            item = model.item(i, 0)
            if item is None:
                item = QStandardItem()
            item.setData(k, Qt.DisplayRole)
            item.setTextAlignment(Qt.AlignCenter)
            model.setItem(i, 0, item)
            item = model.item(i, 1)
            if item is None:
                item = QStandardItem()
            item.setData(fmt.format(score), Qt.DisplayRole)
            bar_ratio = 0.95 * (score - worst_score) / score_span
            item.setData(bar_ratio, gui.TableBarItem.BarRole)
            model.setItem(i, 1, item)
        self.table_view.resizeRowsToContents()

        self.table_view.selectRow(best_run)
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
            self.send_data(row)

    def send_data(self, row=None):
        if self.optimize_k:
            if row is None:
                row = self.selected_row()
            km = self.optimization_runs[row][1]
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
        X, Y, metas = self.data.X, self.data.Y, self.data.metas
        if self.place_cluster_ids == self.OUTPUT_CLASS:
            if classes:
                meta_attrs += classes
                metas = np.hstack((metas, Y.reshape(len(self.data), 1)))
            classes = [clust_var]
            Y = clust_ids
        elif self.place_cluster_ids == self.OUTPUT_ATTRIBUTE:
            attributes += (clust_var, )
            X = np.hstack((X, clust_ids))
        else:
            meta_attrs += (clust_var, )
            metas = np.hstack((metas, clust_ids))

        domain = Domain(attributes, classes, meta_attrs)
        new_table = Table(domain, X, Y, metas, self.data.W)

        centroids = Table(Domain(km.pre_domain.attributes), km.centroids)

        self.send("Annotated Data", new_table)
        self.send("Centroids", centroids)

    def set_data(self, data):
        self.data = data
        if data is None:
            self.table_view.setRowCount(0)
        else:
            self.data = data
            self.run()

    def sendReport(self):
        settings = [("Distance measure",
                     self.distanceMeasures[self.distanceMeasure][0]),
                    ("Initialization",
                     self.initializations[self.initializationType][0]),
                    ("Restarts",
                     self.restarts)]
        if self.optimized:
            self.reportSettings("Settings", settings)
            self.reportSettings("Optimization",
                                [("Minimum num. of clusters",
                                  self.optimizationFrom),
                                 ("Maximum num. of clusters",
                                  self.optimizationTo),
                                 ("Scoring method",
                                  self.scoringMethods[self.scoring][0])])
        else:
            self.reportSettings("Settings",
                                settings + [("Number of clusters (K)",
                                             self.K)])

        self.reportData(self.data)
        if self.optimized:
            import OWReport

            self.reportSection("Cluster size optimization report")
            self.reportRaw(OWReport.reportTable(self.table_view))


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWKMeans()
    d = Table("iris.tab")
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()


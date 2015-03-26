"""
<name>k-Means Clustering</name>
<description>k-means clustering.</description>
<icon>icons/kMeansClustering.svg</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2300</priority>
"""

import math
import random

from PyQt4.QtGui import QGridLayout, QSizePolicy
from PyQt4.QtCore import Qt, QTimer

import Orange
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import colorpalette, itemmodels

class OWKMeans(widget.OWWidget):
    name = "K Means"
    description = ("K Means clustering.")
    icon = "icons/KMeans.svg"
    priority = 2100

    inputs = [("Data", Orange.data.table, "set_data")]

    outputs = [("Annotated Data", Orange.data.Table),
               ("Centroids", Orange.data.Table)]

    INIT_KMEANS, INIT_RANDOM = range(2)
    INIT_METHODS = "Initialize with KMeans++", "Random initialization"

    SILHOUETTE_H, SILHOUETTE, INTERCLUSTER, DISTANCES = range(4)
    SCORING_METHODS = ("Silhouette (heuristic)", "Silhouette",
                       "Between cluster distance", "Distance to centroids")

    OUTPUT_CLASS, OUTPUT_ATTRIBUTE, OUTPUT_META = range(3)
    OUTPUT_METHODS = ("Append cluster id as class",
                      "Append cluster id as feature",
                      "Append cluster id as meta")

    k = Setting(8)
    k_from = Setting(2)
    k_to = Setting(8)
    optimize_k = Setting(False)
    max_iterations = Setting(300)
    n_init = Setting(10)
    smart_init = Setting(INIT_KMEANS)
    scoring = Setting(SILHOUETTE_H)
    append_cluster_ids = Setting(True)
    place_cluster_ids = Setting(OUTPUT_CLASS)
    output_name = Setting("Cluster")
    auto_run = Setting(True)


    def __init__(self):
        super().__init__()

        self.data = None
        self.km = None

        box = gui.widgetBox(self.controlArea, "Number of Clusters")
        layout =QGridLayout()
        bg = gui.radioButtonsInBox(
            box, self, "optimize_k", [], orientation=layout,
            callback=self.set_optimization)

        layout.addWidget(
            gui.appendRadioButton(bg, "Fixed", addToLayout=False),
            1, 1)
        self.fixedSpinBox = gui.spin(
            None, self, "k", minv=2, maxv=30,
            callback=self.update, callbackOnReturn=True)
        layout.addWidget(self.fixedSpinBox, 1, 2)

        layout.addWidget(
            gui.appendRadioButton(bg, "Optimized", addToLayout=False), 2, 1)
        layout.addWidget(gui.widgetLabel(None, "From: "), 3, 1, Qt.AlignRight)
        layout.addWidget(gui.spin(
            None, self, "k_from", minv=2, maxv=30,
            callback=self.update, callbackOnReturn=True), 3, 2)
        layout.addWidget(gui.widgetLabel(None, "To: "), 4, 1, Qt.AlignRight)
        self.fixedSpinBox = gui.spin(
            None, self, "k_to", minv=2, maxv=30,
            callback=self.update, callbackOnReturn=True)
        layout.addWidget(self.fixedSpinBox, 4, 2)

        layout.addWidget(gui.widgetLabel(None, "Scoring: "),
                         5, 1, Qt.AlignRight)
        layout.addWidget(
            gui.comboBox(
                None, self, "scoring", label="Scoring",
                items=self.SCORING_METHODS, callback=self.update), 5, 2)

        box = gui.widgetBox(self.controlArea, "Initialization")
        gui.comboBox(
            box, self, "smart_init", items=self.INIT_METHODS,
            callback=self.update)

        gui.spin(box, self, "n_init", label="Re-runs", minv=1, maxv=100,
                   callback=self.update, callbackOnReturn=True)
        gui.spin(box, self, "max_iterations", label="Maximal iterations",
                 minv=100, maxv=1000, step=100,
                 callback=self.update, callbackOnReturn=True)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.comboBox(box, self, "place_cluster_ids", items=self.OUTPUT_METHODS)
        gui.lineEdit(box, self, "output_name", orientation="horizontal")

        gui.auto_commit(self.controlArea, self, "auto_run", "Run",
                        checkbox_label="Run after any change")
        gui.rubber(self.controlArea)

        # display of clustering results
        self.optimizationReportBox = gui.widgetBox(self.mainArea)
        self.tableBox = gui.widgetBox(
            self.optimizationReportBox, "Optimization Report")
        """
        self.table = QTableView(
            self.tableBox, selectionMode=QTableWidget.SingleSelection)

        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["k", "Best", "Score"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setItemDelegateForColumn(
            2, OWGUI.TableBarItem(self, self.table))

        self.table.setItemDelegateForColumn(
            1, OWGUI.IndicatorItemDelegate(self))

        self.table.setSizePolicy(QSizePolicy.MinimumExpanding,
                                 QSizePolicy.MinimumExpanding)

        self.connect(self.table,
                     SIGNAL("itemSelectionChanged()"),
                     self.tableItemSelected)

        self.setSizePolicy(QSizePolicy.Preferred,
                           QSizePolicy.Preferred)
        """
        self.mainArea.setSizePolicy(QSizePolicy.MinimumExpanding,
                                    QSizePolicy.MinimumExpanding)

        gui.rubber(self.topWidgetPart)
        self.update_optimization_gui()

    def adjustSize(self):
        self.ensurePolished()
        s = self.sizeHint()
        self.resize(s)

    def hideOptResults(self):
        self.mainArea.hide()
        QTimer.singleShot(100, self.adjustSize)

    def showOptResults(self):
        self.mainArea.show()
        QTimer.singleShot(100, self.adjustSize)

    def sizeHint(self):
        s = self.leftWidgetPart.sizeHint()
        if self.optimized and not self.mainArea.isHidden():
            s.setWidth(s.width() + self.mainArea.sizeHint().width() + \
                       self.childrenRect().x() * 4)
        return s

    def update_optimization_gui(self):
        self.fixedSpinBox.setDisabled(bool(self.optimized))
        self.optimizationBox.setDisabled(not bool(self.optimized))
        if self.optimized:
            self.showOptResults()
        else:
            self.hideOptResults()

    def updateOptimizationFrom(self):
        self.optimizationTo = max([self.optimizationFrom + 1,
                                   self.optimizationTo])
        self.update()

    def updateOptimizationTo(self):
        self.optimizationFrom = min([self.optimizationFrom,
                                     self.optimizationTo - 1])
        self.update()

    def set_optimization(self):
        self.updateOptimizationGui()
        self.update()

    def runOptimization(self):
        if self.optimizationTo > len(set(self.data)):
            self.error("Not enough unique data instances (%d) for given "
                       "number of clusters (%d)." % \
                       (len(set(self.data)), self.optimizationTo))
            return

        random.seed(0)
        data = self.data
        nstart = self.restarts
        initialization = self.initializations[self.initializationType][1]
        distance = self.distanceMeasures[self.distanceMeasure][1]
        scoring = self.scoringMethods[self.scoring][1]
        try:
            self.progressBarInit()
            Ks = range(self.optimizationFrom, self.optimizationTo + 1)
            outer_callback_count = len(Ks) * self.restarts
            outer_callback_state = {"restart": 0}
            optimizationRun = []
            for k in Ks:
                def outer_progress(km):
                    outer_callback_state["restart"] += 1
                    self.progressBarSet(
                        100.0 * outer_callback_state["restart"] \
                        / outer_callback_count
                    )

                def inner_progress(km):
                    estimate = self.progressEstimate(km)
                    self.progressBarSet(min(estimate / outer_callback_count + \
                                            outer_callback_state["restart"] * \
                                            100.0 / outer_callback_count,
                                            100.0))

                kmeans = orngClustering.KMeans(
                    data,
                    centroids=k,
                    minscorechange=0,
                    nstart=nstart,
                    initialization=initialization,
                    distance=distance,
                    scoring=scoring,
                    outer_callback=outer_progress,
                    inner_callback=inner_progress
                    )
                optimizationRun.append((k, kmeans))

                if nstart == 1:
                    outer_progress(None)

            self.optimizationRun = optimizationRun
            minimize = getattr(scoring, "minimize", False)
            self.optimizationRunSorted = \
                    sorted(optimizationRun,
                           key=lambda item: item[1].score,
                           reverse=minimize)

            self.progressBarFinished()

            self.bestRun = self.optimizationRunSorted[-1]
            self.showResults()
            self.sendData()
        except Exception as ex:
            self.error(0, "An error occurred while running optimization. "
                          "Reason: " + str(ex))
            raise

    def cluster(self):
        if self.K > len(set(self.data)):
            self.error("Not enough unique data instances (%d) for given "
                       "number of clusters (%d)." % \
                       (len(set(self.data)), self.K))
            return
        random.seed(0)

        self.km = orngClustering.KMeans(
            centroids=self.K,
            minscorechange=0,
            nstart=self.restarts,
            initialization=self.initializations[self.initializationType][1],
            distance=self.distanceMeasures[self.distanceMeasure][1],
            scoring=self.scoringMethods[self.scoring][1],
            inner_callback=self.clusterCallback,
            )
        self.progressBarInit()
        self.km(self.data)
        self.sendData()
        self.progressBarFinished()

    def clusterCallback(self, km):
        norm = math.log(len(self.data), 10)
        if km.iteration < norm:
            self.progressBarSet(80.0 * km.iteration / norm)
        else:
            self.progressBarSet(80.0 + 0.15 * \
                                (1.0 - math.exp(norm - km.iteration)))

    def progressEstimate(self, km):
        norm = math.log(len(km.data), 10)
        if km.iteration < norm:
            return min(80.0 * km.iteration / norm, 90.0)
        else:
            return min(80.0 + 0.15 * (1.0 - math.exp(norm - km.iteration)),
                       90.0)

    def scoreFmt(self, score, max_decimals=10):
        if score > 0 and score < 1:
            fmt = "%%.%if" % min(int(abs(math.log(max(score, 1e-10)))) + 2,
                                 max_decimals)
        else:
            fmt = "%.1f"
        return fmt

    def showResults(self):
        self.table.setRowCount(len(self.optimizationRun))
        scoring = self.scoringMethods[self.scoring][1]
        minimize = getattr(scoring, "minimize", False)

        bestScore = self.bestRun[1].score
        worstScore = self.optimizationRunSorted[0][1].score

        if minimize:
            bestScore, worstScore = worstScore, bestScore

        scoreSpan = (bestScore - worstScore) or 1

        for i, (k, run) in enumerate(self.optimizationRun):
            item = OWGUI.tableItem(self.table, i, 0, k)
            item.setData(Qt.TextAlignmentRole, QVariant(Qt.AlignCenter))

            item = OWGUI.tableItem(self.table, i, 1, None)
            item.setData(OWGUI.IndicatorItemDelegate.IndicatorRole,
                         QVariant((k, run) == self.bestRun))

            item.setData(Qt.TextAlignmentRole, QVariant(Qt.AlignCenter))

            fmt = self.scoreFmt(run.score)
            item = OWGUI.tableItem(self.table, i, 2, fmt % run.score)
            barRatio = 0.95 * (run.score - worstScore) / scoreSpan

            item.setData(OWGUI.TableBarItem.BarRole, QVariant(barRatio))
            if (k, run) == self.bestRun:
                self.table.selectRow(i)

        for i in range(2):
            self.table.resizeColumnToContents(i)

        self.table.show()

        if minimize:
            self.tableBox.setTitle("Optimization Report (smaller is better)")
        else:
            self.tableBox.setTitle("Optimization Report (bigger is better)")

        QTimer.singleShot(0, self.adjustSize)

    def run(self):
        self.error(0)
        if not self.data:
            return
        if self.optimized:
            self.runOptimization()
        else:
            self.cluster()

    def update(self):
        if self.runAnyChange:
            self.run()
        else:
            self.settingsChanged = True

    def tableItemSelected(self):
        selectedItems = self.table.selectedItems()
        rows = set([item.row() for item in selectedItems])
        if len(rows) == 1:
            row = rows.pop()
            self.sendData(self.optimizationRun[row][1])

    def sendData(self, km=None):
        if km is None:
            km = self.bestRun[1] if self.optimized else self.km
        if not self.data or not km:
            self.send("Data", None)
            self.send("Centroids", None)
            return

        clustVar = orange.EnumVariable(self.classifyName,
                                       values=["C%d" % (x + 1) \
                                               for x in range(km.k)])

        origDomain = self.data.domain
        if self.addIdAs == 0:
            domain = orange.Domain(origDomain.attributes, clustVar)
            if origDomain.classVar:
                domain.addmeta(orange.newmetaid(), origDomain.classVar)
            aid = -1
        elif self.addIdAs == 1:
            domain = orange.Domain(origDomain.attributes + [clustVar],
                                   origDomain.classVar)
            aid = len(origDomain.attributes)
        else:
            domain = orange.Domain(origDomain.attributes,
                                   origDomain.classVar)
            aid = orange.newmetaid()
            domain.addmeta(aid, clustVar)

        domain.addmetas(origDomain.getmetas())

        # construct a new data set, with a class as assigned by
        # k-means clustering
        new = orange.ExampleTable(domain, self.data)
        for ex, midx in izip(new, km.clusters):
            ex[aid] = midx

        centroids = orange.ExampleTable(domain, km.centroids)
        for i, c in enumerate(centroids):
            c[aid] = i
            if origDomain.classVar:
                c[origDomain.classVar] = "?"

        self.send("Data", new)
        self.send("Centroids", centroids)

    def setData(self, data):
        "Handle data from the input signal."
        self.runButton.setEnabled(bool(data))
        if not data:
            self.data = None
            self.table.setRowCount(0)
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
            self.reportRaw(OWReport.reportTable(self.table))


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication
    a = QApplication(sys.argv)
    ow = OWKMeans()
#    d = orange.ExampleTable("iris.tab")
#    ow.setData(d)
    ow.show()
    a.exec()
    ow.saveSettings()


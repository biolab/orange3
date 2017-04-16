import numpy as np

from PyQt4 import QtGui

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.clustering import DBSCAN
from Orange import distance


class OWDBSCAN(widget.OWWidget):
    name = "DBSCAN"
    description = "Density-based spatial clustering."
    icon = "icons/DBSCAN.svg"
    priority = 2150

    inputs = [("Data", Table, "set_data")]

    outputs = [("Annotated Data", Table, widget.Default)]

    OUTPUT_CLASS, OUTPUT_ATTRIBUTE, OUTPUT_META = range(3)
    OUTPUT_METHODS = ("Class", "Feature", "Meta")
    METRICS = [
        ("Euclidean", "euclidean"),
        ("Manhattan", "manhattan"),
        # ("Cosine", distance.Cosine),
        ("Jaccard", distance.Jaccard),
        # ("Spearman", distance.SpearmanR),
        # ("Spearman absolute", distance.SpearmanRAbsolute),
        # ("Pearson", distance.PearsonR),
        # ("Pearson absolute", distance.PearsonRAbsolute),
    ]

    min_samples = Setting(5)
    eps = Setting(0.5)
    metric_idx = Setting(0)
    append_cluster_ids = Setting(True)
    place_cluster_ids = Setting(OUTPUT_CLASS)
    output_name = Setting("Cluster")
    auto_run = Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.db = None

        box = gui.widgetBox(self.controlArea, "Parameters")
        gui.spin(box, self, "min_samples", 1, 100, 1, callback=self._invalidate,
                 label="Core point neighbors")
        gui.doubleSpin(box, self, "eps", 0.1, 0.9, 0.1,
                       callback=self._invalidate,
                       label="Neighborhood distance")

        box = gui.widgetBox(self.controlArea, self.tr("Distance Metric"))
        gui.comboBox(box, self, "metric_idx",
                     items=list(zip(*self.METRICS))[0],
                     callback=self._invalidate)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.comboBox(box, self, "place_cluster_ids",
                     label="Append cluster id as ", orientation="horizontal",
                     callback=self.send_data, items=self.OUTPUT_METHODS)
        gui.lineEdit(box, self, "output_name",
                     label="Name: ", orientation="horizontal",
                     callback=self.send_data)

        gui.auto_commit(self.controlArea, self, "auto_run", "Run",
                        checkbox_label="Run after any change  ",
                        orientation="horizontal")
        gui.rubber(self.controlArea)

        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

    def adjustSize(self):
        self.ensurePolished()
        self.resize(self.leftWidgetPart.sizeHint())

    def check_data_size(self):
        if len(self.data) < 2:
            self.error("Not enough unique data instances "
                       "({}).".format(len(self.data)))
            return False
        return True

    def commit(self):
        self.error()
        if not self.data:
            return
        self.cluster()

    def cluster(self):
        if not self.check_data_size():
            return
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.METRICS[self.metric_idx][1]
        )(self.data)
        self.send_data()

    def send_data(self, row=None):
        model = self.model
        if not self.data or not self.model:
            self.send("Data", None)
            return

        clusters = [c if c >= 0 else None for c in model.labels_]
        k = len(set(clusters) - {None})
        clusters = np.array(clusters).reshape(len(self.data), 1)
        core_samples = set(model.core_sample_indices_)
        in_core = np.array([0 if (i in core_samples) else 1
                            for i in range(len(self.data))])
        in_core = in_core.reshape(len(self.data), 1)

        clust_var = DiscreteVariable(
            self.output_name, values=["C%d" % (x + 1) for x in range(k)])
        in_core_var = ContinuousVariable("DBSCAN Core")

        domain = self.data.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        X, Y, metas = self.data.X, self.data.Y, self.data.metas

        if self.place_cluster_ids == self.OUTPUT_CLASS:
            if classes:
                meta_attrs += classes
                metas = np.hstack((metas, Y.reshape(len(self.data), 1)))
            classes = [clust_var]
            Y = clusters
        elif self.place_cluster_ids == self.OUTPUT_ATTRIBUTE:
            attributes += (clust_var, )
            X = np.hstack((X, clusters))
        else:
            meta_attrs += (clust_var, )
            metas = np.hstack((metas, clusters))
        meta_attrs += (in_core_var, )
        metas = np.hstack((metas, in_core))

        domain = Domain(attributes, classes, meta_attrs)
        new_table = Table(domain, X, Y, metas, self.data.W)

        self.send("Annotated Data", new_table)

    def set_data(self, data):
        self.data = data
        if self.data is None:
            self.send("Annotated Data", None)
        self.commit()

    def _invalidate(self):
        self.commit()


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWDBSCAN()
    d = Table("iris.tab")
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()

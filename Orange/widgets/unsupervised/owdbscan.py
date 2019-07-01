import sys

import numpy as np
from AnyQt.QtWidgets import QLayout, QApplication
from AnyQt.QtCore import Qt

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, DiscreteVariable
from Orange.clustering import DBSCAN
from Orange import distance
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import Msg


class OWDBSCAN(widget.OWWidget):
    name = "DBSCAN"
    description = "Density-based spatial clustering."
    icon = "icons/DBSCAN.svg"
    priority = 2150

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Error(widget.OWWidget.Error):
        not_enough_instances = Msg("Not enough unique data instances. "
                                   "At least two are required.")

    METRICS = [
        ("Euclidean", "euclidean"),
        ("Manhattan", "manhattan"),
        ("Cosine", distance.Cosine),
        ("Jaccard", distance.Jaccard),
        # ("Spearman", distance.SpearmanR),
        # ("Spearman absolute", distance.SpearmanRAbsolute),
        # ("Pearson", distance.PearsonR),
        # ("Pearson absolute", distance.PearsonRAbsolute),
    ]

    min_samples = Setting(5)
    eps = Setting(0.5)
    metric_idx = Setting(0)
    auto_commit = Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.db = None
        self.model = None

        box = gui.widgetBox(self.controlArea, "Parameters")
        gui.spin(box, self, "min_samples", 1, 100, 1, callback=self._invalidate,
                 label="Core point neighbors")
        gui.doubleSpin(box, self, "eps", 0.1, 10, 0.01,
                       callback=self._invalidate,
                       label="Neighborhood distance")

        box = gui.widgetBox(self.controlArea, self.tr("Distance Metric"))
        gui.comboBox(box, self, "metric_idx",
                     items=list(zip(*self.METRICS))[0],
                     callback=self._invalidate)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Apply",
                        orientation=Qt.Horizontal)
        gui.rubber(self.controlArea)

        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

    def adjustSize(self):
        self.ensurePolished()
        self.resize(self.controlArea.sizeHint())

    def check_data_size(self):
        if len(self.data) < 2:
            self.Error.not_enough_instances()
            return False
        return True

    def commit(self):
        self.cluster()

    def cluster(self):
        if not self.check_data_size():
            return
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.METRICS[self.metric_idx][1]
        ).get_model(self.data)
        self.send_data()

    def send_data(self):
        model = self.model

        clusters = [c if c >= 0 else np.nan for c in model.labels]
        k = len(set(clusters) - {np.nan})
        clusters = np.array(clusters).reshape(len(self.data), 1)
        core_samples = set(model.projector.core_sample_indices_)
        in_core = np.array([1 if (i in core_samples) else 0
                            for i in range(len(self.data))])
        in_core = in_core.reshape(len(self.data), 1)

        clust_var = DiscreteVariable(
            "Cluster", values=["C%d" % (x + 1) for x in range(k)])
        in_core_var = DiscreteVariable("DBSCAN Core", values=["0", "1"])

        domain = self.data.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        x, y, metas = self.data.X, self.data.Y, self.data.metas

        meta_attrs += (clust_var, )
        metas = np.hstack((metas, clusters))
        meta_attrs += (in_core_var, )
        metas = np.hstack((metas, in_core))

        domain = Domain(attributes, classes, meta_attrs)
        new_table = Table(domain, x, y, metas, self.data.W)

        self.Outputs.annotated_data.send(new_table)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if self.data is None:
            self.Outputs.annotated_data.send(None)
        self.Error.clear()
        if self.data is None:
            return
        self.unconditional_commit()

    def _invalidate(self):
        self.commit()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWDBSCAN()
    d = Table("iris.tab")
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()

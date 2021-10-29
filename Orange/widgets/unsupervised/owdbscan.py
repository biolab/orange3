import sys
from itertools import chain

import numpy as np
from AnyQt.QtWidgets import QApplication
from AnyQt.QtGui import QColor
from sklearn.metrics import pairwise_distances

from Orange.preprocess import Normalize, Continuize, SklImpute
from Orange.widgets import widget, gui
from Orange.widgets.utils.slidergraph import SliderGraph
from Orange.widgets.settings import Setting
from Orange.data import Table, DiscreteVariable
from Orange.data.util import get_unique_names
from Orange.clustering import DBSCAN
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import Msg


DEFAULT_CUT_POINT = 0.1
PREPROCESSORS = [Continuize(), Normalize(), SklImpute()]
EPS_BOTTOM_LIMIT = 0.01


def get_kth_distances(data, metric, k=5):
    """
    The function computes the epsilon parameter for DBSCAN through method
    proposed in the paper.
    Parameters
    ----------
    data : Orange.data.Table
        Visualisation coordinates - embeddings
    metric : callable or str
        The metric to compute the distance.
    k : int
        Number kth observed neighbour

    Returns
    -------
    np.ndarray
        Epsilon parameter for DBSCAN
    """
    x = data.X
    if x.shape[0] > 1000:  # subsample
        x = x[np.random.randint(x.shape[0], size=1000), :]

    dist = pairwise_distances(x, metric=metric)
    k = min(k+1, len(data) - 1)  # k+1 since first one is item itself
    kth_point = np.argpartition(dist, k, axis=1)[:, k]
    kth_dist = np.sort(dist[np.arange(0, len(kth_point)), kth_point])[::-1]

    return kth_dist


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
        ("Manhattan", "cityblock"),
        ("Cosine", "cosine")
    ]

    min_samples = Setting(4)
    eps = Setting(0.5)
    metric_idx = Setting(0)
    normalize = Setting(True)
    auto_commit = Setting(True)
    k_distances = None
    cut_point = None

    def __init__(self):
        super().__init__()

        self.data = None
        self.data_normalized = None
        self.db = None
        self.model = None

        box = gui.widgetBox(self.controlArea, "Parameters")
        gui.spin(box, self, "min_samples", 1, 100, 1,
                 callback=self._min_samples_changed,
                 label="Core point neighbors")
        gui.doubleSpin(box, self, "eps", EPS_BOTTOM_LIMIT, 1000, 0.01,
                       callback=self._eps_changed,
                       label="Neighborhood distance")

        box = gui.widgetBox(self.controlArea, self.tr("Distance Metric"))
        gui.comboBox(box, self, "metric_idx",
                     items=list(zip(*self.METRICS))[0],
                     callback=self._metirc_changed)
        gui.checkBox(box, self, "normalize", "Normalize features",
                     callback=self._on_normalize_changed)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")
        gui.rubber(self.controlArea)

        self.controlArea.layout().addStretch()

        self.plot = SliderGraph(
            x_axis_label="Data items sorted by score",
            y_axis_label="Distance to the k-th nearest neighbour",
            callback=self._on_cut_changed
        )

        self.mainArea.layout().addWidget(self.plot)

    def check_data_size(self, data):
        if data is None:
            return False
        if len(data) < 2:
            self.Error.not_enough_instances()
            return False
        return True

    @gui.deferred
    def commit(self):
        self.cluster()

    def cluster(self):
        if not self.check_data_size(self.data):
            return
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.METRICS[self.metric_idx][1]
        ).get_model(self.data_normalized)
        self.send_data()

    def _compute_and_plot(self, cut_point=None):
        self._compute_kdistances()
        if cut_point is None:
            self._compute_cut_point()
        self._plot_graph()

    def _plot_graph(self):
        nonzero = np.sum(self.k_distances > EPS_BOTTOM_LIMIT)
        self.plot.update(np.arange(len(self.k_distances)),
                         [self.k_distances],
                         colors=[QColor('red')],
                         cutpoint_x=self.cut_point,
                         selection_limit=(0, nonzero - 1))

    def _compute_kdistances(self):
        self.k_distances = get_kth_distances(
            self.data_normalized, metric=self.METRICS[self.metric_idx][1],
            k=self.min_samples
        )

    def _compute_cut_point(self):
        self.cut_point = int(DEFAULT_CUT_POINT * len(self.k_distances))
        self.eps = self.k_distances[self.cut_point]

        mask = self.k_distances >= EPS_BOTTOM_LIMIT
        if self.eps < EPS_BOTTOM_LIMIT and sum(mask):
            self.eps = np.min(self.k_distances[mask])
            self.cut_point = self._find_nearest_dist(self.eps)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not self.check_data_size(data):
            data = None
        self.data = self.data_normalized = data
        if self.data is None:
            self.Outputs.annotated_data.send(None)
            self.plot.clear_plot()
            return

        if self.data is None:
            return

        self._preprocess_data()

        self._compute_and_plot()
        self.commit.now()

    def _preprocess_data(self):
        self.data_normalized = self.data
        for pp in PREPROCESSORS:
            if isinstance(pp, Normalize) and not self.normalize:
                continue
            self.data_normalized = pp(self.data_normalized)

    def send_data(self):
        model = self.model

        clusters = [c if c >= 0 else np.nan for c in model.labels]
        k = len(set(clusters) - {np.nan})
        clusters = np.array(clusters)
        core_samples = set(model.projector.core_sample_indices_)
        in_core = np.array([1 if (i in core_samples) else 0
                            for i in range(len(self.data))])

        domain = self.data.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        names = [var.name for var in chain(attributes, classes, meta_attrs) if var]

        u_clust_var = get_unique_names(names, "Cluster")
        clust_var = DiscreteVariable(
            u_clust_var, values=["C%d" % (x + 1) for x in range(k)])

        u_in_core = get_unique_names(names + [u_clust_var], "DBSCAN Core")
        in_core_var = DiscreteVariable(u_in_core, values=("0", "1"))

        new_table = self.data.add_column(clust_var, clusters, to_metas=True)
        new_table = new_table.add_column(in_core_var, in_core, to_metas=True)

        self.Outputs.annotated_data.send(new_table)

    def _invalidate(self):
        self.commit.deferred()

    def _find_nearest_dist(self, value):
        array = np.asarray(self.k_distances)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _eps_changed(self):
        # find the closest value to eps
        if self.data is None:
            return
        self.cut_point = self._find_nearest_dist(self.eps)
        self.plot.set_cut_point(self.cut_point)
        self._invalidate()

    def _metirc_changed(self):
        if self.data is not None:
            self._compute_and_plot()
            self._invalidate()

    def _on_cut_changed(self, value):
        # cut changed by means of a cut line over the scree plot.
        self.cut_point = value
        self.eps = self.k_distances[value]

        self.commit.deferred()

    def _min_samples_changed(self):
        if self.data is None:
            return
        self._compute_and_plot(cut_point=self.cut_point)
        self._invalidate()

    def _on_normalize_changed(self):
        if not self.data:
            return
        self._preprocess_data()
        self._compute_and_plot()
        self._invalidate()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWDBSCAN()
    d = Table("iris.tab")
    ow.set_data(d)
    ow.show()
    a.exec()
    ow.saveSettings()

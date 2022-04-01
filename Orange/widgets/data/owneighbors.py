import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.preprocess import RemoveNaNColumns, Impute
from Orange import distance
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

METRICS = [
    ("Euclidean", distance.Euclidean),
    ("Manhattan", distance.Manhattan),
    ("Mahalanobis", distance.Mahalanobis),
    ("Cosine", distance.Cosine),
    ("Jaccard", distance.Jaccard),
    ("Spearman", distance.SpearmanR),
    ("Absolute Spearman", distance.SpearmanRAbsolute),
    ("Pearson", distance.PearsonR),
    ("Absolute Pearson", distance.PearsonRAbsolute),
]


class OWNeighbors(OWWidget):
    name = "Neighbors"
    description = "Compute nearest neighbors in data according to reference."
    icon = "icons/Neighbors.svg"
    category = "Unsupervised"
    replaces = ["orangecontrib.prototypes.widgets.owneighbours.OWNeighbours"]

    class Inputs:
        data = Input("Data", Table)
        reference = Input("Reference", Table)

    class Outputs:
        data = Output("Neighbors", Table)

    class Info(OWWidget.Warning):
        removed_references = \
            Msg("Input data includes reference instance(s).\n"
                "Reference instances are excluded from the output.")

    class Warning(OWWidget.Warning):
        all_data_as_reference = \
            Msg("Every data instance is same as some reference")

    class Error(OWWidget.Error):
        diff_domains = Msg("Data and reference have different features")

    n_neighbors: int
    distance_index: int

    n_neighbors = Setting(10)
    limit_neighbors = Setting(True)
    distance_index = Setting(0)
    auto_apply = Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.reference = None
        self.distances = None

        box = gui.vBox(self.controlArea, box=True)
        gui.comboBox(
            box, self, "distance_index", orientation=Qt.Horizontal,
            label="Distance metric: ", items=[d[0] for d in METRICS],
            callback=self.recompute)
        gui.spin(
            box, self, "n_neighbors", label="Limit number of neighbors to:",
            step=1, spinType=int, minv=0, maxv=100, checked='limit_neighbors',
            # call apply by gui.auto_commit, pylint: disable=unnecessary-lambda
            checkCallback=self.commit.deferred,
            callback=self.commit.deferred)

        self.apply_button = gui.auto_apply(self.buttonsArea, self)

    @Inputs.data
    def set_data(self, data):
        self.controls.n_neighbors.setMaximum(len(data) if data else 100)
        self.data = data

    @Inputs.reference
    def set_ref(self, refs):
        self.reference = refs

    def handleNewSignals(self):
        self.compute_distances()
        self.commit.now()

    def recompute(self):
        self.compute_distances()
        self.commit.deferred()

    def compute_distances(self):
        self.Error.diff_domains.clear()
        if not self.data or not self.reference:
            self.distances = None
            return
        if set(self.reference.domain.attributes) != \
                set(self.data.domain.attributes):
            self.Error.diff_domains()
            self.distances = None
            return

        metric = METRICS[self.distance_index][1]
        n_ref = len(self.reference)

        # comparing only attributes, no metas and class-vars
        new_domain = Domain(self.data.domain.attributes)
        reference = self.reference.transform(new_domain)
        data = self.data.transform(new_domain)

        all_data = Table.concatenate([reference, data], 0)
        pp_all_data = Impute()(RemoveNaNColumns()(all_data))
        pp_reference, pp_data = pp_all_data[:n_ref], pp_all_data[n_ref:]
        self.distances = metric(pp_data, pp_reference).min(axis=1)

    @gui.deferred
    def commit(self):
        indices = self._compute_indices()

        if indices is None:
            neighbors = None
        else:
            neighbors = self._data_with_similarity(indices)
        self.Outputs.data.send(neighbors)

    def _compute_indices(self):
        self.Warning.all_data_as_reference.clear()
        self.Info.removed_references.clear()

        if self.distances is None:
            return None

        inrefs = np.isin(self.data.ids, self.reference.ids)
        if np.all(inrefs):
            self.Warning.all_data_as_reference()
            return None
        if np.any(inrefs):
            self.Info.removed_references()

        dist = np.copy(self.distances)
        dist[inrefs] = np.max(dist) + 1
        up_to = len(dist) - np.sum(inrefs)
        if self.limit_neighbors and self.n_neighbors < up_to:
            up_to = self.n_neighbors
        return np.argpartition(dist, up_to - 1)[:up_to]

    def _data_with_similarity(self, indices):
        data = self.data
        varname = get_unique_names(data.domain, "distance")
        metas = data.domain.metas + (ContinuousVariable(varname), )
        domain = Domain(data.domain.attributes, data.domain.class_vars, metas)
        data_metas = self.distances[indices].reshape((-1, 1))
        if data.domain.metas:
            data_metas = np.hstack((data.metas[indices], data_metas))
        neighbors = Table(domain, data.X[indices], data.Y[indices], data_metas)
        neighbors.ids = data.ids[indices]
        neighbors.attributes = self.data.attributes
        return neighbors


if __name__ == "__main__":  # pragma: no cover
    iris = Table("iris.tab")
    WidgetPreview(OWNeighbors).run(
        set_data=iris,
        set_ref=iris[:1])

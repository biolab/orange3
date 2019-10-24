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

    replaces = ["orangecontrib.prototypes.widgets.owneighbours.OWNeighbours"]

    class Inputs:
        data = Input("Data", Table)
        reference = Input("Reference", Table)

    class Outputs:
        data = Output("Neighbors", Table)

    class Warning(OWWidget.Warning):
        all_data_as_reference = \
            Msg("Every data instance is same as some reference")

    class Error(OWWidget.Error):
        diff_domains = Msg("Data and reference have different features")

    n_neighbors: int
    distance_index: int

    n_neighbors = Setting(10)
    distance_index = Setting(0)
    exclude_reference = Setting(True)
    auto_apply = Setting(True)

    want_main_area = False
    buttons_area_orientation = Qt.Vertical

    def __init__(self):
        super().__init__()

        self.data = None
        self.reference = None
        self.distances = None

        box = gui.vBox(self.controlArea, box=True)
        gui.comboBox(
            box, self, "distance_index", orientation=Qt.Horizontal,
            label="Distance: ", items=[d[0] for d in METRICS],
            callback=self.recompute)
        gui.spin(
            box, self, "n_neighbors", label="Number of neighbors:",
            step=1, spinType=int, minv=0, maxv=100,
            # call apply by gui.auto_commit, pylint: disable=unnecessary-lambda
            callback=lambda: self.apply())
        gui.checkBox(
            box, self, "exclude_reference",
            label="Exclude rows (equal to) references",
            # call apply by gui.auto_commit, pylint: disable=unnecessary-lambda
            callback=lambda: self.apply())

        self.apply_button = gui.auto_apply(self.controlArea, self, commit=self.apply)

    def _set_input_summary(self):
        n_data, n_refs = 0, 0
        if self.data is not None and self.reference is not None:
            n_data = len(self.data)
            n_refs = len(self.reference)
        elif self.data is not None:
            n_data = len(self.data)
        elif self.reference is not None:
            n_refs = len(self.reference)
        else:
            self.info.set_input_summary(self.info.NoInput)
            return

        inst = f"{n_data} | {n_refs} "
        if n_data > 0 and n_refs > 0:
            details = f"{n_data} data instances on input; " \
                    f"{n_refs} reference instances on input "
        elif n_data > 0 and n_refs == 0:
            details = f"{n_data} data instances on input; " \
                      f"No reference instances on input "
        elif n_data == 0 and n_refs > 0:
            details = f"No data instances on input; " \
                      f"{n_refs} reference instances on input "

        self.info.set_input_summary(inst, details)

    def _set_output_summary(self):
        if self.data is None:
            self.Outputs.data.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return

        indices = self._compute_indices()
        if np.any(indices):
            neighbors = self._data_with_similarity(indices)
            self.Outputs.data.send(neighbors)
            self.info.set_output_summary(str(len(neighbors)))

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self._set_input_summary()

    @Inputs.reference
    def set_ref(self, refs):
        self.reference = refs
        self._set_input_summary()

    def handleNewSignals(self):
        self.compute_distances()
        self.unconditional_apply()

    def recompute(self):
        self.compute_distances()
        self.apply()

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

    def apply(self):
        indices = self._compute_indices()
        if indices is None:
            neighbors = None
        else:
            neighbors = self._data_with_similarity(indices)
        self.Outputs.data.send(neighbors)
        self._set_output_summary()

    def _compute_indices(self):
        self.Warning.all_data_as_reference.clear()
        dist = self.distances
        if dist is None:
            return None
        if self.exclude_reference:
            non_ref = dist > 1e-5
            skip = len(dist) - non_ref.sum()
            up_to = min(self.n_neighbors + skip, len(dist))
            if skip >= up_to:
                self.Warning.all_data_as_reference()
                return None
            indices = np.argpartition(dist, up_to - 1)[:up_to]
            return indices[non_ref[indices]]
        else:
            up_to = min(self.n_neighbors, len(dist))
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
        neighbors.attributes = self.data.attributes
        return neighbors


if __name__ == "__main__":  # pragma: no cover
    iris = Table("iris.tab")
    WidgetPreview(OWNeighbors).run(
        set_data=iris,
        set_ref=iris[:1])

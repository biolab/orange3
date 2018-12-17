import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable
from Orange.preprocess import RemoveNaNColumns, Impute
from Orange import distance
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget
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

    n_neighbors = Setting(10)
    distance_index = Setting(0)
    exclude_reference = Setting(True)
    auto_apply = Setting(True)

    want_main_area = False
    buttons_area_orientation = Qt.Vertical

    _data_info_default = "No data."
    _ref_info_default = "No reference."

    def __init__(self):
        super().__init__()

        self.data = None
        self.reference = None
        self.dist = None
        box = gui.vBox(self.controlArea, "Info")
        self.data_info_label = gui.widgetLabel(box, self._data_info_default)
        self.ref_info_label = gui.widgetLabel(box, self._ref_info_default)

        box = gui.vBox(self.controlArea, "Settings")
        self.distance_combo = gui.comboBox(
            box, self, "distance_index", orientation=Qt.Horizontal,
            label="Distance: ", items=[d[0] for d in METRICS],
            callback=self.compute_distance)

        check_box = gui.hBox(box)
        self.exclude_ref_label = gui.label(
            check_box, self, "Exclude references:")
        self.exclude_ref_check = gui.checkBox(
            check_box, self, "exclude_reference", label="",
            callback=self.apply)

        box = gui.vBox(self.controlArea, "Output")
        self.nn_spin = gui.spin(
            box, self, "n_neighbors", label="Neighbors:", step=1, spinType=int,
            minv=0, maxv=100, callback=self.apply)

        box = gui.hBox(self.controlArea, True)
        self.apply_button = gui.auto_commit(box, self, "auto_apply", "&Apply",
                                            box=False, commit=self.apply)

    @Inputs.data
    def set_data(self, data):
        text = self._data_info_default if data is None \
            else "{} data instances on input.".format(len(data))
        self.data = data
        self.data_info_label.setText(text)
        self.apply()

    @Inputs.reference
    def set_ref(self, reference):
        text = self._ref_info_default if reference is None \
            else "{} reference instances on input.".format(len(reference))
        self.reference = reference
        self.ref_info_label.setText(text)
        self.apply()

    def compute_distance(self):
        distance = METRICS[self.distance_index][1]
        n_data, n_ref = len(self.data), len(self.reference)
        all_data = Table.concatenate([self.reference, self.data], 0)
        pp_all_data = Impute()(RemoveNaNColumns()(all_data))
        pp_data, pp_reference = pp_all_data[n_ref:], pp_all_data[:n_ref]
        self.dist = np.array(distance(np.vstack((pp_data, pp_reference)))[
                             :n_data, n_data:])

    def handleNewSignals(self):
        if self.data is None or self.reference is None:
            self.Outputs.data.send(None)
            return
        self.compute_distance()
        self.apply()

    def apply(self):
        sorted_indices = list(np.argsort(self.dist.flatten()))[::-1]
        indices = []
        while len(sorted_indices) > 0 and len(indices) < self.n_neighbors:
            index = int(sorted_indices.pop() / len(self.reference))
            if (self.data[index] not in self.reference or
                    not self.exclude_reference) and index not in indices:
                indices.append(index)
                self._add_similarity(self.data, self.dist)
        neighbors = self._add_similarity(self.data[indices], self.dist[indices])
        neighbors.attributes = self.data.attributes
        self.Outputs.data.send(self.data)

    @staticmethod
    def _add_similarity(data, dist):
        dist = np.min(dist, axis=1)[:, None]
        metas = data.domain.metas + (ContinuousVariable("similarity"),)
        domain = Domain(data.domain.attributes, data.domain.class_vars, metas)
        data_metas = np.hstack((data.metas, 100 * (1 - dist / np.max(dist))))
        return Table(domain, data.X, data.Y, data_metas)


if __name__ == "__main__": # pragma: no cover
    data = Table("iris.tab")
    WidgetPreview(OWNeighbors).run(
        set_data=data,
        set_ref=data[:1])
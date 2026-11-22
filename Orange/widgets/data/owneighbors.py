from encodings import normalize_encoding
from typing import Optional

import numpy as np
import bottleneck as bn

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout, QLabel

from orangewidget.utils.signals import LazyValue

from Orange.data import Table, Domain, ContinuousVariable
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.util import get_unique_names
from Orange.preprocess import RemoveNaNColumns, Impute
from Orange.misc.distmatrix import DistMatrix
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.unsupervised.utils import (
    MetricDefs, EuclideanNormalized, Euclidean, Manhattan, Cosine,Mahalanobis,
    Pearson, PearsonAbsolute, Spearman, SpearmanAbsolute, Jaccard
)


class OWNeighbors(OWWidget):
    name = "Neighbors"
    description = "Compute nearest neighbors in data according to reference."
    icon = "icons/Neighbors.svg"
    category = "Unsupervised"
    replaces = ["orangecontrib.prototypes.widgets.owneighbours.OWNeighbours"]
    keywords = "knn, nearest neighbors, distance, similarity"

    class Inputs:
        data = Input("Data", Table)
        distances = Input("Distances", DistMatrix)
        reference = Input("Reference", Table)

    class Outputs:
        data = Output("Neighbors", Table)
        data_w_distances = Output("Data with Distances", Table)

    class Info(OWWidget.Warning):
        removed_references = \
            Msg("Input data includes reference instance(s).\n"
                "Reference instances are not considered as neighbours.")

    class Warning(OWWidget.Warning):
        all_data_as_reference = \
            Msg("Every data instance is same as some reference")
        imputing_data = Msg("Missing values were imputed")
        ignored_references = \
            Msg("Some reference instances are not in the distance matrix and were ignored")

    class Error(OWWidget.Error):
        diff_number = \
            Msg("Distance matrix and data table have mismatching number of items")
        diff_domains = \
            Msg("Data and reference have different features")
        not_same_table = \
            Msg("Distance matrix is not computed from a (single) table.")
        no_references = \
            Msg("Distance matrix does not include any reference instance(s).")

    settingsHandler = DomainContextHandler()
    n_neighbors: int = Setting(10)
    metric_id: int = Setting(EuclideanNormalized)
    label: Optional[StringVariable] = ContextSetting(None)
    selected_item_hint: Optional[str] = Setting(None)
    include_reference = Setting(False)
    transform_to_similarity = Setting(False)
    auto_apply = Setting(True)

    # Backward compatibility for check box to limit the number of neighbours.
    # The new GUI sets it to True and does not expose it
    limit_neighbors = Setting(True)
    backward_compatibility_limit = Setting(False, schema_only=True)

    want_main_area = False
    resizing_enabled = False

    settings_version = 3

    def __init__(self):
        super().__init__()

        self.data: Optional[Table] = None
        self.reference: Optional[Table] = None
        self.distance_matrix: Optional[DistMatrix] = None

        self.label: Optional[StringVariable] = None
        self.selected_item: Optional[str] = None

        # 1D array of the same length as data/distance_matrix,
        # with distances to the closest reference
        self.distances: Optional[np.ndarray] = None

        self._metric_box()
        self._reference_box()
        self._options_box()
        self.apply_button = gui.auto_apply(self.buttonsArea, self)

    def _metric_box(self):
        for self._distance_index, metric in enumerate(MetricDefs.values()):
            if metric.id == self.metric_id:
                break
        else:
            self._distance_index = 0

        self.metric_box = gui.vBox(self.controlArea, addSpaceBefore=False)
        self.metric_box.setContentsMargins(0, 0, 0, 4)
        box = gui.vBox(
            self.metric_box, box="Distance Metric", addSpaceBefore=False)
        gui.comboBox(
            box, self, "_distance_index", orientation=Qt.Horizontal,
            items=[metric.name for metric in MetricDefs.values()],
            tooltips=[metric.tooltip for metric in MetricDefs.values()],
            callback=self._metric_changed
        )

    def _reference_box(self):
        self.reference_box = gui.vBox(self.controlArea)
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.reference_box.setContentsMargins(0, 0, 0, 4)
        box = gui.widgetBox(
            self.reference_box, box="Reference Example", orientation=layout)
        self.variable_model = DomainModel(
            placeholder="Row index",
            valid_types=StringVariable, order=(DomainModel.METAS, )
        )
        lb = gui.comboBox(
            None, self, "label",
            model=self.variable_model, callback=self._label_changed,
            toolTip="Select reference example based on the value of a variable."
        )
        lb.label = QLabel("Variable:")
        layout.addRow(lb.label, lb)

        self.label_model = PyListModel()
        cb = gui.comboBox(
            None, self, "selected_item",
            model=self.label_model, callback=self._item_changed,
            toolTip="Select reference example based on the value of a variable."
        )
        layout.addRow("Value:", cb)

    def _options_box(self):
        box = gui.vBox(self.controlArea, box=True, addSpaceBefore=False)
        if self.backward_compatibility_limit:
            kwargs = {
                "checked": 'limit_neighbors',
                "checkCallback": self.commit.deferred,
                "label": "Limit number of neighbors to:"
            }
        else:
            kwargs = {
                "label": "Number of neighbors:",
            }
        gui.spin(
            box, self, "n_neighbors", step=1, spinType=int, minv=0, maxv=100,
            alignment=Qt.AlignRight,
            callback=self.commit.deferred,
            tooltip="Number of neighbors, excluding the reference example, if appended",
            **kwargs
        )
        gui.checkBox(
            box, self, "include_reference",
            label="Append reference example(s)",
            callback=self.commit.deferred,
            toolTip="Append the reference example to the output table of neighbors"
        )
        gui.checkBox(
            box, self, "transform_to_similarity",
            label="Output similarity instead of distance",
            callback=self.commit.deferred,
            toolTip="Subtract all distances from the maximum distance"
        )

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.data = data

    @Inputs.distances
    def set_distances(self, distances: Optional[DistMatrix]):
        self.Error.not_same_table.clear()
        if distances is not None \
                and distances.col_items is not None \
                and distances.row_items is not distances.col_items:
            self.Error.not_same_table()
            distances = None

        self.metric_box.setHidden(distances is not None)
        self.distance_matrix = distances

    @Inputs.reference
    def set_ref(self, refs: Optional[Table]):
        self.reference_box.setHidden(refs is not None)
        self.reference = refs

    @property
    def mismatching_input_sizes(self):
        return self.distance_matrix is not None and self.data is not None \
                and len(self.distance_matrix) != len(self.data)

    @property
    def act_data(self):
        if self.data is not None:
            return self.data
        if self.distance_matrix is not None:
            return self.distance_matrix.row_items
        return None

    def handleNewSignals(self):
        self.Error.diff_number(shown=self.mismatching_input_sizes)

        data = self.act_data
        if not data:
            self.controls.n_neighbors.setMaximum(100)
            domain = None
        else:
            self.controls.n_neighbors.setMaximum(len(data))
            domain = data.domain

        self.closeContext()

        self.label = None
        self.selected_item = None
        self.variable_model.set_domain(domain)
        has_labels = len(self.variable_model) > 1
        lb = self.controls.label
        lb.setDisabled(not has_labels)
        lb.label.setDisabled(not has_labels)
        self.label = self.variable_model[1] if has_labels else None
        self.openContext(domain)
        self._update_labels()

        self.compute_distances()
        self.commit.now()

    def _metric_changed(self):
        self.metric_id = list(MetricDefs.values())[self._distance_index].id
        self.compute_distances()
        self.commit.deferred()

    def _update_labels(self):
        if self.label is not None:
            labels = self.act_data.get_column(self.label)
        elif self.act_data is not None:
            labels = range(1, 1 + len(self.act_data))
        else:
            labels = []
        self.label_model[:] = [str(x) for x in labels]

        if self.selected_item_hint is not None \
                and self.selected_item_hint in self.label_model:
            self.selected_item = self.selected_item_hint
        else:
            self.selected_item = self.label_model[0] if len(self.label_model) > 0 \
                else None

    def _label_changed(self):
        self._update_labels()
        self.compute_distances()
        self.commit.deferred()

    def _item_changed(self):
        self.selected_item_hint = self.selected_item
        self.compute_distances()
        self.commit.deferred()

    def compute_distances(self):
        self.Error.diff_domains.clear()
        self.Warning.imputing_data.clear()
        self.Warning.ignored_references.clear()
        self.Error.no_references.clear()

        if not self.act_data or self.mismatching_input_sizes:
            self.distances = None
        elif self.distance_matrix is not None:
            if self.reference is not None:
                self.distances = self._compute_from_matrix_reference()
            else:
                self.distances = self._compute_from_matrix_selection()
        else:
            if self.reference is not None:
                self.distances = self._compute_from_data_reference()
            else:
                self.distances = self._compute_from_data_selection()

    def _compute_from_data_reference(self):
        assert self.data is not None
        assert self.reference is not None

        if set(self.reference.domain.attributes) != \
                set(self.data.domain.attributes):
            self.Error.diff_domains()
            return None

        reference = self.reference.transform(self.data.domain)
        return self._compute_from_data_instance(reference)

    def _compute_from_data_selection(self):
        assert self.data is not None

        item_index = self.label_model.indexOf(self.selected_item)
        reference = self.data[[item_index]]
        return self._compute_from_data_instance(reference)

    def _compute_from_data_instance(self, reference: Table):
        metric_def = MetricDefs[self.metric_id]
        metric = metric_def.metric
        data = self.data
        if not metric.supports_missing \
                and (bn.anynan(data.X) or bn.anynan(reference.X)):
            self.Warning.imputing_data()
            data = Impute()(RemoveNaNColumns()(data))

        kwargs = {"impute": True}
        if metric_def.normalize:
            kwargs["normalize"] = True
        return metric(data, reference, **kwargs).min(axis=1)

    def _compute_from_matrix_selection(self):
        assert self.distance_matrix is not None

        row = self.label_model.indexOf(self.selected_item)
        return self._compute_from_matrix_row([row])

    def _compute_from_matrix_reference(self):
        assert self.distance_matrix is not None
        assert self.reference is not None

        rows = np.isin(self.distance_matrix.row_items.ids, self.reference.ids)
        nrows = np.sum(rows)
        if nrows == 0:
            self.Error.no_references()
            return None
        if nrows != len(self.reference):
            self.Warning.ignored_references()
            return None
        return self._compute_from_matrix_row(rows)

    def _compute_from_matrix_row(self, rows):
        return self.distance_matrix[rows].min(axis=0)

    @gui.deferred
    def commit(self):
        indices = self._compute_indices()

        if indices is None:
            neighbors = None
        else:
            neighbors = self._data_with_similarity(indices)
            neighbors.name = self.act_data.name + " (neighbors)"

        if self.distances is None:
            distance_table = None
        else:
            _, domain = self._domain_with_distance()
            length = len(self.act_data)
            distance_table = LazyValue[Table](
                lambda:  self._data_with_similarity(np.arange(length), False),
                length=length, domain=domain)

        self.Outputs.data.send(neighbors)
        self.Outputs.data_w_distances.send(distance_table)

    def _compute_indices(self):
        self.Warning.all_data_as_reference.clear()
        self.Info.removed_references.clear()

        if self.distances is None:
            return None

        if self.reference:
            inrefs = np.isin(self.act_data.ids, self.reference.ids)
            if np.all(inrefs):
                self.Warning.all_data_as_reference()
                return None
            if np.any(inrefs):
                self.Info.removed_references()
        else:
            inrefs = np.full(len(self.act_data), False)
            inrefs[self.label_model.indexOf(self.selected_item)] = True

        dist = np.copy(self.distances)
        dist[inrefs] = np.max(dist) + 1
        up_to = len(dist) - np.sum(inrefs)
        # self.limit_neighbors is for backward compatibility;
        # The new GUI sets it to True and does not expose it
        if self.limit_neighbors and self.n_neighbors < up_to:
            up_to = self.n_neighbors
        # get indexes of N neighbours in unsorted order - faster than argsort
        idx = np.argpartition(dist, up_to - 1)[:up_to]
        # sort selected N neighbours according to distances
        sorted_subset_idx = np.argsort(dist[idx])
        # map sorted indexes back to original index space
        return idx[sorted_subset_idx]

    def _domain_with_distance(self):
        domain = self.act_data.domain
        dist_var = ContinuousVariable(get_unique_names(domain, "distance"))
        metas = domain.metas + (dist_var, )
        return dist_var, Domain(domain.attributes, domain.class_vars, metas)

    def _data_with_similarity(self, indices, include_reference=None):
        dist_var, domain = self._domain_with_distance()
        data = self.act_data
        neighbours = data.from_table(domain, data, row_indices=indices)
        distances = self.distances[indices]
        if self.transform_to_similarity:
            distances = np.max(distances) - distances

        if include_reference is None:
            include_reference = self.include_reference
        if include_reference:
            if self.reference:
                reference = self.reference
            else:
                reference = data[[self.label_model.indexOf(self.selected_item)]]
            neighbours = Table.concatenate([
                neighbours,
                reference.transform(neighbours.domain)])
            distances = np.hstack((distances, [np.nan] * len(reference)))
        with neighbours.unlocked(neighbours.metas):
            if distances.size > 0:
                neighbours.set_column(dist_var, distances)
        return neighbours

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            metric_idx = settings.pop("distance_index", 0)
            old_metrics = [Euclidean, Manhattan, Mahalanobis, Cosine, Jaccard,
                           Spearman, SpearmanAbsolute, Pearson, PearsonAbsolute]
            metric_id = old_metrics[metric_idx]
            settings["metric_id"] = metric_id
        if  version < 3:
            if not settings.get("limit_neighbors", True):
                settings["backward_compatibility_limit"] = True


if __name__ == "__main__":  # pragma: no cover
    from Orange import distance  # pylint: disable=import-error

    iris = Table("iris.tab")
    zoo = Table("zoo.tab")
    distances = distance.Euclidean(zoo)
    # WidgetPreview(OWNeighbors).run(set_data=iris, set_ref=iris[:3])
    WidgetPreview(OWNeighbors).run(set_data=zoo)
    # WidgetPreview(OWNeighbors).run(set_distances=distances, set_ref=zoo[:1])
    # WidgetPreview(OWNeighbors).run(set_distances=distances)

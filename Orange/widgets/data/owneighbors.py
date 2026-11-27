from typing import Literal

import numpy as np
import bottleneck as bn

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout, QLabel

from orangewidget.utils.signals import LazyValue

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
        imputing_data = \
            Msg("Missing values were imputed")
        ignored_references = \
            Msg("Some reference instances are not in the distance matrix and were ignored")
        no_distance_matrix_data = \
            Msg("Distance matrix does not contain a data table.")
        neighbor_by_indices = \
            Msg("Rows from data are assumed to match the rows in distance matrix")

    class Error(OWWidget.Error):
        diff_number = \
            Msg("Distance matrix and data table have mismatching number of items.\n"
                "If the matrix that does not include a related data table, it is\n"
                "assumed that its rows correspond to the rows of the input table.")
        dist_data_id_mismatch = \
            Msg("Distance matrix and data table do not refer to the same data.")
        diff_domains_or_missing = \
            Msg("Reference has different features than data "
                "or missing values that cannot be imputed.")
        not_same_table = \
            Msg("Distance matrix is not computed from a (single) table.")
        no_references = \
            Msg("Distance matrix does not include any reference instance(s).")
        missing_distances = \
            Msg("Distance matrix does not contain distances between "
                "(all) rows of a data table.")
        metric_failed = \
            Msg("Distance metric failed to compute distances:\n{}")

    settingsHandler = DomainContextHandler()
    n_neighbors: int = Setting(10)
    metric_id: int = Setting(EuclideanNormalized)
    label: StringVariable | None = ContextSetting(None)
    selected_item_hint: str | None = Setting(None)
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

        self.data: Table | None = None
        self.reference: Table | None = None
        self.distance_matrix: DistMatrix | None = None

        self.label: StringVariable | None = None
        self.selected_item: str | None = None

        # 1D array of the same length as data/distance_matrix,
        # with distances to the closest reference
        self.distances: np.ndarray | None = None

        # Map from table index to distance matrix index,
        # - or None if N/A
        # - or False if invalid, in which case the widget shows an error.
        self.mapping: np.ndarray | None | Literal[False] = None

        self._metric_box()
        self._reference_box()
        gui.separator(self.controlArea)
        self._options_box()
        self.apply_button = gui.auto_apply(self.buttonsArea, self)

    def _metric_box(self):
        for self._distance_index, metric in enumerate(MetricDefs.values()):
            if metric.id == self.metric_id:
                break
        else:
            self._distance_index = 0
            self.metric_id = next(iter(MetricDefs.values())).id

        box = gui.vBox(self.controlArea, box="Distance Metric")
        self.metric_box = gui.comboBox(
            box, self, "_distance_index", orientation=Qt.Horizontal,
            items=[metric.name for metric in MetricDefs.values()],
            tooltips=[metric.tooltip for metric in MetricDefs.values()],
            callback=self._metric_changed
        )

        self.metric_info = gui.widgetLabel(box, "")
        self.metric_info.setHidden(True)

    def _reference_box(self):
        box = gui.vBox(self.controlArea, box="Reference Example")

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.reference_box = gui.widgetBox(
            box, orientation=layout, addSpaceBefore=False)
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

        self.reference_info = gui.widgetLabel(
            box, "Reference is provided on the input.")
        self.reference_info.setHidden(True)

    def _options_box(self):
        box = gui.vBox(self.controlArea, box=True)
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
            tooltip="Number of neighbors, excluding the reference example",
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
    def set_data(self, data: Table | None):
        self.data = data

    @Inputs.distances
    def set_distances(self, distances: DistMatrix | None):
        self.Error.not_same_table.clear()
        if distances is not None \
                and distances.col_items is not None \
                and distances.row_items is not distances.col_items:
            self.Error.not_same_table()
            distances = None

        self.distance_matrix = distances

    @Inputs.reference
    def set_ref(self, refs: Table | None):
        self.reference = refs

    @property
    def mismatching_input_sizes(self):
        return self.data is not None \
            and self.distance_matrix is not None \
            and not isinstance(self.distance_matrix.row_items, Table) \
            and len(self.distance_matrix) != len(self.data)

    @property
    def act_data(self):
        if self.data is not None:
            return self.data
        if self.distance_matrix is not None \
                and isinstance(self.distance_matrix.row_items, Table):
            return self.distance_matrix.row_items
        return None

    @property
    def act_domain(self):
        if self.act_data:
            return self.act_data.domain
        else:
            return None

    @property
    def dimension(self):
        if self.act_data is not None:
            return len(self.act_data)
        if self.distance_matrix is not None:
            return len(self.distance_matrix)
        return 0

    def handleNewSignals(self):
        self.Error.diff_number(shown=self.mismatching_input_sizes)

        self.controls.n_neighbors.setMaximum(self.dimension or 100)

        self.closeContext()
        self._update_variable_model()
        self.openContext(self.act_domain)
        self._update_labels()

        self._update_info()

        self._update_mapping()
        self.compute_distances()
        self.commit.now()

    def _update_mapping(self):
        """
        Set the mapping from data table's rows to distance matrix.

        ... or to None, if there is not data or no matrix or no data in matrix
        ... or to False, if mapping cannot be established because some rows
            from the data do not appear in the matrix
        """
        self.Error.missing_distances.clear()
        self.Error.dist_data_id_mismatch.clear()

        if self.data is None \
                or self.distance_matrix is None \
                or not isinstance(self.distance_matrix.row_items, Table) \
                or (len(self.data) == len(self.distance_matrix) and
                    np.all(self.data.ids == self.distance_matrix.row_items.ids)):
            self.mapping = None
            return

        data_ids = self.data.ids
        dist_ids = self.distance_matrix.row_items.ids
        nincluded_ids = len(set(data_ids) & set(dist_ids))
        if nincluded_ids == 0:
            self.mapping = False
            self.Error.dist_data_id_mismatch()
            return
        elif nincluded_ids != len(data_ids):
            self.mapping = False
            self.Error.missing_distances()
            return

        ids = {id: i for i, id in enumerate(dist_ids)}
        self.mapping = np.array([ids[id] for id in data_ids])

    def _update_info(self):
        self.metric_box.setHidden(self.distance_matrix is not None)
        self.metric_info.setHidden(self.distance_matrix is None)
        if self.distance_matrix is not None:
            if self.data is not None:
                self.metric_info.setText("""
                    <p style = "line-height: 130%;" >
                      Neighbours are selected from input data,<br/>
                      based on the distance matrix provided as input.
                   </p>""")
            else:
                self.metric_info.setText(
                    "Distances are provided on the input.")

        self.reference_box.setHidden(self.reference is not None)
        self.reference_info.setHidden(self.reference is None)

    def _metric_changed(self):
        self.metric_id = list(MetricDefs.values())[self._distance_index].id
        self.compute_distances()
        self.commit.deferred()

    def _update_variable_model(self):
        self.label = None
        self.selected_item = None
        self.variable_model.set_domain(self.act_domain)
        has_labels = len(self.variable_model) > 1
        lb = self.controls.label
        lb.setDisabled(not has_labels)
        lb.label.setDisabled(not has_labels)
        self.label = self.variable_model[1] if has_labels else None

    def _update_labels(self):
        if self.label is not None:
            labels = self.act_data.get_column(self.label)
        else:
            labels = range(1, 1 + self.dimension)
        self.label_model[:] = [str(x) for x in labels]

        if self.selected_item_hint is not None \
                and self.selected_item_hint in self.label_model:
            self.selected_item = self.selected_item_hint
        else:
            if len(self.label_model) == 0:
                self.selected_item = None
            else:
                self.selected_item = self.label_model[0]
                self.selected_item_hint = self.selected_item

    def _label_changed(self):
        self._update_labels()
        self.compute_distances()
        self.commit.deferred()

    def _item_changed(self):
        self.selected_item_hint = self.selected_item
        self.compute_distances()
        self.commit.deferred()

    def compute_distances(self):
        """
        Set self.distances to distances from the reference example

        If self.data is present, the computed distances correspond to its rows,
        even if the distances are taken from the pre-computed input distances
        (i.e. they are mapped through matching id's).

        If there is no data, distances are taken from distances in the input.
        """
        self.Error.diff_domains_or_missing.clear()
        self.Warning.imputing_data.clear()
        self.Warning.ignored_references.clear()
        self.Error.no_references.clear()
        self.Error.metric_failed.clear()

        if not self.data and self.distance_matrix is None \
                or self.mismatching_input_sizes \
                or self.mapping is False:
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
        # Case: no distance matrix, compute distances between table and reference
        assert self.data is not None
        assert self.reference is not None

        return self._compute_from_data_instance(self.reference)

    def _compute_from_data_selection(self):
        # Case: no distance matrix, no reference, compute distance from chosen row
        assert self.data is not None

        item_index = self.label_model.indexOf(self.selected_item)
        reference = self.data[[item_index]]
        return self._compute_from_data_instance(reference)

    def _compute_from_data_instance(self, reference: Table):
        # Common function for the above two:
        # compute min distances between self.data and the given table
        metric_def = MetricDefs[self.metric_id]
        metric = metric_def.metric
        data = self.data
        if not metric.supports_missing \
                and (bn.anynan(data.X) or bn.anynan(reference.X)):
            self.Warning.imputing_data()
            data = Impute()(RemoveNaNColumns()(data))

        # This also takes care of imputation if necessary
        reference = reference.transform(data.domain)

        if not metric.supports_missing and bn.anynan(reference.X) \
                or bn.allnan(reference.X):
            self.Error.diff_domains_or_missing()
            return None

        kwargs = {"impute": True}
        if metric_def.normalize:
            kwargs["normalize"] = True
        try:
            return metric(data, reference, **kwargs).min(axis=1)
        except Exception as exc:  # pylint: disable=broad-except
            self.Error.metric_failed(str(exc))
            return None

    def _compute_from_matrix_selection(self):
        # Case: distance matrix and no reference,
        # take the selected row and map it to table if necessary
        assert self.distance_matrix is not None

        row = self.label_model.indexOf(self.selected_item)
        if self.mapping is not None:
            # Map from table index to matrix index
            row = self.mapping[row]
        return self._compute_from_matrix_row([row])

    def _compute_from_matrix_reference(self):
        # Case: distance matrix and reference example
        # Find and return the corresponding row, take it and map to table if necessary
        assert self.distance_matrix is not None
        assert self.reference is not None

        if not isinstance(self.distance_matrix.row_items, Table) \
                and self.data is None:
            self.Error.no_references()
            return None

        if self.distance_matrix is not None \
                and isinstance(self.distance_matrix.row_items, Table):
            act_ids = self.distance_matrix.row_items.ids
        else:
            # If distance_matrix has no data but we have a reference and data,
            # we know the sizes match and we assume pairwise correspondence,
            # so we take ids from data
            assert not self.mismatching_input_sizes
            act_ids = self.data.ids

        rows = np.isin(act_ids, self.reference.ids)
        nrows = np.sum(rows)
        if nrows == 0:
            self.Error.no_references()
            return None
        if nrows != len(self.reference):
            self.Warning.ignored_references()
        return self._compute_from_matrix_row(rows)

    def _compute_from_matrix_row(self, rows):
        # The common function for the above two functions:
        # Compute the minimum over all references, map to table if necessary
        distances = self.distance_matrix[rows].min(axis=0)
        if self.mapping is not None:
            distances = distances[self.mapping]
        return distances

    @gui.deferred
    def commit(self):
        self.Warning.no_distance_matrix_data.clear()
        self.Warning.neighbor_by_indices.clear()

        indices = self._compute_indices()

        if indices is None:
            neighbors = None
        else:
            neighbors = self._data_with_similarity(indices)
            if self.act_data:
                neighbors.name = self.act_data.name + " (neighbors)"
            else:
                neighbors.name = "Neighbors' indices"

        if self.distances is None:
            distance_table = None
        else:
            _, domain = self._domain_with_distance()
            distance_table = LazyValue[Table](
                lambda: self._data_with_similarity(np.arange(self.dimension),
                                                   include_reference=False),
                length=self.dimension, domain=domain)

        self.Outputs.data.send(neighbors)
        self.Outputs.data_w_distances.send(distance_table)

    def _compute_indices(self):
        """
        Compute the indices of the smallest k-distances in self.distances
        Exclude the index of the reference example
        """
        self.Warning.all_data_as_reference.clear()
        self.Info.removed_references.clear()

        if self.distances is None:
            return None

        # Compute:
        # - `inrefs`: a mask or index of reference example(s)
        # - `ninrefs`: the number of (actual) reference examples
        if self.reference is None:
            inrefs = self.label_model.indexOf(self.selected_item)
            ninrefs = 1
            self.Info.removed_references()
        else:
            ref_ids = self.reference.ids
            inrefs = np.isin(self.act_data.ids, ref_ids)
            ninrefs = np.sum(inrefs)
            if np.all(inrefs):
                self.Warning.all_data_as_reference()
                return None
            if np.any(inrefs):
                self.Info.removed_references()

        dist = np.copy(self.distances)
        dist[inrefs] = np.max(dist) + 1
        up_to = len(dist) - ninrefs
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
        if not self.act_data:
            dist_var = ContinuousVariable("distance")
            domain = Domain([ContinuousVariable("row_number"), dist_var])
            return dist_var, domain

        domain = self.act_domain
        dist_var = ContinuousVariable(get_unique_names(domain, "distance"))
        metas = domain.metas + (dist_var, )
        return dist_var, Domain(domain.attributes, domain.class_vars, metas)

    def _data_with_similarity(self, indices, include_reference=None):
        dist_var, domain = self._domain_with_distance()
        data = self.act_data
        distances = self.distances[indices]
        if self.transform_to_similarity:
            distances = np.max(distances) - distances

        if data is None:
            self.Warning.no_distance_matrix_data()
            return Table.from_numpy(
                domain,
                np.vstack((indices + 1, distances)).T)

        # If we have data and matrix, match the id's
        if self.data is not None \
                and self.distance_matrix is not None \
                and not isinstance(self.distance_matrix.row_items, Table):
            self.Warning.neighbor_by_indices()

        neighbours = data.from_table(domain, data, row_indices=indices)
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


def main():  # pragma: no cover
    # pylint: disable=import-outside-toplevel,unused-variable
    from Orange import distance

    iris = Table("iris.tab")
    zoo = Table("zoo.tab")
    distances = distance.Euclidean(zoo)
    #WidgetPreview(OWNeighbors).run(set_data=iris, set_ref=iris[:3])
    WidgetPreview(OWNeighbors).run(set_data=zoo)
    # WidgetPreview(OWNeighbors).run(set_distances=distances, set_ref=zoo[:1])
    # WidgetPreview(OWNeighbors).run(set_distances=distances)
    # WidgetPreview(OWNeighbors).run(set_distances=distances, set_data=zoo[50:])


if __name__ == "__main__":  # pragma: no cover
    main()

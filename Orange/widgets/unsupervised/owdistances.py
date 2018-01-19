from AnyQt.QtCore import Qt
from scipy.sparse import issparse
import bottleneck as bn

import Orange.data
import Orange.misc
from Orange import distance
from Orange.widgets import gui, settings
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Msg, Input, Output


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


class OWDistances(OWWidget):
    name = "Distances"
    description = "Compute a matrix of pairwise distances."
    icon = "icons/Distance.svg"

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        distances = Output("Distances", Orange.misc.DistMatrix, dynamic=False)

    settings_version = 2

    axis = settings.Setting(0)        # type: int
    metric_idx = settings.Setting(0)  # type: int

    #: Use normalized distances if the metric supports it.
    #: The default is `True`, expect when restoring from old pre v2 settings
    #: (see `migrate_settings`).
    normalized_dist = settings.Setting(True)  # type: bool
    autocommit = settings.Setting(True)       # type: bool

    want_main_area = False
    buttons_area_orientation = Qt.Vertical

    class Error(OWWidget.Error):
        no_continuous_features = Msg("No numeric features")
        dense_metric_sparse_data = Msg("{} requires dense data.")
        distances_memory_error = Msg("Not enough memory")
        distances_value_error = Msg("Problem in calculation:\n{}")

    class Warning(OWWidget.Warning):
        ignoring_discrete = Msg("Ignoring categorical features")
        imputing_data = Msg("Missing values were imputed")

    def __init__(self):
        super().__init__()

        self.data = None

        gui.radioButtons(self.controlArea, self, "axis", ["Rows", "Columns"],
                         box="Distances between", callback=self._invalidate
                        )
        box = gui.widgetBox(self.controlArea, "Distance Metric")
        self.metrics_combo = gui.comboBox(
            box, self, "metric_idx",
            items=[m[0] for m in METRICS],
            callback=self._metric_changed
        )
        self.normalization_check = gui.checkBox(
            box, self, "normalized_dist", "Normalized",
            callback=self._invalidate,
            tooltip=("All dimensions are (implicitly) scaled to a common"
                     "scale to normalize the influence across the domain.")
        )
        _, metric = METRICS[self.metric_idx]
        self.normalization_check.setEnabled(metric.supports_normalization)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")
        self.layout().setSizeConstraint(self.layout().SetFixedSize)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data = data
        self.refresh_metrics()
        self.unconditional_commit()

    def refresh_metrics(self):
        sparse = self.data is not None and issparse(self.data.X)
        for i, metric in enumerate(METRICS):
            item = self.metrics_combo.model().item(i)
            item.setEnabled(not sparse or metric[1].supports_sparse)

    def commit(self):
        # pylint: disable=invalid-sequence-index
        metric = METRICS[self.metric_idx][1]
        dist = self.compute_distances(metric, self.data)
        self.Outputs.distances.send(dist)

    def compute_distances(self, metric, data):
        def _check_sparse():
            # pylint: disable=invalid-sequence-index
            if issparse(data.X) and not metric.supports_sparse:
                self.Error.dense_metric_sparse_data(METRICS[self.metric_idx][0])
                return False

        def _fix_discrete():
            nonlocal data
            if data.domain.has_discrete_attributes() and (
                    issparse(data.X) and getattr(metric, "fallback", None)
                    or not metric.supports_discrete
                    or self.axis == 1):
                if not data.domain.has_continuous_attributes():
                    self.Error.no_continuous_features()
                    return False
                self.Warning.ignoring_discrete()
                data = distance.remove_discrete_features(data)

        def _fix_missing():
            nonlocal data
            if not metric.supports_missing and bn.anynan(data.X):
                self.Warning.imputing_data()
                data = distance.impute(data)

        self.clear_messages()
        if data is None:
            return
        for check in (_check_sparse, _fix_discrete, _fix_missing):
            if check() is False:
                return
        try:
            if metric.supports_normalization and self.normalized_dist:
                return metric(data, axis=1 - self.axis, impute=True,
                              normalize=True)
            else:
                return metric(data, axis=1 - self.axis, impute=True)
        except ValueError as e:
            self.Error.distances_value_error(e)
        except MemoryError:
            self.Error.distances_memory_error()

    def _invalidate(self):
        self.commit()

    def _metric_changed(self):
        metric = METRICS[self.metric_idx][1]
        self.normalization_check.setEnabled(metric.supports_normalization)
        self._invalidate()

    def send_report(self):
        # pylint: disable=invalid-sequence-index
        self.report_items((
            ("Distances Between", ["Rows", "Columns"][self.axis]),
            ("Metric", METRICS[self.metric_idx][0])
        ))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None or version < 2 and "normalized_dist" not in settings:
            # normalize_dist is set to False when restoring settings from
            # an older version to preserve old semantics.
            settings["normalized_dist"] = False

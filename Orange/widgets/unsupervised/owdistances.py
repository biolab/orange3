from scipy.sparse import issparse
import bottleneck as bn

from AnyQt.QtCore import Qt

import Orange.data
import Orange.misc
from Orange import distance
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output


METRICS = [
    ("Euclidean", distance.Euclidean),
    ("Manhattan", distance.Manhattan),
    ("Cosine", distance.Cosine),
    ("Jaccard", distance.Jaccard),
    ("Spearman", distance.SpearmanR),
    ("Absolute Spearman", distance.SpearmanRAbsolute),
    ("Pearson", distance.PearsonR),
    ("Absolute Pearson", distance.PearsonRAbsolute),
    ("Hamming", distance.Hamming),
    ("Mahalanobis", distance.Mahalanobis),
    ('Bhattacharyya', distance.Bhattacharyya)
]


class InterruptException(Exception):
    pass


class DistanceRunner:
    @staticmethod
    def run(data: Orange.data.Table, metric: distance, normalized_dist: bool,
            axis: int, state: TaskState) -> Orange.misc.DistMatrix:
        if data is None:
            return None

        def callback(i: float) -> bool:
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        state.set_status("Calculating...")
        kwargs = {"axis": 1 - axis, "impute": True, "callback": callback}
        if metric.supports_normalization and normalized_dist:
            kwargs["normalize"] = True
        return metric(data, **kwargs)


class OWDistances(OWWidget, ConcurrentWidgetMixin):
    name = "Distances"
    description = "Compute a matrix of pairwise distances."
    icon = "icons/Distance.svg"
    keywords = []

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        distances = Output("Distances", Orange.misc.DistMatrix, dynamic=False)

    settings_version = 3

    axis = Setting(0)        # type: int
    metric_idx = Setting(0)  # type: int

    #: Use normalized distances if the metric supports it.
    #: The default is `True`, expect when restoring from old pre v2 settings
    #: (see `migrate_settings`).
    normalized_dist = Setting(True)  # type: bool
    autocommit = Setting(True)       # type: bool

    want_main_area = False
    resizing_enabled = False

    class Error(OWWidget.Error):
        no_continuous_features = Msg("No numeric features")
        no_binary_features = Msg("No binary features")
        dense_metric_sparse_data = Msg("{} requires dense data.")
        distances_memory_error = Msg("Not enough memory")
        distances_value_error = Msg("Problem in calculation:\n{}")
        data_too_large_for_mahalanobis = Msg(
            "Mahalanobis handles up to 1000 {}.")

    class Warning(OWWidget.Warning):
        ignoring_discrete = Msg("Ignoring categorical features")
        ignoring_nonbinary = Msg("Ignoring non-binary features")
        imputing_data = Msg("Missing values were imputed")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data = None

        gui.radioButtons(
            self.controlArea, self, "axis", ["Rows", "Columns"],
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
                     "scale to normalize the influence across the domain."),
            stateWhenDisabled=False, attribute=Qt.WA_LayoutUsesWidgetRect
        )
        _, metric = METRICS[self.metric_idx]
        self.normalization_check.setEnabled(metric.supports_normalization)

        gui.auto_apply(self.buttonsArea, self, "autocommit")

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.cancel()
        self.data = data
        self.refresh_metrics()
        self.commit.now()

    def refresh_metrics(self):
        sparse = self.data is not None and issparse(self.data.X)
        for i, metric in enumerate(METRICS):
            item = self.metrics_combo.model().item(i)
            item.setEnabled(not sparse or metric[1].supports_sparse)

    @gui.deferred
    def commit(self):
        # pylint: disable=invalid-sequence-index
        metric = METRICS[self.metric_idx][1]
        self.compute_distances(metric, self.data)

    def compute_distances(self, metric, data):
        def _check_sparse():
            # pylint: disable=invalid-sequence-index
            if issparse(data.X) and not metric.supports_sparse:
                self.Error.dense_metric_sparse_data(METRICS[self.metric_idx][0])
                return False
            return True

        def _fix_discrete():
            nonlocal data
            if data.domain.has_discrete_attributes() \
                    and metric is not distance.Jaccard \
                    and (issparse(data.X) and getattr(metric, "fallback", None)
                         or not metric.supports_discrete
                         or self.axis == 1):
                if not data.domain.has_continuous_attributes():
                    self.Error.no_continuous_features()
                    return False
                self.Warning.ignoring_discrete()
                data = distance.remove_discrete_features(data, to_metas=True)
            return True

        def _fix_nonbinary():
            nonlocal data
            if metric is distance.Jaccard and not issparse(data.X):
                nbinary = sum(a.is_discrete and len(a.values) == 2
                              for a in data.domain.attributes)
                if not nbinary:
                    self.Error.no_binary_features()
                    return False
                elif nbinary < len(data.domain.attributes):
                    self.Warning.ignoring_nonbinary()
                    data = distance.remove_nonbinary_features(data,
                                                              to_metas=True)
            return True

        def _fix_missing():
            nonlocal data
            if not metric.supports_missing and bn.anynan(data.X):
                self.Warning.imputing_data()
                data = distance.impute(data)
            return True

        def _check_tractability():
            if metric is distance.Mahalanobis:
                if self.axis == 1:
                    # when computing distances by columns, we want < 100 rows
                    if len(data) > 1000:
                        self.Error.data_too_large_for_mahalanobis("rows")
                        return False
                else:
                    if len(data.domain.attributes) > 1000:
                        self.Error.data_too_large_for_mahalanobis("columns")
                        return False
            return True

        self.clear_messages()
        if data is not None:
            for check in (_check_sparse, _check_tractability,
                          _fix_discrete, _fix_missing, _fix_nonbinary):
                if not check():
                    data = None
                    break

        self.start(DistanceRunner.run, data, metric,
                   self.normalized_dist, self.axis)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Orange.misc.DistMatrix):
        assert isinstance(result, Orange.misc.DistMatrix) or result is None
        self.Outputs.distances.send(result)

    def on_exception(self, ex):
        if isinstance(ex, ValueError):
            self.Error.distances_value_error(ex)
        elif isinstance(ex, MemoryError):
            self.Error.distances_memory_error()
        elif isinstance(ex, InterruptException):
            pass
        else:
            raise ex

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def _invalidate(self):
        self.commit.deferred()

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
        if version is None or version < 3:
            # Mahalanobis was moved from idx = 2 to idx = 9
            metric_idx = settings["metric_idx"]
            if metric_idx == 2:
                settings["metric_idx"] = 9
            elif 2 < metric_idx <= 9:
                settings["metric_idx"] -= 1


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDistances).run(Orange.data.Table("iris"))

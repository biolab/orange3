from typing import NamedTuple, Dict, Type, Optional

from AnyQt.QtWidgets import QButtonGroup, QRadioButton
from AnyQt.QtCore import Qt
from scipy.sparse import issparse
import bottleneck as bn

import Orange.data
import Orange.misc
from Orange import distance
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output


Euclidean, EuclideanNormalized, Manhattan, ManhattanNormalized, Cosine, \
    Mahalanobis, Hamming, \
    Pearson, PearsonAbsolute, Spearman, SpearmanAbsolute, Jaccard = range(12)


class MetricDef(NamedTuple):
    id: int  # pylint: disable=invalid-name
    name: str
    tooltip: str
    metric: Type[distance.Distance]
    normalize: bool = False


MetricDefs: Dict[int, MetricDef] = {
    metric.id: metric for metric in (
        MetricDef(EuclideanNormalized, "Euclidean (normalized)",
                  "Square root of summed difference between normalized values",
                  distance.Euclidean, normalize=True),
        MetricDef(Euclidean, "Euclidean",
                  "Square root of summed difference between values",
                  distance.Euclidean),
        MetricDef(ManhattanNormalized, "Manhattan (normalized)",
                  "Sum of absolute differences between normalized values",
                  distance.Manhattan, normalize=True),
        MetricDef(Manhattan, "Manhattan",
                  "Sum of absolute differences between values",
                  distance.Manhattan),
        MetricDef(Mahalanobis, "Mahalanobis",
                  "Mahalanobis distance",
                  distance.Mahalanobis),
        MetricDef(Hamming, "Hamming", "Hamming distance",
                  distance.Hamming),
        MetricDef(Cosine, "Cosine", "Cosine distance",
                  distance.Cosine),
        MetricDef(Pearson, "Pearson",
                  "Pearson correlation; distance = 1 - ρ/2",
                  distance.PearsonR),
        MetricDef(PearsonAbsolute, "Pearson (absolute)",
                  "Absolute value of Pearson correlation; distance = 1 - |ρ|",
                  distance.PearsonRAbsolute),
        MetricDef(Spearman, "Spearman",
                  "Pearson correlation; distance = 1 - ρ/2",
                  distance.PearsonR),
        MetricDef(SpearmanAbsolute, "Spearman (absolute)",
                  "Absolute value of Pearson correlation; distance = 1 - |ρ|",
                  distance.SpearmanRAbsolute),
        MetricDef(Jaccard, "Jaccard", "Jaccard distance",
                  distance.Jaccard)
    )
}


class InterruptException(Exception):
    pass


class DistanceRunner:
    @staticmethod
    def run(data: Orange.data.Table, metric: distance, normalized_dist: bool,
            axis: int, state: TaskState) -> Optional[Orange.misc.DistMatrix]:
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
    keywords = "distances"

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        distances = Output("Distances", Orange.misc.DistMatrix, dynamic=False)

    settings_version = 4

    axis: int = Setting(0)
    metric_id: int = Setting(EuclideanNormalized)
    autocommit: bool = Setting(True)

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
        unsupported_sparse = Msg("Some metrics don't support sparse data\n"
                                 "and were disabled: {}")
        imputing_data = Msg("Missing values were imputed")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data = None

        gui.radioButtons(
            self.controlArea, self, "axis", ["Rows", "Columns"],
            box="Compare", orientation=Qt.Horizontal, callback=self._invalidate
        )
        box = gui.hBox(self.controlArea, "Distance Metric")
        self.metric_buttons = QButtonGroup()
        width = 0
        for i, metric in enumerate(MetricDefs.values()):
            if i % 6 == 0:
                vb = gui.vBox(box)
            b = QRadioButton(metric.name)
            b.setChecked(self.metric_id == metric.id)
            b.setToolTip(metric.tooltip)
            vb.layout().addWidget(b)
            width = max(width, b.sizeHint().width())
            self.metric_buttons.addButton(b, metric.id)
        for b in self.metric_buttons.buttons():
            b.setFixedWidth(width)

        self.metric_buttons.idClicked.connect(self._metric_changed)

        gui.auto_apply(self.buttonsArea, self, "autocommit")


    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.cancel()
        self.data = data
        self.refresh_radios()
        self.commit.now()

    def _metric_changed(self, id_):
        self.metric_id = id_
        self._invalidate()

    def refresh_radios(self):
        sparse = self.data is not None and issparse(self.data.X)
        unsupported_sparse = []
        for metric in MetricDefs.values():
            button = self.metric_buttons.button(metric.id)
            no_sparse = sparse and not metric.metric.supports_sparse
            button.setEnabled(not no_sparse)
            if no_sparse:
                unsupported_sparse.append(metric.name)
        self.Warning.unsupported_sparse(", ".join(unsupported_sparse),
                                        shown=bool(unsupported_sparse))

    @gui.deferred
    def commit(self):
        self.compute_distances(self.data)

    def compute_distances(self, data):
        def _check_sparse():
            # pylint: disable=invalid-sequence-index
            if issparse(data.X) and not metric.supports_sparse:
                self.Error.dense_metric_sparse_data(metric_def.name)
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
                if self.axis == 0:
                    # when computing distances by columns, we want < 1000 rows
                    if len(data) > 1000:
                        self.Error.data_too_large_for_mahalanobis("rows")
                        return False
                else:
                    if len(data.domain.attributes) > 1000:
                        self.Error.data_too_large_for_mahalanobis("columns")
                        return False
            return True

        metric_def = MetricDefs[self.metric_id]
        metric = metric_def.metric
        self.clear_messages()
        if data is not None:
            for check in (_check_sparse, _check_tractability,
                          _fix_discrete, _fix_missing, _fix_nonbinary):
                if not check():
                    data = None
                    break

        self.start(DistanceRunner.run, data, metric,
                   metric_def.normalize, self.axis)

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

    def send_report(self):
        # pylint: disable=invalid-sequence-index
        self.report_items((
            ("Distances Between", ["Rows", "Columns"][self.axis]),
            ("Metric", MetricDefs[self.metric_id].name)
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
        if version < 4:
            metric_idx = settings.pop("metric_idx")
            metric_id = [Euclidean, Manhattan, Cosine, Jaccard,
                         Spearman, SpearmanAbsolute, Pearson, PearsonAbsolute,
                         Hamming, Mahalanobis, Euclidean][metric_idx]
            if settings.pop("normalized_dist", False):
                metric_id = {Euclidean: EuclideanNormalized,
                             Manhattan: ManhattanNormalized}.get(metric_id,
                                                                 metric_id)
            settings["metric_id"] = metric_id


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDistances).run(Orange.data.Table("iris"))

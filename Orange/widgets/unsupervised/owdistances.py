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

    axis = settings.Setting(0)
    metric_idx = settings.Setting(0)
    autocommit = settings.Setting(True)

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
        self.metrics_combo = gui.comboBox(self.controlArea, self, "metric_idx",
                                          box="Distance Metric",
                                          items=[m[0] for m in METRICS],
                                          callback=self._invalidate
                                         )
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
            return metric(data, axis=1 - self.axis, impute=True)
        except ValueError as e:
            self.Error.distances_value_error(e)
        except MemoryError:
            self.Error.distances_memory_error()

    def _invalidate(self):
        self.commit()

    def send_report(self):
        # pylint: disable=invalid-sequence-index
        self.report_items((
            ("Distances Between", ["Rows", "Columns"][self.axis]),
            ("Metric", METRICS[self.metric_idx][0])
        ))

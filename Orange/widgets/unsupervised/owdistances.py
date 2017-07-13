from AnyQt.QtCore import Qt
from scipy.sparse import issparse

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
        dense_metric_sparse_data = Msg("Selected metric does not support sparse data")
        empty_data = Msg("Empty data set")
        mahalanobis_error = Msg("{}")
        distances_memory_error = Msg("Not enough memory.")
        distances_value_error = Msg("Error occurred while calculating distances\n{}")

    class Warning(OWWidget.Warning):
        ignoring_discrete = Msg("Ignoring categorical features")
        imputing_data = Msg("Imputing missing values")

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
        box = gui.auto_commit(self.buttonsArea, self, "autocommit", "Apply",
                              box=False, checkbox_label="Apply automatically")
        box.layout().insertWidget(0, self.report_button)
        box.layout().insertSpacing(1, 8)

        self.layout().setSizeConstraint(self.layout().SetFixedSize)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """
        Set the input data set from which to compute the distances
        """
        self.data = data
        self.refresh_metrics()
        self.unconditional_commit()

    def refresh_metrics(self):
        """
        Refresh available metrics depending on the input data's sparsenes
        """
        sparse = self.data is not None and issparse(self.data.X)
        for i, metric in enumerate(METRICS):
            item = self.metrics_combo.model().item(i)
            item.setEnabled(not sparse or metric[1].supports_sparse)

        self._checksparse()

    def _checksparse(self):
        # Check the current metric for input data compatibility and set/clear
        # appropriate informational GUI state
        self.Error.dense_metric_sparse_data(
            shown=self.data is not None and issparse(self.data.X) and
            not METRICS[self.metric_idx][1].supports_sparse)

    def commit(self):
        metric = METRICS[self.metric_idx][1]
        dist = self.compute_distances(metric, self.data)
        self.Outputs.distances.send(dist)

    def compute_distances(self, metric, data):
        def checks(metric, data):
            if data is None:
                return
            if issparse(data.X):
                if not metric.supports_sparse:
                    self.Error.dense_metric_sparse_data()
                    return
                remove_discrete = metric.fallback is not None \
                        and self.data.domain.has_discrete_attributes()
            else:  # not sparse
                remove_discrete = \
                    not metric.supports_discrete or self.axis == 1 \
                    and self.data.domain.has_discrete_attributes()
            if remove_discrete:
                data = distance._preprocess(data, impute=False)
                if data.X.size:
                    self.Warning.ignoring_discrete()
            if not data.X.size:
                self.Error.empty_data()
                return
            return data

        self.clear_messages()
        data = checks(metric, data)
        if data is None:
            return
        try:
            met = metric(data, data, 1 - self.axis, impute=True)
        except ValueError as e:
            self.Error.distances_value_error(e)
            return
        except MemoryError:
            self.Error.distances_memory_error()
            return
        return met

    def _invalidate(self):
        self._checksparse()
        self.commit()

    def send_report(self):
        self.report_items((
            ("Distances Between", ["Rows", "Columns"][self.axis]),
            ("Metric", METRICS[self.metric_idx][0])
        ))

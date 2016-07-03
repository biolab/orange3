import numpy
from PyQt4.QtCore import Qt
from scipy.sparse import issparse

import Orange.data
import Orange.misc
from Orange import distance
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Msg

METRICS = [
    distance.Euclidean,
    distance.Manhattan,
    distance.Mahalanobis,
    distance.Cosine,
    distance.Jaccard,
    distance.SpearmanR,
    distance.SpearmanRAbsolute,
    distance.PearsonR,
    distance.PearsonRAbsolute,
]


class OWDistances(widget.OWWidget):
    name = "Distances"
    description = "Compute a matrix of pairwise distances."
    icon = "icons/Distance.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Distances", Orange.misc.DistMatrix)]

    axis = settings.Setting(0)
    metric_idx = settings.Setting(0)
    autocommit = settings.Setting(False)

    want_main_area = False
    buttons_area_orientation = Qt.Vertical

    class Error(widget.OWWidget.Error):
        no_continuous_features = Msg("No continuous features")
        empty_data = Msg("Empty data (shape = {})")

    class Warning(widget.OWWidget.Warning):
        ignoring_discrete = Msg("Ignoring discrete features")

    def __init__(self):
        super().__init__()

        self.data = None

        gui.radioButtons(self.controlArea, self, "axis", ["Rows", "Columns"],
                         box="Distances between", callback=self._invalidate
        )
        self.metrics_combo = gui.comboBox(self.controlArea, self, "metric_idx",
                                          box="Distance Metric",
                                          items=[m.name for m in METRICS],
                                          callback=self._invalidate
        )
        box = gui.auto_commit(self.buttonsArea, self, "autocommit", "Apply",
                              box=False, checkbox_label="Apply automatically")
        box.layout().insertWidget(0, self.report_button)
        box.layout().insertSpacing(1, 8)

        self.layout().setSizeConstraint(self.layout().SetFixedSize)

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
            item.setEnabled(not sparse or metric.supports_sparse)

        self._checksparse()

    def _checksparse(self):
        # Check the current metric for input data compatibility and set/clear
        # appropriate informational GUI state
        metric = METRICS[self.metric_idx]
        data = self.data
        if data is not None and issparse(data.X) and \
                not metric.supports_sparse:
            self.error(2, "Selected metric does not support sparse data")
        else:
            self.error(2)

    def commit(self):
        self.warning(1)
        self.error(1)
        metric = METRICS[self.metric_idx]
        distances = None
        data = self.data
        if data is not None and issparse(data.X) and \
                not metric.supports_sparse:
            data = None
        self.clear_messages()

        if data is not None:
            if isinstance(metric, distance.MahalanobisDistance):
                metric.fit(self.data, axis=1-self.axis)

            if not any(a.is_continuous for a in self.data.domain.attributes):
                self.Error.no_continuous_features()
                data = None
            elif any(a.is_discrete for a in self.data.domain.attributes) or \
                    (not issparse(self.data.X) and numpy.any(numpy.isnan(self.data.X))):
                data = distance._preprocess(self.data)
                if len(self.data.domain.attributes) - len(data.domain.attributes) > 0:
                    self.Warning.ignoring_discrete()
            else:
                data = self.data

        if data is not None:
            shape = (len(data), len(data.domain.attributes))
            if numpy.product(shape) == 0:
                self.Error.empty_data(shape)
            else:
                distances = metric(data, data, 1 - self.axis, impute=True)

        self.send("Distances", distances)

    def _invalidate(self):
        self._checksparse()
        self.commit()

    def send_report(self):
        self.report_items((
            ("Distances Between", ["Rows", "Columns"][self.axis]),
            ("Metric", METRICS[self.metric_idx].name)
        ))

from inspect import getmembers

import numpy
from PyQt4.QtCore import Qt
from scipy.sparse import issparse

import Orange.data
import Orange.misc
from Orange import distance
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.sql import check_sql_input

DENSE_METRICS = [obj for name, obj in getmembers(distance,
                                                 lambda x: isinstance(x, distance.Distance))]
SPARSE_METRICS = list(filter(lambda x: x.supports_sparse, DENSE_METRICS))


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

    def __init__(self):
        super().__init__()

        self.data = None
        self.available_metrics = DENSE_METRICS

        gui.radioButtons(self.controlArea, self, "axis", ["Rows", "Columns"],
                         box="Distances between", callback=self._invalidate
        )
        self.metrics_combo = gui.comboBox(self.controlArea, self, "metric_idx",
                                          box="Distance Metric",
                                          items=[m.name for m in self.available_metrics],
                                          callback=self._invalidate
        )
        box = gui.auto_commit(self.buttonsArea, self, "autocommit", "Apply",
                              box=False, checkbox_label="Apply automatically")
        box.layout().insertWidget(0, self.report_button)
        box.layout().insertSpacing(1, 8)

        self.layout().setSizeConstraint(self.layout().SetFixedSize)

    @check_sql_input
    def set_data(self, data):
        self.data = data
        self.refresh_metrics()
        self.unconditional_commit()

    def refresh_metrics(self):
        sparse = self.data and issparse(self.data.X)
        self.available_metrics = SPARSE_METRICS if sparse else DENSE_METRICS

        self.metrics_combo.clear()
        self.metric_idx = 0
        for m in self.available_metrics:
            self.metrics_combo.addItem(m.name)

    def commit(self):
        self.warning()
        self.error()

        data = distances = None
        if self.data is not None:
            metric = self.available_metrics[self.metric_idx]
            if isinstance(metric, distance.MahalanobisDistance):
                metric.fit(self.data, axis=1-self.axis)

            if not any(a.is_continuous for a in self.data.domain.attributes):
                self.error("No continuous features")
                data = None
            elif any(a.is_discrete for a in self.data.domain.attributes) or \
                    (not issparse(self.data.X) and numpy.any(numpy.isnan(self.data.X))):
                data = distance._preprocess(self.data)
                if len(self.data.domain.attributes) - len(data.domain.attributes) > 0:
                    self.warning("Ignoring discrete features")
            else:
                data = self.data

        if data is not None:
            shape = (len(data), len(data.domain.attributes))
            if numpy.product(shape) == 0:
                self.error("Empty data (shape == {})".format(shape))
            else:
                distances = metric(data, data, 1 - self.axis, impute=True)

        self.send("Distances", distances)

    def _invalidate(self):
        self.commit()

    def send_report(self):
        self.report_items((
            ("Distances Between", ["Rows", "Columns"][self.axis]),
            ("Metric", self.available_metrics[self.metric_idx].name)
        ))

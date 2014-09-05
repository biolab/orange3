import numpy
from sklearn.metrics import pairwise

import Orange.data
from Orange.widgets import widget, gui, settings


_METRICS = [
    ("Euclidean", pairwise.euclidean_distances),
    ("Manhattan", pairwise.manhattan_distances)
]


class OWDistances(widget.OWWidget):
    name = "Distances"
    description = "Compute a matrix of pairwise distances."
    icon = "icons/Distance.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Distances", numpy.ndarray)]

    axis = settings.Setting(0)
    metric_idx = settings.Setting(0)
    autocommit = settings.Setting(False)

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self._invalidated = False

        box = gui.widgetBox(self.controlArea, self.tr("Distances Between"))
        gui.radioButtons(
            box, self, "axis",
            [self.tr("rows"), self.tr("columns")],
            callback=self._invalidate
        )

        box = gui.widgetBox(self.controlArea, self.tr("Distance Metric"))
        gui.comboBox(box, self, "metric_idx",
                     items=["Euclidean", "Manhattan"],
                     callback=self._invalidate)

        box = gui.widgetBox(self.controlArea, self.tr("Commit"))
        cb = gui.checkBox(box, self, "autocommit", "Commit on any change")
        b = gui.button(box, self, "Apply", callback=self.commit)
        gui.setStopper(self, b, cb, "_invalidated", callback=self.commit)

        self.layout().setSizeConstraint(self.layout().SetFixedSize)

    def set_data(self, data):
        self.data = data
        self.commit()

    def commit(self):
        distances = None
        if self.data is not None:
            metric = _METRICS[self.metric_idx][1]
            X = self.data.X
            if self.axis == 1:
                X = X.T
            distances = metric(X, X)

        self.send("Distances", distances)

    def _invalidate(self):
        if self.autocommit:
            self.commit()
        else:
            self._invalidated = True

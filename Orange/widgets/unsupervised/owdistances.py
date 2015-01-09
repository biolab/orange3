import Orange.data
import Orange.misc
from Orange.widgets import widget, gui, settings
from Orange import distance


_METRICS = [
    ("Euclidean", distance.Euclidean),
    ("Manhattan", distance.Manhattan),
    ("Cosine", distance.Cosine),
    ("Jaccard", distance.Jaccard),
    ("Spearman", distance.SpearmanR),
    ("Spearman absolute", distance.SpearmanRAbsolute),
    ("Pearson", distance.PearsonR),
    ("Pearson absolute", distance.PearsonRAbsolute),
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
                     items=list(zip(*_METRICS))[0],
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
            X = self.data
            X = distance._impute(X)
            distances = metric(X, X, 1-self.axis)

        self.send("Distances", distances)

    def _invalidate(self):
        if self.autocommit:
            self.commit()
        else:
            self._invalidated = True


from sklearn import neighbors

import Orange.data
import Orange.classification
from Orange.data.continuizer import DomainContinuizer

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


class KNNLearner(Orange.classification.Fitter):
    def __init__(self, n_neighbors=5, metric="euclidean", normalize=True):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.normalize = normalize

    def _domain_continuizer(self, data):
        multinomial = continuous = None
        if any(map(is_discrete, data.domain.attributes)):
            multinomial = DomainContinuizer.FrequentIsBase
        if self.normalize:
            continuous = DomainContinuizer.NormalizeBySD
        if multinomial is not None or continuous is not None:
            return DomainContinuizer(multinomial_treatment=multinomial,
                                     normalize_continuous=continuous)
        else:
            return None

    def __call__(self, data):
        dc = self._domain_continuizer(data)
        if dc is not None:
            domain = dc(data)
            data = Orange.data.Table.from_table(domain, data)

        return super().__call__(data)

    def fit(self, X, Y, W):
        skclf = neighbors.KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric=self.metric
        )
        skclf.fit(X, Y.ravel())
        return KNNClassifier(skclf)


class KNNClassifier(Orange.classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob


class OWKNNLearner(widget.OWWidget):

    name = "K Nearest Neighbors"
    description = "K Nearest Neighbors"
    icon = "icons/KNN.svg"
    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Learner", KNNLearner), ("Classifier", KNNClassifier)]

    want_main_area = False

    learner_name = Setting("kNN")
    n_neighbors = Setting(5)
    metric_index = Setting(0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        box = gui.widgetBox(self.controlArea, "Learner/Classifier Name")
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, "Neighbors")
        gui.spin(box, self, "n_neighbors", 1, 100, label="Number of neighbors")

        box = gui.widgetBox(box, "Metric")
        box.setFlat(True)

        gui.comboBox(box, self, "metric_index",
                     items=["Euclidean", "Manhattan", "Maximal"])

        self.metrics = ["euclidean", "manhattan", "chebyshev"]

        gui.button(self.controlArea, self, "Apply",
                   callback=self.apply, default=True)

        self.setMinimumWidth(250)
        layout = self.layout()
        self.layout().setSizeConstraint(layout.SetFixedSize)

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()

    def apply(self):
        learner = KNNLearner(
            n_neighbors=self.n_neighbors,
            metric=self.metrics[self.metric_index]
        )
        learner.name = self.learner_name
        classifier = None
        if self.data is not None:
            classifier = learner(self.data)
            classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

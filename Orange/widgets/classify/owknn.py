from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification import KNNLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWKNNLearner(OWBaseLearner):
    name = "Nearest Neighbors"
    description = "Predict according to the nearest training instances."
    icon = "icons/KNN.svg"
    priority = 20

    LEARNER = KNNLearner

    weights = ["uniform", "distance"]
    metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

    learner_name = Setting("kNN")
    n_neighbors = Setting(5)
    metric_index = Setting(0)
    weight_type = Setting(0)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Neighbors")
        self.n_neighbors_spin = gui.spin(
            box, self, "n_neighbors", 1, 100, label="Number of neighbors:",
            alignment=Qt.AlignRight, callback=self.settings_changed)
        self.metrics_combo = gui.comboBox(
            box, self, "metric_index", orientation=Qt.Horizontal,
            label="Metric:", items=[i.capitalize() for i in self.metrics],
            callback=self.settings_changed)
        self.weights_combo = gui.comboBox(
            box, self, "weight_type", orientation=Qt.Horizontal,
            label="Weight:", items=[i.capitalize() for i in self.weights],
            callback=self.settings_changed)

    def create_learner(self):
        return self.LEARNER(
            n_neighbors=self.n_neighbors,
            metric=self.metrics[self.metric_index],
            weights=self.weights[self.weight_type],
            preprocessors=self.preprocessors
        )

    def get_learner_parameters(self):
        return (("Number of neighbours", self.n_neighbors),
                ("Metric", self.metrics[self.metric_index].capitalize()),
                ("Weight", self.weights[self.weight_type].capitalize()))


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWKNNLearner()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

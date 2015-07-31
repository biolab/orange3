from Orange.data import Table
from Orange.classification import KNNLearner, SklModel
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWKNNLearner(widget.OWWidget):
    name = "Nearest Neighbors"
    description = "k-nearest neighbors classification algorithm."
    icon = "icons/KNN.svg"
    inputs = [("Data", Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", KNNLearner), ("Classifier", SklModel)]

    want_main_area = False
    resizing_enabled = False

    learner_name = Setting("kNN")
    n_neighbors = Setting(5)
    metric_index = Setting(0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, "Learner/Classifier Name")
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, "Neighbors")
        gui.spin(box, self, "n_neighbors", 1, 100, label="Number of neighbors")

        box = gui.widgetBox(box, "Metric")
        box.setFlat(True)

        gui.comboBox(box, self, "metric_index",
                     items=["Euclidean", "Manhattan", "Maximal", "Mahalanobis"])
        self.metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

        gui.button(self.controlArea, self, "Apply",
                   callback=self.apply, default=True)

        self.apply()

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        learner = KNNLearner(
            n_neighbors=self.n_neighbors,
            metric=self.metrics[self.metric_index],
            preprocessors=self.preprocessors
        )
        learner.name = self.learner_name
        classifier = None

        if self.data is not None:
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)


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

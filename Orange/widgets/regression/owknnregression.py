"""
"""

from Orange.data import Table
from  Orange.regression.knn import KNNRegressionLearner
from Orange.regression import SklModel
from Orange.preprocess.preprocess import Preprocess

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWKNNRegression(widget.OWWidget):
    name = "Nearest Neighbors"
    description = "k-nearest neighbours regression algorithm."
    icon = "icons/kNearestNeighbours.svg"
    priority = 20

    inputs = [("Data", Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", KNNRegressionLearner),
               ("Predictor", SklModel)]

    want_main_area = False

    learner_name = Setting("k Nearest Neighbors Regression")
    n_neighbors = Setting(5)
    metric_index = Setting(0)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.preprocessors = ()
        self.data = None

        box = gui.widgetBox(self.controlArea, "Learner/Model Name")
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, "Neighbors")
        gui.spin(box, self, "n_neighbors", 1, 100, label="Number of neighbors")

        box = gui.widgetBox(box, "Metric")
        box.setFlat(True)
        box.layout().setContentsMargins(0, 0, 0, 0)

        gui.comboBox(box, self, "metric_index",
                     items=["Euclidean", "Manhattan", "Maximal", "Mahalanobis"])
        self.metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

        gui.button(self.controlArea, self, "Apply",
                   callback=self.apply, default=True)

        layout = self.layout()
        self.layout().setSizeConstraint(layout.SetFixedSize)

        self.apply()

    def set_data(self, data):
        """Set input training dataset."""
        self.data = data
        if data is not None:
            self.apply()

    def set_preprocessor(self, preproc):
        """Set preprocessor to apply on training data."""
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        """
        Construct the learner and apply it on the training data if available.
        """
        learner = KNNRegressionLearner(
            n_neighbors=self.n_neighbors,
            metric=self.metrics[self.metric_index],
            preprocessors=self.preprocessors
        )
        learner.name = self.learner_name
        model = None
        if self.data is not None:
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                model = learner(self.data)
                model.name = self.learner_name

        self.send("Learner", learner)
        self.send("Predictor", model)


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWKNNRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

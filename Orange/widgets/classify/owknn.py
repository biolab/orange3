import Orange.data
from Orange.classification import KNNLearner, SklModel
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


class OWKNNLearner(widget.OWWidget):

    name = "K Nearest Neighbors"
    description = "K Nearest Neighbors"
    icon = "icons/KNN.svg"
    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", KNNLearner), ("Classifier", SklModel)]

    want_main_area = False
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

        self.setMinimumWidth(250)
        layout = self.layout()
        self.layout().setSizeConstraint(layout.SetFixedSize)

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
            classifier = learner(self.data)
            classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

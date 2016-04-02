from PyQt4.QtGui import QHBoxLayout
from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.classification import KNNLearner, SklModel
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWProvidesLearner
from Orange.widgets.utils.sql import check_sql_input


class OWKNNLearner(OWProvidesLearner, widget.OWWidget):
    name = "Nearest Neighbors"
    description = "k-nearest neighbors classification algorithm."
    icon = "icons/KNN.svg"
    inputs = [("Data", Table, "set_data")] + OWProvidesLearner.inputs
    outputs = [("Learner", KNNLearner), ("Classifier", SklModel)]

    want_main_area = False
    resizing_enabled = False

    weights = ["uniform", "distance"]
    metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

    learner_name = Setting("kNN")
    n_neighbors = Setting(5)
    metric_index = Setting(0)
    weight_type = Setting(0)

    def __init__(self):
        super().__init__()
        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, "Learner/Classifier Name")
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, "Neighbors")
        gui.spin(box, self, "n_neighbors", 1, 100, label="Number of neighbors",
                 alignment=Qt.AlignRight)
        gui.comboBox(box, self, "metric_index", label="Metric",
                     orientation="horizontal",
                     items=[i.capitalize() for i in self.metrics])
        gui.comboBox(box, self, "weight_type", label='Weight',
                     orientation="horizontal",
                     items=[i.capitalize() for i in self.weights])

        g = QHBoxLayout()
        self.controlArea.layout().addLayout(g)
        apply = gui.button(None, self, "Apply",
                           callback=self.apply, default=True)
        g.layout().addWidget(self.report_button)
        g.layout().addWidget(apply)
        self.apply()

    @check_sql_input
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()

    LEARNER = KNNLearner

    def apply(self):
        learner = self.LEARNER(
            n_neighbors=self.n_neighbors,
            metric=self.metrics[self.metric_index],
            weights=self.weights[self.weight_type],
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

    def send_report(self):
        self.report_items((("Name", self.learner_name),))
        self.report_items("Model parameters", (
            ("Number of neighbours", self.n_neighbors),
            ("Metric", self.metrics[self.metric_index].capitalize()),
            ("Weight", self.weights[self.weight_type].capitalize())))
        if self.data:
            self.report_data("Data", self.data)


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

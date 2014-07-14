import Orange.data
from Orange.regression import linear
from Orange.widgets import widget, settings, gui


class OWLinearRegression(widget.OWWidget):
    name = "Linear Regression"
    description = ""
    icon = "icons/LinearRegression.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Learner", linear.RidgeRegressionLearner),
               ("Predictor", linear.LinearModel)]

    learner_name = settings.Setting("Linear Regression")
    alpha = settings.Setting(1.0)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None

        box = gui.widgetBox(self.controlArea, "Learner/Predictor Name")
        gui.lineEdit(box, self, "learner_name")
        gui.button(self.controlArea, self, "Apply", callback=self.apply,
                   default=True)

        self.apply()

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()

    def apply(self):
        learner = linear.RidgeRegressionLearner(
            alpha=self.alpha)
        learner.name = self.learner_name
        predictor = None
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = self.learner_name

        self.send("Learner", learner)
        self.send("Predictor", predictor)

from Orange.widgets import widget, settings, gui

import Orange.data
from Orange.regression import mean


class OWMean(widget.OWWidget):
    name = "Mean Learner"
    description = ""
    icon = "icons/Mean.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Learner", mean.MeanFitter), ("Predictor", mean.MeanModel)]

    learner_name = settings.Setting("Mean Learner")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None

        box = gui.widgetBox(self.controlArea, "Learner Name")
        gui.lineEdit(box, self, "learner_name")
        gui.button(self.controlArea, self, "Apply", callback=self.apply,
                   default=True)
        self.apply()

    def set_data(self, data):
        self.error(0)
        if data is not None:
            if not isinstance(data.domain.class_var,
                              Orange.data.ContinuousVariable):
                data = None
                self.error(0, "Continuous class variable expected.")

        self.data = data
        self.apply()

    def apply(self):
        learner = mean.MeanFitter()
        learner.name = self.learner_name
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = learner.name
        else:
            predictor = None

        self.send("Learner", learner)
        self.send("Predictor", predictor)

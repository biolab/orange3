from Orange.widgets import widget, settings, gui

import Orange.data
from Orange.regression import mean
from Orange.preprocess.preprocess import Preprocess


class OWMean(widget.OWWidget):
    name = "Mean Learner"
    description = "Regression to the average class value from the training set."
    icon = "icons/Mean.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", mean.MeanLearner), ("Predictor", mean.MeanModel)]

    learner_name = settings.Setting("Mean Learner")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, "Learner Name")
        gui.lineEdit(box, self, "learner_name")
        gui.button(self.controlArea, self, "Apply", callback=self.apply,
                   default=True)
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
        learner = mean.MeanLearner(preprocessors=self.preprocessors)
        learner.name = self.learner_name
        predictor = None
        if self.data is not None:
            try:
                self.warning(0)
                predictor = learner(self.data)
                predictor.name = learner.name
            except ValueError as err:
                self.warning(0, str(err))

        self.send("Learner", learner)
        self.send("Predictor", predictor)

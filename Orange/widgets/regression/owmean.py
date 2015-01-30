from Orange.widgets import widget, settings, gui

import Orange.data
from Orange.regression import mean


class OWMean(widget.OWWidget):
    name = "Mean Learner"
    description = ""
    icon = "icons/Mean.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Orange.data.preprocess.Preprocess,
               "set_preprocessor")]
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
        self.error(0)
        if data is not None:
            if not isinstance(data.domain.class_var,
                              Orange.data.ContinuousVariable):
                data = None
                self.error(0, "Continuous class variable expected.")

        self.data = data
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
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = learner.name
        else:
            predictor = None

        self.send("Learner", learner)
        self.send("Predictor", predictor)

"""
Naive Bayes Learner

"""

import Orange.data
import Orange.classification.naive_bayes
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui, settings


class OWNaiveBayes(widget.OWWidget):
    name = "Naive Bayes"
    description = "Naive Bayesian classifier."
    icon = "icons/NaiveBayes.svg"
    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [
       ("Learner", Orange.classification.naive_bayes.NaiveBayesLearner),
       ("Classifier", Orange.classification.naive_bayes.NaiveBayesModel)
    ]

    want_main_area = False

    learner_name = settings.Setting("Naive Bayes")

    def __init__(self):
        super().__init__()

        # GUI
        gui.lineEdit(gui.widgetBox(self.controlArea, self.tr("Name")),
                     self, "learner_name")

        gui.rubber(self.controlArea)

        gui.button(self.controlArea, self, self.tr("&Apply"),
                   callback=self.apply)

        self.data = None
        self.preprocessors = None

        self.initialize()

    def initialize(self):
        """
        Initialize the widget's state.
        """
        learner = Orange.classification.naive_bayes.NaiveBayesLearner()
        learner.name = self.learner_name
        self.send("Learner", learner)
        self.send("Classifier", None)

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()
        else:
            self.send("Classifier", None)

    def apply(self):
        classifier = None
        learner = Orange.classification.naive_bayes.NaiveBayesLearner(
            preprocessors=self.preprocessors)

        learner.name = self.learner_name

        if self.data is not None:
            classifier = learner(self.data)
            classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

"""
Naive Bayes Learner

"""

import Orange.data
import Orange.classification.naive_bayes
from Orange.widgets import widget, gui, settings


class OWNaiveBayes(widget.OWWidget):
    name = "Naive Bayes"
    icon = "icons/NaiveBayes.svg"
    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [
       ("Learner", Orange.classification.naive_bayes.BayesLearner),
       ("Classifier", Orange.classification.naive_bayes.BayesClassifier)
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

        self.initialize()

    def initialize(self):
        """
        Initialize the widget's state.
        """
        learner = Orange.classification.naive_bayes.BayesLearner()
        learner.name = self.learner_name
        self.send("Learner", learner)
        self.send("Classifier", None)

    def setData(self, data):
        self.data = data
        if data is not None:
            self.apply()
        else:
            self.send("Classifier", None)

    def apply(self):
        classifier = None
        learner = Orange.classification.naive_bayes.BayesLearner()
        learner.name = self.learner_name

        if self.data is not None:
            classifier = learner(self.data)
            classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

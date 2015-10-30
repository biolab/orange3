"""
Naive Bayes Learner

"""

from  Orange.data import Table
from Orange.classification.naive_bayes import NaiveBayesLearner, NaiveBayesModel
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, gui, settings


class OWNaiveBayes(widget.OWWidget):
    name = "Naive Bayes"
    description = "Naive Bayesian classifier."
    icon = "icons/NaiveBayes.svg"
    inputs = [("Data", Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", NaiveBayesLearner),
               ("Classifier", NaiveBayesModel)]

    want_main_area = False
    resizing_enabled = False

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
        learner = NaiveBayesLearner()
        learner.name = self.learner_name
        self.send("Learner", learner)
        self.send("Classifier", None)

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()
        else:
            self.send("Classifier", None)

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        learner = NaiveBayesLearner(
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
    ow = OWNaiveBayes()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

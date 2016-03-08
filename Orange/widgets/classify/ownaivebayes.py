"""Naive Bayes Learner
"""

from Orange.data import Table
from Orange.classification.naive_bayes import NaiveBayesLearner
from Orange.widgets import settings
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWNaiveBayes(OWBaseLearner):
    name = "Naive Bayes"
    description = "Naive Bayesian classifier."
    icon = "icons/NaiveBayes.svg"
    priority = 70

    LEARNER = NaiveBayesLearner
    OUTPUT_MODEL_NAME = "Classifier"

    want_main_area = False
    resizing_enabled = False

    learner_name = settings.Setting("Naive Bayes")

    def create_learner(self):
        return self.LEARNER(
            preprocessors=self.preprocessors
        )


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

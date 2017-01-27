"""Naive Bayes Learner
"""

from Orange.data import Table
from Orange.classification.naive_bayes import NaiveBayesLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWNaiveBayes(OWBaseLearner):
    name = "Naive Bayes"
    description = "A fast and simple probabilistic classifier based on " \
                  "Bayes' theorem with the assumption of feature independence. "
    icon = "icons/NaiveBayes.svg"
    priority = 70

    LEARNER = NaiveBayesLearner


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWNaiveBayes()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

"""Naive Bayes Learner
"""

from Orange.data import Table
from Orange.classification.naive_bayes import NaiveBayesLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWNaiveBayes(OWBaseLearner):
    name = "Naive Bayes"
    description = "A fast and simple probabilistic classifier based on " \
                  "Bayes' theorem with the assumption of feature independence."
    icon = "icons/NaiveBayes.svg"
    replaces = [
        "Orange.widgets.classify.ownaivebayes.OWNaiveBayes",
    ]
    priority = 70

    LEARNER = NaiveBayesLearner


if __name__ == "__main__":  # pragma: no cover
    OWNaiveBayes.test_run(Table("iris"))

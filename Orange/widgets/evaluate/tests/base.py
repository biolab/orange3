from unittest.mock import Mock

import numpy as np

from Orange import classification, evaluation
from Orange.data import Table, Domain, DiscreteVariable
from Orange.evaluation import Results
from Orange.evaluation.performance_curves import Curves
from Orange.tests import test_filename

from Orange.widgets.tests.base import WidgetTest


class EvaluateTest(WidgetTest):
    def setUp(self):
        super().setUp()

        n, p = (0, 1)
        actual, probs = np.array([
            (p, .8), (n, .7), (p, .6), (p, .55), (p, .54), (n, .53), (n, .52),
            (p, .51), (n, .505), (p, .4), (n, .39), (p, .38), (n, .37),
            (n, .36), (n, .35), (p, .34), (n, .33), (p, .30), (n, .1)]).T
        self.curves = Curves(actual, probs)
        probs2 = (probs + 1) / 2
        self.curves2 = Curves(actual, probs2)
        pred = probs > 0.5
        pred2 = probs2 > 0.5
        probs = np.vstack((1 - probs, probs)).T
        probs2 = np.vstack((1 - probs2, probs2)).T
        domain = Domain([], DiscreteVariable("y", values=("a", "b")))
        self.results = Results(
            domain=domain,
            actual=actual,
            folds=np.array([Ellipsis]),
            models=np.array([[Mock(), Mock()]]),
            row_indices=np.arange(19),
            predicted=np.array((pred, pred2)),
            probabilities=np.array([probs, probs2]))

        self.lenses = data = Table(test_filename("datasets/lenses.tab"))
        majority = classification.MajorityLearner()
        majority.name = "majority"
        knn3 = classification.KNNLearner(n_neighbors=3)
        knn3.name = "knn-3"
        knn1 = classification.KNNLearner(n_neighbors=1)
        knn1.name = "knn-1"
        self.lenses_results = evaluation.TestOnTestData(
            store_data=True, store_models=True)(
                data=data[::2], test_data=data[1::2],
                learners=[majority, knn3, knn1])
        self.lenses_results.learner_names = ["majority", "knn-3", "knn-1"]

    def test_many_evaluation_results(self):
        if not hasattr(self, "widget"):
            return

        data = Table("iris")
        learners = [
            classification.MajorityLearner(),
            classification.LogisticRegressionLearner(),
            classification.TreeLearner(),
            classification.SVMLearner(),
            classification.KNNLearner(),
            classification.CN2Learner(),
            classification.SGDClassificationLearner(),
            classification.RandomForestLearner(),
            classification.NaiveBayesLearner(),
            classification.SGDClassificationLearner()
        ]
        res = evaluation.CrossValidation(k=2, store_data=True)(data, learners)
        # this is a mixin; pylint: disable=no-member
        self.send_signal("Evaluation Results", res)

from Orange import classification, evaluation
from Orange.data import Table


class EvaluateTest:
    def test_many_evaluation_results(self):
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

from Orange import classification, evaluation
from Orange.data import Table


class EvaluateTest:

    def test_many_evaluation_results(self):
        """
        Now works with more than 9 evaluation results.
        GH-2394 (ROC Analysis)
        GH-2522 (Lift Curve, Calibration Plot)
        """
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
        res = evaluation.CrossValidation(data, learners, k=2, store_data=True)
        self.send_signal("Evaluation Results", res)

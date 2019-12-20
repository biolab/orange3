import numpy as np

from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerClassification", "ModelClassification",
           "SklModelClassification", "SklLearnerClassification"]


class LearnerClassification(Learner):
    learner_adequacy_err_msg = "Discrete class variable expected."

    def check_learner_adequacy(self, domain):
        return domain.has_discrete_class


class ModelClassification(Model):
    pass


class SklModelClassification(SklModel, ModelClassification):
    def predict(self, X):
        prediction = super().predict(X)
        if not isinstance(prediction, tuple):
            return prediction
        values, probs = prediction

        class_vars = self.domain.class_vars
        max_values = max(len(cv.values) for cv in class_vars)
        if max_values == probs.shape[-1]:
            return values, probs

        if not self.supports_multiclass:
            probs = probs[:, np.newaxis, :]
        probs_ext = np.zeros((len(probs), len(class_vars), max_values))
        for c, used_vals in enumerate(self.used_vals):
            for i, cv in enumerate(used_vals):
                probs_ext[:, c, cv] = probs[:, c, i]
        if not self.supports_multiclass:
            probs_ext = probs_ext[:, 0, :]
        return values, probs_ext


class SklLearnerClassification(SklLearner, LearnerClassification):
    __returns__ = SklModelClassification

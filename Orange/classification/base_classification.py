import numpy as np

from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerClassification", "ModelClassification",
           "SklModelClassification", "SklLearnerClassification"]


class LearnerClassification(Learner):
    learner_adequacy_err_msg = "Discrete class variable expected."

    def check_learner_adequacy(self, domain):
        return domain.has_discrete_class or domain.class_var is None


class ModelClassification(Model):
    pass


class SklLearnerClassification(SklLearner, LearnerClassification):
    pass


class SklModelClassification(SklModel, ModelClassification):
    def __call__(self, data, ret=Model.Value):
        prediction = super().__call__(data, ret=ret)

        if ret == Model.Value:
            return prediction

        if ret == Model.Probs:
            probs = prediction
        else:  # ret == Model.ValueProbs
            value, probs = prediction

        # Expand probability predictions for class values which are not present
        if ret != self.Value:
            n_class = len(self.domain.class_vars)
            max_values = max(len(cv.values) for cv in self.domain.class_vars)
            if max_values != probs.shape[-1]:
                if not self.supports_multiclass:
                    probs = probs[:, np.newaxis, :]
                probs_ext = np.zeros((len(probs), n_class, max_values))
                for c in range(n_class):
                    i = 0
                    class_values = len(self.domain.class_vars[c].values)
                    for cv in range(class_values):
                        if (i < len(self.used_vals[c]) and
                                    cv == self.used_vals[c][i]):
                            probs_ext[:, c, cv] = probs[:, c, i]
                            i += 1
                if self.supports_multiclass:
                    probs = probs_ext
                else:
                    probs = probs_ext[:, 0, :]

        if ret == Model.Probs:
            return probs
        else:  # ret == Model.ValueProbs
            return value, probs

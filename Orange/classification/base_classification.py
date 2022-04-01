from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerClassification", "ModelClassification",
           "SklModelClassification", "SklLearnerClassification"]


class LearnerClassification(Learner):

    def incompatibility_reason(self, domain):
        reason = None
        if len(domain.class_vars) > 1 and not self.supports_multiclass:
            reason = "Too many target variables."
        elif not domain.has_discrete_class:
            reason = "Categorical class variable expected."
        return reason


class ModelClassification(Model):
    pass


class SklModelClassification(SklModel, ModelClassification):
    pass


class SklLearnerClassification(SklLearner, LearnerClassification):
    __returns__ = SklModelClassification

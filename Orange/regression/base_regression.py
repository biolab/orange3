from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerRegression", "ModelRegression",
           "SklModelRegression", "SklLearnerRegression"]


class LearnerRegression(Learner):

    def incompatibility_reason(self, domain):
        reason = None
        if len(domain.class_vars) > 1 and not self.supports_multiclass:
            reason = "Too many target variables."
        elif not domain.has_continuous_class:
            reason = "Numeric class variable expected."
        return reason


class ModelRegression(Model):
    pass


class SklModelRegression(SklModel, ModelRegression):
    pass


class SklLearnerRegression(SklLearner, LearnerRegression):
    __returns__ = SklModelRegression

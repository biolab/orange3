from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerRegression", "ModelRegression",
           "SklModelRegression", "SklLearnerRegression"]


class LearnerRegression(Learner):
    learner_adequacy_err_msg = "Continuous class variable expected."

    def check_learner_adequacy(self, domain):
        return domain.has_continuous_class or domain.class_var is None


class ModelRegression(Model):
    pass


class SklLearnerRegression(SklLearner, LearnerRegression):
    pass


class SklModelRegression(SklModel, ModelRegression):
    pass

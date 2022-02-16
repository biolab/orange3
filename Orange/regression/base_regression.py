from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerRegression", "ModelRegression",
           "SklModelRegression", "SklLearnerRegression"]


class LearnerRegression(Learner):

    def check_learner_adequacy(self, domain):
        is_adequate = True
        if len(domain.class_vars) > 1:
            is_adequate = False
            self.learner_adequacy_err_msg = "Too many target variables."
        elif not domain.has_continuous_class:
            is_adequate = False
            self.learner_adequacy_err_msg = "Numeric class variable expected."
        return is_adequate


class ModelRegression(Model):
    pass


class SklModelRegression(SklModel, ModelRegression):
    pass


class SklLearnerRegression(SklLearner, LearnerRegression):
    __returns__ = SklModelRegression

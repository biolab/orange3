from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerClassification", "ModelClassification",
           "SklModelClassification", "SklLearnerClassification"]


class LearnerClassification(Learner):

    def check_learner_adequacy(self, domain):
        is_adequate = True
        if len(domain.class_vars) > 1:
            is_adequate = False
            self.learner_adequacy_err_msg = "Too many target variables."
        elif not domain.has_discrete_class:
            is_adequate = False
            self.learner_adequacy_err_msg = "Categorical class variable expected."
        return is_adequate


class ModelClassification(Model):
    pass


class SklModelClassification(SklModel, ModelClassification):
    pass


class SklLearnerClassification(SklLearner, LearnerClassification):
    __returns__ = SklModelClassification

from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerClassification", "ModelClassification",
           "SklModelClassification", "SklLearnerClassification"]


class LearnerClassification(Learner):
    learner_adequacy_err_msg = "Categorical class variable expected."

    def check_learner_adequacy(self, domain):
        return domain.has_discrete_class


class ModelClassification(Model):
    pass


class SklModelClassification(SklModel, ModelClassification):
    pass


class SklLearnerClassification(SklLearner, LearnerClassification):
    __returns__ = SklModelClassification

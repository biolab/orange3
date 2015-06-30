from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["LearnerRegression", "ModelRegression",
           "SklModelRegression", "SklLearnerRegression"]


class LearnerRegression(Learner):
    def __call__(self, data):
        check_learner_adequacy(data.domain)
        return super().__call__(data)


class ModelRegression(Model):
    pass


class SklLearnerRegression(SklLearner):
    def __call__(self, data):
        check_learner_adequacy(data.domain)
        return super().__call__(data)


class SklModelRegression(SklModel):
    def __call__(self, data, ret=Model.Value):
        return super().__call__(data, ret=ret)


def check_learner_adequacy(domain):
    if domain.has_discrete_class:
        raise ValueError("Continuous class variable expected.")

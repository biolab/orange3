import numpy as np

from Orange.base import Learner, Model
from Orange.modelling import Fitter
from Orange.classification import LogisticRegressionLearner
from Orange.classification.base_classification import LearnerClassification
from Orange.data import Domain, ContinuousVariable, Table
from Orange.evaluation import CrossValidation
from Orange.regression import RidgeRegressionLearner
from Orange.regression.base_regression import LearnerRegression


__all__ = ['StackedLearner', 'StackedClassificationLearner',
           'StackedRegressionLearner', 'StackedFitter']


class StackedModel(Model):
    def __init__(self, models, aggregate, use_prob=True, domain=None):
        super().__init__(domain=domain)
        self.models = models
        self.aggregate = aggregate
        self.use_prob = use_prob

    def predict_storage(self, data):
        if self.use_prob:
            probs = [m(data, Model.Probs) for m in self.models]
            X = np.hstack(probs)
        else:
            pred = [m(data) for m in self.models]
            X = np.column_stack(pred)
        Y = np.repeat(np.nan, X.shape[0])
        stacked_data = data.transform(self.aggregate.domain)
        with stacked_data.unlocked():
            stacked_data.X = X
            stacked_data.Y = Y
        return self.aggregate(
            stacked_data, Model.ValueProbs if self.use_prob else Model.Value)


class StackedLearner(Learner):
    """
    Constructs a stacked model by fitting an aggregator
    over the results of base models.

    K-fold cross-validation is used to get predictions of the base learners
    and fit the aggregator to obtain a stacked model.

    Args:
        learners (list):
            list of `Learner`s used for base models

        aggregate (Learner):
            Learner used to fit the meta model, aggregating predictions
            of base models

        k (int):
            number of folds for cross-validation

    Returns:
        instance of StackedModel
    """

    __returns__ = StackedModel

    def __init__(self, learners, aggregate, k=5, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.learners = learners
        self.aggregate = aggregate
        self.k = k
        self.params = vars()

    def fit_storage(self, data):
        cv = CrossValidation(k=self.k)
        res = cv(data, self.learners)
        if data.domain.class_var.is_discrete:
            X = np.hstack(res.probabilities)
            use_prob = True
        else:
            X = res.predicted.T
            use_prob = False
        dom = Domain([ContinuousVariable('f{}'.format(i + 1))
                      for i in range(X.shape[1])],
                     data.domain.class_var)
        stacked_data = data.transform(dom).copy()
        with stacked_data.unlocked_reference():
            stacked_data.X = X
            stacked_data.Y = res.actual
        models = [l(data) for l in self.learners]
        aggregate_model = self.aggregate(stacked_data)
        return StackedModel(models, aggregate_model, use_prob=use_prob,
                            domain=data.domain)


class StackedClassificationLearner(StackedLearner, LearnerClassification):
    """
    Subclass of StackedLearner intended for classification tasks.

    Same as the super class, but has a default
    classification-specific aggregator (`LogisticRegressionLearner`).
    """

    def __init__(self, learners, aggregate=LogisticRegressionLearner(), k=5,
                 preprocessors=None):
        super().__init__(learners, aggregate, k=k, preprocessors=preprocessors)


class StackedRegressionLearner(StackedLearner, LearnerRegression):
    """
    Subclass of StackedLearner intended for regression tasks.

    Same as the super class, but has a default
    regression-specific aggregator (`RidgeRegressionLearner`).
    """
    def __init__(self, learners, aggregate=RidgeRegressionLearner(), k=5,
                 preprocessors=None):
        super().__init__(learners, aggregate, k=k, preprocessors=preprocessors)


class StackedFitter(Fitter):
    __fits__ = {'classification': StackedClassificationLearner,
                'regression': StackedRegressionLearner}

    def __init__(self, learners, **kwargs):
        kwargs['learners'] = learners
        super().__init__(**kwargs)


if __name__ == '__main__':
    import Orange
    iris = Table('iris')
    knn = Orange.modelling.KNNLearner()
    tree = Orange.modelling.TreeLearner()
    sl = StackedFitter([tree, knn])
    m = sl(iris[::2])
    print(m(iris[1::2], Model.Value))

    housing = Table('housing')
    sl = StackedFitter([tree, knn])
    m = sl(housing[::2])
    print(list(zip(housing[1:10:2].Y, m(housing[1:10:2], Model.Value))))

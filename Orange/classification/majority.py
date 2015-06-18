from numpy import tile, array

from Orange import data
from Orange.classification import Learner, Model
from Orange.statistics import distribution

__all__ = ["MajorityLearner"]


class MajorityLearner(Learner):
    """
    A majority classifier. Always returns most frequent class from the
    training set, regardless of the attribute values from the test data
    instance. Returns class value distribution if class probabilities
    are requested. Can be used as a baseline when comparing classifiers.
    """

    name = 'majority'

    def fit_storage(self, dat):
        if not dat.domain.has_discrete_class:
            raise ValueError("classification.MajorityLearner expects a domain with a "
                             "(single) discrete variable")
        dist = distribution.get_distribution(dat, dat.domain.class_var)
        N = dist.sum()
        if N > 0:
            dist /= N
        else:
            dist.fill(1 / len(dist))
        return ConstantModel(dist=dist)


class ConstantModel(Model):
    """
    A classification model that returns a given class value.
    """
    def __init__(self, dist):
        """
        Constructs `Orange.classification.MajorityModel` that always returns majority value of given distribution.

        If no or empty distribution given, constructs a model that returns equal probabilities for each class value.

        :param dist: domain for the `Table`
        :type dist: Orange.statistics.distribution.Discrete
        :return: regression model that returns majority value
        :rtype: Orange.classification.Model
        """
        self.dist = array(dist)

    def predict(self, X):
        """
        Returns majority class for each given instance in X.

        :param X: data table for which to make predictions
        :type X: Orange.data.Table
        :return: predicted value
        :rtype: vector of majority values
        """
        return tile(self.dist, (len(X), 1))

    def __str__(self):
        return 'ConstantModel {}'.format(self.dist)

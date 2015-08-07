from numpy import tile, array, zeros
from hashlib import sha1

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

    In the special case of uniform class distribution within the training data,
    class value is selected randomly. In order to produce consistent results on
    the same data set, this value is selected based on hash of the class vector.
    """

    name = 'majority'

    def fit_storage(self, dat):
        if not dat.domain.has_discrete_class:
            raise ValueError("classification.MajorityLearner expects a domain "
                             "with a  (single) discrete variable")
        dist = distribution.get_distribution(dat, dat.domain.class_var)
        N = dist.sum()
        if N > 0:
            dist /= N
        else:
            dist.fill(1 / len(dist))

        if all(array(dist) == dist[0]):
            unif_maj = int(sha1(bytes(dat.Y)).hexdigest(), 16) % len(dist)
        else:
            unif_maj = None
        return ConstantModel(dist=dist, unif_maj=unif_maj)


class ConstantModel(Model):
    """
    A classification model that returns a given class value.
    """
    def __init__(self, dist, unif_maj=None):
        """
        Constructs `Orange.classification.MajorityModel` that always
        returns majority value of given distribution.

        If no or empty distribution given, constructs a model that returns equal
        probabilities for each class value.

        :param dist: domain for the `Table`
        :param unif_maj: majority class for the special case of uniform
            class distribution in the training data
        :type dist: Orange.statistics.distribution.Discrete
        :return: regression model that returns majority value
        :rtype: Orange.classification.Model
        """
        self.dist = array(dist)
        self.unif_maj = unif_maj

    def predict(self, X):
        """
        Returns majority class for each given instance in X.

        :param X: data table for which to make predictions
        :type X: Orange.data.Table
        :return: predicted value
        :rtype: vector of majority values
        """
        probs = tile(self.dist, (len(X), 1))
        if self.unif_maj is not None:
            value = tile(self.unif_maj, (len(X), ))
            return value, probs
        return probs

    def __str__(self):
        return 'ConstantModel {}'.format(self.dist)

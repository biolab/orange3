from numpy import tile, array

from Orange import classification, data
from Orange.statistics import distribution


class MajorityFitter(classification.Fitter):
    def fit_storage(self, dat):
        """
        Constructs `Orange.classification.majority.ConstantClassifier` from given data.

        :param dat: table of data
        :type dat: Orange.data.Table
        :return: classification model, which always returns majority value
        :rtype: Orange.classification.majority.ConstantClassifier
        """

        if not isinstance(dat.domain.class_var, data.DiscreteVariable):
            raise ValueError("classification.MajorityFitter expects a domain with a "
                             "(single) discrete variable")
        dist = distribution.get_distribution(dat, dat.domain.class_var)
        N = dist.sum()
        if N > 0:
            dist /= N
        else:
            dist.fill(1 / len(dist))
        return ConstantClassifier(dist=dist)

class ConstantClassifier(classification.Model):
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
        return 'ConstantClassifier {}'.format(self.dist)

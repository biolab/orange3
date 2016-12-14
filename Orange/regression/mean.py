import numpy

from Orange.regression import Learner, Model
from Orange.data import ContinuousVariable
from Orange.statistics import distribution

__all__ = ["MeanLearner"]


class MeanLearner(Learner):
    """
    Fit a regression model that returns the average response (class) value.
    """
    def fit_storage(self, data):
        """
        Construct a :obj:`MeanModel` by computing the mean value of the given
        data.

        :param data: data table
        :type data: Orange.data.Table
        :return: regression model, which always returns mean value
        :rtype: :obj:`MeanModel`
        """
        if not data.domain.has_continuous_class:
            raise ValueError("regression.MeanLearner expects a domain with a "
                             "(single) continuous variable")
        dist = distribution.get_distribution(data, data.domain.class_var)
        return MeanModel(dist)


# noinspection PyMissingConstructor
class MeanModel(Model):
    """
    A regression model that returns the average response (class) value.
    Instances can be constructed directly, by passing a distribution to the
    constructor, or by calling the :obj:`MeanLearner`.

    .. automethod:: __init__

    """
    def __init__(self, dist, domain=None):
        """
        Construct :obj:`Orange.regression.MeanModel` that always returns the
        mean value computed from the given distribution.

        If the distribution is empty, it constructs a model that returns zero.

        :param dist: domain for the `Table`
        :type dist: Orange.statistics.distribution.Continuous
        :return: regression model that returns mean value
        :rtype: :obj:`MeanModel`
        """
        # Don't call super().__init__ because it will raise an error since
        # domain is None.
        self.domain = domain
        self.dist = dist
        if dist.any():
            self.mean = self.dist.mean()
        else:
            self.mean = 0.0

    # noinspection PyPep8Naming
    def predict(self, X):
        """
        Return predictions (that is, the same mean value) for each given
        instance in `X`.

        :param X: data for which to make predictions
        :type X: :obj:`numpy.ndarray`
        :return: a vector of predictions
        :rtype: :obj:`numpy.ndarray`
        """
        return numpy.full(len(X), self.mean)

    def __str__(self):
        return 'MeanModel({})'.format(self.mean)

MeanLearner.__returns__ = MeanModel

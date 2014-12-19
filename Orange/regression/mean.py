import numpy

from Orange import classification, data
from Orange.statistics import distribution

class MeanFitter(classification.Fitter):
    """
    Fits a regression model that returns the average response (class) value.
    """
    def fit_storage(self, dat):
        """
        Constructs `Orange.regression.MeanModel` from given data.

        :param dat: table of data
        :type dat: Orange.data.Table
        :return: regression model, which always returns mean value
        :rtype: Orange.regression.mean.MeanModel
        """
        if not isinstance(dat.domain.class_var, data.ContinuousVariable):
            raise ValueError("regression.MeanFitter expects a domain with a "
                             "(single) continuous variable")
        dist = distribution.get_distribution(dat, dat.domain.class_var)
        return MeanModel(dist)

class MeanModel(classification.Model):
    """
    A regression model that returns the average response (class) value.
    """
    def __init__(self, dist):
        """
        Constructs `Orange.regression.MeanModel` that always returns mean value of given distribution.

        If no or empty distribution given, constructs a model that returns zero.

        :param dist: domain for the `Table`
        :type dist: Orange.statistics.distribution.Continuous
        :return: regression model that returns mean value
        :rtype: Orange.regression.Model
        """
        self.dist = dist
        if dist.any():
            self.mean = self.dist.mean()
        else:
            self.mean = 0.0

    def predict(self, X):
        """
        Returns mean value for each given instance in X.

        :param X: data table for which to make predictions
        :type X: Orange.data.Table
        :return: predicted value
        :rtype: vector of mean values
        """
        return numpy.zeros(X.shape[0]) + self.mean

    def __str__(self):
        return 'MeanModel {}'.format(self.mean)

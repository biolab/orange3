import numpy as np
import Orange.data
from ..data.value import Value
from ..data.instance import Instance

class Learner:
    pass


class Classifier:
    """
    Pure virtual methods
    --------------------

    Derived classes should overload at least one of the following methods:
    :method:`predict_inst_class`, :method:`predict_inst_prob`,
    :method:`predict_inst`, :method:`predict_X`. Most operations will be
    much faster if the classifier also overloads at least one of
    :method:`predict_data_class`, :method:`predict_data_prob`,
    :method:`predict_data`, :method:`predict_X`.

    .. method: predict_inst_class(inst)

        Predict the class for instance `inst` of type
        :class:`~Orange.data.Instance`.

    .. method: predict_inst_prob(inst)

        Predict probability distribution for instance `inst` of
        type :class:`~Orange.data.Instance`. Result is a vector of
        normalized probabilities.

    .. method: predict_inst(inst)

        Return a tuple with predicted value and probability distribution (as
        a normalized vector with probabilities) for instance
        `inst` of type :class:`~Orange.data.Instance`.

    .. method: predict_X(X)

        Predict probability distributions for data instances in 2d array X.
        Result is a two-dimensional array, with rows containing normalized
        probabilities.

    .. method: predict_table_class(data)

        Predict classes of data instances given as an :class:`Orange.data.Table`.
        Result is a vector with the length equal to the number of instances.

    .. method: predict_table_prob(data)

        Predict probability distributions for data instances given as an
        :class:`Orange.data.Table`. Result is a two-dimensional vector,
        array, with rows containing normalized probabilities.

    .. method: predict_table(data)

        Return a tuple of a vector with class predictions and a two-dimensional
        array, with rows containing normalized probabilities.
    """

    Class = 0
    Prob = 1
    ClassProb =2

    def __init__(self, domain):
        self.domain = domain

    predict_inst_class = None
    predict_inst_prob = None
    predict_inst = None
    predict_table_class = None
    predict_table_prob = None
    predict_table = None
    predict_X = None

    def _predict_instance(self, inst, ret=Class):
        """Internal function; do not implement or call directly"""
        if ret == Classifier.Class:
            if self.predict_inst_class:
                value = self.predict_inst_class(inst)
            elif self.predict_inst:
                value = self.predict_inst(inst)[0]
            elif self.predict_inst_prob:
                value = np.argmax(self.predict_inst_prob(inst))
            elif self.predict_X:
                value = np.argmax(self.predict_X(np.atleast_2d(inst._values))[0])
                raise SystemError("invalid classifier")
            return Value(self.domain.class_var, value)
        elif ret == Classifier.Prob:
            if self.predict_inst_prob:
                return self.predict_inst_prob(inst)
            elif self.predict_X:
                return self.predict_X(np.atleast_2d(inst._values))[0]
            elif self.predict_inst:
                return self.predict_inst(inst)[1]
            elif self.predict_inst_class:
                dist = np.zeros(len(self.domain.class_var.values))
                dist[self.predict_inst_class(inst)] = 1
                return dist
            else:
                raise SystemError("invalid classifier")
        else:
            if self.predict_inst:
                return self.predict_inst(inst)
            elif self.predict_inst_prob:
                dist = self.predict_inst_prob(inst)
                return np.argmax(dist), dist
            elif self.predict_X:
                dist = self.predict_X(np.atleast_2d(inst))
                return np.argmax(dist), dist
            elif self.predict_inst_class:
                value = self.predict_inst_class(inst)
                dist = np.zeros(len(self.domain.class_var.values))
                dist[value] = 1
                return value, dist
            else:
                raise SystemError("invalid classifier")


    def _predict_X(self, x, ret=Class):
        """Internal function; do not implement or call directly"""
        prediction = self.predict_X(x)
        if x.ndim == 1:
            if ret == Classifier.Class:
                return np.argmax(prediction[0])
            elif ret == Classifier.Prob:
                return prediction[0]
            else:
                return np.argmax(prediction[0]), prediction[0]
        else:
            if ret == Classifier.Class:
                return np.argmax(prediction, axis=1)
            elif ret == Classifier.Prob:
                return prediction
            else:
                return np.argmax(prediction[0], axis=1), prediction


    def _predict_data(self, data, ret=Class):
        ...


    def __call__(self, data, ret=Class):
        if 0 <= ret <= 2:
            raise ValueError("invalid value of 'ret'")

        if self.predict_X and isinstance(data, np.array):
            return self._predict_X(data.X, ret)

        if isinstance(data, Instance):
            if data.domain != self.domain:
                inst = Instance(self.domain, data, ret)
            return self._predict_instance(data, ret)

        if isinstance(data, np.array):
            if data.ndim == 1:
                inst = Instance(self.domain, data)
                return self._predict_instance(inst, ret)
            else:
                data = data.Table.new_from_domain(self.domain, data)
                return self._predict_data(data, ret)

        if isinstance(data, data.Table):
            if data.domain != self.domain:
                data = data.Table.new_from_table(self.domain, data)
            return self._predict_data(data, ret)

        raise TypeError("invalid arguments for classifier")
import numpy as np

class Results:
    def __init__(self, data, nmethods, nrows=None):
        self.data = None
        self.models = None
        nclasses = len(data.domain.class_var.values)
        if nrows is None:
            nrows = len(data)
        try:
            dtype = data.Y.dtype
        except:
            dtype = np.float32
        self.correct = np.empty(nrows, dtype=dtype)
        self.predictions = np.empty((nmethods, nrows), dtype=dtype)
        self.probabilities = np.empty((nmethods, nrows, nclasses),
                                      dtype=np.float32)
        self.row_indices = np.empty(nrows, dtype=np.int32)


class Testing:
    def __new__(cls, data=None, learners=None, **kwargs):
        """

        :param store_data:
        :param store_models:
        :return:
        """
        self = super().__new__(cls)
        self.store_data = kwargs.pop('store_data', False)
        self.store_classifiers = kwargs.pop('store_models', False)

        if data or learners:
            self = self.__init__(**kwargs)
            return self(data, learners)
        return self


class CrossValidation(Testing):
    def __init__(self, k=10, **kwargs):
        self.k = kwargs.pop('k', 10)

    def __call__(self, data, fitters):
        from sklearn import cross_validation
        n = len(data)
        indices = cross_validation.KFold(n, self.k, shuffle=True)
        results = Results(data, len(fitters))

        for train, test in indices:
            for i, fitter in enumerate(fitters):
                model = fitter(data[train])
                values, probs = model(data[test], model.ValueProbs)
                results.predictions[i][test, :] = values
                results.probabilities[i][test, :] = probs
        return results

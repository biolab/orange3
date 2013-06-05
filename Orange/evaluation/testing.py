import numpy as np


class Results:
    def __init__(self, data, nmethods, nrows):
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
    def __new__(cls, store_data=False, store_models=False):
        """

        :param store_data:
        :param store_models:
        :return:
        """
        self = super().__new__(cls)
        self.store_data = store_data
        self.store_classifiers = store_models
        return self



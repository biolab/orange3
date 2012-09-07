import collections
import numpy as np
import bottleneck as bn
from Orange import classification
from Orange import data

class MajorityLearner(classification.Learner):
    def __call__(self, data):
        if len(data.domain.class_vars) > 1:
            raise NotImplementedError(
                "Majority learner does not support multiple classes")
        class_var = data.domain.class_var
        y = data.Y
        if isinstance(data.domain.class_var, data.ContinuousVariable):
            return DefaultClassifier(data.domain, bn.nanmedian(y))
        else:
            n_values = data.domain.class_var.values()
            if y.dtype != int:
                nans = y.isnan()
                y = np.array(y, dtype=int)
                y[nans] = len(n_values)
            if data.W.shape[-1] == 0:
                distr = np.bincount(y, minlength=n_values)
            else:
                distr = np.bincount(y, data.W, minlength=n_values)
            distr = np.asarray(distr, np.float)[:n_values]
            return DefaultClassifier(data.domain, distr=distr)


class DefaultClassifier(classification.Classifier):
    def __init__(self, value=None, distr=None):
        if value is None:
            mx = np.max(distr)
            value = [i for i, e in enumerate(distr) if e == mx]
        self.value = value[0] if len(value) == 1 else value
        self.distr = distr

    def __call__(self, x):
        if isinstance(x, data.Instance):
            if isinstance(self.value, collections.Sequence):
                return self.value[x.checksum() % len(self.value)]
            else:
                return self.value
        else:
            if isinstance(self.value, collections.Sequence):
                rand = np.random.RandomState(x.checksum())
                return self.value[rand.randint(len(self.value), size=len(x))]
            else:
                return np.tile(self.value, len(x))

    def p(self, x):
        if self.distr is None:
            raise ValueError("Distribution is unknown")
        if isinstance(x, data.Instance):
            return self.distr
        else:
            return np.tile(self.distr, len(x)).reshape(len(x), -1)

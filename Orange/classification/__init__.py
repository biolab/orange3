import copy
import numpy as np
from Orange import data
from ..data.value import Value

class Fitter:
    def __call__(self, data):
        model = copy.deepcopy(self)
        model.fit(data.X)
        return model


class Model:
    Value = 0
    Probs = 1
    ValueProbs =2

    def __init__(self, domain):
        self.domain = domain

    def predict(self, data):
        raise TypeError("Descendants of Model must overload method predict")

    def __call__(self, data, ret=Value):
        if not 0 <= ret <= 2:
            raise ValueError("invalid value of argument 'ret'")
        if ret > 0 and any(isinstance(v, data.ContinuousVariable)
                           for v in self.domain.class_vars):
            raise ValueError("cannot predict continuous distributions")

        # Call the predictor
        if isinstance(data, np.array):
            prediction = self.predict(np.atleast_2d(data))
        elif isinstance(data, data.Instance):
            if data.domain != self.domain:
                data = data.Instance(self.domain, data)
            prediction = self.predict(np.atleast_2d(data._values))
        elif isinstance(data, data.Table):
            if data.domain != self.domain:
                data = data.Table.new_from_table(self.domain, data)
            prediction = self.predict(data.X)
        else:
            raise TypeError("Unrecognized argument (instance of '%s')",
                            type(data).__name__)

        # Parse the result into value and probs
        multitarget = len(self.domain.class_vars) > 1
        if isinstance(prediction, tuple):
            value, probs = prediction
        elif prediction.ndim == 1 + multitarget:
            value, probs = prediction, None
        elif prediction.ndim == 2 + multitarget:
            value, probs = None, prediction
        else:
            raise TypeError("model returned a %i-dimensional array",
                            prediction.ndim)

        # Ensure that we have what we need to return
        if ret != Model.Probs and value is None:
            value = np.argmax(probs, axis=1)
        if ret != Model.Value and probs is None:
            if multitarget:
                max_card = max(len(c.values) for c in self.domain.class_vars)
                probs = np.zeros(value.shape + (max_card,), float)
                for i, cvar in enumerate(self.domain.class_vars):
                    for cval in len(cvar.values):
                        probs[i, value[:, i]==cval] = 1
            else:
                probs = np.zeros((len(value), len(self.domain.class_var.values)),
                                 float)
                for cval in probs.shape[1]:
                    probs[value==cval] = 1
            return probs

        # Return what we need to
        if ret == Model.Value:
            if isinstance(data, data.Instance) and not multitarget:
                value = Value(self.domain.class_var, value[0])
            return value
        if ret == Model.Probs:
            return probs
        else: # ret == Model.ValueProbs
            return value, probs

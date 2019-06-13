from Orange.classification import Model

__all__ = ["ModelWithThreshold"]


class ModelWithThreshold(Model):
    def __init__(self, wrapped_model, threshold, target_class=1):
        super().__init__(wrapped_model.domain, wrapped_model.original_domain)
        self.name = f"{wrapped_model.name}, thresh={threshold:.2f}"
        self.wrapped_model = wrapped_model
        self.threshold = threshold
        self.target_class = target_class

    def __call__(self, data, ret=Model.Value):
        probs = self.wrapped_model(data, ret=Model.Probs)
        if ret == Model.Probs:
            return probs
        vals = probs[:, self.target_class].flatten() > self.threshold
        if ret == Model.Value:
            return vals
        else:
            return vals, probs

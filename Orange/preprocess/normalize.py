from Orange.data import ContinuousVariable, Domain
from Orange.statistics import distribution
from .transformation import Normalizer as Norm
from .preprocess import Normalize
import warnings

__all__ = ["Normalizer"]


class Normalizer:
    def __init__(self,
                 zero_based=True,
                 norm_type=Normalize.NormalizeBySD,
                 transform_class=False):
        self.zero_based = zero_based
        self.norm_type = norm_type
        self.transform_class = transform_class

    def __call__(self, data):
        dists = distribution.get_distributions(data)
        new_attrs = [self.normalize(dists[i], var) for
                     (i, var) in enumerate(data.domain.attributes)]
        new_class_vars = data.domain.class_vars
        if self.transform_class:
            attr_len = len(data.domain.attributes)
            new_class_vars = [self.normalize(dists[i + attr_len], var) for
                              (i, var) in enumerate(data.domain.class_vars)]
        domain = Domain(new_attrs, new_class_vars, data.domain.metas)
        return data.from_table(domain, data)

    def normalize(self, dist, var):
        if not isinstance(var, ContinuousVariable):
            return var
        elif self.norm_type == Normalize.NormalizeBySD:
            return self.normalize_by_sd(dist, var)
        elif self.norm_type == Normalize.NormalizeBySpan:
            return self.normalize_by_span(dist, var)

    def normalize_by_sd(self, dist, var):
        new_var = ContinuousVariable(var.name)
        avg, sd = dist.mean(), dist.standard_deviation()
        if sd == 0:
            sd = 1
        new_var.compute_value = Norm(var, avg, 1 / sd)
        return new_var

    def normalize_by_span(self, dist, var):
        new_var = ContinuousVariable(var.name)
        dma, dmi = dist.max(), dist.min()
        diff = dma - dmi
        if diff < 1e-15:
            diff = 1
        if self.zero_based:
            new_var.compute_value = Norm(var, dmi, 1 / diff)
        else:
            new_var.compute_value = Norm(var, (dma + dmi) / 2,
                                         2 / diff)
        return new_var

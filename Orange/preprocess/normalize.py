from Orange.data import ContinuousVariable, Domain
from Orange.statistics import distribution
from Orange.util import Reprable
from .preprocess import Normalize
from .transformation import Normalizer as Norm

__all__ = ["Normalizer"]


class Normalizer(Reprable):
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
        return data.transform(domain)

    def normalize(self, dist, var):
        if not var.is_continuous:
            return var
        elif self.norm_type == Normalize.NormalizeBySD:
            return self.normalize_by_sd(dist, var)
        elif self.norm_type == Normalize.NormalizeBySpan:
            return self.normalize_by_span(dist, var)

    def normalize_by_sd(self, dist, var):
        avg, sd = (dist.mean(), dist.standard_deviation()) if dist.size else (0, 1)
        if sd == 0:
            sd = 1
        return ContinuousVariable(var.name, compute_value=Norm(var, avg, 1 / sd), sparse=var.sparse)

    def normalize_by_span(self, dist, var):
        dma, dmi = dist.max(), dist.min()
        diff = dma - dmi
        if diff < 1e-15:
            diff = 1
        if self.zero_based:
            return ContinuousVariable(
                var.name,
                compute_value=Norm(var, dmi, 1 / diff),
                sparse=var.sparse)
        else:
            return ContinuousVariable(
                var.name,
                compute_value=Norm(var, (dma + dmi) / 2, 2 / diff),
                sparse=var.sparse)

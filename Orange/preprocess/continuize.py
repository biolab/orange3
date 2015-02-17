from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.statistics import distribution
from .transformation import Identity, Indicator, Indicator1, Normalizer
from .preprocess import Continuize

__all__ = ["DomainContinuizer", "MultinomialTreatment"]


class DomainContinuizer:
    def __new__(cls, data=None, zero_based=True,
                multinomial_treatment=Continuize.Indicators,
                normalize_continuous=None,
                transform_class=False):
        self = super().__new__(cls)
        self.zero_based = zero_based
        self.multinomial_treatment = multinomial_treatment
        if normalize_continuous is None:
            self.normalize_continuous = Continuize.Leave
        else:
            self.normalize_continuous = normalize_continuous
        self.transform_class = transform_class

        return self if data is None else self(data)

    def __call__(self, data):
        def transform_discrete(var):
            if (len(var.values) < 2 or
                    treat == Continuize.Remove or
                    treat == Continuize.RemoveMultinomial and
                    len(var.values) > 2):
                return []
            if treat == Continuize.AsOrdinal:
                new_var = ContinuousVariable(var.name)
                new_var.compute_value = Identity(var)
                return [new_var]
            if treat == Continuize.AsNormalizedOrdinal:
                new_var = ContinuousVariable(var.name)
                n_values = max(1, len(var.values))
                if self.zero_based:
                    new_var.compute_value = \
                        Normalizer(var, 0, 1 / (n_values - 1))
                else:
                    new_var.compute_value = \
                        Normalizer(var, (n_values - 1) / 2, 2 / (n_values - 1))
                return [new_var]

            new_vars = []
            if treat == Continuize.Indicators:
                base = -1
            elif treat in (Continuize.FirstAsBase,
                           Continuize.RemoveMultinomial):
                base = max(var.base_value, 0)
            else:
                base = dists[var_ptr].modus()
            ind_class = [Indicator1, Indicator][self.zero_based]
            for i, val in enumerate(var.values):
                if i == base:
                    continue
                new_var = ContinuousVariable(
                    "{}={}".format(var.name, val))
                new_var.compute_value = ind_class(var, i)
                new_vars.append(new_var)
            return new_vars

        def transform_continuous(var):
            if self.normalize_continuous == Continuize.Leave:
                return var
            elif self.normalize_continuous == Continuize.NormalizeBySpan:
                new_var = ContinuousVariable(var.name)
                dma, dmi = dists[var_ptr].max(), dists[var_ptr].min()
                diff = dma - dmi
                if diff < 1e-15:
                    diff = 1
                if self.zero_based:
                    new_var.compute_value = Normalizer(var, dmi, 1 / diff)
                else:
                    new_var.compute_value = Normalizer(var, (dma + dmi) / 2,
                                                       2 / diff)
                return new_var
            elif self.normalize_continuous == Continuize.NormalizeBySD:
                new_var = ContinuousVariable(var.name)
                avg = dists[var_ptr].mean()
                sd = dists[var_ptr].standard_deviation()
                new_var.compute_value = Normalizer(var, avg, 1 / sd)
                return new_var

        def transform_list(s):
            nonlocal var_ptr
            new_vars = []
            for var in s:
                if isinstance(var, DiscreteVariable):
                    new_vars += transform_discrete(var)
                    if needs_discrete:
                        var_ptr += 1
                else:
                    new_var = transform_continuous(var)
                    if new_var is not None:
                        new_vars.append(new_var)
                        if needs_continuous:
                            var_ptr += 1
            return new_vars

        treat = self.multinomial_treatment
        transform_class = self.transform_class

        domain = data if isinstance(data, Domain) else data.domain
        if treat == Continuize.ReportError and any(
                isinstance(var, DiscreteVariable) and len(var.values) > 2
                for var in domain):
            raise ValueError("data has multinomial attributes")
        needs_discrete = (treat == Continuize.FrequentAsBase and
                          domain.has_discrete_attributes(transform_class))
        needs_continuous = (not self.normalize_continuous == Continuize.Leave
                            and
                            domain.has_continuous_attributes(transform_class))
        if needs_discrete or needs_continuous:
            if isinstance(data, Domain):
                raise TypeError("continuizer requires data")
            dists = distribution.get_distributions(
                data, not needs_discrete, not needs_continuous)
        var_ptr = 0
        new_attrs = transform_list(domain.attributes)
        if transform_class:
            new_classes = transform_list(domain.class_vars)
        else:
            new_classes = domain.class_vars
        return Domain(new_attrs, new_classes, domain.metas)

MultinomialTreatment = Continuize.MultinomialTreatment

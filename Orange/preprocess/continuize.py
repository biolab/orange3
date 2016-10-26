from Orange.data import ContinuousVariable, Domain
from Orange.statistics import distribution
from Orange.misc.reprable import Reprable
from .transformation import Identity, Indicator, Indicator1, Normalizer
from .preprocess import Continuize

__all__ = ["DomainContinuizer", "MultinomialTreatment"]


class DomainContinuizer(Reprable):
    def __new__(cls, data=None, zero_based=True,
                multinomial_treatment=Continuize.Indicators,
                transform_class=False):
        self = super().__new__(cls)
        self.zero_based = zero_based
        self.multinomial_treatment = multinomial_treatment
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
                new_var = ContinuousVariable(var.name,
                                             compute_value=Identity(var))
                return [new_var]
            if treat == Continuize.AsNormalizedOrdinal:
                n_values = max(1, len(var.values))
                if self.zero_based:
                    return [ContinuousVariable(var.name, compute_value=Normalizer(var, 0, 1 / (n_values - 1)))]
                else:
                    return [ContinuousVariable(var.name, compute_value=Normalizer(var, (n_values - 1) / 2, 2 / (n_values - 1)))]

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
                    "{}={}".format(var.name, val),
                    compute_value=ind_class(var, i))
                new_vars.append(new_var)
            return new_vars

        def transform_list(s):
            nonlocal var_ptr
            new_vars = []
            for var in s:
                if var.is_discrete:
                    new_vars += transform_discrete(var)
                    if needs_discrete:
                        var_ptr += 1
                else:
                    new_var = var
                    if new_var is not None:
                        new_vars.append(new_var)
                        if needs_continuous:
                            var_ptr += 1
            return new_vars

        treat = self.multinomial_treatment
        transform_class = self.transform_class

        domain = data if isinstance(data, Domain) else data.domain
        if (treat == Continuize.ReportError and
                any(var.is_discrete and len(var.values) > 2 for var in domain)):
            raise ValueError("data has multinomial attributes")
        needs_discrete = (treat == Continuize.FrequentAsBase and
                          domain.has_discrete_attributes(transform_class))
        needs_continuous = False
        if needs_discrete:
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

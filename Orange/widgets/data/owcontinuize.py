from PyQt4 import QtGui

import Orange.data
from Orange.statistics import distribution
from Orange.preprocess import Continuize
from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting


class OWContinuize(widget.OWWidget):
    name = "Continuize"
    description = ("Turns discrete attributes into continuous and, " +
                   "optionally, normalizes the continuous values.")
    icon = "icons/Continuize.svg"
    author = "Martin Frlin"
    category = "Data"
    keywords = ["data", "continuize"]

    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Data", Orange.data.Table)]

    want_main_area = False

    multinomial_treatment = Setting(0)
    zero_based = Setting(1)
    continuous_treatment = Setting(0)
    class_treatment = Setting(0)

    transform_class = Setting(False)

    autosend = Setting(True)

    multinomial_treats = (
        ("Target or First value as base", Continuize.FirstAsBase),
        ("Most frequent value as base", Continuize.FrequentAsBase),
        ("One attribute per value", Continuize.Indicators),
        ("Ignore multinomial attributes", Continuize.RemoveMultinomial),
        ("Remove all discrete attributes", Continuize.Remove),
        ("Treat as ordinal", Continuize.AsOrdinal),
        ("Divide by number of values", Continuize.AsNormalizedOrdinal))

    continuous_treats = (
        ("Leave them as they are", Continuize.Leave),
        ("Normalize by span", Continuize.NormalizeBySpan),
        ("Normalize by standard deviation", Continuize.NormalizeBySD))

    class_treats = (
        ("Leave it as it is", Continuize.Leave),
        ("Treat as ordinal", Continuize.AsOrdinal),
        ("Divide by number of values", Continuize.AsNormalizedOrdinal),
        ("One class per value", Continuize.Indicators),
    )

    value_ranges = ["from -1 to 1", "from 0 to 1"]

    def __init__(self, parent=None):
        widget.OWWidget.__init__(self, parent)

        box = gui.widgetBox(self.controlArea, "Multinomial attributes")
        gui.radioButtonsInBox(
            box, self, "multinomial_treatment",
            btnLabels=[x[0] for x in self.multinomial_treats],
            callback=self.settings_changed)

        box = gui.widgetBox(self.controlArea, "Continuous attributes")
        gui.radioButtonsInBox(
            box, self, "continuous_treatment",
            btnLabels=[x[0] for x in self.continuous_treats],
            callback=self.settings_changed)

        box = gui.widgetBox(self.controlArea, "Discrete class attribute")
        gui.radioButtonsInBox(
            box, self, "class_treatment",
            btnLabels=[t[0] for t in self.class_treats],
            callback=self.settings_changed)

        zbbox = gui.widgetBox(self.controlArea, "Value range")

        gui.radioButtonsInBox(
            zbbox, self, "zero_based",
            btnLabels=self.value_ranges,
            callback=self.settings_changed)

        gui.auto_commit(self.controlArea, self, "autosend", "Apply")

        self.data = None
        self.resize(150, 300)

    def settings_changed(self):
        self.commit()

    def setData(self, data):
        self.data = data
        if data is None:
            self.send("Data", None)
        else:
            self.unconditional_commit()

    def constructContinuizer(self):
        conzer = DomainContinuizer(
            zero_based=self.zero_based,
            multinomial_treatment=self.multinomial_treats[self.multinomial_treatment][1],
            continuous_treatment=self.continuous_treats[self.continuous_treatment][1],
            class_treatment=self.class_treats[self.class_treatment][1]
        )
        return conzer

    # def sendPreprocessor(self):
    #     continuizer = self.constructContinuizer()
    #     self.send("Preprocessor", PreprocessedLearner(
    #         lambda data, weightId=0, tc=(self.targetValue if self.classTreatment else -1):
    #             Table(continuizer(data, weightId, tc)
    #                 if data.domain.class_var and isinstance(self.data.domain.class_var, DiscreteVariable)
    #                 else continuizer(data, weightId), data)))

    def commit(self):
        continuizer = self.constructContinuizer()
        if self.data is not None:
            domain = continuizer(self.data)
            data = Table.from_table(domain, self.data)
            self.send("Data", data)
        else:
            self.send("Data", None)

    def sendReport(self):
        self.reportData(self.data, "Input data")
        self.reportSettings(
            "Settings",
            [("Multinominal attributes",
              self.multinomial_treats[self.multinomial_treatment][0]),
             ("Continuous attributes",
              self.continuous_treats[self.continuous_treatment][0]),
             ("Class", self.class_treats[self.class_tereatment][0]),
             ("Value range", self.value_ranges[self.zero_based])])


from Orange.preprocess.transformation import \
    Identity, Indicator, Indicator1, Normalizer

from functools import partial, wraps, reduce


# flip:: (a * b -> c) -> (b * a -> c)
def flip(func):
    """Flip parameter order"""
    return wraps(func)(lambda a, b: func(b, a))


is_discrete = partial(flip(isinstance), Orange.data.DiscreteVariable)
is_continuous = partial(flip(isinstance), Orange.data.ContinuousVariable)


class WeightedIndicator(Indicator):
    def __init__(self, variable, value, weight=1.0):
        super().__init__(variable, value)
        self.weight = weight

    def transform(self, c):
        t = super().transform(c) * self.weight
        if self.weight != 1.0:
            t *= self.weight
        return t


class WeightedIndicator_1(Indicator1):
    def __init__(self, variable, value, weight=1.0):
        super().__init__(variable, value)
        self.weight = weight

    def transform(self, c):
        t = super().transform(c) * self.weight
        if self.weight != 1.0:
            t *= self.weight
        return t


def make_indicator_var(source, value_ind, weight=None, zero_based=True):
    var = Orange.data.ContinuousVariable(
        "{}={}".format(source.name, source.values[value_ind])
    )
    if zero_based and weight is None:
        indicator = Indicator(source, value=value_ind)
    elif zero_based:
        indicator = WeightedIndicator(source, value=value_ind, weight=weight)
    elif weight is None:
        indicator = Indicator1(source, value=value_ind)
    else:
        indicator = WeightedIndicator_1(source, value=value_ind, weight=weight)
    var.compute_value = indicator
    return var


def dummy_coding(var, base_value=-1, zero_based=True):
    N = len(var.values)
    if base_value == -1:
        base_value = var.base_value if var.base_value >= 0 else 0
    assert 0 <= base_value < len(var.values)
    return [make_indicator_var(var, i, zero_based=zero_based)
            for i in range(N) if i != base_value]


def one_hot_coding(var, zero_based=True):
    N = len(var.values)
    return [make_indicator_var(var, i, zero_based=zero_based)
            for i in range(N)]


def continuize_domain(data_or_domain,
                      multinomial_treatment=Continuize.Indicators,
                      continuous_treatment=Continuize.Leave,
                      class_treatment=Continuize.Leave,
                      zero_based=True):

    if isinstance(data_or_domain, Orange.data.Domain):
        data, domain = None, data_or_domain
    else:
        data, domain = data_or_domain, data_or_domain.domain

    def needs_dist(var, mtreat, ctreat):
        "Does the `var` need a distribution given specified flags"
        if isinstance(var, Orange.data.DiscreteVariable):
            return mtreat == Continuize.FrequentAsBase
        elif isinstance(var, Orange.data.ContinuousVariable):
            return ctreat != Continuize.Leave
        else:
            raise ValueError

    # Compute the column indices which need a distribution.
    attr_needs_dist = [needs_dist(var, multinomial_treatment,
                                  continuous_treatment)
                       for var in domain.attributes]
    cls_needs_dist = [needs_dist(var, class_treatment, Continuize.Leave)
                      for var in domain.class_vars]

    columns = [i for i, needs in enumerate(attr_needs_dist + cls_needs_dist)
               if needs]

    if columns:
        if data is None:
            raise TypeError("continuizer requires data")
        dist = distribution.get_distributions_for_columns(data, columns)
    else:
        dist = []

    dist_iter = iter(dist)

    newattrs = [continuize_var(var, next(dist_iter) if needs_dist else None,
                               multinomial_treatment, continuous_treatment,
                               zero_based)
                for var, needs_dist in zip(domain.attributes, attr_needs_dist)]

    newclass = [continuize_var(var,
                               next(dist_iter) if needs_dist else None,
                               class_treatment, Continuize.Remove,
                               zero_based)
                for var, needs_dist in zip(domain.class_vars, cls_needs_dist)]

    newattrs = reduce(list.__iadd__, newattrs, [])
    newclass = reduce(list.__iadd__, newclass, [])
    return Orange.data.Domain(newattrs, newclass, domain.metas)


def continuize_var(var,
                   data_or_dist=None,
                   multinomial_treatment=Continuize.Indicators,
                   continuous_treatment=Continuize.Leave,
                   zero_based=True):

    if isinstance(var, Orange.data.ContinuousVariable):
        if continuous_treatment == Continuize.NormalizeBySpan:
            return [normalize_by_span(var, data_or_dist, zero_based)]
        elif continuous_treatment == Continuize.NormalizeBySD:
            return [normalize_by_sd(var, data_or_dist)]
        else:
            return [var]

    elif isinstance(var, Orange.data.DiscreteVariable):
        if len(var.values) > 2 and \
                multinomial_treatment == Continuize.ReportError:
            raise ValueError("{0.name} is a multinomial variable".format(var))
        if len(var.values) < 2 or \
                multinomial_treatment == Continuize.Remove or \
                (multinomial_treatment == Continuize.RemoveMultinomial
                 and len(var.values) > 2):
            return []
        elif multinomial_treatment == Continuize.AsOrdinal:
            return [ordinal_to_continuous(var)]
        elif multinomial_treatment == Continuize.AsNormalizedOrdinal:
            return [ordinal_to_normalized_continuous(var, zero_based)]
        elif multinomial_treatment == Continuize.Indicators:
            return one_hot_coding(var, zero_based)
        elif multinomial_treatment == Continuize.FirstAsBase or \
                multinomial_treatment == Continuize.RemoveMultinomial:
            return dummy_coding(var, zero_based=zero_based)
        elif multinomial_treatment == Continuize.FrequentAsBase:
            dist = _ensure_dist(var, data_or_dist)
            modus = dist.modus()
            return dummy_coding(var, base_value=modus, zero_based=zero_based)
        elif multinomial_treatment == Continuize.Leave:
            return [var]
        else:
            raise NotImplementedError  # ValueError??


def _ensure_dist(var, data_or_dist):
    if isinstance(data_or_dist, distribution.Discrete):
        if not is_discrete(var):
            raise TypeError
        return data_or_dist
    elif isinstance(data_or_dist, distribution.Continuous):
        if not is_continuous(var):
            raise TypeError
        return data_or_dist
    elif isinstance(data_or_dist, Orange.data.Storage):
        return distribution.get_distribution(data_or_dist, var)
    else:
        raise ValueError("Need a distribution or data.")


def normalized_var(var, translate, scale):
    new_var = Orange.data.ContinuousVariable(var.name)
    norm = Normalizer(var, translate, scale)
    new_var.compute_value = norm
    return new_var


def ordinal_to_continuous(var):
    new_var = Orange.data.ContinuousVariable(var.name)
    new_var.compute_value = Identity(var)
    return new_var


def ordinal_to_normalized_continuous(var, zero_based=True):
    n_values = len(var.values)
    if zero_based:
        return normalized_var(var, 0, 1 / (n_values - 1))
    else:
        return normalized_var(var, (n_values - 1) / 2, 2 / (n_values - 1))


def normalize_by_span(var, data_or_dist, zero_based=True):
    dist = _ensure_dist(var, data_or_dist)
    v_max, v_min = dist.max(), dist.min()
    span = v_max - v_min
    if span < 1e-15:
        span = 1

    if zero_based:
        return normalized_var(var, v_min, 1 / span)
    else:
        return normalized_var(var, (v_min + v_max) / 2, 2 / span)


def normalize_by_sd(var, data_or_dist):
    dist = _ensure_dist(var, data_or_dist)
    mean, sd = dist.mean(), dist.standard_deviation()
    return normalized_var(var, mean, 1 / sd)


class DomainContinuizer:
    def __new__(cls, data=None, zero_based=True,
                multinomial_treatment=Continuize.Indicators,
                continuous_treatment=Continuize.Leave,
                class_treatment=Continuize.Leave):
        self = super().__new__(cls)
        self.zero_based = zero_based
        self.multinomial_treatment = multinomial_treatment
        self.continuous_treatment = continuous_treatment
        self.class_treatment = class_treatment
        return self if data is None else self(data)

    def __call__(self, data):
        treat = self.multinomial_treatment
        if isinstance(data, Orange.data.Domain):
            domain, data = data, None
        else:
            domain = data.domain

        if treat == Continuize.ReportError and \
                any(isinstance(var, Orange.data.DiscreteVariable) and
                    len(var.values) > 2
                    for var in domain):
            raise ValueError("Domain has multinomial attributes")

        newdomain = continuize_domain(
            data or domain,
            self.multinomial_treatment,
            self.continuous_treatment,
            self.class_treatment,
            self.zero_based
        )
        return newdomain


if __name__ == "__main__":
    import sys
    a = QtGui.QApplication(sys.argv)
    ow = OWContinuize()
    data = Table("lenses")
    ow.setData(data)
    ow.show()
    a.exec_()
    ow.saveSettings()

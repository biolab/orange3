from datetime import date
from html import escape

from AnyQt.QtCore import Qt

from orangewidget.utils.signals import summarize, PartialSummary

from Orange.data import (
    StringVariable, DiscreteVariable, ContinuousVariable, TimeVariable,
    Table
)

from Orange.evaluation import Results
from Orange.misc import DistMatrix
from Orange.preprocess import Preprocess, PreprocessorList
from Orange.preprocess.score import Scorer
from Orange.widgets.utils.signals import AttributeList
from Orange.base import Model, Learner


def format_variables_string(variables):
    """
    A function that formats the descriptive part of the input/output summary for
    either features, targets or metas of the input dataset.

    :param variables: Features, targets or metas of the input dataset
    :return: A formatted string
    """
    if not variables:
        return '—'

    agg = []
    for var_type_name, var_type in [('categorical', DiscreteVariable),
                                    ('numeric', ContinuousVariable),
                                    ('time', TimeVariable),
                                    ('string', StringVariable)]:
        # Disable pylint here because a `TimeVariable` is also a
        # `ContinuousVariable`, and should be labelled as such. That is why
        # it is necessary to check the type this way instead of using
        # `isinstance`, which would fail in the above case
        var_type_list = [v for v in variables if type(v) is var_type]  # pylint: disable=unidiomatic-typecheck
        if var_type_list:
            agg.append((var_type_name, len(var_type_list)))

    attrs, counts = list(zip(*agg))
    if len(attrs) > 1:
        var_string = [f'{i} {j}' for i, j in zip(counts, attrs)]
        var_string = f'{sum(counts)} ({", ".join(var_string)})'
    elif counts[0] == 1:
        var_string = attrs[0]
    else:
        var_string = f'{counts[0]} {attrs[0]}'
    return var_string


def _plural(number):
    return 's' * (number % 100 != 1)


# `format` is a good name for the argument, pylint: disable=redefined-builtin
def format_summary_details(data, format=Qt.PlainText):
    """
    A function that forms the entire descriptive part of the input/output
    summary.

    :param data: A dataset
    :type data: Orange.data.Table
    :return: A formatted string
    """
    if data is None:
        return ""

    if format == Qt.PlainText:
        def b(s):
            return s
    else:
        def b(s):
            return f"<b>{s}</b>"

    features = format_variables_string(data.domain.attributes)
    targets = format_variables_string(data.domain.class_vars)
    metas = format_variables_string(data.domain.metas)

    features_missing = missing_values(data.has_missing_attribute()
                                      and data.get_nan_frequency_attribute())
    n_features = len(data.domain.variables) + len(data.domain.metas)
    name = getattr(data, "name", None)
    if name == "untitled":
        name = None
    basic = f'{len(data):n} instance{_plural(len(data))}, ' \
            f'{n_features} variable{_plural(n_features)}'

    if format == Qt.PlainText:
        details = \
            (f"{name}: " if name else "") + basic \
            + f'\nFeatures: {features} {features_missing}' \
            + f'\nTarget: {targets}'
        if data.domain.metas:
            details += f'\nMetas: {metas}'
    else:
        descs = []
        if name:
            descs.append(_nobr(f"<b><u>{escape(name)}</u></b>: {basic}"))
        else:
            descs.append(_nobr(f'{basic}'))

        if data.domain.variables:
            descs.append(_nobr(f'Features: {features} {features_missing}'))
        if data.domain.class_vars:
            descs.append(_nobr(f"Target: {targets}"))
        if data.domain.metas:
            descs.append(_nobr(f"Metas: {metas}"))

        details = '<br/>'.join(descs)

    return details


def missing_values(value):
    if value:
        return f'({value*100:.1f}% missing values)'
    else:
        return '(no missing values)'


def format_multiple_summaries(data_list, type_io='input'):
    """
    A function that forms the entire descriptive part of the input/output
    summary for widgets that have more than one input/output.

    :param data_list: A list of tuples for each input/output dataset where the
    first element of the tuple is the name of the dataset (can be omitted)
    and the second is the dataset
    :type data_list: list(tuple(str, Orange.data.Table))
    :param type_io: A string that indicates weather the input or output data
    is being formatted
    :type type_io: str

    :return A formatted summary
    :rtype str
    """

    def new_line(text):
        return text.replace('\n', '<br>')

    full_details = []
    for (name, data) in data_list:
        if data:
            details = new_line(format_summary_details(data))
        else:
            details = f'No data on {type_io}.'
        full_details.append(details if not name else f'{name}:<br>{details}')
    return '<hr>'.join(full_details)


def _name_of(object):
    return _nobr(getattr(object, 'name', type(object).__name__))


def _nobr(s):
    return f"<nobr>{s}</nobr>"


@summarize.register
def summarize_(data: Table):
    return PartialSummary(
        data.approx_len(),
        format_summary_details(data, format=Qt.RichText))


@summarize.register
def summarize_(matrix: DistMatrix):  # pylint: disable=function-redefined
    n, m = matrix.shape
    return PartialSummary(f"{n}×{m}", _nobr(f"{n}×{m} distance matrix"))


@summarize.register
def summarize_(results: Results):  # pylint: disable=function-redefined
    nmethods, ninstances = results.predicted.shape
    summary = f"{nmethods}×{ninstances}"
    details = f"{nmethods} method{_plural(nmethods)} " \
              f"on {ninstances} test instance{_plural(ninstances)}"
    return PartialSummary(summary, _nobr(details))


@summarize.register
def summarize_(attributes: AttributeList):  # pylint: disable=function-redefined
    n = len(attributes)
    if n == 0:
        details = "empty list"
    elif n <= 3:
        details = _nobr(", ".join(var.name for var in attributes))
    else:
        details = _nobr(", ".join(var.name for var in attributes[:2]) +
                       f" and {n - 2} others")
    return PartialSummary(n, details)


@summarize.register
def summarize_(preprocessor: Preprocess):
    if isinstance(preprocessor, PreprocessorList):
        if preprocessor.preprocessors:
            details = "<br/>".join(map(_name_of, preprocessor.preprocessors))
        else:
            details = _nobr(f"{_name_of(preprocessor)} (empty)")
    else:
        details = _name_of(preprocessor)
    return PartialSummary("🄿", details)


def summarize_by_name(type_, symbol):
    @summarize.register
    def summarize_(model: type_):
        return PartialSummary(symbol, _name_of(model))


summarize_by_name(Model, "&#9924;" if date.today().month == 12 else "🄼")
summarize_by_name(Learner, "🄻")
summarize_by_name(Scorer, "🅂")

from datetime import date
from html import escape
from typing import Union

from AnyQt.QtCore import Qt

from Orange.widgets.utils.localization import pl
from orangewidget.utils.signals import summarize, PartialSummary, LazyValue
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.tableview import TableView
from Orange.widgets.utils.distmatrixmodel import \
    DistMatrixModel, DistMatrixView

from Orange.data import (
    StringVariable, DiscreteVariable, ContinuousVariable, TimeVariable,
    Table, Domain
)

from Orange.evaluation import Results
from Orange.misc import DistMatrix
from Orange.preprocess import Preprocess, PreprocessorList
from Orange.preprocess.score import Scorer
from Orange.widgets.utils.signals import AttributeList
from Orange.base import Model, Learner


COMPUTE_NANS_LIMIT = 1e7


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


# `format` is a good name for the argument, pylint: disable=redefined-builtin
def format_summary_details(data: Union[Table, Domain],
                           format=Qt.PlainText, missing=None):
    """
    A function that forms the entire descriptive part of the input/output
    summary.

    :param data: A dataset
    :type data: Orange.data.Table or Orange.data.Domain
    :return: A formatted string
    """
    if data is None:
        return ""

    features_missing = "" if missing is None else missing_values(missing)
    if isinstance(data, Domain):
        domain = data
        name = None
        basic = ""
    else:
        assert isinstance(data, Table)
        domain = data.domain
        if not features_missing and \
                len(data) * len(domain.attributes) < COMPUTE_NANS_LIMIT:
            features_missing \
                = missing_values(data.get_nan_frequency_attribute())
        name = getattr(data, "name", None)
        if name == "untitled":
            name = None
        basic = f'{len(data):n} {pl(len(data), "instance")}, '

    n_features = len(domain.variables) + len(domain.metas)
    basic += f'{n_features} {pl(n_features, "variable")}'

    features = format_variables_string(domain.attributes)
    features = f'Features: {features}{features_missing}'

    targets = format_variables_string(domain.class_vars)
    targets = f'Target: {targets}'

    metas = format_variables_string(domain.metas)
    metas = f'Metas: {metas}'

    if format == Qt.PlainText:
        details = f"{name}: " if name else "Table with "
        details += f"{basic}\n{features}\n{targets}"
        if domain.metas:
            details += f"\n{metas}"
    else:
        descs = []
        if name:
            descs.append(_nobr(f"<b><u>{escape(name)}</u></b>: {basic}"))
        else:
            descs.append(_nobr(f"Table with {basic}"))

        if domain.variables:
            descs.append(_nobr(features))
        if domain.class_vars:
            descs.append(_nobr(targets))
        if domain.metas:
            descs.append(_nobr(metas))

        details = '<br/>'.join(descs)

    return details


def missing_values(value):
    if value:
        return f' ({value*100:.1f}% missing values)'
    elif value is None:
        return ''
    else:
        return ' (no missing values)'


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
def summarize_table(data: Table):  # pylint: disable=function-redefined
    return PartialSummary(
        data.approx_len(),
        format_summary_details(data, format=Qt.RichText),
        lambda: _table_previewer(data))


@summarize.register
def summarize_table(data: LazyValue[Table]):
    if data.is_cached:
        return summarize(data.get_value())

    length = getattr(data, "length", "?")
    details = format_summary_details(data.domain, format=Qt.RichText,
                                     missing=getattr(data, "missing", None)) \
        if hasattr(data, "domain") else "data available, but not prepared yet"
    return PartialSummary(
        length,
        details,
        lambda: _table_previewer(data.get_value()))


def _table_previewer(data):
    view = TableView(selectionMode=TableView.NoSelection)
    view.setModel(TableModel(data))
    return view


@summarize.register
def summarize_matrix(matrix: DistMatrix):  # pylint: disable=function-redefined
    def previewer():
        view = DistMatrixView(selectionMode=TableView.NoSelection)
        model = DistMatrixModel()
        model.set_data(matrix)
        col_labels = matrix.get_labels(matrix.col_items)
        row_labels = matrix.get_labels(matrix.row_items)
        if matrix.is_symmetric() and (
                (col_labels is None) is not (row_labels is None)):
            if col_labels is None:
                col_labels = row_labels
            else:
                row_labels = col_labels
        if col_labels is None:
            col_labels = [str(x) for x in range(w)]
        if row_labels is None:
            row_labels = [str(x) for x in range(h)]
        model.set_labels(Qt.Horizontal, col_labels)
        model.set_labels(Qt.Vertical, row_labels)
        view.setModel(model)

        return view

    h, w = matrix.shape
    return PartialSummary(
        f"{w}×{h}",
        _nobr(f"{w}×{h} distance matrix"),
        previewer
    )


@summarize.register
def summarize_results(results: Results):  # pylint: disable=function-redefined
    nmethods, ninstances = results.predicted.shape
    summary = f"{nmethods}×{ninstances}"
    details = f"{nmethods} {pl(nmethods, 'method')} " \
              f"on {ninstances} test {pl(ninstances, 'instance')}"
    return PartialSummary(summary, _nobr(details))


@summarize.register
def summarize_attributes(attributes: AttributeList):  # pylint: disable=function-redefined
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
def summarize_preprocessor(preprocessor: Preprocess):  # pylint: disable=function-redefined
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

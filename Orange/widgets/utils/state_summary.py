from Orange.data import StringVariable, DiscreteVariable, ContinuousVariable, \
    TimeVariable


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


def format_summary_details(data):
    """
    A function that forms the entire descriptive part of the input/output
    summary.

    :param data: A dataset
    :type data: Orange.data.Table
    :return: A formatted string
    """
    def _plural(number):
        return 's' * (number != 1)

    details = ''
    if data:
        features = format_variables_string(data.domain.attributes)
        targets = format_variables_string(data.domain.class_vars)
        metas = format_variables_string(data.domain.metas)

        features_missing = missing_values(data.has_missing_attribute()
                                          and data.get_nan_frequency_attribute())
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = \
            f'{len(data)} instance{_plural(len(data))}, ' \
            f'{n_features} variable{_plural(n_features)}\n' \
            f'Features: {features} {features_missing}\n' \
            f'Target: {targets}\nMetas: {metas}'
    return details


def missing_values(value):
    if value:
        return f'({value*100:.1f}% missing values)'
    else:
        return '(No missing values)'


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

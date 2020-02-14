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
            not_shown = ' (not shown)' if issubclass(var_type, StringVariable)\
                else ''
            agg.append((f'{var_type_name}{not_shown}', len(var_type_list)))

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

        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = \
            f'{len(data)} instance{_plural(len(data))}, ' \
            f'{n_features} variable{_plural(n_features)}\n' \
            f'Features: {features}\nTarget: {targets}\nMetas: {metas}'

    return details

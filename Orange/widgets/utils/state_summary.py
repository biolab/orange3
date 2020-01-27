from Orange.data import StringVariable, DiscreteVariable, ContinuousVariable, \
    TimeVariable


def format_variables_string(variables):
    """
    A function that formats the descriptive part of the input/output summary for
    either features, targets or metas of the input dataset.

    :param variables: Features, targets or metas of the input dataset
    :return: A formatted string
    """
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
            shown = var_type in (StringVariable,)
            agg.append(
                (f'{len(var_type_list)} ' + var_type_name +
                 f"{['', ' (not shown)'][shown]}",
                 len(var_type_list)))

    if not agg:
        return '—'

    attrs, counts = list(zip(*agg))
    if len(attrs) > 1:
        var_string = ', '.join(attrs[:-1]) + ', ' + attrs[-1]
        return f'{sum(counts)} (' + var_string + ')'
    elif sum(counts) == 1:
        var_string = attrs[0][2:]
        return var_string
    else:
        types = [s for s in ['categorical', 'numeric', 'time', 'string'] if
                 s in attrs[0]]
        ind = attrs[0].find(types[0])
        var_string = attrs[0][ind:]
        return f'{sum(counts)} ' + var_string


def format_summary_details(data):
    """
    A function that forms the entire descriptive part of the input/output
    summary.

    :param data: A dataset
    :type data: Orange.data.Table
    :return: A formatted string
    """
    def _plural(number):
        return number, 's' * (number != 1)

    details = ''
    if data:
        features = format_variables_string(data.domain.attributes)
        targets = format_variables_string(data.domain.class_vars)
        metas = format_variables_string(data.domain.metas)

        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = \
            f'{_plural(len(data))[0]} instance{_plural(len(data))[1]}, ' \
            f'{_plural(n_features)[0]} feature{_plural(n_features)[1]}\n' \
            f'Features: ' + features + '\n' + \
            f'Target: ' + targets + '\n' + \
            f'Metas: ' + metas

    return details

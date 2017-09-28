import re

import numpy as np
from Orange.data import Domain, DiscreteVariable

ANNOTATED_DATA_SIGNAL_NAME = "Data"
ANNOTATED_DATA_FEATURE_NAME = "Selected"
RE_FIND_INDEX = "(^{} \()(\d{{1,}})(\)$)"


def add_columns(domain, attributes=(), class_vars=(), metas=()):
    """Construct a new domain with new columns added to the specified place

    Parameters
    ----------
    domain : Domain
        source domain
    attributes
        list of variables to append to attributes from source domain
    class_vars
        list of variables to append to class_vars from source domain
    metas
        list of variables to append to metas from source domain

    Returns
    -------
    Domain
    """
    attributes = domain.attributes + tuple(attributes)
    class_vars = domain.class_vars + tuple(class_vars)
    metas = domain.metas + tuple(metas)
    return Domain(attributes, class_vars, metas)


def get_indices(names, name):
    """
    Return list of indices which occur in a names list for a given name.
    :param names: list of strings
    :param name: str
    :return: list of indices
    """
    return [int(a.group(2)) for x in names
            for a in re.finditer(RE_FIND_INDEX.format(name), x)]


def get_next_name(names, name):
    """
    Returns next 'possible' attribute name. The name should not be duplicated
    and is generated using name parameter, appended by smallest possible index.

    :param names: list
    :param name: str
    :return: str
    """
    indexes = get_indices(names, name)
    if name not in names and not indexes:
        return name
    return "{} ({})".format(name, max(indexes, default=1) + 1)


def get_unique_names(names, proposed):
    """
    Returns unique names of variables. Variables which are duplicate get appended by
    unique index which is the same in all proposed variable names in a list.
    :param names: list of strings
    :param proposed: list of strings
    :return: list of strings
    """
    if len([name for name in proposed if name in names]):
        max_index = max([max(get_indices(names, name),
                             default=1) for name in proposed], default=1)
        for i, name in enumerate(proposed):
            proposed[i] = "{} ({})".format(name, max_index + 1)
    return proposed


def create_annotated_table(data, selected_indices):
    """
    Returns data with concatenated flag column. Flag column represents
    whether data instance has been selected (Yes) or not (No), which is
    determined in selected_indices parameter.

    :param data: Table
    :param selected_indices: list or ndarray
    :return: Table
    """
    if data is None:
        return None
    names = [var.name for var in data.domain.variables + data.domain.metas]
    name = get_next_name(names, ANNOTATED_DATA_FEATURE_NAME)
    domain = add_columns(data.domain, metas=[DiscreteVariable(name, ("No", "Yes"))])
    annotated = np.zeros((len(data), 1))
    if selected_indices is not None:
        annotated[selected_indices] = 1
    table = data.transform(domain)
    table[:, name] = annotated
    return table


def create_groups_table(data, selection):
    if data is None:
        return None
    names = [var.name for var in data.domain.variables + data.domain.metas]
    name = get_next_name(names, ANNOTATED_DATA_FEATURE_NAME)
    metas = data.domain.metas + (
        DiscreteVariable(
            name,
            ["Unselected"] + ["G{}".format(i + 1)
                              for i in range(np.max(selection))]),
    )
    domain = Domain(data.domain.attributes, data.domain.class_vars, metas)
    table = data.transform(domain)
    table.metas[:, len(data.domain.metas):] = \
        selection.reshape(len(data), 1)
    return table

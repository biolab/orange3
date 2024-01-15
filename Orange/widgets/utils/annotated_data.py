from typing import Union

import numpy as np

from orangewidget.utils.signals import LazyValue

from Orange.data import Domain, DiscreteVariable, Table
from Orange.data.util import get_unique_names

ANNOTATED_DATA_SIGNAL_NAME = "Data"
ANNOTATED_DATA_FEATURE_NAME = "Selected"


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


def domain_with_annotation_column(
        data: Union[Table, Domain],
        values=("No", "Yes"),
        var_name=ANNOTATED_DATA_FEATURE_NAME):
    domain = data if isinstance(data, Domain) else data.domain
    var = DiscreteVariable(get_unique_names(domain, var_name), values)
    class_vars, metas = domain.class_vars, domain.metas
    if not domain.class_vars:
        class_vars += (var, )
    else:
        metas += (var, )
    return Domain(domain.attributes, class_vars, metas), var


def _table_with_annotation_column(data, values, column_data, var_name):
    domain, var = domain_with_annotation_column(data, values, var_name)
    if not data.domain.class_vars:
        column_data = column_data.reshape((len(data), ))
    else:
        column_data = column_data.reshape((len(data), 1))
    table = data.transform(domain)
    with table.unlocked(table.Y if not data.domain.class_vars else table.metas):
        table[:, var] = column_data
    return table


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
    annotated = np.zeros((len(data), 1))
    if selected_indices is not None:
        annotated[selected_indices] = 1
    return _table_with_annotation_column(
        data, ("No", "Yes"), annotated, ANNOTATED_DATA_FEATURE_NAME)


def lazy_annotated_table(data, selected_indices):
    domain, _ = domain_with_annotation_column(data)
    return LazyValue[Table](
        lambda: create_annotated_table(data, selected_indices),
        length=len(data), domain=domain)


def create_groups_table(data, selection,
                        include_unselected=True,
                        var_name=ANNOTATED_DATA_FEATURE_NAME,
                        values=None):
    if data is None:
        return None
    values, max_sel = group_values(selection, include_unselected, values)
    if include_unselected:
        # Place Unselected instances in the "last group", so that the group
        # colors and scatter diagram marker colors will match
        mask = (selection != 0)
        selection = selection.copy()
        selection[mask] = selection[mask] - 1
        selection[~mask] = selection[~mask] = max_sel
    else:
        mask = np.flatnonzero(selection)
        data = data[mask]
        selection = selection[mask] - 1
    return _table_with_annotation_column(data, values, selection, var_name)


def lazy_groups_table(data, selection, include_unselected=True,
                      var_name=ANNOTATED_DATA_FEATURE_NAME, values=None):
    length = len(data) if include_unselected else np.sum(selection != 0)
    values, _ = group_values(selection, include_unselected, values)
    domain, _ = domain_with_annotation_column(data, values, var_name)
    return LazyValue[Table](
        lambda: create_groups_table(data, selection, include_unselected,
                                    var_name, values),
        length=length, domain=domain
    )


def group_values(selection, include_unselected, values):
    max_sel = np.max(selection)
    if values is None:
        values = ["G{}".format(i + 1) for i in range(max_sel)]
        if include_unselected:
            values.append("Unselected")
    return values, max_sel

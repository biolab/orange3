import re
import numpy as np
from Orange.data import Table, Domain, DiscreteVariable

ANNOTATED_DATA_SIGNAL_NAME = "Data"
ANNOTATED_DATA_FEATURE_NAME = "Selected"


def _get_next_name(names, name):
    """
    Returns next 'possible' attribute name. The name should not be duplicated
    and is generated using name parameter, appended by smallest possible index.

    :param names: list
    :param name: str
    :return: str
    """
    indexes = [int(a.group(2)) for x in names
               for a in re.finditer("(^{} \()(\d{{1,}})(\)$)".format(name), x)]
    if name not in names and not indexes:
        return name
    return "{} ({})".format(name, max(indexes, default=1) + 1)


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
    name = _get_next_name(names, ANNOTATED_DATA_FEATURE_NAME)
    metas = data.domain.metas + (DiscreteVariable(name, ("No", "Yes")),)
    domain = Domain(data.domain.attributes, data.domain.class_vars, metas)
    annotated = np.zeros((len(data), 1))
    if selected_indices is not None:
        annotated[selected_indices] = 1
    return Table(domain, data.X, data.Y,
                 metas=np.hstack((data.metas, annotated)))

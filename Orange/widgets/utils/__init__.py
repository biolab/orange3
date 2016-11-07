from Orange.data.variable import TimeVariable
from Orange.util import deepgetattr


def vartype(var):
    if var.is_discrete:
        return 1
    elif var.is_continuous:
        if isinstance(var, TimeVariable):
            return 4
        return 2
    elif var.is_string:
        return 3
    else:
        return 0


def progress_bar_milestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])


def getdeepattr(obj, attr, *arg, **kwarg):
    if isinstance(obj, dict):
        return obj.get(attr)
    return deepgetattr(obj, attr, *arg, **kwarg)


def to_html(str):
    return str.replace("<=", "&#8804;").replace(">=", "&#8805;").\
        replace("<", "&#60;").replace(">", "&#62;").replace("=\\=", "&#8800;")

getHtmlCompatibleString = to_html


def checksum(x):
    if x is None:
        return None
    try:
        return x.checksum()
    except:
        return float('nan')


def get_variable_values_sorted(variable):
    """
    Return a list of sorted values for given attribute, if all its values can be
    cast to int's.
    """
    if variable.is_continuous:
        return []
    try:
        return sorted(variable.values, key=int)
    except ValueError:
        return variable.values

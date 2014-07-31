from functools import reduce
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable


def vartype(var):
    if isinstance(var, DiscreteVariable):
        return 1
    elif isinstance(var, ContinuousVariable):
        return 2
    elif isinstance(var, StringVariable):
        return 3
    else:
        return 0


def progress_bar_milestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])


def getdeepattr(obj, attr, *arg, **kwarg):
    if isinstance(obj, dict):
        return obj.get(attr)
    try:
        return reduce(lambda o, n: getattr(o, n), attr.split("."), obj)
    except:
        if arg:
            return arg[0]
        if kwarg:
            return kwarg["default"]
        raise AttributeError("'%s' has no attribute '%s'" % (obj, attr))

def getHtmlCompatibleString(strVal):
    return strVal.replace("<=", "&#8804;").replace(">=","&#8805;").replace("<", "&#60;").replace(">","&#62;").replace("=\\=", "&#8800;")

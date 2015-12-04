from functools import reduce


def vartype(var):
    if var.is_discrete:
        return 1
    elif var.is_continuous:
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
    try:
        return reduce(getattr, attr.split("."), obj)
    except AttributeError:
        if arg:
            return arg[0]
        if kwarg:
            return kwarg["default"]
        raise

def getHtmlCompatibleString(strVal):
    return strVal.replace("<=", "&#8804;").replace(">=","&#8805;").replace("<", "&#60;").replace(">","&#62;").replace("=\\=", "&#8800;")

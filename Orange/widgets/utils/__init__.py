from functools import reduce


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

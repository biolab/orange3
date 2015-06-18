"""Various small utilities that might be useful everywhere"""


def abstract(obj):
    """Designate decorated class or method abstract."""
    if isinstance(obj, type):
        old__new__ = obj.__new__

        def _refuse__new__(cls, *args, **kwargs):
            if cls == obj:
                raise NotImplementedError("Can't instantiate abstract class " + obj.__name__)
            return old__new__(cls, *args, **kwargs)

        obj.__new__ = _refuse__new__
        return obj
    else:
        cls_name = obj.__qualname__.rsplit('.', 1)[0]
        def _refuse__call__(*args, **kwargs):
            raise NotImplementedError("Can't call abstract method {} of class {}"
                                      .format(obj.__name__, cls_name))
        return _refuse__call__

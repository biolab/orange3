import inspect


class Reprable:
    def __repr__(self):
        """ A helper class that, when inherited, creates an executable __repr__
        String based on the arguments passed to the object's __init__ function.
        """
        try:
            argspec = inspect.getargspec(self.__class__.__init__)
            args = list(argspec.args)
            defaults = []
            if defaults is not None:
                if len(defaults) < len(args):
                    padding = [None] * (len(args) - len(defaults))
                    defaults = padding + defaults
            else:
                defaults = [None] * len(args)

        except AttributeError:
            # __init__ doesn't have __code__, so the class doesn't override
            # the constructor
            return type(self).__name__ + '()'
        else:
            return "{}({})".format(
                self.__class__.__name__,
                # add arguments strings in for each argument to the object's __init__
                ", ".join("{}={}".format(arg, repr(getattr(self, arg)))
                    for i, arg in enumerate(args)
                    if arg != "self" and
                    defaults[i] != repr(getattr(self, arg))
                )
            )

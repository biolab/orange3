class Reprable:
    def __repr__(self):
        """ A helper class that, when inherited, creates an executable __repr__
        String based on the arguments passed to the preprocessor's __init__ function.
        """
        try:
            args = self.__class__.__init__.__code__.co_varnames[1:]
        except AttributeError:
            # __init__ doesn't have __code__, so the class doesn't override
            # the constructor
            return type(self).__name__ + '()'
        else:
            return "{}({})".format(
                self.__class__.__name__,
                ", ".join("{}={}".format(arg, repr(getattr(self, arg))) for i, arg in enumerate(args) if
                    self.__class__.__init__.__defaults__[i-1] != getattr(self, arg))
            )


class CallReprable:
    def __init__(self):
        return None

    def __repr__(self):
        """ A helper class that, when inherited, creates an executable __repr__
        String based on the arguments passed to the preprocessor's __init__ function.
        """
        try:
            args = self.__class__.__call__.__code__.co_varnames[1:]
        except AttributeError:
            # __call__ doesn't have __code__, so the class doesn't override
            # the constructor
            return type(self).__name__ + '()'
        else:
            return "{}({})".format(
                self.__class__.__name__,
                ", ".join("{}={}".format(arg, repr(getattr(self, arg))) for i, arg in enumerate(args) if
                    self.__class__.__call__.__defaults__[i-1] != getattr(self, arg))
            )

import warnings

from Orange.data import Table
from Orange.util import OrangeDeprecationWarning


class _RefuseDataInConstructor:
    # TODO: drop this as soon as possible
    def __new__(cls, *args, **kwargs):
        if (args and isinstance(args[0], Table) or
                isinstance(kwargs.get('data'), Table)):
            data = kwargs.pop('data', None)
            if data is None and args:
                data, args = args[0], args[1:]

            obj = super().__new__(cls)
            obj.__init__(*args, **kwargs)

            warnings.warn(
                'Passing data into {0} constructor is deprecated. Instead, '
                'first make an instance, i.e. {0}(), then call it with data.'
                    .format(cls.__name__), OrangeDeprecationWarning,
                stacklevel=2)

            return obj(data)

        return super().__new__(cls)

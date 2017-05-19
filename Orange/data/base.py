from collections.abc import Sequence, Iterable

import numpy as np
from scipy.sparse import issparse

from Orange.data import Domain, Variable
from Orange.util import deprecated, Registry, abstract


class Expression:
    """
    Roughly replaces Orange.data.filter.Filter.
    """
    def __init__(self, e1, op=None, e2=None):
        self.e1 = e1
        self.op = op
        self.e2 = e2

    def __str__(self):
        return '({} {} {})'.format(self.e1, self.op, self.e2)

    def __and__(self, other):
        assert isinstance(other, Expression)
        return Expression(self, 'AND', other)

    def __or__(self, other):
        assert isinstance(other, Expression)
        return Expression(self, 'OR', other)

    def __eq__(self, other):
        assert isinstance(other, (Expression, int, float))
        return Expression(self, 'EQ', other)

    def __ne__(self, other):
        assert isinstance(other, (Expression, int, float))
        return Expression(self, 'NE', other)

    def __le__(self, other):
        assert isinstance(other, (Expression, int, float))
        return Expression(self, 'LE', other)

    def __lt__(self, other):
        assert isinstance(other, (Expression, int, float))
        return Expression(self, 'LT', other)

    def __gt__(self, other):
        assert isinstance(other, (Expression, int, float))
        return Expression(self, 'GT', other)

    def __ge__(self, other):
        assert isinstance(other, (Expression, int, float))
        return Expression(self, 'GE', other)

    def between(self, interval):
        assert len(interval) == 2
        return Expression(self, 'BETWEEN', interval)

    def outside(self, interval):
        assert len(interval) == 2
        return Expression(self, 'OUTSIDE', interval)

    def startswith(self, affix):
        assert isinstance(affix, str)
        return Expression(self, 'STARTSWITH', affix)

    def endswith(self, affix):
        assert isinstance(affix, str)
        return Expression(self, 'ENDSWITH', affix)

    def is_in(self, collection):
        assert isinstance(collection, Iterable)
        return Expression(self, 'IN', collection)

    def match(self, regex):
        assert isinstance(regex, str)
        return Expression(self, 'MATCH', regex)


class _DriverMeta(Registry):
    pass


class _Driver(metaclass=_DriverMeta):
    """
    This is the abstract driver interface. Specific drivers for different
    data storages must override its methods.
    Examples: NumpyDriver, SqlDriver, SparseDriver
    """
    def __new__(cls, *args, **kwargs):
        """Accepts all args and kwargs as Table does."""
        # See if any of the subtypes can handle the passed arguments better,
        # e.g. if SqlDriver can open 'postgresql://user:pass@host/db'
        if cls is _Driver:
            for subcls in cls.registry.values():
                if not hasattr(subcls, '__new__'):
                    continue
                try:
                    obj = subcls(*args, **kwargs)
                    if obj is not None:
                        return obj
                except ValueError:
                    pass
            raise ValueError('No driver apt to codec for passed arguments')

        # Default implementation sets the ids
        self = super().__new__(cls)
        ids = kwargs.pop('ids', ())
        if ids:
            self._ids = ids
        return self

    @property
    def domain(self):
        return self._domain
    @property
    def ids(self):
        return self._ids
    @property
    def X(self):
        """
        When X is too large to hold in memory (e.g. in SqlDriver), a
        reasonable approximation is used. The algorithms should prefer,
        but not restrict themselves to operate through the
        `Table.__getitem__()` (i.e. `Driver.select()`) interface.
        """
        return self._X
    @property
    def Y(self):
        return self._Y
    @property
    def y(self):
        return self._Y[:, 0]
    @property
    def M(self):
        return self._M
    @property
    def weights(self):
        return self._weights

    def __len__(self):
        return len(self.X)

    class ExprProcessor:
        """
        This class should be extended in driver subclasses with methods
        matching possible Expression.op values.
        """
        @classmethod
        def process(cls, driver, expr):
            """
            This method should not necessarily be overridden. It just reduces
            a possibly complex Expression query.
            """
            e1, e2 = expr.e1, expr.e2
            e1 = (cls.process(e1) if isinstance(e1, Expression) else
                  cls.column_data(driver, e1) if isinstance(e1, Variable) else
                  e1)
            e2 = (cls.process(e2) if isinstance(e2, Expression) else
                  cls.column_data(driver, e2) if isinstance(e2, Variable) else
                  e2)
            return getattr(cls, expr.op)(e1, e2)

        @staticmethod
        def column_data(driver, var):
            """
            This method may need to be overridden if data is actually stored
            in other than X, Y, M (e.g. in SqlDriver).
            """
            d = driver.select(..., [var])
            return (d.X if len(d.X) else d.Y if len(d.Y) else d.M)[:, 0]

    @abstract
    def select(self, rows, cols):
        """
        Return a driver with subset view on current data, matching `rows` and
        `cols`. When overriding, make sure to account for all possible inputs.

        Parameters
        ----------
        rows: list or slice or Ellipsis or Expression
            `rows` can be a list of row indices, a `slice` object, an
            `Ellipsis` object if all the rows are to be selected, or an
            `Expression` object for more complex, data-dependant row selection.
        cols: list or slice or Ellipsis
            `cols` can be an a list of domain indices or str variable names,
            a `slice` object or an `Ellipsis` object for all the columns.

        Returns
        -------
        driver: Driver
            A driver that matches the selection.
        """
        raise NotImplementedError

    @abstract
    def iter(self, is_transposed):
        """
        Return an iterator through rows if not `is_transposed` else columns.
        """
        raise NotImplementedError

    """ The self-explanatory interface below is proposed. """

    @abstract
    def copy(self):
        raise NotImplementedError

    @abstract
    def concatenate(self, rest, axis):
        raise NotImplementedError

    @abstract
    def contingency(self, attr):
        raise NotImplementedError

    @abstract
    def distribution(self, attr):
        raise NotImplementedError

    @abstract
    def min(self, X, col_idx):
        raise NotImplementedError

    @abstract
    def max(self, X, col_idx):
        raise NotImplementedError


class NumpyDriver(_Driver):
    def __new__(cls, domain=None, X=(), Y=(), M=(), weights=(), ids=()):
        self = super().__new__(cls)

        n_inst = {0, len(X), len(Y), len(M), len(weights), len(ids)}
        n_inst.remove(0)
        if len(n_inst) > 1:
            raise ValueError('X or Y or M or weights not all of same size')
        n_inst = n_inst.pop() if n_inst else 0

        X = np.asarray(X if X is not None else ())
        Y = np.asarray(Y if Y is not None else ())
        M = np.asarray(M if M is not None else ())
        weights = np.asarray(weights if weights is not None else ())
        ids = np.asarray(ids if len(ids) else tuple(range(n_inst)))  # TODO

        if not X.size:
            X = X.reshape((n_inst, 0))
        if not Y.size:
            Y = Y.reshape((n_inst, 0))
        if not M.size:
            M = M.reshape((n_inst, 0))
        if not weights.size:
            weights = weights.reshape((n_inst, 0))

        self._domain = domain = domain or Domain.from_numpy(X, Y, M)
        if (X.shape[1] != len(domain.attributes) or
            Y.shape[1] != len(domain.class_vars) or
            M.shape[1] != len(domain.metas)):
            raise ValueError("X or Y or M don't match domain")
        self._X = X
        self._Y = Y
        self._M = M
        self._weights = weights
        self._ids = ids
        return self

    def iter(self, is_transposed):
        return iter(self.X.T if is_transposed else self.X)

    def contingency(self, attr):
        ...

    def distribution(self, attr):
        ...

    def min(self, X, attr):
        return X[:, attr].min(0)

    def max(self, X, attr):
        return X[:, attr].max(0)

    class _ExprProcessor(_Driver.ExprProcessor):
        GT = np.ndarray.__gt__
        AND = np.ndarray.__and__
        ...

    def select(self, rows, cols):
        if cols is ...:
            X_cols, Y_cols, M_cols = ..., ..., ...
        else:
            X_cols, Y_cols, M_cols = self.domain.index(cols)
        if isinstance(rows, Expression):
            if rows.e1.op is None:  # TODO DELETE once Variable does Expressions
                rows.e1 = rows.e1.e1

            rows = self._ExprProcessor.process(self, rows)
        return self.__class__(self.domain[cols],
                              self.X[rows, ...][..., X_cols] if self.X.size else (),
                              self.Y[rows, ...][..., Y_cols] if self.Y.size else (),
                              self.M[rows, ...][..., M_cols] if self.M.size else (),
                              weights=self.weights[rows] if self.weights.size else (),
                              ids=self.ids[rows])

class SQLDriver(_Driver):
    SUPPORTED_SCHEMES = (
        'postgresql',
        'mysql',
        'oracle',
        'mssql',
        'sqlite',
    )
    import re
    IS_SQL_URL = re.compile('^({})'.format('|'.join(SUPPORTED_SCHEMES)) +
                            '(\+\w+)?'  # Optional driver part
                            '://').match
    def __new__(cls, *args, **kwargs):
        if not (len(args) == 1 and
                isinstance(args[0], str) and
                cls.IS_SQL_URL(args[0])):
            return None
        ...


class SparseDriver(_Driver):
    def __new__(cls, *args, **kwargs):
        if not any(issparse(i) for i in args):
            return None
        ...


class TableMeta(type(Sequence), Registry):
    pass


class Table(Sequence, metaclass=TableMeta):
    """
    This is the base class that holds and transmits tabular data.
    Subclasses (like Corpus, Relation, Timeseries) should NOT need to
    override ANY of Table's methods (it's the back-end drivers that do the
    heavy lifting anyway) but may wish to introduce their own `__init__()`.
    """

    class Attrs:
        """
        Holds Table's dynamic attributes, like table.name or table.url.
        Only so as to avoid the parent namespace and have some control over
        the user-set attributes.
        """
        def __init__(self, driver):
            self.__driver = driver
        def __setattr__(self, key, value):
            super().__setattr__(key, value)
        def __getattr__(self, item):
            return getattr(self.__driver, item, None)

    # Forward-declare for __setattr__
    _driver = None
    _domain = None
    attr = None

    @property
    def driver(self):
        return self._driver

    def __new__(cls, *args, **kwargs):
        """
        Examples
        --------
        Table('iris')
        Table('/path/to/data.csv')
        Table('http://example.org/data.tab')
        Table('postgresql://u:p@example.org:14463/db', schema='foo')
        Table(domain, X, Y, M, weights, ids)
        Table(table)  # no copy, only cast
        """
        if not args:
            raise ValueError('{} called without positional arguments'.format(
                cls.__name__))

        arg = args[0]
        if len(args) == 1 and not kwargs:
            # Cast args[0] into cls
            if isinstance(arg, (Table, _Driver)):
                self = super().__new__(cls)
                self._driver = getattr(arg, 'driver', arg)
                self._domain = arg.domain
                return self
            if isinstance(arg, str):
                if arg.startswith(('http://', 'https://', 'ftp://', 'file://')):
                    return cls._from_url(arg)
                return cls._from_file(arg)

        Driver = kwargs.pop('driver', _Driver)
        self = super().__new__(cls)
        self._driver = driver = Driver(*args, **kwargs)
        self._domain = kwargs.pop('domain', arg if isinstance(arg, Domain) else driver.domain)
        self.attr = Table.Attrs(driver)
        return self

    @property
    def ids(self):
        return self._driver.ids

    @property
    def d(self):
        return self.domain
    @property
    def domain(self):
        return self._domain

    @property
    def X(self):
        return self._driver.X

    @property
    def Y(self):
        return self._driver.Y

    @property
    def y(self):
        return self._driver.y

    @property
    def M(self):
        return self._driver.M
    @property
    @deprecated
    def metas(self):
        return self.M

    @property
    def weights(self):
        return self._driver.weights
    @property
    @deprecated
    def W(self):
        return self.weights

    def __getitem__(self, key):
        rows, cols = None, ...
        if isinstance(key, tuple) and len(key) == 2:
            # 2D slicing
            rows, cols = None, None  # Invalid, raise Exception below
            row_key, col_key = key
            if isinstance(row_key, int):
                rows = slice(row_key, row_key + 1)
            elif isinstance(row_key, slice):
                rows = ... if row_key.indices(len(self)) == (0, len(self), 1) else row_key
            elif row_key is ...:
                rows = ...
            elif isinstance(row_key, Expression):
                rows = row_key
            elif isinstance(row_key, np.ndarray) and row_key.dtype == bool:
                rows = row_key.nonzero()[0]
            elif (isinstance(row_key, Sequence) and
                      isinstance(row_key and row_key[0], int)):
                rows = row_key

            if isinstance(col_key, (str, Variable)):
                cols = [col_key]
            elif isinstance(col_key, slice):
                cols = ... if col_key.indices(len(self.domain)) == (0, len(self.domain), 1) else col_key
            elif col_key is ...:
                cols = ...
            elif isinstance(col_key, np.ndarray) and col_key.dtype == bool:
                cols = col_key.nonzero()[0]
            elif (isinstance(col_key, Sequence) and
                      isinstance(col_key and col_key[0], (str, Variable))):
                cols = col_key

        # 1D indexing
        elif isinstance(key, int):
            # A single row
            rows = slice(key, key + 1)
        elif isinstance(key, (str, Variable)):
            # A single col
            cols = slice(key, key + 1)
        elif isinstance(key, slice):
            # Rows slice
            rows = ... if key.indices(len(self)) == (0, len(self), 1) else key
        elif key is ...:
            rows = ...
        elif isinstance(key, Expression):
            rows = key
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            # Rows selection mask
            rows = key.nonzero()[0]
        elif isinstance(key, np.ndarray) and key.dtype == str:
            # Cols Variable names
            cols = key
        elif isinstance(key, Sequence):  # Testing for sequence should be last
            item0 = key and key[0]
            if isinstance(item0, (str, Variable)):
                # Cols Variables or Variable names
                cols = key
            elif isinstance(item0, int):
                # Rows indices
                rows = key

        if not (rows or cols) or rows is None or cols is None:
            raise TypeError("Can't index table with {}".format(key))

        return self.__class__(self._driver.select(rows, cols))

    def __setitem__(self, item):
        raise TypeError("Don't set data on table. Instead, construct the table anew!")

    def __setattr__(self, attr, value):
        if not (hasattr(self, attr) or hasattr(self.__class__, attr)):
            raise TypeError("Don't set attributes on table. Instead, set table.attr.<attribute>!")
        super().__setattr__(attr, value)

    def __len__(self):
        return len(self._driver)

    def __iter__(self):
        return self._driver.iter(self.__is_transposed)

    def __contains__(self, attr):
        return attr in self.domain

    @property
    def shape(self):
        tup = (len(self), len(self.domain))
        return tup if not self.__is_transposed else tup[::-1]

    __is_transposed = False

    @property
    def T(self):
        self.__is_transposed = not self.__is_transposed
        return self

    def contingency(self, attr1, attr2):
        return self._driver.contingency(attr1, attr2)

    def distribution(self, attr):
        return self._driver.distribution(attr)

    @staticmethod
    def concatenate(tables, axis=0):
        table, *rest = tables
        return table.driver.concatenate(rest, axis)

    def min(self, attr):
        assert attr in self.domain
        return self._driver.min(attr)

    def max(self, attr):
        assert attr in self.domain
        return self._driver.max(attr)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return self._driver.copy()

    @classmethod
    def _from_file(cls, filename):
        ...

    @classmethod
    def _from_url(cls, url):
        ...

    """
    Compatibility with the legacy interface
    """

    @classmethod
    def from_file(cls, filename):
        return cls(filename)

    @classmethod
    def from_url(cls, url):
        return cls(url)

    @classmethod
    @deprecated
    def from_numpy(cls, domain, X, Y=(), metas=(), W=()):
        return cls(domain, X, Y, metas, weights=W)

    @classmethod
    @deprecated
    def from_list(cls, domain, rows, weights=()):
        # FIXME
        return cls(domain, rows, weights=weights)

    @classmethod
    @deprecated
    def from_table(cls, domain, source, row_indices=...):
        return cls(source[row_indices, domain])

    @classmethod
    @deprecated
    def from_table_rows(cls, source, row_indices=...):
        return cls(source[row_indices])

    @classmethod
    @deprecated
    def from_domain(cls, domain, nrows=0, weights=()):
        return cls(domain, weights=weights)


class Corpus(Table):
    pass


if __name__ == '__main__':

    data = Table(None, np.random.random((10, 5)))
    data = Corpus(data)

    data2 = data[1:-1, ['Feature 2', 'Feature 3']]
    assert data2.shape == (8, 2)

    data3 = data[Expression(data.domain['Feature 2']) > .5, ['Feature 1', 'Feature 3']]
    assert len(data3) < len(data) and data3.shape[1] == 2
    assert isinstance(data3, Corpus)

    # The idea with Expression is to be able to:
    #
    #    data[data.domain.some_var > .5]
    #
    # Needs only a slight adaptation in Variable.

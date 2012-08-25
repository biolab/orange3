import os
import random
import zlib
from collections import MutableSequence, Iterable
from numbers import Real

import numpy as np
import bottleneck as bn

from ..data import Value, Instance, RowInstance, Domain, Unknown
from ..data.io import TabDelimReader

EllipsisType = type(...)

class Table(MutableSequence):
    def __new__(cls, *args, **argkw):
        if not args:
            if not args and not argkw:
                self = super().__new__(cls)
            elif "filename" in argkw:
                self = cls.read_data(argkw["filename"])
        elif len(args) == 1:
            if isinstance(args[0], str):
                self = Table.read_data(args[0])
        else:
            raise ValueError("Invalid arguments for Table.__new__")
        self.clear_cache()
        return self

    @staticmethod
    def new_from_domain(domain, n_rows, weights=True):
        self = Table.__new__(Table)
        self.domain = domain
        self.n_rows = n_rows
        self._X = np.zeros((n_rows, len(domain.attributes)))
        self._Y = np.zeros((n_rows, len(domain.class_vars)))
        if weights:
            self._W = np.ones(n_rows)
        else:
            self._W = None
        self._metas = np.empty((n_rows, len(self.domain.metas)), object)
        return self


    @staticmethod
    def new_as_reference(domain, table, row_indices=..., col_indices=...):
        self = Table.__new__(Table)
        self.domain = domain
        self._X = table._X[row_indices, col_indices]
        self._Y = table._Y[row_indices]
        self._metas = table._metas[row_indices]
        if table._W:
            self._W = np.array(table._W[row_indices])
        return self


    def is_view(self):
        return self._X.base is not None and \
               self._Y.base is not None and \
               self._metas.base is not None and \
               (self._W is None or self._W.base is not None)

    def is_copy(self):
        return self._X.base is None and \
               self._Y.base is None and \
               self._metas.base is None and \
               (self._W is None or self._W.base is None)

    def ensure_copy(self):
        if self._X.base:
            self._X = self._X.copy()
        if self._Y.base:
            self._Y = self._Y.copy()
        if self._metas.base:
            self._metas = self._metas.copy()
        if self._W and self._W.base:
            self._W = self._W.copy()

    def getX(self):
        return self._X

    def getY(self):
        return self._Y

    def get_metas(self):
        return self._metas

    def get_weights(self):
        return self._W

    X = property(getX)
    Y = property(getY)
    W = property(get_weights)
    metas = property(get_metas)

    def clear_cache(self, what="XYWm"):
        pass

    def set_weights(self, val=1):
        if self._W:
            self._W[:] = val
        else:
            self._W = np.empty(len(self))
            self._W.fill(val)

    @staticmethod
    def read_data(filename):
        ext = os.path.splitext(filename)[1]
        if not ext:
            for ext in [".tab"]:
                if os.path.exists(filename + ext):
                    filename += ext
                    break
        if not os.path.exists(filename):
            raise IOError('File "{}" is not found'.format(filename))
        if ext == ".tab":
            return TabDelimReader().read_file(filename)
        else:
            raise IOError('Extension "{}" is not recognized'.format(filename))


    def _compute_col_indices(self, col_idx):
        """Return a list of new attributes and column indices,
           or (None, self.col_indices) if no new domain needs to be constructed"""
        if isinstance(col_idx, np.ndarray) and col_idx.dtype == bool:
            return [attr for attr, c in zip(self.domain, col_idx) if c], \
                   np.nonzero(col_idx)
        elif isinstance(col_idx, slice):
            s = len(self.domain.variables)
            start, end, stride = col_idx.indices(s)
            if col_idx.indices(s) == (0, s, 1):
                return None, self.col_indices
            else:
                return self.domain.attributes[col_idx], \
                       self.arange(start, end, stride)
        elif isinstance(col_idx, Iterable):
            attributes = [self.domain[col] for col in col_idx]
            if attributes == self.domain.attributes:
                return None, self.col_indices
            return attributes, np.fromiter(
                (self.domain.index(attr) for attr in attributes), int)
        else:
            if isinstance(col_idx, int):
                attr = self.domain[col_idx]
            else:
                attr = self.domain[col_idx]
                col_idx =  self.domain.index(attr)
            return [attr], np.array([col_idx])


    def __getitem__(self, key):
        if isinstance(key, int):
            return RowInstance(self, key)
        if not isinstance(key, tuple):
            return Table.new_as_reference(self.domain, self, key)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, int):
            if isinstance(col_idx, slice):
                columns = range(*col_idx.indices(len(self.domain.attributes)))
            elif isinstance(col_idx, Iterable) and not isinstance(col_idx, str):
                columns = [col if isinstance(col, int) else self.domain.index(col)
                           for col in col_idx]
            else:
                columns = None
            if columns:
                row = self._X[row_idx]
                return [Value(self.domain[col], row[col]) for col in columns]

            # single row, single column
            if not isinstance(col_idx, int):
                col_idx = self.domain.index(col_idx)
            var = self.domain[col_idx]
            if col_idx >= 0:
                if col_idx < len(self.domain.attributes):
                    return Value(var, self._X[row_idx, col_idx])
                else:
                    return Value(var,
                        self._Y[row_idx, col_idx - len(self.domain.attributes)])
            else:
                return Value(var, self._metas[row_idx, -1-col_idx])

        # multiple rows: construct a new table
        attributes, col_indices = self._compute_col_indices(col_idx)
        if attributes:
            domain = Domain(attributes, self.domain.class_vars)
        else:
            domain = self.table.domain
        return Table.new_as_reference(domain, self, row_idx, col_indices)


    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            if isinstance(value, Real):
                self._X[key, :] = value
            self.domain.convert_to_row(value,
                self._X[key], self._Y[key], self._metas[key])
            self.clear_cache()
            return

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")
        row_idx, col_idx = key

        # single row
        if isinstance(row_idx, int):
            if isinstance(col_idx, slice):
                col_idx = range(*slice.indices(col_idx, self._X.shape[1]))
            if not isinstance(col_idx, str) and isinstance(col_idx, Iterable):
                ...
                return

            if not isinstance(value, int):
                value = self.domain[col_idx].to_val(value)
            if not isinstance(col_idx, int):
                col_idx = self.domain.index(col_idx)
            if col_idx >= 0:
                self._X[row_idx, col_idx] = value
                self.clear_cache("X")
            else:
                self._metas[row_idx, -1-col_idx] = value
                self.clear_cache("m")

        # multiple rows, multiple columns
        attributes, col_indices = self._compute_col_indices(col_idx)
        if col_indices is ...:
            col_indices = range(len(self.domain))
        n_attrs = self._X.shape[1]
        if isinstance(value, str):
            if not attributes:
                attributes = self.domain.attributes
            for var, col in zip(attributes, col_indices):
                if 0 <= col < n_attrs:
                    self._X[row_idx, col] = var.to_val(value)
                elif col >= n_attrs:
                    self._Y[row_idx, col - n_attrs] = var.to_val(value)
                else:
                    self._metas[row_idx, -1-col] = var.to_val(value)
        else:
            attr_cols = np.fromiter(
                (col for col in col_indices if 0 <= col < n_attrs), int)
            class_cols = np.fromiter(
                (col - n_attrs for col in col_indices if col >= n_attrs), int)
            meta_cols = np.fromiter(
                (-1-col for col in col_indices if col < col), int)
            if value is None:
                value = Unknown
            if not isinstance(value, Real) and attr_cols or class_cols:
                raise TypeError("Ordinary attributes can only have primitive values")
            if len(attr_cols):
                self._X[row_idx, attr_cols] = value
            if len(class_cols):
                self._Y[row_idx, class_cols] = value
            if len(meta_cols):
                self._metas[row_idx, meta_cols] = value
        if any(0 <= col < n_attrs for col in col_indices):
            self.clear_cache("X")
        if any(col >= n_attrs for col in col_indices):
            self.clear_cache("Y")
        if any(col < 0 for col in col_indices):
            self.clear_cache("m")


    def __delitem__(self, key):
        if key is ...:
            key = range(len(self))
        self._X = np.delete(self._X, key, axis=0)
        self._Y = np.delete(self._Y, key, axis=0)
        self._metas = np.delete(self._metas, key, axis=0)
        if self._W is not None:
            self._W = np.delete(self._W, key, axis=0)
        self.clear_cache()


    def clear(self):
        del self[...]

    def __len__(self):
        return len(self._X)

    def insert(self, key, value):
        ...


    def random_example(self):
        n_examples = len(self)
        if not n_examples:
            raise IndexError("Table is empty")
        return self[random.randint(0, n_examples-1)]

    def total_weight(self):
        if self._W is None:
            return len(self)
        return sum(self._W)


    def has_missing(self):
        return bn.anynan(self._X)

    def has_missing_class(self):
        return bn.anynan(self.Y)

    def checksum(self, include_metas=False):
        #TODO: check whether .data builds a new buffer; try avoiding it
        cs = 1
        cs = zlib.adler32(self._X.data)
        cs = zlib.adler32(self._Y.data, cs)
        cs = zlib.adler32(self._metas.data, cs)
        if self._W:
            cs = zlib.adler32(self._metas.data, cs)
        return cs

    def shuffle(self):
        # TODO: write a function in Cython that would do this in place
        ind = np.range(len(self._X))
        np.random.shuffle(ind)
        self._X = self._X[ind]
        self._Y = self._Y[ind]
        self._metas = self._metas[ind]
        if self._W is not None:
            self._W = self._W[ind]
        self.clear_cache()

    def sort(self, attributes=None):
        if attributes is not None:
            attributes, col_indices = self._compute_col_indices(attributes)
        if not attributes:
            ranks = bn.nanrankdata(self._X, axis=0)
        else:
            if np.any(col_indices < 0):
                data = np.hstack((self._X, self._Y, self._metas))
            else:
                data = np.hstack((self._X, self._Y))
            ranks = bn.nanrankdata(data[col_indices], axis=0)
            del data
        print(ranks.shape, self.n_rows, len(self._X))
        if self.row_indices is ...:
            rank_row = np.hstack((ranks, np.arange(self.n_rows)))
        else:
            rank_row = np.hstack((ranks, self.row_indices))
        rank_row = np.sort(rank_row, axis=0)
        print(rank_row)
        self.row_indices = rank_row[:, 1]
        self.clear_cache()

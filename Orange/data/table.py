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
        if domain.metas:
            self._metas = np.empty((n_rows, len(self.domain.metas)), object)
        else:
            self._metas = None
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
        return (self._X is None or self._X.base) and \
               (self._Y is None or self._Y.base) and \
               (self._W is None or self._W.base) and \
               (self._metas is None or self._metas.base)

    def is_copy(self):
        return (self._X is None or not self._X.base) and \
               (self._Y is None or not self._Y.base) and \
               (self._W is None or not self._W.base) and \
               (self._metas is None or not self._metas.base)

    def ensure_copy(self):
        if self._X and self._X.base:
            self._X = self._X.copy()
        if self._Y and self._Y.base:
            self._Y = self._Y.copy()
        if self._W and self._W.base:
            self._W = self._W.copy()
        if self._metas and self._metas.base:
            self._metas = self._metas.copy()

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


    def _getitem_from_row(self, row_idx, col_idx):
        # single row, multiple columns
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
                    self._X[row_idx, col_idx - len(self.domain.attributes)])
        else:
            return Value(var, self._metas[row_idx, -1-col_idx])
        #TODO implement metas


    def __getitem__(self, key):
        if isinstance(key, int):
            return RowInstance(self, key)
        if not isinstance(key, tuple):
            return Table.new_as_reference(self.domain, self, key)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, int):
            return self._getitem_from_row(row_idx, col_idx)

        attributes, col_indices = self._compute_col_indices(col_idx)
        if attributes:
            domain = Domain(attributes, self.domain.class_vars)
        else:
            domain = self.table.domain
        return Table.new_as_reference(domain, self, row_idx, col_indices)


    def _setitem_row(self, row_idx, value):
        if isinstance(value, Real):
            self._X[row_idx, :] = value
        elif isinstance(value, Instance):
            # TODO domain conversion
            self._X[row_idx, :] = value._values[:self._X.shape[1]]
            self._Y[row_idx, :] = value._values[self._X.shape[1]:]
            if value.metas:
                self._metas[row_idx] = value.metas
        self._X[row_idx, :] = value[:self._X.shape[1]]
        if len(value) > value[:self._X.shape[1]]:
            self_Y[row_idx, :] = value[self._X.shape[1]:]
        self.clear_cache("X Y")


    def _setitem_row_col(self, row_idx, col_idx, value):
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


    def __setitem__(self, key, value):
        # single index -- one row
        if isinstance(key, int):
            self._setitem_row(key, value)

        # single index -- multiple rows
        if not isinstance(key, tuple):
            self._set_item_row(key, value)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")
        row_idx, col_idx = key

        # single row
        if isinstance(row_idx, int):
            return self._setitem_row_col(row_idx, col_idx, value)

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
        if self._X is not None:
            self._X = np.delete(self._X, key, axis=0)
        if self._Y is not None:
            self._Y = np.delete(self._Y, key, axis=0)
        if self._W is not None:
            self._W = np.delete(self._W, key, axis=0)
        if self._metas is not None:
            self._metas = np.delete(self._metas, key, axis=0)
        self.clear_cache()


    def clear(self):
        del self[...]

    def __len__(self):
        if self._X is not None:
            return len(self._X)
        if self._Y is not None:
            return len(self._Y)
        if self._metas is not None:
            return len(self._metas)
        raise ValueError("Invalid Table (no data)")

    def insert(self, key, value):
        ...


    def convert(self, example):
        if example.domain == self.domain:
            return example
        if self.last_domain is not example.domain:
            self.last_domain = self.known_domains.get(example.domain, None)
            if not self.last_domain:
                pass


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
        if self._X is None:
            return False
        return bn.anynan(self._X)

    def has_missing_class(self):
        if self._Y is None:
            return False
        return bn.anynan(self.Y)

    def checksum(self, include_metas=False):
        #TODO: check whether .data builds a new buffer; try avoiding it
        cs = 1
        if self._X is not None:
            cs = zlib.adler32(self.X.data)
        if self._Y is not None:
            cs = zlib.adler32(self.Y.data, cs)
        if include_metas and self._metas is not None:
            cs = zlib.adler32(self.metas.data, cs)
        return cs

    def shuffle(self):
        if self._X is not None:
            np.random.shuffle(self._X)
        if self._Y is not None:
            np.random.shuffle(self._Y)
        if self._W is not None:
            np.random.shuffle(self._W)
        if self._metas is not None:
            np.random.shuffle(self._metas)
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

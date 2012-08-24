import os
import random
import zlib
from collections import MutableSequence, Iterable
from numbers import Real

import numpy as np
import bottleneck as bn

from ..data import Value, Example, RowExample, Domain, Unknown
from ..data.io import TabDelimReader
from ..misc import IdemMap

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
        self.row_indices = ...
        self.col_indices = ...
        if domain.metas:
            self._metas = np.empty((n_rows, len(self.domain.metas)), object)
        else:
            self._metas = None
        return self


    @staticmethod
    def new_as_reference(domain, table, row_indices=..., col_indices=...):
        self = Table.__new__(Table)
        self.domain = domain
        self._X = table._X
        self._Y = table._Y
        self._metas = table._metas
        if table._W:
            if row_indices is None:
                self._W = np.array(table._W)
            else:
                self._W = table._W[row_indices]
        if not isinstance(row_indices, (np.ndarray, EllipsisType)):
            self.row_indices = np.array(row_indices)
        else:
            self.row_indices = row_indices
        self.col_indices = col_indices
        if self.row_indices is not ...:
            self.n_rows = len(row_indices)
        elif self._X is not None:
            self.n_rows = self._X.shape[1]
        elif self._Y is not None:
            self.n_rows = self._Y.shape[-1]
        else:
            self.n_rows = 0
        return self


    def getX(self):
        if self._Xcache is None and self._X is not None:
            self._Xcache = self._X[self.row_indices, self.col_indices]
        return self._Xcache

    def getY(self):
        if self._Ycache is None and self._Y is not None:
            self._Ycache = self._Y[self.row_indices]
        return self._Ycache

    def get_metas(self):
        if self._metas_cache is None and self._metas is not None:
            self._metas_cache = self._metas[self.row_indices]
        return self._metas_cache

    def get_weights(self):
        if self._Wcache is None and self._W is not None:
            self._Wcache = self._W[self.row_indices]
        return self._Wcache

    def clear_cache(self, what=None):
        if what:
            for w in what:
                if w == "X":
                    self._Xcache = None
                elif w == "Y":
                    self._Ycache = None
                elif w == "m":
                    self._metas_cache = None
                elif w == "w":
                    self._Wcache = None
        else:
            self._Xcache = self._Ycache = self._metas_cache = self._Wcache = None

    X = property(getX)
    Y = property(getY)
    W = property(get_weights)
    metas = property(get_metas)

    def set_weights(self, val=1):
        if self._W:
            self._W[:] = val
        else:
            self._W = np.empty(len(self))
            self._W.fill(val)
        self._Wcache = None

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


    def _compute_row_indices(self, row_idx):
        if isinstance(row_idx, slice):
            start, stop, stride = row_idx.indices(self.n_rows)
            if (start, stop, stride) == (0, self.n_rows, 1):
                return self.row_indices
            if self.row_indices is ...:
                return np.arange(start, stop, stride)
            else:
                return self.row_indices[start:stop:stride]
        if isinstance(row_idx, np.ndarray) and row_idx.dtype == bool:
            if self.row_indices is ...:
                return np.nonzero(row_idx)
            else:
                return self.row_indices[row_idx]
        if isinstance(row_idx, Iterable):
            if self.row_indices is ...:
                return np.fromiter(row_idx, int)
            else:
                return self.row_indices[np.fromiter(row_idx, int)]
        raise IndexError("Invalid index type")


    def _compute_col_indices(self, col_idx):
        """Return a list of new attributes and column indices,
           or (None, self.col_indices) if no new domain needs to be constructed"""
        if isinstance(col_idx, np.ndarray) and col_idx.dtype == bool:
            return [attr for attr, c in zip(self.domain, col_idx) if c], \
                   np.nonzero(col_idx)
        elif isinstance(col_idx, slice):
            if self.col_indices is not ...:
                return self.domain[col_idx], self.col_indices(col_idx)
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
            if self.col_indices is ...:
                return attributes, np.fromiter(
                    (self.domain.index(attr) for attr in attributes), int)
            else:
                return attributes, np.fromiter(
                    (self.col_indices[self.domain.index(attr)]
                        for attr in attributes), int)
        else:
            if isinstance(col_idx, int):
                attr = self.domain[col_idx]
            else:
                attr = self.domain[col_idx]
                col_idx =  self.domain.index(attr)
            if self.col_indices is not ...:
                col_idx = self.col_indices[col_idx]
            return [attr], np.array([col_idx])


    def _getitem_from_row(self, row_idx, col_idx):
        # single row, multiple columns
        if isinstance(col_idx, slice):
            columns = range(*col_idx.indices(len(self.domain.attributes)))
        elif isinstance(col_idx, Iterable) and \
             not isinstance(col_idx, str):
            columns = [col if isinstance(col, int) else self.domain.index(col)
                       for col in col_idx]
        else:
            columns = None
        if columns:
            row = self._X[row_idx]
            if self.col_indices is ...:
                return [Value(self.domain[col], row[col]) for col in columns]
            else:
                return [Value(self.domain[col], row[self.col_indices[col]])
                        for col in columns]

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
        # single index -- one row
        if isinstance(key, int):
            if self.row_indices is not ...:
                key = self.row_indices[key]
            return RowExample(self, key)

        # single index -- multiple rows
        if not isinstance(key, tuple):
            return Table.new_as_reference(
                self.domain, self,
                self._compute_row_indices(key))

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")
        row_idx, col_idx = key

        # single row
        if isinstance(row_idx, int):
            if self.row_indices is not ...:
                row_idx = self.row_indices[row_idx]
            return self._getitem_from_row(row_idx, col_idx)

        # multiple rows, multiple columns;
        # return a new table, create new domain if needed
        attributes, col_indices = self._compute_col_indices(col_idx)
        if attributes:
            domain = Domain(attributes, self.domain.class_vars)
        else:
            domain = self.table.domain

        row_indices = self._compute_row_indices(row_idx)
        return Table.new_as_reference(domain, self, row_indices, col_indices)



    def _setitem_row(self, row_idx, value):
        if isinstance(value, Real):
            self._X[row_idx, :] = value
        elif isinstance(value, Example):
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
            if self.row_indices is ...:
                self._setitem_row(key, value)
            else:
                self._setitem_row(self.row_indices[key], value)

        # single index -- multiple rows
        if not isinstance(key, tuple):
            self._set_item_row(self._compute_row_indices(key), value)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")
        row_idx, col_idx = key

        # single row
        if isinstance(row_idx, int):
            if self.row_indices is not ...:
                row_idx = self.row_indices[row_idx]
            return self._setitem_row_col(row_idx, col_idx, value)

        # multiple rows, multiple columns
        row_indices = self._compute_row_indices(row_idx)
        attributes, col_indices = self._compute_col_indices(col_idx)
        print(attributes, col_indices)
        if col_indices is ...:
            col_indices = range(len(self.domain))
        n_attrs = self._X.shape[1]
        if isinstance(value, str):
            if not attributes:
                attributes = self.domain.attributes
            for var, col in zip(attributes, col_indices):
                if 0 <= col < n_attrs:
                    self._X[row_indices, col] = var.to_val(value)
                elif col >= n_attrs:
                    self._Y[row_indices, col - n_attrs] = var.to_val(value)
                else:
                    self._metas[row_indices, -1-col] = var.to_val(value)
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
            print(attr_cols, bool(attr_cols), row_indices)
            if len(attr_cols):
                self._X[row_indices, attr_cols] = value
            if len(class_cols):
                self._Y[row_indices, class_cols] = value
            if len(meta_cols):
                self._metas[row_indices, meta_cols] = value
            print(self._X)
        if any(0 <= col_indices < n_attrs for i in col_indices):
            self.clear_cache("X")
        if any(i >= n_attrs for i in col_indices):
            self.clear_cache("Y")
        if any(i < 0 for i in col_indices):
            self.clear_cache("m")


    def __delitem__(self, key):
        if self.row_indices is ...:
            if isinstance(key, int):
                if key > self.n_rows:
                    raise IndexError("Row {} is out of range".format(key))
                self.row_indices = np.hstack((np.arange(key),
                                              np.arange(key+1, self.n_rows)))
            elif isinstance(key, slice):
                start, end, stride = key.indices(self.n_rows)
                if stride == 1:
                    self.row_indices = np.hstack((np.arange(start),
                                                  np.arange(end,self.n_rows)))
                else:
                    self.row_indices = np.hstack((
                        np.arange(start),
                        np.arange(start, end)[np.arange(end-start) % stride != 0],
                        np.arange(end, self.n_rows)))
            elif isinstance(key, np.ndarray) and key.dtype == bool:
                self.row_indices = np.logical_not(key).nonzero()
            elif isinstance(key, Iterable):
                key = np.fromiter(key, int)
                self.row_indices = np.setdiff1d(np.arange(self.n_rows, key))
            else:
                raise IndexError("Invalid index type")
        else:
            if isinstance(key, np.ndarray) and key.dtype == bool:
                self.row_indices = self.row_indices[key]
            else:
                if isinstance(key, Iterable):
                    key = sorted(key, reverse=True)
                if isinstance(key, (int, slice, np.ndarray, list)):
                    self.row_indices = np.delete(self.row_indices, key)
                else:
                    raise IndexError("Invalid index type")
        self.n_rows = len(self.row_indices)
        self.clear_cache()


    def clear(self):
        self.row_indices = np.array([], int)
        self.n_rows = 0
        self.clear_cache()

    def __len__(self):
        return self.n_rows

    def insert(self, key, value):
        ...


    def convert(self, example):
        if example.domain == self.domain:
            return example
        if self.last_domain is not example.domain:
            self.last_domain = self.known_domains.get(example.domain, None)
            if not self.last_domain:


    def random_example(self):
        n_examples = len(self)
        print(n_examples)
        if not n_examples:
            raise IndexError("Table is empty")
        return self[random.randint(0, n_examples-1)]

    def total_weight(self):
        if self._W is None:
            return self.n_rows
        return sum(self._W[self.row_indices])


    def has_missing(self):
        if self._X is None:
            return False
        return bn.anynan(self.X)

    def has_missing_class(self):
        if self._Y is None:
            return False
        return bn.anynan(self.Y)

    def checksum(self, include_metas=False):
        #TODO: check whether .data builds a new buffer; try avoiding it
        #TODO: is there a better way to do it?!
        cs = 1
        if self._X is not None:
            cs = zlib.adler32(self.X.data)
        if self._Y is not None:
            cs = zlib.adler32(self.Y.data, cs)
        if include_metas and self._metas is not None:
            cs = zlib.adler32(self.metas.data, cs)
        return cs

    def shuffle(self):
        if self.row_indices is ...:
            self.row_indices = np.arange(self.n_rows)
        np.random.shuffle(self.row_indices)
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

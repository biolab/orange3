import os
import random
import zlib
from collections import MutableSequence, Iterable
from numbers import Real

import numpy as np
import bottleneck as bn

from ..data import Value, Instance, Domain, Unknown
from ..data.io import TabDelimReader

class RowInstance(Instance):
    def __init__(self, table, row_index):
        super().__init__(table.domain)
        if table._X is not None:
            self._x = table._X[row_index]
            self._values = list(self._x)
        else:
            self._x = None
            self._values = []
        if table._Y is not None:
            self._y = table._Y[row_index]
            self._values += list(self._y)
        else:
            self._y = None
        self.row_index = row_index
        self._metas = table._metas is not None and table._metas[row_index]
        self.table = table

    def _check_single_class(self):
        if not self.domain.class_vars:
            raise TypeError("Domain has no class variable")
        elif len(self.domain.class_vars) > 1:
            raise TypeError("Domain has multiple class variables")

    def get_class(self):
        self._check_single_class()
        if self.table.domain.class_var:
            return Value(self.table.domain.class_var, self._y[0])

    def set_class(self, value):
        self._check_single_class()
        if not isinstance(value, Real):
            self._y[0] = self.table.domain.class_var.to_val(value)
        else:
            self._y[0] = value

    def get_classes(self):
        return (Value(var, value) for var, value in
            zip(self.table.domain.class_vars, self._y))

    def set_weight(self, weight):
        if self.table._W is None:
            self.table.set_weights()
        self.table._W[self.row_index] = weight

    def get_weight(self):
        if not self.table._W:
            raise ValueError("Instances in the referenced table have no weights")
        return self.table._W[self.row_index]

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            key = self.domain.index(key)
        if isinstance(value, str):
            var = self.domain[key]
            value = var.to_val(value)
        if key >= 0:
            if not isinstance(value, Real):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            if key < len(self._x):
                self._x[key] = value
            else:
                self._y[key - len(self._x)] = value
        else:
            self._metas[-1-key] = value



class Table(MutableSequence):
    def __new__(cls, *args, **argkw):
        self = None
        if not args:
            if not args and not argkw:
                self = super().__new__(cls)
            elif "filename" in argkw:
                self = cls.read_data(argkw["filename"])
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                self = Table.read_data(args[0])
            elif isinstance(arg, Domain):
                self = Table.new_from_domain(arg, 0)
        if self is None:
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
            self._W = np.empty((n_rows, 0))
        self._metas = np.empty((n_rows, len(self.domain.metas)), object)
        return self


    @staticmethod
    def new_as_reference(domain, table, row_indices=..., col_indices=...):
        self = Table.__new__(Table)
        self.domain = domain
        self._X = table._X[row_indices, col_indices]
        self._Y = table._Y[row_indices]
        self._metas = table._metas[row_indices]
        self._W = np.array(table._W[row_indices])
        return self


    def is_view(self):
        return self._X.base is not None and \
               self._Y.base is not None and \
               self._metas.base is not None and \
               self._W.base is not None

    def is_copy(self):
        return self._X.base is None and \
               self._Y.base is None and \
               self._metas.base is None and \
               self._W.base is None

    def ensure_copy(self):
        if self._X.base is not None:
            self._X = self._X.copy()
        if self._Y.base is not None:
            self._Y = self._Y.copy()
        if self._metas.base is not None:
            self._metas = self._metas.copy()
        if self._W.base is not None:
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

    def clear_cache(self, _="XYWm"):
        pass

    def set_weights(self, val=1):
        if self._W.shape[-1]:
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


    def convert_to_row(self, example, key):
        domain = self.domain
        if isinstance(example, Instance):
            if example.domain == domain:
                if isinstance(example, RowInstance):
                    self._X[key] = example._x
                    self._Y[key] = example._y
                else:
                    self._X[key] = example._values[:len(domain.attributes)]
                    self._Y[key] = example._values[len(domain.attributes):]
                self._metas[key] = example._metas
                return
            c = self.domain.get_conversion(example.domain)
            self._X[key] = [example._values[i] if isinstance(i, int) else
                    (Unknown if not i else i(example)) for i in c.attributes]
            self._Y[key] = [example._values[i] if isinstance(i, int) else
                    (Unknown if not i else i(example)) for i in c.classes]
            self._metas[key] = [example._values[i] if isinstance(i, int) else
                    (Unknown if not i else i(example)) for i in c.metas]
        else:
            self._X[key] = [var.to_val(val)
                    for var, val in zip(domain.attributes, example)]
            self._Y[key] = [var.to_val(val)
                    for var, val in zip(domain.class_vars, example[len(domain.attributes):])]
            self._metas[key] = Unknown

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
            self.convert_to_row(value, key)
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
        self._W = np.delete(self._W, key, axis=0)
        self.clear_cache()


    def clear(self):
        del self[...]

    def __len__(self):
        return len(self._X)


    def __str__(self):
        s = "[" + ",\n ".join(str(ex) for ex in self[:5])
        if len(self) > 5:
            s += ",\n ..."
        s += "\n]"
        return s

    def resize_all(self, new_length):
        old_length = len(self._X)
        if old_length == new_length:
            return
        try:
            self._X.resize(new_length, self._X.shape[1])
            self._Y.resize(new_length, self._Y.shape[1])
            self._metas.resize(new_length, self._metas.shape[1])
            if self._W.ndim == 2:
                self._W.resize((new_length, 0))
            else:
                self._W.resize(new_length)
        except Exception:
            if len(self._X) == new_length:
                self._X.resize(old_length, self._X.shape[1])
            if len(self._Y) == new_length:
                self._Y.resize(old_length, self._Y.shape[1])
            if len(self._metas) == new_length:
                self._metas.resize(old_length, self._metas.shape[1])
            if len(self._W) == new_length:
                if self._W.ndim == 2:
                    self._W.resize((old_length, 0))
                else:
                    self._W.resize(old_length)
            raise

    def extend(self, examples):
        old_length = len(self)
        self.resize_all(old_length + len(examples))
        try:
            # shortcut
            if isinstance(examples, Table) and examples.domain == self.domain:
                self._X[old_length:] = examples._X
                self._Y[old_length:] = examples._Y
                self._metas[old_length:]  = examples._metas
                if self._W.shape[-1]:
                    if examples._W.shape[-1]:
                        self._W[old_length:] = examples._W
                    else:
                        self._W[old_length:] = 1
            else:
                for i, example in enumerate(examples):
                    self[old_length + i] = example
        except Exception:
            self.resize_all(old_length)
            raise

    def insert(self, key, value):
        if key < 0:
            key += len(self)
        if key < 0 or key > len(self):
            raise IndexError("Index out of range")
        self.resize_all(len(self) + 1)
        if key < len(self):
            self._X[key+1:] = self._X[key:-1]
            self._Y[key+1:] = self._Y[key:-1]
            self._metas[key+1:] = self._metas[key:-1]
            self._W[key+1:] = self._W[key:-1]
        try:
            self.convert_to_row(value, key)
            if self._W.shape[-1]:
                self._W[key] = 1
        except Exception:
            self._X[key:-1] = self._X[key+1:]
            self._Y[key:-1] = self._Y[key+1:]
            self._metas[key:-1] = self._metas[key+1:]
            self._W[key:-1] = self._W[key+1:]
            self.resize_all(len(self)-1)
            raise


    def append(self, value):
        self.insert(len(self), value)

    def random_example(self):
        n_examples = len(self)
        if not n_examples:
            raise IndexError("Table is empty")
        return self[random.randint(0, n_examples-1)]

    def total_weight(self):
        if self._W.shape[-1]:
            return sum(self._W)
        return len(self)


    def has_missing(self):
        return bn.anynan(self._X)

    def has_missing_class(self):
        return bn.anynan(self.Y)

    def checksum(self, include_metas=True):
        #TODO: check whether .data builds a new buffer; try avoiding it
        cs = zlib.adler32(self._X.data)
        cs = zlib.adler32(self._Y.data, cs)
        if include_metas:
            cs = zlib.adler32(self._metas.data, cs)
        cs = zlib.adler32(self._W.data, cs)
        return cs

    def shuffle(self):
        # TODO: write a function in Cython that would do this in place
        ind = np.arange(len(self._X))
        np.random.shuffle(ind)
        self._X = self._X[ind]
        self._Y = self._Y[ind]
        self._metas = self._metas[ind]
        self._W = self._W[ind]
        self.clear_cache()

    def sort(self, attributes=None):
        # TODO Does not work
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
        if self.row_indices is ...:
            rank_row = np.hstack((ranks, np.arange(self.n_rows)))
        else:
            rank_row = np.hstack((ranks, self.row_indices))
        rank_row = np.sort(rank_row, axis=0)
        self.row_indices = rank_row[:, 1]
        self.clear_cache()


    #TODO fast mapping of entire example tables, not just examples
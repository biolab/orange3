import os
import random
import zlib
from collections import MutableSequence, Iterable
from itertools import chain
from numbers import Real
import operator
from functools import reduce

import numpy as np
import bottleneck as bn

from .instance import *
from Orange.data import domain as orange_domain, io, variable

dataset_dirs = ['']

class RowInstance(Instance):
    def __init__(self, table, row_index):
        super().__init__(table.domain)
        self._x = table._X[row_index]
        self._y = table._Y[row_index]
        self._values = np.hstack((self._x, self._y))
        self._metas = table._metas[row_index]
        self.row_index = row_index
        self.table = table

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
        self._values[len(self.table.domain.attributes)] = self._y[0]

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
                self._values[key] = self._x[key] = value
            else:
                self._values[key] = self._y[key - len(self._x)] = value
        else:
            self._metas[-1-key] = value


class Columns:
    def __init__(self, domain):
        for v in chain(domain, domain.metas):
            setattr(self, v.name.replace(" ", "_"), v)

class Table(MutableSequence):
    def __new__(cls, *args, **argkw):
        if not args and not argkw:
            return super().__new__(cls)
        if "filename" in argkw:
           return cls.read_data(argkw["filename"])
        try:
            if isinstance(args[0], str):
                return cls.read_data(args[0])
            if isinstance(args[0], Domain) and len(args) == 1:
                return cls.new_from_domain(args[0])
            if all(isinstance(arg, np.ndarray) for arg in args):
                domain = cls.create_anonymous_domain(*args[:3])
                return cls.new_from_numpy(domain, *args)
            if isinstance(args[0], Domain) and \
               all(isinstance(arg, np.ndarray) for arg in args[1:]):
                return cls.new_from_numpy(*args)
        except IndexError:
            pass
        raise ValueError("Invalid arguments for Table.__new__")

    @staticmethod
    def create_anonymous_domain(X, Y=None, metas=None):
        attr_vars = [variable.ContinuousVariable(name="Feature %i" % a) for a in range(X.shape[1])]
        class_vars = []
        if Y is not None:
            for i, class_ in enumerate(Y.T):
                values = np.unique(class_)
                if len(values) < 20:
                    class_vars.append(variable.DiscreteVariable(name="Class %i" % i, values=list(map(int, values))))
                else:
                    class_vars.append(variable.ContinuousVariable(name="Class %i" % i))
        meta_vars = [variable.StringVariable(name="Meta %i" % m) for m in range(metas.shape[1])] if metas is not None else []

        domain = orange_domain.Domain(attr_vars, class_vars, meta_vars)
        domain.anonymous = True
        return domain

    @staticmethod
    def new_from_domain(domain, n_rows=0, weights=True):
        assert(len(domain.class_vars) <= 1)
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
    def compose_cols_from(table, row_indices, src_cols, n_rows):
        #TODO: handle get_value_from
        if not len(src_cols):
            return np.zeros((n_rows, 0), dtype=table._X.dtype)

        n_src_attrs = len(table.domain.attributes)
        if all(0 <= x < n_src_attrs for x in src_cols):
            return table._X[row_indices, src_cols]
        if all(x < 0 for x in src_cols):
            return table._metas[row_indices, [-1-x for x in src_cols]]
        if all(x >= n_src_attrs for x in src_cols):
            return table._Y[row_indices, [x-n_src_attrs for x in src_cols]]

        a = np.empty((n_rows, len(src_cols)), dtype=table._X.dtype)
        for i, col in enumerate(src_cols):
            if col < 0:
                a[:, i] = table._metas[row_indices, -1 - col]
            elif col < n_src_attrs:
                a[:, i] = table._X[row_indices, col]
            else:
                a[:, i] = table._Y[row_indices, col - n_src_attrs]
        return a


    @staticmethod
    def new_from_table(domain, table, row_indices=...):
        if domain == table.domain:
            return Table.new_from_table_rows(table, row_indices)

        if isinstance(row_indices, slice):
            start, stop, stride = row_indices.indices(len(table._X))
            n_rows = (stop - start) / stride
            if n_rows < 0:
                n_rows = 0
        elif row_indices is ...:
            n_rows = len(table._X)
        else:
            n_rows = len(row_indices)

        self = Table.__new__(Table)
        self.domain = domain
        conversion = domain.get_conversion(table.domain)
        self._X = Table.compose_cols_from(
            table, row_indices, conversion.attributes, n_rows)
        self._Y = Table.compose_cols_from(
            table, row_indices, conversion.class_vars, n_rows)
        self._metas = Table.compose_cols_from(
            table, row_indices, conversion.metas, n_rows)
        self._W = np.array(table._W[row_indices])
        return self


    @staticmethod
    def new_from_table_rows(table, row_indices):
        self = Table.__new__(Table)
        self.domain = table.domain
        self._X = table._X[row_indices]
        self._Y = table._Y[row_indices]
        self._metas = table._metas[row_indices]
        self._W = table._W[row_indices]
        return self

    @staticmethod
    def new_from_numpy(domain, X, Y=None, metas=None, W=None):
        #assert(len(domain.class_vars) <= 1)
        if Y is None:
            Y = X[:, len(domain.attributes):]
            X = X[:, :len(domain.attributes)]
        if metas is None:
            metas = np.empty((X.shape[0], 0), object)
        if W is None:
            W = np.empty((X.shape[0], 0))

        if X.shape[1] != len(domain.attributes):
            raise ValueError("Invalid number of variable columns ({} != {}".
                format(X.shape[1], len(domain.attributes)))
        if Y.shape[1] != len(domain.class_vars):
            raise ValueError("Invalid number of class columns ({} != {}".
                format(Y.shape[1], len(domain.class_vars)))
        if metas.shape[1] != len(domain.metas):
            raise ValueError("Invalid number of meta attribute columns ({} != {}".
                format(metas.shape[1], len(domain.metas)))
        if not X.shape[0] == Y.shape[0] == metas.shape[0] == W.shape[0]:
            raise ValueError("Parts of data contain different numbers of rows.")

        self = Table.__new__(Table)
        self.domain = domain
        self._X = X
        self._Y = Y
        self._metas = metas
        self._W = W
        self.n_rows = len(self._X)
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
    domain = None

    columns= property(lambda self: Columns(self.domain))

    def clear_cache(self, _="XYWm"):
        pass

    def set_weights(self, val=1):
        if self._W.shape[-1]:
            self._W[:] = val
        else:
            self._W = np.empty(len(self))
            self._W.fill(val)

    def has_weights(self):
        return self._W.shape[-1] != 0

    @staticmethod
    def read_data(filename):
        for dir in dataset_dirs:
            ext = os.path.splitext(filename)[1]
            absolute_filename = os.path.join(dir, filename)
            if not ext:
                for ext in [".tab"]:
                    if os.path.exists(absolute_filename + ext):
                        absolute_filename += ext
                        break
            if os.path.exists(absolute_filename):
                break

        if not os.path.exists(absolute_filename):
            raise IOError('File "{}" is not found'.format(absolute_filename))
        if ext == ".tab":
            return io.TabDelimReader().read_file(absolute_filename)
        else:
            raise IOError('Extension "{}" is not recognized'.format(absolute_filename))


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
                    (Unknown if not i else i(example)) for i in c.class_vars]
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
        if col_idx is ...:
            return None, None
        if isinstance(col_idx, np.ndarray) and col_idx.dtype == bool:
            return [attr for attr, c in zip(self.domain, col_idx) if c], \
                   np.nonzero(col_idx)
        elif isinstance(col_idx, slice):
            s = len(self.domain.variables)
            start, end, stride = col_idx.indices(s)
            if col_idx.indices(s) == (0, s, 1):
                return None, None
            else:
                return self.domain.variables[col_idx], \
                       np.arange(start, end, stride)
        elif isinstance(col_idx, Iterable) and not isinstance(col_idx, str):
            attributes = [self.domain[col] for col in col_idx]
            if attributes == self.domain.attributes:
                return None, None
            return attributes, np.fromiter(
                (self.domain.index(attr) for attr in attributes), int)
        elif isinstance(col_idx, int):
            attr = self.domain[col_idx]
        else:
            attr = self.domain[col_idx]
            col_idx =  self.domain.index(attr)
        return [attr], np.array([col_idx])


    def __getitem__(self, key):
        if isinstance(key, int):
            return RowInstance(self, key)
        if not isinstance(key, tuple):
            return Table.new_from_table_rows(self, key)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, int):
            try:
                col_idx = self.domain.index(col_idx)
                var = self.domain[col_idx]
                if 0 <= col_idx < len(self.domain.attributes):
                    return Value(var, self._X[row_idx, col_idx])
                elif col_idx >= len(self.domain.attributes):
                    return Value(var, self._Y[row_idx, col_idx - len(self.domain.attributes)])
                elif col_idx < 0:
                    return Value(var, self._metas[row_idx, -1-col_idx])
            except TypeError:
                row_idx = [row_idx]

        # multiple rows OR single row but multiple columns: construct a new table
        attributes, col_indices = self._compute_col_indices(col_idx)
        if attributes is not None:
            n_attrs = len(self.domain.attributes)
            r_attrs = [attributes[i] for i, col in enumerate(col_indices) if 0 <= col < n_attrs]
            r_classes = [attributes[i] for i, col in enumerate(col_indices) if col >= n_attrs]
            r_metas = [attributes[i] for i, col in enumerate(col_indices) if col < 0]
            domain = Domain(r_attrs, r_classes)
            domain.metas = r_metas
        else:
            domain = self.domain
        return Table.new_from_table(domain, self, row_idx)


    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            if isinstance(value, Real):
                self._X[key, :] = value
                return
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
                #TODO implement
                return

            if not isinstance(value, int):
                value = self.domain[col_idx].to_val(value)
            if not isinstance(col_idx, int):
                col_idx = self.domain.index(col_idx)
            if col_idx >= 0:
                if col_idx < self._X.shape[1]:
                    self._X[row_idx, col_idx] = value
                    self.clear_cache("X")
                else:
                    self._Y[row_idx, col_idx - self._X.shape[1]] = value
                    self.clear_cache("Y")
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
        return bn.anynan(self._X) or bn.anynan(self._Y)

    def has_missing_class(self):
        return bn.anynan(self.Y)

    def checksum(self, include_metas=True):
        cs = zlib.adler32(self._X)
        cs = zlib.adler32(self._Y, cs)
        if include_metas:
            cs = zlib.adler32(self._metas, cs)
        cs = zlib.adler32(self._W, cs)
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

    def get_column_view(self, index):
        if not isinstance(index, int):
            index = self.domain.index(index)
        if index >= 0:
            if index < self._X.shape[1]:
                return self._X[:, index]
            else:
                return self._Y[:, index - self._X.shape[1]]
        else:
            return self._metas[:, -1-index]

    def filter_is_defined(self, check=None, negate=False):
        #TODO implement checking by columns
        retain = np.logical_or(bn.anynan(self._X, axis=1),
                               bn.anynan(self._Y, axis=1))
        if not negate:
            retain = np.logical_not(retain)
        return Table.new_from_table_rows(self, retain)

    def filter_has_class(self, negate=False):
        retain = bn.anynan(self._Y, axis=1)
        if not negate:
            retain = np.logical_not(retain)
        return Table.new_from_table_rows(self, retain)

    def filter_random(self, prob, negate=False):
        retain = np.zeros(len(self), dtype=bool)
        if prob < 1:
            prob *= len(self)
        if negate:
            retain[prob:] = True
        else:
            retain[:prob] = True
        np.random.shuffle(retain)
        return Table.new_from_table_rows(self, retain)

    def filter_same_value(self, position, value, negate=False):
        if not isinstance(value, Real):
            value = self.domain[position].to_val(value)
        sel = self.get_column_view(position) == value
        if negate:
            sel = np.logical_not(sel)
        return Table.new_from_table_rows(self, sel)

    def filter_values(self, filt):
        from Orange.data import filter
        if isinstance(filt, filter.Values):
            conditions = filt.conditions
            conjunction = filt.conjunction
        else:
            conditions = [filt]
            conjunction = True
        if conjunction:
            sel = np.ones(len(self), dtype=bool)
        else:
            sel = np.zeros(len(self), dtype=bool)

        for f in conditions:
            col = self.get_column_view(f.position)
            if isinstance(f, filter.FilterDiscrete):
                if conjunction:
                    s2 = np.zeros(len(self))
                    for val in f.values:
                        if not isinstance(val, Real):
                            val = self.domain[f.position].to_val(val)
                        s2 += (col==val)
                    sel *= s2
                else:
                    for val in f.values:
                        if not isinstance(val, Real):
                            val = self.domain[f.position].to_val(val)
                        sel += (col==val)
            elif isinstance(f, filter.FilterStringList):
                if not f.case_sensitive:
                    col = np.char.lower(np.array(col, dtype=str))
                    vals = [val.lower() for val in f.values]
                else:
                    vals = f.values
                if conjunction:
                    sel *= reduce(operator.add, (col==val for val in vals))
                else:
                    sel = reduce(operator.add, (col==val for val in vals), sel)
            elif isinstance(f, (filter.FilterContinuous, filter.FilterString)):
                if isinstance(f, filter.FilterString) and not f.case_sensitive:
                    col = np.char.lower(np.array(col, dtype=str))
                    fmin = f.min.lower()
                    if f.oper in [f.Operator.Between, f.Operator.Outside]:
                        fmax = f.max.lower()
                else:
                    fmin, fmax = f.min, f.max
                if f.oper == f.Operator.Equal:
                    col = (col == fmin)
                elif f.oper == f.Operator.NotEqual:
                    col = (col != fmin)
                elif f.oper == f.Operator.Less:
                    col = (col < fmin)
                elif f.oper == f.Operator.LessEqual:
                    col = (col <= fmin)
                elif f.oper == f.Operator.Greater:
                    col = (col > fmin)
                elif f.oper == f.Operator.GreaterEqual:
                    col = (col >= fmin)
                elif f.oper == f.Operator.Between:
                    col = (col >= fmin) * (col <= fmax)
                elif f.oper == f.Operator.Outside:
                    col = (col < fmin) + (col > fmax)
                elif not isinstance(f, filter.FilterString):
                    raise TypeError("Invalid operator")
                elif f.oper == f.Operator.Contains:
                    col = np.fromiter((fmin in e for e in col), dtype=bool)
                elif f.oper == f.Operator.BeginsWith:
                    col = np.fromiter((e.startswith(fmin) for e in col), dtype=bool)
                elif f.oper == f.Operator.EndsWith:
                    col = np.fromiter((e.endswith(fmin) for e in col), dtype=bool)
                else:
                    raise TypeError("Invalid operator")
                if conjunction:
                    sel *= col
                else:
                    sel += col
            else:
                raise TypeError("Invalid filter")
        return Table.new_from_table_rows(self, sel)


import operator
import os
import sys
import threading
import warnings
import weakref
import zlib
from collections.abc import Iterable, Sequence, Sized
from contextlib import contextmanager
from functools import reduce
from itertools import chain
from numbers import Real, Integral
from threading import Lock
from typing import List, TYPE_CHECKING

import bottleneck as bn
import numpy as np

from scipy import sparse as sp
from scipy.sparse import issparse, csc_matrix

import Orange.data  # import for io.py
from Orange.data import (
    _contingency, _valuecount,
    Domain, Variable, Storage, StringVariable, Unknown, Value, Instance,
    ContinuousVariable, DiscreteVariable, MISSING_VALUES,
    DomainConversion)
from Orange.data.util import SharedComputeValue, \
    assure_array_dense, assure_array_sparse, \
    assure_column_dense, assure_column_sparse, get_unique_names_duplicates
from Orange.misc.collections import frozendict
from Orange.statistics.util import bincount, countnans, contingency, \
    stats as fast_stats, sparse_has_implicit_zeros, sparse_count_implicit_zeros, \
    sparse_implicit_zero_weights
from Orange.util import deprecated, OrangeDeprecationWarning, dummy_callback
if TYPE_CHECKING:
    # import just for type checking - avoid circular import
    from Orange.data.aggregate import OrangeTableGroupBy

__all__ = ["dataset_dirs", "get_sample_datasets_dir", "RowInstance", "Table"]


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '..', 'datasets')
    return os.path.realpath(dataset_dir)


dataset_dirs = ['', get_sample_datasets_dir()]


class _ThreadLocal(threading.local):
    def __init__(self):
        super().__init__()
        # Domain conversion cache used in Table.from_table. It is defined
        # here instead of as a class variable of a Table so that caching also works
        # with descendants of Table.
        self.conversion_cache = None
        self.domain_cache = None


_thread_local = _ThreadLocal()


def _idcache_save(cachedict, keys, value):
    cachedict[tuple(map(id, keys))] = \
        value, [weakref.ref(k) for k in keys]


def _idcache_restore(cachedict, keys):
    shared, weakrefs = cachedict.get(tuple(map(id, keys)), (None, []))
    for r in weakrefs:
        if r() is None:
            return None
    return shared


class DomainTransformationError(Exception):
    pass


class RowInstance(Instance):
    sparse_x = None
    sparse_y = None
    sparse_metas = None
    _weight = None

    def __init__(self, table, row_index):
        """
        Construct a data instance representing the given row of the table.
        """
        self.table = table
        self._domain = table.domain
        self.row_index = row_index
        self.id = table.ids[row_index]
        self._x = table.X[row_index]
        if sp.issparse(self._x):
            self.sparse_x = sp.csr_matrix(self._x)
            self._x = np.asarray(self._x.todense())[0]
        self._y = table._Y[row_index]
        if sp.issparse(self._y):
            self.sparse_y = sp.csr_matrix(self._y)
            self._y = np.asarray(self._y.todense())[0]
        self._y = np.atleast_1d(self._y)
        self._metas = table.metas[row_index]
        if sp.issparse(self._metas):
            self.sparse_metas = sp.csr_matrix(self._metas)
            self._metas = np.asarray(self._metas.todense())[0]

    @property
    def weight(self):
        if not self.table.has_weights():
            return 1
        return self.table.W[self.row_index]

    @weight.setter
    def weight(self, weight):
        if not self.table.has_weights():
            self.table.set_weights()
        self.table.W[self.row_index] = weight

    def set_class(self, value):
        # pylint: disable=protected-access
        self._check_single_class()
        if not isinstance(value, Real):
            value = self.table.domain.class_var.to_val(value)
        if self.sparse_y:
            self.table._Y[self.row_index, 0] = value
        else:
            self.table._Y[self.row_index] = value
            if self.table._Y.ndim == 1:  # if _y is not a view
                self._y[0] = value

    def __setitem__(self, key, value):
        if not isinstance(key, Integral):
            key = self._domain.index(key)
        if isinstance(value, str):
            var = self._domain[key]
            value = var.to_val(value)
        if key >= 0:
            if not isinstance(value, Real):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            if key < len(self._x):
                # write to self.table.X to support table unlocking for live instances
                self.table.X[self.row_index, key] = value
                if self.sparse_x is not None:
                    self._x[key] = value
            else:
                if self.sparse_y is not None:
                    self.table._Y[self.row_index, key - len(self._x)] = value
                else:
                    self.table._Y[self.row_index] = value
                    if self.table._Y.ndim == 1:  # if _y is not a view
                        self._y[0] = value
        else:
            self.table.metas[self.row_index, -1 - key] = value
            if self.sparse_metas is not None:
                self._metas[-1 - key] = value

    def _str(self, limit):
        def sp_values(matrix, variables):
            if not sp.issparse(matrix):
                if matrix.ndim == 1:
                    matrix = matrix[:, np.newaxis]
                return Instance.str_values(matrix[row], variables, limit)

            row_entries, idx = [], 0
            while idx < len(variables):
                # Make sure to stop printing variables if we limit the output
                if limit and len(row_entries) >= 5:
                    break

                var = variables[idx]
                if var.is_discrete or matrix[row, idx]:
                    row_entries.append("%s=%s" % (var.name, var.str_val(matrix[row, idx])))

                idx += 1

            s = ", ".join(row_entries)

            if limit and idx < len(variables):
                s += ", ..."

            return s

        table = self.table
        domain = table.domain
        row = self.row_index
        s = "[" + sp_values(table.X, domain.attributes)
        if domain.class_vars:
            s += " | " + sp_values(table.Y, domain.class_vars)
        s += "]"
        if self._domain.metas:
            s += " {" + sp_values(table.metas, domain.metas) + "}"
        return s

    def __str__(self):
        return self._str(False)

    def __repr__(self):
        return self._str(True)


class Columns:
    def __init__(self, domain):
        for v in chain(domain.variables, domain.metas):
            setattr(self, v.name.replace(" ", "_"), v)


class _ArrayConversion:

    def __init__(self, target, src_cols, variables, is_sparse, source_domain):
        self.target = target
        self.src_cols = src_cols
        self.is_sparse = is_sparse
        self.subarray_from = self._can_copy_all(src_cols, source_domain)
        self.variables = variables
        dtype = np.float64
        if any(isinstance(var, StringVariable) for var in self.variables):
            dtype = object
        self.dtype = dtype
        self.row_selection_needed = any(not isinstance(x, Integral)
                                        for x in src_cols)

    def _can_copy_all(self, src_cols, source_domain):
        n_src_attrs = len(source_domain.attributes)
        if all(isinstance(x, Integral) and 0 <= x < n_src_attrs
               for x in src_cols):
            return "X"
        if all(isinstance(x, Integral) and x < 0 for x in src_cols):
            return "metas"
        if all(isinstance(x, Integral) and x >= n_src_attrs
               for x in src_cols):
            return "Y"

    def get_subarray(self, source, row_indices, n_rows):
        if not len(self.src_cols):
            if self.is_sparse:
                return sp.csr_matrix((n_rows, 0), dtype=source.X.dtype)
            else:
                return np.zeros((n_rows, 0), dtype=source.X.dtype)

        match_density = assure_array_sparse if self.is_sparse else assure_array_dense
        n_src_attrs = len(source.domain.attributes)
        if self.subarray_from == "X":
            arr = match_density(_subarray(source.X, row_indices, self.src_cols))
        elif self.subarray_from == "metas":
            arr = match_density(_subarray(source.metas, row_indices,
                                          [-1 - x for x in self.src_cols]))
        elif self.subarray_from == "Y":
            Y = source.Y if source.Y.ndim == 2 else source.Y[:, None]
            arr = match_density(_subarray(
                Y, row_indices,
                [x - n_src_attrs for x in self.src_cols]))
        else:
            assert False
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype)
        assert arr.ndim == 2 or self.subarray_from == "Y" and arr.ndim == 1
        return arr

    def get_columns(self, source, row_indices, n_rows, out=None, target_indices=None):
        n_src_attrs = len(source.domain.attributes)

        data = []
        sp_col = []
        sp_row = []
        match_density = (
            assure_column_sparse if self.is_sparse else assure_column_dense
        )

        # converting to csc before instead of each column is faster
        # do not convert if not required
        if any(isinstance(x, int) for x in self.src_cols):
            X = source.X
            Y = source.Y
            if Y.ndim == 1:
                Y = Y[:, None]
            if self.is_sparse:
                X = csc_matrix(X)
                Y = csc_matrix(Y)

        if self.row_selection_needed:
            if row_indices is ...:
                sourceri = source
            else:
                sourceri = source[row_indices]

        shared_cache = _thread_local.conversion_cache
        for i, col in enumerate(self.src_cols):
            if col is None:
                col_array = match_density(
                    np.full((n_rows, 1), self.variables[i].Unknown)
                )
            elif not isinstance(col, Integral):
                if isinstance(col, SharedComputeValue):
                    shared = _idcache_restore(shared_cache, (col.compute_shared, source))
                    if shared is None:
                        shared = col.compute_shared(sourceri)
                        _idcache_save(shared_cache, (col.compute_shared, source), shared)
                    col_array = match_density(
                        col(sourceri, shared_data=shared))
                else:
                    col_array = match_density(col(sourceri))
            elif col < 0:
                col_array = match_density(
                    source.metas[row_indices, -1 - col]
                )
            elif col < n_src_attrs:
                col_array = match_density(X[row_indices, col])
            else:
                col_array = match_density(
                    Y[row_indices, col - n_src_attrs]
                )

            if self.is_sparse:
                # col_array should be coo matrix
                data.append(col_array.data)
                sp_col.append(np.full(len(col_array.data), i))
                sp_row.append(col_array.indices)  # row indices should be same
            else:
                out[target_indices, i] = col_array

        if self.is_sparse:
            # creating csr directly would need plenty of manual work which
            # would probably slow down the process - conversion coo to csr
            # is fast
            out = sp.coo_matrix(
                (np.hstack(data), (np.hstack(sp_row), np.hstack(sp_col))),
                shape=(n_rows, len(self.src_cols)),
                dtype=self.dtype
            )
            out = out.tocsr()

        return out


class _FromTableConversion:

    def __init__(self, source, destination):
        conversion = DomainConversion(source, destination)

        self.X = _ArrayConversion("X", conversion.attributes,
                                  destination.attributes, conversion.sparse_X,
                                  source)
        self.Y = _ArrayConversion("Y", conversion.class_vars,
                                  destination.class_vars, conversion.sparse_Y,
                                  source)
        self.metas = _ArrayConversion("metas", conversion.metas,
                                      destination.metas, conversion.sparse_metas,
                                      source)

        self.subarray = []
        self.columnwise = []

        for part in [self.X, self.Y, self.metas]:
            if part.subarray_from is None:
                self.columnwise.append(part)
            else:
                self.subarray.append(part)


# noinspection PyPep8Naming
class Table(Sequence, Storage):

    LOCKING = None
    """ If the class attribute LOCKING is True, tables will throw exceptions
    on in-place modifications unless unlocked explicitly. LOCKING is supposed
    to be set to True for testing to help us find bugs. If set to False
    or None, no safeguards are in place. Two different values are used for
    the same behaviour to distinguish the unchanged default (None) form
    explicit deactivation (False) that some add-ons might need. """

    __file__ = None
    name = "untitled"

    domain = Domain([])
    _X = _Y = _metas = _W = np.zeros((0, 0))  # pylint: disable=invalid-name
    ids = np.zeros(0)
    ids.setflags(write=False)
    attributes = frozendict()

    _Unlocked_X_val, _Unlocked_Y_val, _Unlocked_metas_val, _Unlocked_W_val = 1, 2, 4, 8
    _Unlocked_X_ref, _Unlocked_Y_ref, _Unlocked_metas_ref, _Unlocked_W_ref = 16, 32, 64, 128
    _unlocked = 0xff  # pylint: disable=invalid-name

    @property
    def columns(self):
        """
        A class whose attributes contain attribute descriptors for columns.
        For a table `table`, setting `c = table.columns` will allow accessing
        the table's variables with, for instance `c.gender`, `c.age` ets.
        Spaces are replaced with underscores.
        """
        return Columns(self.domain)

    _next_instance_id = 0
    _next_instance_lock = Lock()

    def _check_unlocked(self, partflag):
        if not self._unlocked & partflag:
            raise ValueError("Table is read-only unless unlocked")

    @property
    def X(self):  # pylint: disable=invalid-name
        return self._X

    @X.setter
    def X(self, value):
        self._check_unlocked(self._Unlocked_X_ref)
        self._X = _dereferenced(value)
        self._update_locks()

    @property
    def Y(self):  # pylint: disable=invalid-name
        return self._Y

    @Y.setter
    def Y(self, value):
        self._check_unlocked(self._Unlocked_Y_ref)
        if sp.issparse(value) and len(self) != value.shape[0]:
            value = value.T
        if sp.issparse(value):
            value = _dereferenced(value.toarray())
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0].copy()  # no views!
        self._Y = value
        self._update_locks()

    @property
    def metas(self):
        return self._metas

    @metas.setter
    def metas(self, value):
        self._check_unlocked(self._Unlocked_metas_ref)
        self._metas = _dereferenced(value)
        self._update_locks()

    @property
    def W(self):  # pylint: disable=invalid-name
        return self._W

    @W.setter
    def W(self, value):
        self._check_unlocked(self._Unlocked_W_ref)
        self._W = value
        self._update_locks()

    def __setstate__(self, state):
        # Backward compatibility with pickles before table locking

        def no_view(x):
            # Some arrays can be unpickled as views; ensure they are not
            if isinstance(x, np.ndarray) and x.base is not None:
                return x.copy()
            return x

        self._initialize_unlocked()  # __dict__ seems to be cleared before calling __setstate__
        with self.unlocked_reference():
            for k in ("X", "W", "metas"):
                if k in state:
                    setattr(self, k, no_view(state.pop(k)))
            if "_Y" in state:
                setattr(self, "Y", no_view(state.pop("_Y")))  # state["_Y"] is a 2d array
            self.__dict__.update(state)

    def __getstate__(self):
        # Compatibility with pickles before table locking:
        # return the same state as before table lock
        state = self.__dict__.copy()
        for k in ["X", "metas", "W"]:
            if "_" + k in state:  # Check existence; SQL tables do not contain them
                state[k] = state.pop("_" + k)
        # before locking, _Y was always a 2d array: save it as such
        if "_Y" in state:
            y = state.pop("_Y")
            y2d = y.reshape(-1, 1) if y.ndim == 1 else y
            state["_Y"] = y2d
        state.pop("_unlocked", None)
        return state

    def _lock_parts_val(self):
        return ((self._X, self._Unlocked_X_val, "X"),
                (self._Y, self._Unlocked_Y_val, "Y"),
                (self._metas, self._Unlocked_metas_val, "metas"),
                (self._W, self._Unlocked_W_val, "weights"))

    def _lock_parts_ref(self):
        return ((self._X, self._Unlocked_X_ref, "X"),
                (self._Y, self._Unlocked_Y_ref, "Y"),
                (self._metas, self._Unlocked_metas_ref, "metas"),
                (self._W, self._Unlocked_W_ref, "weights"))

    def _initialize_unlocked(self):
        if Table.LOCKING:
            self._unlocked = 0
        else:
            self._unlocked = sum(f for _, f, _ in (self._lock_parts_val() + self._lock_parts_ref()))

    def _update_locks(self, force=False, lock_bases=()):
        if not Table.LOCKING:
            return

        def sync(*xs):
            for x in xs:
                # no need to make empty arrays writable, as nothing can get written
                if writeable and x.size == 0:
                    continue
                try:
                    undo_on_fail.append((x, x.flags.writeable))
                    x.flags.writeable = writeable
                except ValueError:
                    if force \
                            and writeable \
                            and x.base is not None \
                            and not x.base.flags.writeable:
                        x.base.flags.writeable = writeable
                        x.flags.writeable = writeable
                        forced_bases.append(x.base)
                    else:
                        raise

        forced_bases = []
        undo_on_fail = []
        for base in lock_bases:
            base.flags.writeable = False
        try:
            for part, flag, _ in self._lock_parts_val():
                if part is None:
                    continue
                writeable = bool(self._unlocked & flag)
                if sp.isspmatrix_csr(part) or sp.isspmatrix_csc(part):
                    sync(part.data, part.indices, part.indptr)
                elif sp.isspmatrix_coo(part):
                    sync(part.data, part.row, part.col)
                elif sp.issparse(part):
                    raise ValueError("Unsupported sparse data type")
                else:
                    sync(part)
        except:
            for part, flag in undo_on_fail:
                part.flags.writeable = flag
            raise
        return tuple(forced_bases)

    def __unlocked(self, *parts, force=False, reference_only=False):
        prev_state = self._unlocked
        if reference_only:
            lock_parts = self._lock_parts_ref()
        else:
            lock_parts = self._lock_parts_val() + self._lock_parts_ref()
        for part, flag, _ in lock_parts:
            if not parts or any(ppart is part for ppart in parts):
                self._unlocked |= flag
        try:
            forced_bases = self._update_locks(force)
            yield
        finally:
            self._unlocked = prev_state
            self._update_locks(lock_bases=forced_bases)

    def force_unlocked(self, *parts):
        """
        Unlocking without any checks.

        Use with extreme caution. This is meant primarily for 3rd party
        functions in Cython that expect read-write buffer, but do not
        actually modify it. the given parts (default: all parts) of the table.

        The function will still fail to unlock and raise an exception if the
        table contains view to another table.
        """
        return contextmanager(self.__unlocked)(*parts, force=True)

    def unlocked_reference(self, *parts):
        """
        Unlock references to the given parts (default: all parts) of the table.

        The caller must ensure that the table is safe to modify.
        """
        return contextmanager(self.__unlocked)(*parts, reference_only=True)

    def unlocked(self, *parts):
        """
        Unlock the given parts (default: all parts) of the table.

        The caller must ensure that the table is safe to modify. The function
        will raise an exception if the table contains view to other table.
        """
        def can_unlock(x):
            if sp.issparse(x):
                return can_unlock(x.data)
            return x.flags.writeable or x.flags.owndata or x.size == 0

        for part, flag, name in self._lock_parts_val():
            if not flag & self._unlocked \
                    and (not parts or any(ppart is part for ppart in parts)) \
                    and part is not None and not can_unlock(part):
                raise ValueError(f"'{name}' is a view into another table "
                                 "and cannot be unlocked")
        return contextmanager(self.__unlocked)(*parts)

    def __new__(cls, *args, **kwargs):
        def warn_deprecated(method):
            warnings.warn("Direct calls to Table's constructor are deprecated "
                          "and will be removed. Replace this call with "
                          f"Table.{method}", OrangeDeprecationWarning,
                          stacklevel=3)

        if not args:
            if not kwargs:
                return super().__new__(cls)
            else:
                raise TypeError("Table() must not be called directly")

        if isinstance(args[0], str):
            if len(args) > 1:
                raise TypeError("Table(name: str) expects just one argument")
            if args[0].startswith('https://') or args[0].startswith('http://'):
                return cls.from_url(args[0], **kwargs)
            else:
                return cls.from_file(args[0], **kwargs)

        elif isinstance(args[0], Table):
            if len(args) > 1:
                raise TypeError("Table(table: Table) expects just one argument")
            return cls.from_table(args[0].domain, args[0], **kwargs)
        elif isinstance(args[0], Domain):
            domain, args = args[0], args[1:]
            if not args:
                warn_deprecated("from_domain")
                return cls.from_domain(domain, **kwargs)
            if isinstance(args[0], Table):
                warn_deprecated("from_table")
                return cls.from_table(domain, *args, **kwargs)
            elif isinstance(args[0], list):
                warn_deprecated("from_list")
                return cls.from_list(domain, *args, **kwargs)
        else:
            warnings.warn("Omitting domain in a call to Table(X, Y, metas), is "
                          "deprecated and will be removed. "
                          "Call Table.from_numpy(None, X, Y, metas) instead.",
                          OrangeDeprecationWarning, stacklevel=2)
            domain = None

        return cls.from_numpy(domain, *args, **kwargs)

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._initialize_unlocked()
        self._update_locks()

    @classmethod
    def from_domain(cls, domain, n_rows=0, weights=False):
        """
        Construct a new `Table` with the given number of rows for the given
        domain. The optional vector of weights is initialized to 1's.

        :param domain: domain for the `Table`
        :type domain: Orange.data.Domain
        :param n_rows: number of rows in the new table
        :type n_rows: int
        :param weights: indicates whether to construct a vector of weights
        :type weights: bool
        :return: a new table
        :rtype: Orange.data.Table
        """
        self = cls()
        self.domain = domain
        self.n_rows = n_rows
        with self.unlocked():
            self.X = np.zeros((n_rows, len(domain.attributes)))
            if len(domain.class_vars) != 1:
                self.Y = np.zeros((n_rows, len(domain.class_vars)))
            else:
                self.Y = np.zeros(n_rows)
            if weights:
                self.W = np.ones(n_rows)
            else:
                self.W = np.empty((n_rows, 0))
            self.metas = np.empty((n_rows, len(self.domain.metas)), object)
            cls._init_ids(self)
            self.attributes = {}
        return self

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        """
        Create a new table from selected columns and/or rows of an existing
        one. The columns are chosen using a domain. The domain may also include
        variables that do not appear in the source table; they are computed
        from source variables if possible.

        The resulting data may be a view or a copy of the existing data.

        :param domain: the domain for the new table
        :type domain: Orange.data.Domain
        :param source: the source table
        :type source: Orange.data.Table
        :param row_indices: indices of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """

        PART = 5000

        new_cache = _thread_local.conversion_cache is None
        try:
            if new_cache:
                _thread_local.conversion_cache = {}
                _thread_local.domain_cache = {}
            else:
                cached = _idcache_restore(_thread_local.conversion_cache, (domain, source))
                if cached is not None:
                    return cached
            if domain is source.domain:
                table = cls.from_table_rows(source, row_indices)
                # assure resulting domain is the instance passed on input
                table.domain = domain
                # since sparse flags are not considered when checking for
                # domain equality, fix manually.
                with table.unlocked_reference():
                    table = assure_domain_conversion_sparsity(table, source)
                return table

            if row_indices is ...:
                n_rows = len(source)
            elif isinstance(row_indices, slice):
                row_indices_range = range(*row_indices.indices(source.X.shape[0]))
                n_rows = len(row_indices_range)
            else:
                n_rows = len(row_indices)

            self = cls()
            self.domain = domain

            table_conversion = \
                _idcache_restore(_thread_local.domain_cache, (domain, source.domain))
            if table_conversion is None:
                table_conversion = _FromTableConversion(source.domain, domain)
                _idcache_save(_thread_local.domain_cache, (domain, source.domain),
                              table_conversion)

            # if an array can be a subarray of the input table, this needs to be done
            # on the whole table, because this avoids needless copies of contents

            with self.unlocked_reference():
                for array_conv in table_conversion.subarray:
                    out = array_conv.get_subarray(source, row_indices, n_rows)
                    setattr(self, array_conv.target, out)

                parts = {}

                for array_conv in table_conversion.columnwise:
                    if array_conv.is_sparse:
                        parts[array_conv.target] = []
                    else:
                        # F-order enables faster writing to the array while accessing and
                        # matrix operations work with same speed (e.g. dot)
                        parts[array_conv.target] = \
                            np.zeros((n_rows, len(array_conv.src_cols)),
                                     order="F", dtype=array_conv.dtype)

                if n_rows <= PART:
                    for array_conv in table_conversion.columnwise:
                        out = array_conv.get_columns(source, row_indices, n_rows,
                                                     parts[array_conv.target],
                                                     ...)
                        setattr(self, array_conv.target, out)
                else:
                    i_done = 0

                    while i_done < n_rows:
                        target_indices = slice(i_done, min(n_rows, i_done + PART))
                        if row_indices is ...:
                            source_indices = target_indices
                        elif isinstance(row_indices, slice):
                            r = row_indices_range[target_indices]
                            source_indices = slice(r.start, r.stop, r.step)
                        else:
                            source_indices = row_indices[target_indices]
                        part_rows = min(n_rows, i_done+PART) - i_done

                        for array_conv in table_conversion.columnwise:
                            out = array_conv.get_columns(source, source_indices, part_rows,
                                                         parts[array_conv.target],
                                                         target_indices)
                            if array_conv.is_sparse:  # dense arrays are populated in-place
                                parts[array_conv.target].append(out)

                        i_done += PART

                        # clear cache after a part is done
                        if new_cache:
                            _thread_local.conversion_cache = {}

                    for array_conv in table_conversion.columnwise:
                        cparts = parts[array_conv.target]
                        out = cparts if not array_conv.is_sparse else sp.vstack(cparts)
                        setattr(self, array_conv.target, out)

                if source.has_weights():
                    self.W = source.W[row_indices]
                else:
                    self.W = np.empty((n_rows, 0))
                self.name = getattr(source, 'name', '')
                if hasattr(source, 'ids'):
                    self.ids = source.ids[row_indices]
                else:
                    cls._init_ids(self)
                self.attributes = getattr(source, 'attributes', {})
                _idcache_save(_thread_local.conversion_cache, (domain, source), self)
            return self
        finally:
            if new_cache:
                _thread_local.conversion_cache = None
                _thread_local.domain_cache = None

    def transform(self, domain):
        """
        Construct a table with a different domain.

        The new table keeps the row ids and other information. If the table
        is a subclass of :obj:`Table`, the resulting table will be of the same
        type.

        In a typical scenario, an existing table is augmented with a new
        column by ::

            domain = Domain(old_domain.attributes + [new_attribute],
                            old_domain.class_vars,
                            old_domain.metas)
            table = data.transform(domain)
            table[:, new_attribute] = new_column

        Args:
            domain (Domain): new domain

        Returns:
            A new table
        """
        return type(self).from_table(domain, self)

    @classmethod
    def from_table_rows(cls, source, row_indices):
        """
        Construct a new table by selecting rows from the source table.

        :param source: an existing table
        :type source: Orange.data.Table
        :param row_indices: indices of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """
        self = cls()
        self.domain = source.domain
        with self.unlocked_reference():
            self.X = source.X[row_indices]
            if self.X.ndim == 1:
                self.X = self.X.reshape(-1, len(self.domain.attributes))
            self.Y = source.Y[row_indices]
            self.metas = source.metas[row_indices]
            if self.metas.ndim == 1:
                self.metas = self.metas.reshape(-1, len(self.domain.metas))
            self.W = source.W[row_indices]
            self.name = getattr(source, 'name', '')
            self.ids = np.array(source.ids[row_indices])
            self.attributes = getattr(source, 'attributes', {})
        return self

    @classmethod
    def from_numpy(cls, domain, X, Y=None, metas=None, W=None,
                   attributes=None, ids=None):
        """
        Construct a table from numpy arrays with the given domain. The number
        of variables in the domain must match the number of columns in the
        corresponding arrays. All arrays must have the same number of rows.
        Arrays may be of different numpy types, and may be dense or sparse.

        :param domain: the domain for the new table
        :type domain: Orange.data.Domain
        :param X: array with attribute values
        :type X: np.array
        :param Y: array with class values
        :type Y: np.array
        :param metas: array with meta attributes
        :type metas: np.array
        :param W: array with weights
        :type W: np.array
        :return:
        """
        X, Y, W = _check_arrays(X, Y, W, dtype='float64')
        metas, = _check_arrays(metas, dtype=object, shape_1=X.shape[0])
        ids, = _check_arrays(ids, dtype=int, shape_1=X.shape[0])

        if domain is None:
            domain = Domain.from_numpy(X, Y, metas)

        if Y is None:
            if not domain.class_vars or sp.issparse(X):
                Y = np.empty((X.shape[0], 0), dtype=np.float64)
            else:
                own_data = X.flags.owndata and X.base is None
                Y = X[:, len(domain.attributes):]
                X = X[:, :len(domain.attributes)]
                if own_data:
                    Y = Y.copy()
                    X = X.copy()
        if metas is None:
            metas = np.empty((X.shape[0], 0), object)
        if W is None or W.size == 0:
            W = np.empty((X.shape[0], 0))
        elif W.shape != (W.size, ):
            W = W.reshape(W.size).copy()

        if X.shape[1] != len(domain.attributes):
            raise ValueError(
                "Invalid number of variable columns ({} != {})".format(
                    X.shape[1], len(domain.attributes))
            )
        if Y.ndim == 1:
            if not domain.class_var:
                raise ValueError(
                    "Invalid number of class columns "
                    f"(1 != {len(domain.class_vars)})")
        elif Y.shape[1] != len(domain.class_vars):
            raise ValueError(
                "Invalid number of class columns ({} != {})".format(
                    Y.shape[1], len(domain.class_vars))
            )
        if metas.shape[1] != len(domain.metas):
            raise ValueError(
                "Invalid number of meta attribute columns ({} != {})".format(
                    metas.shape[1], len(domain.metas))
            )
        if not X.shape[0] == Y.shape[0] == metas.shape[0] == W.shape[0]:
            raise ValueError(
                "Parts of data contain different numbers of rows.")

        self = cls()
        with self.unlocked_reference():
            self.domain = domain
            self.X = X
            self.Y = Y
            self.metas = metas
            self.W = W
            self.n_rows = self.X.shape[0]
            if ids is None:
                cls._init_ids(self)
            else:
                self.ids = ids
            self.attributes = {} if attributes is None else attributes
        return self

    @classmethod
    def from_list(cls, domain, rows, weights=None):
        if weights is not None and len(rows) != len(weights):
            raise ValueError("mismatching number of instances and weights")
        self = cls.from_domain(domain, len(rows), weights is not None)
        all_vars = domain.variables + domain.metas
        nattrs = len(domain.attributes)
        nattrscls = len(domain.variables)
        with self.unlocked():
            for i, row in enumerate(rows):
                if isinstance(row, Instance):
                    row = row.list
                vals = [var.to_val(val) for var, val in zip(all_vars, row)]
                if self.X.size:
                    self.X[i] = vals[:nattrs]
                if self.Y.size:
                    if self._Y.ndim == 1:
                        self._Y[i] = vals[nattrs] if nattrs < len(vals) else np.nan
                    else:
                        self._Y[i] = vals[nattrs:nattrscls]
                # for backward compatibility: allow omittine some (or all) metas
                if self.metas.size:
                    self.metas[i, :len(vals) - nattrscls] = vals[nattrscls:]
            if weights is not None:
                self.W = np.array(weights)
            self.attributes = {}
        return self

    @classmethod
    def _init_ids(cls, obj):
        with cls._next_instance_lock:
            obj.ids = np.array(range(cls._next_instance_id, cls._next_instance_id + obj.X.shape[0]))
            cls._next_instance_id += obj.X.shape[0]

    @classmethod
    def new_id(cls):
        with cls._next_instance_lock:
            id = cls._next_instance_id
            cls._next_instance_id += 1
            return id

    def to_pandas_dfs(self):
        return Orange.data.pandas_compat.table_to_frames(self)

    @staticmethod
    def from_pandas_dfs(xdf, ydf, mdf):
        return Orange.data.pandas_compat.table_from_frames(xdf, ydf, mdf)

    @property
    def X_df(self):
        return Orange.data.pandas_compat.OrangeDataFrame(
            self, orange_role=Role.Attribute
        )

    @X_df.setter
    def X_df(self, df):
        Orange.data.pandas_compat.amend_table_with_frame(
            self, df, role=Role.Attribute
        )

    @property
    def Y_df(self):
        return Orange.data.pandas_compat.OrangeDataFrame(
            self, orange_role=Role.ClassAttribute
        )

    @Y_df.setter
    def Y_df(self, df):
        Orange.data.pandas_compat.amend_table_with_frame(
            self, df, role=Role.ClassAttribute
        )

    @property
    def metas_df(self):
        return Orange.data.pandas_compat.OrangeDataFrame(
            self, orange_role=Role.Meta
        )

    @metas_df.setter
    def metas_df(self, df):
        Orange.data.pandas_compat.amend_table_with_frame(
            self, df, role=Role.Meta
        )

    def save(self, filename):
        """
        Save a data table to a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        """
        ext = os.path.splitext(filename)[1]
        from Orange.data.io import FileFormat
        writer = FileFormat.writers.get(ext)
        if not writer:
            desc = FileFormat.names.get(ext)
            if desc:
                raise IOError(
                    "Writing of {}s is not supported".format(desc.lower()))
            else:
                raise IOError("Unknown file name extension.")
        writer.write_file(filename, self)

    @classmethod
    def from_file(cls, filename, sheet=None):
        """
        Read a data table from a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        :param sheet: Sheet in a file (optional)
        :type sheet: str
        :return: a new data table
        :rtype: Orange.data.Table
        """
        from Orange.data.io import FileFormat

        absolute_filename = FileFormat.locate(filename, dataset_dirs)
        reader = FileFormat.get_reader(absolute_filename)
        reader.select_sheet(sheet)
        data = reader.read()

        # no need to call _init_ids as fuctions from .io already
        # construct a table with .ids

        data.__file__ = absolute_filename
        return data

    @classmethod
    def from_url(cls, url):
        from Orange.data.io import UrlReader
        reader = UrlReader(url)
        data = reader.read()
        return data

    # Helper function for __setitem__:
    # Set the row of table data matrices
    # noinspection PyProtectedMember
    def _set_row(self, example, row):
        # pylint: disable=protected-access
        domain = self.domain
        if isinstance(example, Instance):
            if example.domain == domain:
                self.X[row] = example._x
                if self._Y.ndim == 1:
                    self._Y[row] = float(example._y)
                else:
                    self._Y[row] = np.atleast_1d(example._y)
                self.metas[row] = example._metas
                return

            self.X[row], self._Y[row], self.metas[row] = \
                self.domain.convert(example)
            try:
                self.ids[row] = example.id
            except:
                with type(self)._next_instance_lock:
                    self.ids[row] = type(self)._next_instance_id
                    type(self)._next_instance_id += 1

        else:
            attrs = domain.attributes
            if len(example) != len(domain.variables):
                raise ValueError("invalid length")
            if self._X.size:
                self._X[row] = [var.to_val(val) for var, val in zip(attrs, example)]
            if self._Y.size:
                if self._Y.ndim == 1:
                    self._Y[row] = domain.class_var.to_val(example[len(attrs)])
                else:
                    self._Y[row] = [var.to_val(val)
                                    for var, val in zip(domain.class_vars,
                                                        example[len(attrs):])]
            if self._metas.size:
                self.metas[row] = np.array([var.Unknown for var in domain.metas],
                                       dtype=object)

    def _check_all_dense(self):
        return all(x in (Storage.DENSE, Storage.MISSING)
                   for x in (self.X_density(), self.Y_density(),
                             self.metas_density()))

    def __getitem__(self, key):
        if isinstance(key, Integral):
            return RowInstance(self, key)
        if not isinstance(key, tuple):
            return self.from_table_rows(self, key)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, Integral):
            if isinstance(col_idx, (str, Integral, Variable)):
                col_idx = self.domain.index(col_idx)
                var = self.domain[col_idx]
                if 0 <= col_idx < len(self.domain.attributes):
                    val = self.X[row_idx, col_idx]
                elif col_idx == len(self.domain.attributes) and self._Y.ndim == 1:
                    val = self._Y[row_idx]
                elif col_idx >= len(self.domain.attributes):
                    val = self._Y[row_idx,
                                  col_idx - len(self.domain.attributes)]
                else:
                    val = self.metas[row_idx, -1 - col_idx]
                if isinstance(col_idx, DiscreteVariable) and var is not col_idx:
                    val = col_idx.get_mapper_from(var)(val)
                return Value(var, val)
            else:
                row_idx = [row_idx]

        # multiple rows OR single row but multiple columns:
        # construct a new table
        attributes, col_indices = self.domain._compute_col_indices(col_idx)
        if attributes is not None:
            n_attrs = len(self.domain.attributes)
            r_attrs = [attributes[i]
                       for i, col in enumerate(col_indices)
                       if 0 <= col < n_attrs]
            r_classes = [attributes[i]
                         for i, col in enumerate(col_indices)
                         if col >= n_attrs]
            r_metas = [attributes[i]
                       for i, col in enumerate(col_indices) if col < 0]
            domain = Domain(r_attrs, r_classes, r_metas)
        else:
            domain = self.domain
        return self.from_table(domain, self, row_idx)

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            if isinstance(value, Real):
                self.X[key, :] = value
                return
            self._set_row(value, key)
            return

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")
        row_idx, col_idx = key

        # single row
        if isinstance(row_idx, Integral):
            if isinstance(col_idx, slice):
                col_idx = range(*slice.indices(col_idx, self.X.shape[1]))
            if not isinstance(col_idx, str) and isinstance(col_idx, Iterable):
                col_idx = list(col_idx)
            if not isinstance(col_idx, str) and isinstance(col_idx, Sized):
                if isinstance(value, (Sequence, np.ndarray)):
                    values = value
                elif isinstance(value, Iterable):
                    values = list(value)
                else:
                    raise TypeError("Setting multiple values requires a "
                                    "sequence or numpy array")
                if len(values) != len(col_idx):
                    raise ValueError("Invalid number of values")
            else:
                col_idx, values = [col_idx], [value]
            if isinstance(col_idx, DiscreteVariable) \
                    and self.domain[col_idx] != col_idx:
                values = self.domain[col_idx].get_mapper_from(col_idx)(values)
            for val, col_idx in zip(values, col_idx):
                if not isinstance(val, Integral):
                    val = self.domain[col_idx].to_val(val)
                if not isinstance(col_idx, Integral):
                    col_idx = self.domain.index(col_idx)
                if col_idx >= 0:
                    if col_idx < self.X.shape[1]:
                        self.X[row_idx, col_idx] = val
                    elif self._Y.ndim == 1 and col_idx == self.X.shape[1]:
                        self._Y[row_idx] = val
                    else:
                        self._Y[row_idx, col_idx - self.X.shape[1]] = val
                else:
                    self.metas[row_idx, -1 - col_idx] = val

        # multiple rows, multiple columns
        attributes, col_indices = self.domain._compute_col_indices(col_idx)
        if col_indices is ...:
            col_indices = range(len(self.domain.variables))
        n_attrs = self.X.shape[1]
        if isinstance(value, str):
            if not attributes:
                attributes = self.domain.attributes
            for var, col in zip(attributes, col_indices):
                val = var.to_val(value)
                if 0 <= col < n_attrs:
                    self.X[row_idx, col] = val
                elif col >= n_attrs:
                    if self._Y.ndim == 1 and col == n_attrs:
                        self._Y[row_idx] = val
                    else:
                        self._Y[row_idx, col - n_attrs] = val
                else:
                    self.metas[row_idx, -1 - col] = val
        else:
            attr_cols = np.fromiter(
                (col for col in col_indices if 0 <= col < n_attrs), int)
            class_cols = np.fromiter(
                (col - n_attrs for col in col_indices if col >= n_attrs), int)
            meta_cols = np.fromiter(
                (-1 - col for col in col_indices if col < 0), int)
            if value is None:
                value = Unknown

            if not isinstance(value, (Real, np.ndarray)) and \
                    (len(attr_cols) or len(class_cols)):
                raise TypeError(
                    "Ordinary attributes can only have primitive values")
            if len(attr_cols):
                if self.X.size:
                    self.X[row_idx, attr_cols] = value
            if len(class_cols):
                if self._Y.size:
                    if self._Y.ndim == 1 and np.all(class_cols == 0):
                        if isinstance(value, np.ndarray):
                            yshape = self._Y[row_idx].shape
                            if value.shape != yshape:
                                value = value.reshape(yshape)
                        self._Y[row_idx] = value
                    else:
                        self._Y[row_idx, class_cols] = value
            if len(meta_cols):
                if self._metas.size:
                    self.metas[row_idx, meta_cols] = value

    def __len__(self):
        return self.X.shape[0]

    def __str__(self):
        return "[" + ",\n ".join(str(ex) for ex in self) + "]"

    def __repr__(self):
        head = 5
        if self.is_sparse():
            head = min(self.X.shape[0], head)
        s = "[" + ",\n ".join(repr(ex) for ex in self[:head])
        if len(self) > head:
            s += ",\n ..."
        s += "\n]"
        return s

    @classmethod
    def concatenate(cls, tables, axis=0):
        """
        Concatenate tables into a new table, either vertically or horizontally.

        If axis=0 (vertical concatenate), all tables must have the same domain.

        If axis=1 (horizontal),
        - all variable names must be unique.
        - ids are copied from the first table.
        - weights are copied from the first table in which they are defined.
        - the dictionary of table's attributes are merged. If the same attribute
          appears in multiple dictionaries, the earlier are used.

        Args:
            tables (Table): tables to be joined

        Returns:
            table (Table)
        """
        if axis not in (0, 1):
            raise ValueError("invalid axis")
        if not tables:
            raise ValueError('need at least one table to concatenate')

        if len(tables) == 1:
            return tables[0].copy()

        if axis == 0:
            conc = cls._concatenate_vertical(tables)
        else:
            conc = cls._concatenate_horizontal(tables)

        # TODO: Add attributes = {} to __init__
        conc.attributes = getattr(conc, "attributes", {})
        for table in reversed(tables):
            conc.attributes.update(table.attributes)

        names = [table.name for table in tables if table.name != "untitled"]
        if names:
            conc.name = names[0]
        return conc

    @classmethod
    def _concatenate_vertical(cls, tables):
        def vstack(arrs):
            return [np, sp][any(sp.issparse(arr) for arr in arrs)].vstack(arrs)

        def merge1d(arrs):
            arrs = list(arrs)
            ydims = {arr.ndim for arr in arrs}
            if ydims == {1}:
                return np.hstack(arrs)
            else:
                return vstack([
                    arr if arr.ndim == 2 else np.atleast_2d(arr).T
                    for arr in arrs
                ])

        def collect(attr):
            return [getattr(arr, attr) for arr in tables]

        domain = tables[0].domain
        if any(table.domain != domain for table in tables):
            raise ValueError('concatenated tables must have the same domain')

        conc = cls.from_numpy(
            domain,
            vstack(collect("X")),
            merge1d(collect("Y")),
            vstack(collect("metas")),
            merge1d(collect("W"))
        )
        conc.ids = np.hstack([t.ids for t in tables])
        return conc

    @classmethod
    def _concatenate_horizontal(cls, tables):
        """
        """
        def all_of(objs, names):
            return (tuple(getattr(obj, name) for obj in objs)
                    for name in names)

        def stack(arrs):
            non_empty = tuple(arr if arr.ndim == 2 else arr[:, np.newaxis]
                              for arr in arrs
                              if arr is not None and arr.size > 0)
            return np.hstack(non_empty) if non_empty else None

        doms, Ws = all_of(tables, ("domain", "W"))
        Xs, Ys, Ms = map(stack, all_of(tables, ("X", "Y", "metas")))
        # pylint: disable=undefined-loop-variable
        for W in Ws:
            if W.size:
                break

        parts = all_of(doms, ("attributes", "class_vars", "metas"))
        domain = Domain(*(tuple(chain(*lst)) for lst in parts))
        return cls.from_numpy(domain, Xs, Ys, Ms, W, ids=tables[0].ids)

    def add_column(self, variable, data, to_metas=None):
        """
        Create a new table with an additional column

        Column's name must be unique

        Args:
            variable (Variable): variable for the new column
            data (np.ndarray): data for the new column
            to_metas (bool, optional): if `True` the column is added as meta
                column. Otherwise, primitive variables are added to attributes
                and non-primitive to metas.

        Returns:
            table (Table): a new table with the additional column
        """
        dom = self.domain
        attrs, classes, metavars = dom.attributes, dom.class_vars, dom.metas
        to_metas = to_metas or not variable.is_primitive()
        if to_metas:
            metavars += (variable, )
        else:
            attrs += (variable, )
        domain = Domain(attrs, classes, metavars)
        new_table = self.transform(domain)
        with new_table.unlocked(new_table.metas if to_metas else new_table.X):
            new_table.get_column_view(variable)[0][:] = data
        return new_table

    @deprecated("array.base is not None for each subarray of Orange.data.Table (i.e. X, Y, W, metas)")
    def is_view(self):
        """
        Return `True` if all arrays represent a view referring to another table
        """
        return ((not self._X.shape[-1] or self._X.base is not None) and
                (not self._Y.shape[-1] or self._Y.base is not None) and
                (not self._metas.shape[-1] or self._metas.base is not None) and
                (not self._W.shape[-1] or self._W.base is not None))

    @deprecated("array.base is None for each subarray of Orange.data.Table (i.e. X, Y, W, metas)")
    def is_copy(self):
        """
        Return `True` if the table owns its data
        """
        return ((not self._X.shape[-1] or self._X.base is None) and
                (self._Y.base is None) and
                (self._metas.base is None) and
                (self._W.base is None))

    def is_sparse(self):
        """
        Return `True` if the table stores data in sparse format
        """
        return any(sp.issparse(i) for i in [self._X, self._Y, self._metas])

    def ensure_copy(self):
        """
        Ensure that the table owns its data; copy arrays when necessary.
        """

        def is_view(x):
            if not sp.issparse(x):
                return x.base is not None
            else:
                return x.data.base is not None

        if is_view(self._X):
            self._X = self._X.copy()
        if is_view(self._Y):
            self._Y = self._Y.copy()
        if is_view(self._metas):
            self._metas = self._metas.copy()
        if is_view(self._W):
            self._W = self._W.copy()

    def copy(self):
        """
        Return a copy of the table
        """
        t = self.__class__(self)
        t.ensure_copy()
        return t

    @staticmethod
    def __determine_density(data):
        if data is None:
            return Storage.Missing
        if data is not None and sp.issparse(data):
            return Storage.SPARSE_BOOL if (data.data == 1).all() else Storage.SPARSE
        else:
            return Storage.DENSE

    def X_density(self):
        if not hasattr(self, "_X_density"):
            self._X_density = self.__determine_density(self.X)
        return self._X_density

    def Y_density(self):
        if not hasattr(self, "_Y_density"):
            self._Y_density = self.__determine_density(self._Y)
        return self._Y_density

    def metas_density(self):
        if not hasattr(self, "_metas_density"):
            self._metas_density = self.__determine_density(self.metas)
        return self._metas_density

    def set_weights(self, weight=1):
        """
        Set weights of data instances; create a vector of weights if necessary.
        """
        if not self.W.shape[-1]:
            self.W = np.empty(len(self))
        self.W[:] = weight

    def has_weights(self):
        """Return `True` if the data instances are weighed. """
        return self.W.shape[-1] != 0

    def total_weight(self):
        """
        Return the total weight of instances in the table, or their number if
        they are unweighted.
        """
        if self.W.shape[-1]:
            return sum(self.W)
        return len(self)

    def has_missing(self):
        """Return `True` if there are any missing attribute or class values."""
        missing_x = not sp.issparse(self.X) and bn.anynan(self.X)  # do not check for sparse X
        return missing_x or bn.anynan(self._Y)

    def has_missing_attribute(self):
        """Return `True` if there are any missing attribute values."""
        return not sp.issparse(self.X) and bn.anynan(self.X)  # do not check for sparse X

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return bn.anynan(self._Y)

    def get_nan_frequency_attribute(self):
        if self.X.size == 0:
            return 0
        return np.isnan(self.X).sum() / self.X.size

    def get_nan_frequency_class(self):
        if self.Y.size == 0:
            return 0
        return np.isnan(self._Y).sum() / self._Y.size

    def checksum(self, include_metas=True):
        # TODO: zlib.adler32 does not work for numpy arrays with dtype object
        # (after pickling and unpickling such arrays, checksum changes)
        # Why, and should we fix it or remove it?
        """Return a checksum over X, Y, metas and W."""
        cs = zlib.adler32(np.ascontiguousarray(self._X))
        cs = zlib.adler32(np.ascontiguousarray(self._Y), cs)
        if include_metas:
            cs = zlib.adler32(np.ascontiguousarray(self._metas), cs)
        cs = zlib.adler32(np.ascontiguousarray(self._W), cs)
        return cs

    def shuffle(self):
        """Randomly shuffle the rows of the table."""
        if not self._check_all_dense():
            raise ValueError("Rows of sparse data cannot be shuffled")
        ind = np.arange(self.X.shape[0])
        np.random.shuffle(ind)
        self.X = self.X[ind]
        self._Y = self._Y[ind]
        self.metas = self.metas[ind]
        self.W = self.W[ind]
        self.ids = self.ids[ind]

    def get_column_view(self, index):
        """
        Return a vector - as a view, not a copy - with a column of the table,
        and a bool flag telling whether this column is sparse. Note that
        vertical slicing of sparse matrices is inefficient.

        :param index: the index of the column
        :type index: int, str or Orange.data.Variable
        :return: (one-dimensional numpy array, sparse)
        """

        def rx(M):
            if sp.issparse(M):
                return np.asarray(M.todense())[:, 0], True
            else:
                return M, False

        if isinstance(index, Integral):
            col_index = index
        else:
            col_index = self.domain.index(index)
        if col_index >= 0:
            if col_index < self.X.shape[1]:
                col = rx(self.X[:, col_index])
            elif self._Y.ndim == 1 and col_index == self._X.shape[1]:
                col = rx(self._Y)
            else:
                col = rx(self._Y[:, col_index - self.X.shape[1]])
        else:
            col = rx(self.metas[:, -1 - col_index])

        if isinstance(index, DiscreteVariable) \
                and index.values != self.domain[col_index].values:
            col = index.get_mapper_from(self.domain[col_index])(col[0]), col[1]
            col[0].flags.writeable = False
        return col

    def _filter_is_defined(self, columns=None, negate=False):
        # structure of function is obvious; pylint: disable=too-many-branches
        def _sp_anynan(a):
            return a.indptr[1:] != a[-1:] + a.shape[1]

        if columns is None:
            if sp.issparse(self.X):
                remove = _sp_anynan(self.X)
            else:
                remove = bn.anynan(self.X, axis=1)
            if sp.issparse(self._Y):
                remove += _sp_anynan(self._Y)
            else:
                if self._Y.ndim == 1:
                    remove += np.isnan(self._Y)
                else:
                    remove += bn.anynan(self._Y, axis=1)
            if sp.issparse(self.metas):
                remove += _sp_anynan(self._metas)
            else:
                for i, var in enumerate(self.domain.metas):
                    col = self.metas[:, i].flatten()
                    if var.is_primitive():
                        remove += np.isnan(col.astype(float))
                    else:
                        remove += ~col.astype(bool)
        else:
            remove = np.zeros(len(self), dtype=bool)
            for column in columns:
                col, sparse = self.get_column_view(column)
                if sparse:
                    remove += col == 0
                elif self.domain[column].is_primitive():
                    remove += bn.anynan([col.astype(float)], axis=0)
                else:
                    remove += col.astype(bool)
        retain = remove if negate else np.logical_not(remove)
        return self.from_table_rows(self, retain)

    def _filter_has_class(self, negate=False):
        if sp.issparse(self._Y):
            if negate:
                retain = (self._Y.indptr[1:] !=
                          self._Y.indptr[-1:] + self._Y.shape[1])
            else:
                retain = (self._Y.indptr[1:] ==
                          self._Y.indptr[-1:] + self._Y.shape[1])
        else:
            if self._Y.ndim == 1:
                retain = np.isnan(self._Y)
            else:
                retain = bn.anynan(self._Y, axis=1)
            if not negate:
                retain = np.logical_not(retain)
        return self.from_table_rows(self, retain)

    def _filter_same_value(self, column, value, negate=False):
        if not isinstance(value, Real):
            value = self.domain[column].to_val(value)
        sel = self.get_column_view(column)[0] == value
        if negate:
            sel = np.logical_not(sel)
        return self.from_table_rows(self, sel)

    def _filter_values(self, filter):
        selection = self._values_filter_to_indicator(filter)
        return self.from_table(self.domain, self, selection)

    def _values_filter_to_indicator(self, filter):
        """Return selection of rows matching the filter conditions

        Handles conjunction/disjunction and negate modifiers

        Parameters
        ----------
        filter: Values object containing the conditions

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        from Orange.data.filter import Values

        if isinstance(filter, Values):
            conditions = filter.conditions
            conjunction = filter.conjunction
        else:
            conditions = [filter]
            conjunction = True
        if conjunction:
            sel = np.ones(len(self), dtype=bool)
        else:
            sel = np.zeros(len(self), dtype=bool)

        for f in conditions:
            selection = self._filter_to_indicator(f)

            if conjunction:
                sel *= selection
            else:
                sel += selection

        if filter.negate:
            sel = ~sel
        return sel

    def _filter_to_indicator(self, filter):
        """Return selection of rows that match the condition.

        Parameters
        ----------
        filter: ValueFilter describing the condition

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        from Orange.data.filter import (
            FilterContinuous, FilterDiscrete, FilterRegex, FilterString,
            FilterStringList, IsDefined, Values
        )
        if isinstance(filter, Values):
            return self._values_filter_to_indicator(filter)

        def get_col_indices():
            cols = chain(self.domain.variables, self.domain.metas)
            if isinstance(filter, IsDefined):
                if filter.columns is not None:
                    return list(filter.columns)
                else:
                    return list(cols)

            if filter.column is not None:
                return [filter.column]

            if isinstance(filter, FilterDiscrete):
                raise ValueError("Discrete filter can't be applied across rows")
            if isinstance(filter, FilterContinuous):
                return [col for col in cols if col.is_continuous]
            if isinstance(filter,
                          (FilterString, FilterStringList, FilterRegex)):
                return [col for col in cols if col.is_string]
            raise TypeError("Invalid filter")

        def col_filter(col_idx):
            col = self.get_column_view(col_idx)[0]
            if isinstance(filter, IsDefined):
                if self.domain[col_idx].is_primitive():
                    return ~np.isnan(col.astype(float))
                else:
                    return col.astype(bool)
            if isinstance(filter, FilterDiscrete):
                return self._discrete_filter_to_indicator(filter, col)
            if isinstance(filter, FilterContinuous):
                return self._continuous_filter_to_indicator(filter, col)
            if isinstance(filter, FilterString):
                return self._string_filter_to_indicator(filter, col)
            if isinstance(filter, FilterStringList):
                if not filter.case_sensitive:
                    col = np.char.lower(np.array(col, dtype=str))
                    vals = [val.lower() for val in filter.values]
                else:
                    vals = filter.values
                return reduce(operator.add, (col == val for val in vals))
            if isinstance(filter, FilterRegex):
                return np.vectorize(filter)(col)
            raise TypeError("Invalid filter")

        col_indices = get_col_indices()
        if len(col_indices) == 1:
            sel = col_filter(col_indices[0])
        else:
            sel = np.ones(len(self), dtype=bool)
            for col_idx in col_indices:
                sel *= col_filter(col_idx)

        if isinstance(filter, IsDefined) and filter.negate:
            sel = ~sel
        return sel

    def _discrete_filter_to_indicator(self, filter, col):
        """Return selection of rows matched by the given discrete filter.

        Parameters
        ----------
        filter: FilterDiscrete
        col: np.ndarray

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        if filter.values is None:  # <- is defined filter
            col = col.astype(float)
            return ~np.isnan(col)

        sel = np.zeros(len(self), dtype=bool)
        for val in filter.values:
            if not isinstance(val, Real):
                val = self.domain[filter.column].to_val(val)
            sel += (col == val)
        return sel

    def _continuous_filter_to_indicator(self, filter, col):
        """Return selection of rows matched by the given continuous filter.

        Parameters
        ----------
        filter: FilterContinuous
        col: np.ndarray

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        if filter.oper == filter.IsDefined:
            col = col.astype(float)
            return ~np.isnan(col)

        return self._range_filter_to_indicator(filter, col, filter.min, filter.max)

    def _string_filter_to_indicator(self, filter, col):
        """Return selection of rows matched by the given string filter.

        Parameters
        ----------
        filter: FilterString
        col: np.ndarray

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        if filter.oper == filter.IsDefined:
            return col.astype(bool)

        col = col.astype(str)
        fmin = filter.min or ""
        fmax = filter.max or ""

        if not filter.case_sensitive:
            # convert all to lower case
            col = np.char.lower(col)
            fmin = fmin.lower()
            fmax = fmax.lower()

        if filter.oper == filter.Contains:
            return np.fromiter((fmin in e for e in col),
                               dtype=bool)
        if filter.oper == filter.StartsWith:
            return np.fromiter((e.startswith(fmin) for e in col),
                               dtype=bool)
        if filter.oper == filter.EndsWith:
            return np.fromiter((e.endswith(fmin) for e in col),
                               dtype=bool)

        return self._range_filter_to_indicator(filter, col, fmin, fmax)

    @staticmethod
    def _range_filter_to_indicator(filter, col, fmin, fmax):
        with np.errstate(invalid="ignore"):  # nan's are properly discarded
            if filter.oper == filter.Equal:
                return col == fmin
            if filter.oper == filter.NotEqual:
                return col != fmin
            if filter.oper == filter.Less:
                return col < fmin
            if filter.oper == filter.LessEqual:
                return col <= fmin
            if filter.oper == filter.Greater:
                return col > fmin
            if filter.oper == filter.GreaterEqual:
                return col >= fmin
            if filter.oper == filter.Between:
                return (col >= fmin) * (col <= fmax)
            if filter.oper == filter.Outside:
                return (col < fmin) + (col > fmax)

            raise TypeError("Invalid operator")

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_variance=False):
        if compute_variance:
            raise NotImplementedError("computation of variance is "
                                      "not implemented yet")
        W = self._W if self.has_weights() else None
        rr = []
        stats = []
        if not columns:
            if self.domain.attributes:
                rr.append(fast_stats(self._X, W))
            if self.domain.class_vars:
                rr.append(fast_stats(self._Y, W))
            if include_metas and self.domain.metas:
                rr.append(fast_stats(self.metas, W))
            if len(rr):
                stats = np.vstack(tuple(rr))
        else:
            nattrs = len(self.domain.attributes)
            for column in columns:
                c = self.domain.index(column)
                if 0 <= c < nattrs:
                    S = fast_stats(self._X[:, [c]], W and W[:, [c]])
                elif c >= nattrs:
                    if self._Y.ndim == 1 and c == nattrs:
                        S = fast_stats(self._Y[:, None], W and W[:, None])
                    else:
                        S = fast_stats(self._Y[:, [c - nattrs]], W and W[:, [c - nattrs]])
                else:
                    S = fast_stats(self._metas[:, [-1 - c]], W and W[:, [-1 - c]])
                stats.append(S[0])
        return stats

    def _compute_distributions(self, columns=None):
        if columns is None:
            columns = range(len(self.domain.variables))
        else:
            columns = [self.domain.index(var) for var in columns]

        distributions = []
        X = self.X
        if sp.issparse(X):
            X = X.tocsc()


        W = self.W.ravel() if self.has_weights() else None

        for col in columns:
            variable = self.domain[col]

            # Select the correct data column from X, Y or metas
            if 0 <= col < X.shape[1]:
                x = X[:, col]
            elif col < 0:
                x = self.metas[:, col * (-1) - 1]
                if np.issubdtype(x.dtype, np.dtype(object)):
                    x = x.astype(float)
            elif self._Y.ndim == 1 and col == X.shape[1]:
                x = self._Y
            else:
                x = self._Y[:, col - X.shape[1]]

            if variable.is_discrete:
                dist, unknowns = bincount(x, weights=W, max_val=len(variable.values) - 1)
            elif not x.shape[0]:
                dist, unknowns = np.zeros((2, 0)), 0
            else:
                if W is not None:
                    if sp.issparse(x):
                        arg_sort = np.argsort(x.data)
                        ranks = x.indices[arg_sort]
                        vals = np.vstack((x.data[arg_sort], W[ranks]))
                    else:
                        ranks = np.argsort(x)
                        vals = np.vstack((x[ranks], W[ranks]))
                else:
                    x_values = x.data if sp.issparse(x) else x
                    vals = np.ones((2, x_values.shape[0]))
                    vals[0, :] = x_values
                    vals[0, :].sort()

                dist = np.array(_valuecount.valuecount(vals))
                # If sparse, then 0s will not be counted with `valuecount`, so
                # we have to add them to the result manually.
                if sp.issparse(x) and sparse_has_implicit_zeros(x):
                    if W is not None:
                        zero_weights = sparse_implicit_zero_weights(x, W).sum()
                    else:
                        zero_weights = sparse_count_implicit_zeros(x)
                    zero_vec = [0, zero_weights]
                    dist = np.insert(dist, np.searchsorted(dist[0], 0), zero_vec, axis=1)
                # Since `countnans` assumes vector shape to be (1, n) and `x`
                # shape is (n, 1), we pass the transpose
                unknowns = countnans(x.T, W)
            distributions.append((dist, unknowns))

        return distributions

    def _compute_contingency(self, col_vars=None, row_var=None):
        n_atts = self.X.shape[1]

        if col_vars is None:
            col_vars = range(len(self.domain.variables))
        else:
            col_vars = [self.domain.index(var) for var in col_vars]
        if row_var is None:
            row_var = self.domain.class_var
            if row_var is None:
                raise ValueError("No row variable")

        row_desc = self.domain[row_var]
        if not row_desc.is_discrete:
            raise TypeError("Row variable must be discrete")
        row_indi = self.domain.index(row_var)
        n_rows = len(row_desc.values)
        if 0 <= row_indi < n_atts:
            row_data = self.X[:, row_indi]
        elif row_indi < 0:
            row_data = self.metas[:, -1 - row_indi]
        elif self._Y.ndim == 1 and row_indi == n_atts:
            row_data = self._Y
        else:
            row_data = self._Y[:, row_indi - n_atts]

        W = self.W if self.has_weights() else None

        col_desc = [self.domain[var] for var in col_vars]
        col_indi = [self.domain.index(var) for var in col_vars]

        if any(not (var.is_discrete or var.is_continuous)
               for var in col_desc):
            raise ValueError("Contingency can be computed only for categorical "
                             "and numeric values.")

        # when we select a column in sparse matrix it is still two dimensional
        # and sparse - since it is just a column we can afford to transform
        # it to dense and make it 1D
        if issparse(row_data):
            row_data = row_data.toarray().ravel()
        if row_data.dtype.kind != "f":  # meta attributes can be stored as type object
            row_data = row_data.astype(float)

        contingencies = [None] * len(col_desc)
        for arr, f_cond, f_ind in (
                (self.X, lambda i: 0 <= i < n_atts, lambda i: i),
                (self._Y, lambda i: i >= n_atts, lambda i: i - n_atts),
                (self.metas, lambda i: i < 0, lambda i: -1 - i)):

            arr_indi = [e for e, ind in enumerate(col_indi) if f_cond(ind)]

            vars = [(e, f_ind(col_indi[e]), col_desc[e]) for e in arr_indi]
            disc_vars = [v for v in vars if v[2].is_discrete]
            if disc_vars:
                if sp.issparse(arr):
                    max_vals = max(len(v[2].values) for v in disc_vars)
                    disc_indi = {i for _, i, _ in disc_vars}
                    mask = [i in disc_indi for i in range(arr.shape[1])]
                    conts, nans_cols, nans_rows, nans = contingency(
                        arr, row_data, max_vals - 1, n_rows - 1, W, mask)
                    for col_i, arr_i, var in disc_vars:
                        n_vals = len(var.values)
                        contingencies[col_i] = (
                            conts[arr_i][:, :n_vals], nans_cols[arr_i],
                            nans_rows[arr_i], nans[arr_i])
                else:
                    for col_i, arr_i, var in disc_vars:
                        col = arr if arr.ndim == 1 else arr[:, arr_i]
                        contingencies[col_i] = contingency(
                            col.astype(float),
                            row_data, len(var.values) - 1, n_rows - 1, W)

            cont_vars = [v for v in vars if v[2].is_continuous]
            if cont_vars:
                W_ = None
                if W is not None:
                    W_ = W.astype(dtype=np.float64)
                if sp.issparse(arr):
                    arr = sp.csc_matrix(arr)

                for col_i, arr_i, _ in cont_vars:
                    if sp.issparse(arr):
                        col_data = arr.data[arr.indptr[arr_i]:arr.indptr[arr_i + 1]]
                        rows = arr.indices[arr.indptr[arr_i]:arr.indptr[arr_i + 1]]
                        W_ = None if W_ is None else W_[rows]
                        classes_ = row_data[rows]
                    else:
                        col_data, W_, classes_ = arr[:, arr_i], W_, row_data

                    col_data = col_data.astype(dtype=np.float64)
                    contingencies[col_i] = _contingency.contingency_floatarray(
                        col_data, classes_, n_rows, W_)

        return contingencies

    @classmethod
    def transpose(cls, table, feature_names_column="",
                  meta_attr_name="Feature name", feature_name="Feature",
                  remove_redundant_inst=False, progress_callback=None):
        """
        Transpose the table.

        :param table: Table - table to transpose
        :param feature_names_column: str - name of (String) meta attribute to
            use for feature names
        :param meta_attr_name: str - name of new meta attribute into which
            feature names are mapped
        :param feature_name: str - default feature name prefix
        :param remove_redundant_inst: bool - remove instance that
            represents feature_names_column
        :param progress_callback: callable - to report the progress
        :return: Table - transposed table
        """
        if progress_callback is None:
            progress_callback = dummy_callback
        progress_callback(0, "Transposing...")

        if isinstance(feature_names_column, Variable):
            feature_names_column = feature_names_column.name

        self = cls()
        n_cols, self.n_rows = table.X.shape
        old_domain = table.attributes.get("old_domain")
        table_domain_attributes = list(table.domain.attributes)
        attr_index = None
        if remove_redundant_inst:
            attr_names = [a.name for a in table_domain_attributes]
            if feature_names_column and feature_names_column in attr_names:
                attr_index = attr_names.index(feature_names_column)
                self.n_rows = self.n_rows - 1
                table_domain_attributes.remove(
                    table_domain_attributes[attr_index])

        # attributes
        # - classes and metas to attributes of attributes
        # - arbitrary meta column to feature names
        with self.unlocked_reference():
            self.X = table.X.T
            if attr_index is not None:
                self.X = np.delete(self.X, attr_index, 0)
            if feature_names_column:
                names = [str(row[feature_names_column]) for row in table]
                progress_callback(0.1)
                names = get_unique_names_duplicates(names)
                progress_callback(0.3)
                attributes = [ContinuousVariable(name) for name in names]
            else:
                places = int(np.ceil(np.log10(n_cols))) if n_cols else 1
                attributes = [ContinuousVariable(f"{feature_name} {i:0{places}}")
                              for i in range(1, n_cols + 1)]
            progress_callback(0.4)

            if old_domain is not None and feature_names_column:
                for i, _ in enumerate(attributes):
                    if attributes[i].name in old_domain:
                        var = old_domain[attributes[i].name]
                        attr = ContinuousVariable(var.name) if var.is_continuous \
                            else DiscreteVariable(var.name, var.values)
                        attr.attributes = var.attributes.copy()
                        attributes[i] = attr

            def set_attributes_of_attributes(_vars, _table):
                for i, variable in enumerate(_vars):
                    if variable.name == feature_names_column:
                        continue
                    for j, row in enumerate(_table):
                        value = variable.repr_val(row) if np.isscalar(row) \
                            else row[i] if isinstance(row[i], str) \
                            else variable.repr_val(row[i])

                        if value not in MISSING_VALUES:
                            attributes[j].attributes[variable.name] = value

            set_attributes_of_attributes(table.domain.class_vars, table.Y)
            progress_callback(0.5)
            set_attributes_of_attributes(table.domain.metas, table.metas)

            # weights
            self.W = np.empty((self.n_rows, 0))

            def get_table_from_attributes_of_attributes(_vars, _dtype=float):
                T = np.empty((self.n_rows, len(_vars)), dtype=_dtype)
                for i, _attr in enumerate(table_domain_attributes):
                    for j, _var in enumerate(_vars):
                        val = str(_attr.attributes.get(_var.name, ""))
                        if not _var.is_string:
                            val = np.nan if val in MISSING_VALUES else \
                                _var.values.index(val) if \
                                    _var.is_discrete else float(val)
                        T[i, j] = val
                return T

            # class_vars - attributes of attributes to class - from old domain
            class_vars = []
            if old_domain is not None:
                class_vars = old_domain.class_vars
            self.Y = get_table_from_attributes_of_attributes(class_vars)

            # metas
            # - feature names and attributes of attributes to metas
            self.metas, metas = np.empty((self.n_rows, 0), dtype=object), []
            if meta_attr_name not in [m.name for m in table.domain.metas] and \
                    table_domain_attributes:
                self.metas = np.array([[a.name] for a in table_domain_attributes],
                                      dtype=object)
                metas.append(StringVariable(meta_attr_name))

            names = chain.from_iterable(list(attr.attributes)
                                        for attr in table_domain_attributes)
            names = sorted(set(names) - {var.name for var in class_vars})
            progress_callback(0.6)

            def guessed_var(i, var_name):
                orig_vals = M[:, i]
                val_map, vals, var_type = Orange.data.io.guess_data_type(orig_vals)
                values, variable = Orange.data.io.sanitize_variable(
                    val_map, vals, orig_vals, var_type, {}, name=var_name)
                M[:, i] = values
                return variable

            _metas = [StringVariable(n) for n in names]
            if old_domain is not None:
                _metas = [m for m in old_domain.metas if m.name != meta_attr_name]
            M = get_table_from_attributes_of_attributes(_metas, _dtype=object)
            progress_callback(0.7)
            if old_domain is None:
                _metas = [guessed_var(i, m.name) for i, m in enumerate(_metas)]
            if _metas:
                self.metas = np.hstack((self.metas, M))
                metas.extend(_metas)

            self.domain = Domain(attributes, class_vars, metas)
            progress_callback(0.9)
            cls._init_ids(self)
            self.attributes = table.attributes.copy()
            self.attributes["old_domain"] = table.domain
            progress_callback(1)
            return self

    def to_sparse(self, sparse_attributes=True, sparse_class=False,
                  sparse_metas=False):
        def sparsify(features):
            for f in features:
                f.sparse = True

        new_domain = self.domain.copy()

        if sparse_attributes:
            sparsify(new_domain.attributes)
        if sparse_class:
            sparsify(new_domain.class_vars)
        if sparse_metas:
            sparsify(new_domain.metas)
        return self.transform(new_domain)

    def to_dense(self, dense_attributes=True, dense_class=True,
                 dense_metas=True):
        def densify(features):
            for f in features:
                f.sparse = False

        new_domain = self.domain.copy()

        if dense_attributes:
            densify(new_domain.attributes)
        if dense_class:
            densify(new_domain.class_vars)
        if dense_metas:
            densify(new_domain.metas)
        t = self.transform(new_domain)
        t.ids = self.ids  # preserve indices
        return t

    def groupby(self, columns: List[Variable]) -> "OrangeTableGroupBy":
        """
        Group Table by variables defined in the columns list. Behaviour is
        similar to Pandas groupby.

        Parameters
        ----------
        columns
            List of variables used to determine the groups

        Returns
        -------
        GroupBy object of type OrangeTableGroupBy which holds information about
        groups.
        """
        return Orange.data.aggregate.OrangeTableGroupBy(self, columns)


def _dereferenced(array):
    # CSR and CSC matrices are constructed so that array.data is a
    # view to a base, which prevents unlocking them. Therefore, if
    # sparse matrix doesn't own its data, but its base array is
    # referenced only by this matrix, we copy it. This doesn't
    # increase memory use, but allows unlocking.
    if sp.issparse(array) \
            and array.data.base is not None \
            and sys.getrefcount(array.data.base) == 2:  # 2 = 1 real + 1 for arg
        array.data = array.data.copy()
    return array


def _check_arrays(*arrays, dtype=None, shape_1=None):
    checked = []
    if not len(arrays):
        return checked

    def ninstances(array):
        if hasattr(array, "shape"):
            return array.shape[0]
        else:
            return len(array) if array is not None else 0

    if shape_1 is None:
        shape_1 = ninstances(arrays[0])

    for array in arrays:
        if array is None:
            checked.append(array)
            continue

        if ninstances(array) != shape_1:
            raise ValueError("Leading dimension mismatch (%d != %d)"
                             % (ninstances(array), shape_1))

        if sp.issparse(array):
            if not (sp.isspmatrix_csr(array) or sp.isspmatrix_csc(array)):
                array = array.tocsr()
            array.data = np.asarray(array.data)
            array = _dereferenced(array)
            has_inf = _check_inf(array.data)
        else:
            if dtype is not None:
                array = np.asarray(array, dtype=dtype)
            else:
                array = np.asarray(array)
            has_inf = _check_inf(array)

        if has_inf:
            array[np.isinf(array)] = np.nan
            warnings.warn("Array contains infinity.", RuntimeWarning)
        checked.append(array)

    return checked


def _check_inf(array):
    return array.dtype.char in np.typecodes['AllFloat'] and \
           np.isinf(array.data).any()


def _subarray(arr, rows, cols):
    rows = _optimize_indices(rows, arr.shape[0])
    if arr.ndim == 1:
        return arr[rows]
    cols = _optimize_indices(cols, arr.shape[1])
    return arr[_rxc_ix(rows, cols)]


def _optimize_indices(indices, maxlen):
    """
    Convert integer indices to slice if possible. It only converts increasing
    integer ranges with positive steps and valid starts and ends.
    Only convert valid ends so that invalid ranges will still raise
    an exception.

    Allows numpy to reuse the data array, because it defaults to copying
    if given indices.

    Parameters
    ----------
    indices : 1D sequence, slice or Ellipsis
    """
    if isinstance(indices, slice):
        return indices

    if indices is ...:
        return slice(None, None, 1)

    # a very common case for column selection
    if len(indices) == 1 and not isinstance(indices[0], bool):
        if indices[0] >= 0:
            return slice(indices[0], indices[0] + 1, 1)
        else:
            return slice(indices[0], indices[0] - 1, -1)

    if len(indices) >= 1:
        indices = np.asarray(indices)
        if indices.dtype != bool:
            begin = indices[0]
            end = indices[-1]
            steps = np.diff(indices) if len(indices) > 1 else np.array([1])
            step = steps[0]

            # continuous ranges with constant step and valid start and stop index can be slices
            if np.all(steps == step) and step > 0 and begin >= 0 and end < maxlen:
                return slice(begin, end + step, step)

    return indices


def _rxc_ix(rows, cols):
    """
    Construct an index object to index the `rows` x `cols` cross product.

    Rows and columns can be a 1d bool or int sequence, or a slice.
    The later is a convenience and is interpreted the same
    as `slice(None, None, -1)`

    Parameters
    ----------
    rows : 1D sequence, slice
        Row indices.
    cols : 1D sequence, slice
        Column indices.

    See Also
    --------
    numpy.ix_

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10).reshape(2, 5)
    >>> a[_rxc_ix([0, 1], [3, 4])]
    array([[3, 4],
           [8, 9]])
    >>> a[_rxc_ix([False, True], slice(None, None, 1))]
    array([[5, 6, 7, 8, 9]])

    """
    isslice = (isinstance(rows, slice), isinstance(cols, slice))
    if isslice == (True, True):
        return rows, cols
    elif isslice == (True, False):
        return rows, np.asarray(np.ix_(cols), int).ravel()
    elif isslice == (False, True):
        return np.asarray(np.ix_(rows), int).ravel(), cols
    else:
        r, c = np.ix_(rows, cols)
        return np.asarray(r, int), np.asarray(c, int)


def assure_domain_conversion_sparsity(target, source):
    """
    Assure that the table obeys the domain conversion's suggestions about sparsity.

    Args:
        target (Table): the target table.
        source (Table): the source table.

    Returns:
        Table: with fixed sparsity. The sparsity is set as it is recommended by domain conversion
            for transformation from source to the target domain.
    """
    conversion = DomainConversion(source.domain, target.domain)
    match_density = [assure_array_dense, assure_array_sparse]
    target.X = match_density[conversion.sparse_X](target.X)
    target.Y = match_density[conversion.sparse_Y](target.Y)
    target.metas = match_density[conversion.sparse_metas](target.metas)
    return target


class Role:
    Attribute = 0
    ClassAttribute = 1
    Meta = 2

    @staticmethod
    def get_arr(role, table):
        return table.X if role == Role.Attribute else \
               table.Y if role == Role.ClassAttribute else \
               table.metas

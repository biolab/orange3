import numpy as np

from Orange import data


def _get_variable(variable, dat, attr_name, expected_type=None, expected_name=""):
    failed = False
    if isinstance(variable, data.Variable):
        datvar = getattr(dat, "variable", None)
        if datvar is not None and datvar is not variable:
            raise ValueError("variable does not match the variable in the data")
    elif hasattr(dat, "domain"):
        variable = dat.domain[variable]
    elif hasattr(dat, attr_name):
        variable = dat.variable
    else:
        failed = True
    if failed or (expected_type is not None and not isinstance(variable, expected_type)):
        if not expected_type or isinstance(variable, data.Variable):
            raise ValueError("expected %s variable not %s" % (expected_name, variable))
        else:
            raise ValueError("expected %s, not '%s'" % (
                expected_type.__name__, type(variable).__name__))
    return variable


def _create_discrete(cls, *args):
    return cls(*args)


class Discrete(np.ndarray):
    def __new__(cls, dat, col_variable=None, row_variable=None,
                row_unknowns=None, col_unknowns=None, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None or row_unknowns is not None or \
                    col_unknowns is not None:
                raise TypeError(
                    "incompatible arguments (data storage and 'unknowns'")
            return cls.from_data(dat, col_variable, row_variable)

        if row_variable is not None:
            row_variable = _get_variable(row_variable, dat, "row_variable")
        if col_variable is not None:
            col_variable = _get_variable(col_variable, dat, "col_variable")

        self = super().__new__(cls, dat.shape)
        self.row_variable = row_variable
        self.col_variable = col_variable

        self.col_unknowns = np.zeros(dat.shape[0]) \
            if col_unknowns is None else col_unknowns
        self.row_unknowns = np.zeros(dat.shape[1]) \
            if col_unknowns is None else row_unknowns
        self.unknowns = unknowns or 0

        self[...] = dat

        return self

    @classmethod
    def from_data(cls, data, col_variable, row_variable=None):
        if row_variable is None:
            row_variable = data.domain.class_var
            if row_variable is None:
                raise ValueError(
                    "row_variable needs to be specified (data has no class)")
        row_variable = _get_variable(row_variable, data, "row_variable")
        col_variable = _get_variable(col_variable, data, "col_variable")
        try:
            dist, col_unknowns, row_unknowns, unknowns = \
                data._compute_contingency([col_variable], row_variable)[0]

            self = super().__new__(cls, dist.shape)
            self[...] = dist
            self.col_unknowns = col_unknowns
            self.row_unknowns = row_unknowns
            self.unknowns = unknowns
        except NotImplementedError:
            shape = len(row_variable.values), len(col_variable.values)
            self = super().__new__(cls, shape)
            self[...] = np.zeros(shape)
            self.col_unknowns = np.zeros(shape[0])
            self.row_unknowns = np.zeros(shape[1])
            self.unknowns = 0
            rind = data.domain.index(row_variable)
            cind = data.domain.index(col_variable)
            for row in data:
                rval, cval = row[rind], row[cind]
                w = row.weight
                if np.isnan(rval) and np.isnan(cval):
                    self.unknowns += w
                elif np.isnan(rval):
                    self.row_unknowns[int(cval)] += w
                elif np.isnan(cval):
                    self.col_unknowns[int(rval)] += w
                else:
                    self[int(rval), int(cval)] += w
        self.row_variable = row_variable
        self.col_variable = col_variable
        return self

    @property
    def array_with_unknowns(self):
        """
        This property returns a contingency array with unknowns arrays added
        as a column and row.

        Returns
        -------
        np.array
            Array with concatenated unknowns as a row and column
        """
        return np.vstack(
            (np.hstack((np.array(self), self.col_unknowns.reshape(-1, 1))),
             np.append(self.row_unknowns, self.unknowns)))

    def __eq__(self, other):
        return (
            np.array_equal(self, other) and
            (not hasattr(other, "col_unknowns") or
             np.array_equal(self.col_unknowns, other.col_unknowns)) and
            (not hasattr(other, "row_unknowns") or
             np.array_equal(self.row_unknowns, other.row_unknowns)) and
            (not hasattr(other, "unknowns") or
             self.unknowns == other.unknowns)
        )

    def __getitem__(self, index):
        if isinstance(index, str):
            if len(self.shape) == 2:  # contingency
                index = self.row_variable.to_val(index)
                contingency_row = super().__getitem__(index)
                contingency_row.col_variable = self.col_variable
                return contingency_row
            else:  # Contingency row
                column = self.strides == self.base.strides[:1]
                if column:
                    index = self.row_variable.to_val(index)
                else:
                    index = self.col_variable.to_val(index)

        elif isinstance(index, tuple):
            if isinstance(index[0], str):
                index = (self.row_variable.to_val(index[0]), index[1])
            if isinstance(index[1], str):
                index = (index[0], self.col_variable.to_val(index[1]))
        result = super().__getitem__(index)
        if isinstance(result, Discrete):
            if not isinstance(index, tuple):
                result.col_unknowns = self.col_unknowns[index]
                result.row_unknowns = self.row_unknowns
            else:
                result.col_unknowns = self.col_unknowns[index[0]]
                result.row_unknowns = self.col_unknowns[index[1]]
            result.unknowns = self.unknowns
        if result.strides:
            result.col_variable = self.col_variable
            result.row_variable = self.row_variable
        return result

    def __setitem__(self, index, value):
        if isinstance(index, str):
            index = self.row_variable.to_val(index)
        elif isinstance(index, tuple):
            if isinstance(index[0], str):
                index = (self.row_variable.to_val(index[0]), index[1])
            if isinstance(index[1], str):
                index = (index[0], self.col_variable.to_val(index[1]))
        super().__setitem__(index, value)

    def normalize(self, axis=None):
        t = np.sum(self, axis=axis)
        if t > 1e-6:
            self[:] /= t
            if axis is None or axis == 1:
                self.unknowns /= t

    def __reduce__(self):
        return (
            _create_discrete,
            (Discrete, np.copy(self), self.col_variable, self.row_variable,
             self.col_unknowns, self.row_unknowns, self.unknowns)
        )

    def __array_finalize__(self, obj):
        # defined in __new__, pylint: disable=attribute-defined-outside-init
        """See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if obj is None:
            return
        self.col_variable = getattr(obj, 'col_variable', None)
        self.row_variable = getattr(obj, 'row_variable', None)
        self.col_unknowns = getattr(obj, 'col_unknowns', None)
        self.row_unknowns = getattr(obj, 'row_unknowns', None)
        self.unknowns = getattr(obj, 'unknowns', None)


class Continuous:
    def __init__(self, dat, col_variable=None, row_variable=None,
                 col_unknowns=None, row_unknowns=None, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None or row_unknowns is not None or \
                    col_unknowns is not None:
                raise TypeError(
                    "incompatible arguments (data storage and 'unknowns'")
            self.from_data(dat, col_variable, row_variable)
            return

        if row_variable is not None:
            row_variable = _get_variable(row_variable, dat, "row_variable")
        if col_variable is not None:
            col_variable = _get_variable(col_variable, dat, "col_variable")

        self.values, self.counts = dat

        self.row_variable = row_variable
        self.col_variable = col_variable
        self.col_unknowns = np.zeros(self.counts.shape[1]) \
            if col_unknowns is None else col_unknowns
        self.row_unknowns = np.zeros(self.counts.shape[0]) \
            if row_unknowns is None else row_unknowns
        self.unknowns = unknowns or 0

    def from_data(self, data, col_variable, row_variable=None):
        if row_variable is None:
            row_variable = data.domain.class_var
            if row_variable is None:
                raise ValueError("row_variable needs to be specified (data has no class)")
        self.row_variable = _get_variable(row_variable, data, "row_variable")
        self.col_variable = _get_variable(col_variable, data, "col_variable")
        try:
            conts = data._compute_contingency([col_variable], row_variable)
            (self.values, self.counts), self.col_unknowns, self.row_unknowns, \
            self.unknowns = conts[0]
        except NotImplementedError:
            raise NotImplementedError(
                "Fallback method for computation of contingencies is not implemented yet"
            )

    @property
    def array_with_unknowns(self):
        """
        This function returns the list of all items returned by __getitem__
        with adding a row of row_unknowns together with values.
        """
        # pylint: disable=unnecessary-comprehension
        other_rows = [x for x in self]
        ind = self.row_unknowns > 0
        unknown_rows = np.vstack((self.values[ind], self.row_unknowns[ind]))
        return other_rows + [unknown_rows]

    def __eq__(self, other):
        return (
            np.array_equal(self.values, other.values) and
            np.array_equal(self.counts, other.counts) and
            (not hasattr(other, "col_unknowns") or
             np.array_equal(self.col_unknowns, other.col_unknowns)) and
            (not hasattr(other, "row_unknowns") or
             np.array_equal(self.row_unknowns, other.row_unknowns)) and
            (not hasattr(other, "unknowns") or
             self.unknowns == other.unknowns)
        )

    def __getitem__(self, index):
        """ Return contingencies for a given class value. """
        if isinstance(index, (str, float)):
            index = self.row_variable.to_val(index)
        C = self.counts[index]
        ind = C > 0
        return np.vstack((self.values[ind], C[ind]))

    def __len__(self):
        return self.counts.shape[0]

    def __setitem__(self, index, value):
        raise NotImplementedError(
            "Setting individual class contingencies is not implemented yet. "
            "Set .values and .counts."
        )

    def normalize(self, axis=None):
        if axis is None:
            t = sum(np.sum(x[:, 1]) for x in self)
            if t > 1e-6:
                for x in self:
                    x[:, 1] /= t
        elif axis != 1:
            raise ValueError(
                "contingencies can be normalized only with axis=1 or without axis"
            )
        else:
            for i, x in enumerate(self):
                t = np.sum(x[:, 1])
                if t > 1e-6:
                    x[:, 1] /= t
                    self.unknowns[i] /= t
                else:
                    if self.unknowns[i] > 1e-6:
                        self.unknowns[i] = 1


def get_contingency(dat, col_variable, row_variable=None, col_unknowns=None,
                    row_unknowns=None, unks=None):
    variable = _get_variable(col_variable, dat, "col_variable")
    if variable.is_discrete:
        return Discrete(
            dat, col_variable, row_variable, col_unknowns, row_unknowns, unks)
    elif variable.is_continuous:
        return Continuous(
            dat, col_variable, row_variable, col_unknowns, row_unknowns)
    else:
        raise TypeError(
            "cannot compute distribution of '%s'" % type(variable).__name__)


def get_contingencies(dat, skip_discrete=False, skip_continuous=False):
    vars = dat.domain.attributes
    row_var = dat.domain.class_var
    if row_var is None:
        raise ValueError("data has no target variable")
    if skip_discrete:
        if skip_continuous:
            return []
        columns = [i for i, var in enumerate(vars) if var.is_continuous]
    elif skip_continuous:
        columns = [i for i, var in enumerate(vars) if var.is_discrete]
    else:
        columns = None
    try:
        dist_unks = dat._compute_contingency(columns)
        if columns is None:
            columns = np.arange(len(vars))
        contigs = []
        for col, (cont, col_unk, row_unk, unks) in zip(columns, dist_unks):
            contigs.append(get_contingency(
                cont, vars[col], row_var, col_unk, row_unk, unks))
    except NotImplementedError:
        if columns is None:
            columns = range(len(vars))
        contigs = [get_contingency(dat, i) for i in columns]
    return contigs

import zlib
import math
import numpy as np
from Orange import data


def _get_variable(variable, dat, attr_name,
                  expected_type=None, expected_name=""):
    failed = False
    if isinstance(variable, data.Variable):
        datvar = getattr(dat, "variable", None)
        if datvar is not None and datvar is not variable:
            raise ValueError("variable does not match the variable"
                             "in the data")
    elif hasattr(dat, "domain"):
        variable = dat.domain[variable]
    elif hasattr(dat, attr_name):
        variable = dat.variable
    else:
        failed = True
    if failed or (expected_type is not None and
                  not isinstance(variable, expected_type)):
        if not expected_type or isinstance(variable, data.Variable):
            raise ValueError(
                "expected %s variable not %s" % (expected_name, variable))
        else:
            raise ValueError("expected %s, not '%s'" %
                             (expected_type.__name__, type(variable).__name__))
    return variable


class Contingency(np.ndarray):
    def __new__(cls, dat=None, col_variable=None, row_variable=None, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None:
                raise TypeError(
                    "incompatible arguments (data storage and 'unknowns'")
            return cls.from_data(dat, col_variable, row_variable)

        if row_variable is not None:
            row_variable = _get_variable(row_variable, dat, "row_variable")
            rows = len(row_variable.values)
        else:
            rows = dat.shape[0]
        if col_variable is not None:
            col_variable = _get_variable(col_variable, dat, "col_variable")
            cols = len(col_variable.values)
        else:
            cols = dat.shape[1]

        self = super().__new__(cls, (rows, cols))
        self.row_variable = row_variable
        self.col_variable = col_variable
        if dat is None:
            self[:] = 0
            self.unknowns = unknowns or 0
        else:
            self[...] = dat
            self.unknowns = (unknowns if unknowns is not None
                             else getattr(dat, "unknowns", 0))
        return self


    @classmethod
    def from_data(cls, data, col_variable, row_variable=None):
        if row_variable is None:
            row_variable = data.domain.class_var
            if row_variable is None:
                raise ValueError("row_variable needs to be specified (data"
                                 "has no class)")
        row_variable = _get_variable(row_variable, data, "row_variable")
        col_variable = _get_variable(col_variable, data, "col_variable")
        try:
            dist, unknowns = data._compute_contingency(
                [col_variable], row_variable)[0]
            self = super().__new__(cls, dist.shape)
            self[...] = dist
            self.unknowns = unknowns
        except NotImplementedError:
            self = np.zeros(
                (len(row_variable.values), len(col_variable.values)))
            self.unknowns = 0
            rind = data.domain.index(row_variable)
            cind = data.domain.index(col_variable)
            for row in data:
                rval, cval = row[rind], row[cind]
                if math.isnan(rval):
                    continue
                w = row.weight
                if math.isnan(cval):
                    self.unknowns[cval] += w
                else:
                    self[rval, cval] += w
        self.row_variable = row_variable
        self.col_variable = col_variable
        return self


    def __eq__(self, other):
        return np.array_equal(self, other) and (
            not hasattr(other, "unknowns") or self.unknowns == other.unknowns)


    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.variable.to_val(index)
        elif isinstance(index, tuple):
            if isinstance(index[0], str):
                index = (self.variable.to_val(index[0]), index[1])
            if isinstance(index[1], str):
                index = (index[0], self.variable.to_val(index[1]))
        return super().__getitem__(index)


    def __setitem__(self, index, value):
        if isinstance(index, str):
            index = self.variable.to_val(index)
        elif isinstance(index, tuple):
            if isinstance(index[0], str):
                index = (self.variable.to_val(index[0]), index[1])
            if isinstance(index[1], str):
                index = (index[0], self.variable.to_val(index[1]))
        super().__setitem__(index, value)


    def __hash__(self):
        return zlib.adler32(self) ^ hash(self.unknowns)


    def normalize(self, axis=None):
        t = np.sum(self, axis=axis)
        if t > 1e-6:
            self[:] /= t
            if axis == 0:
                self.unknowns /= t


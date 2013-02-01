from Orange.data import Variable, Storage

def _get_variable(variable, dat):
    if isinstance(variable, Variable):
        datvar = getattr(dat, "variable", None)
        if datvar is not None and datvar is not variable:
            raise ValueError("variable does not match the variable "
                             "in the data")
    elif hasattr(dat, "domain"):
        variable = dat.domain[variable]
    elif hasattr(dat, "variable"):
        variable = dat.variable
    else:
        raise ValueError("invalid specification of variable")
    return variable


class BasicStats:
    def __init__(self, dat=None, variable=None):
        if isinstance(dat, Storage):
            self.from_data(dat, variable)
        elif dat is None:
            self.min = float("inf")
            self.max = float("-inf")
            self.mean = self.var = self.nans = self.non_nans = 0
        else:
            self.min, self.max, self.mean, self.var, self.nans, self.non_nans \
                = dat

    def from_data(self, data, variable):
        variable = _get_variable(variable, data)
        stats = data._compute_basic_stats([variable])
        self.min, self.max, self.mean, self.var, self.nans, self.non_nans \
            = stats[0]

class DomainBasicStats:
    def __init__(self, data, include_metas=False):
        self.domain = data.domain
        self.stats = [BasicStats(s) for s in
                      data._compute_basic_stats(include_metas=include_metas)]

    def __getitem__(self, index):
        """
        Index can be a variable, variable name or an integer. Meta attributes
        can be specified by negative indices or by indices above len(domain).
        """
        if not isinstance(index, int):
            index = self.domain.index(index)
        if index < 0:
            index = len(self.domain) + (-1 - index)
        return self.stats[index]


from Orange.data import Variable, ContinuousVariable

import bottleneck as bn


# TODO: this is all just a quick hack; do it properly -- call data storage etc.

class BasicStats:
    def __init__(self, variable, data):
        if (isinstance(variable, Variable) and
                not isinstance(variable, ContinuousVariable)):
            raise ValueError("variable '{}' is not continuous".
                             format(variable.name))
        col, sparse = data.get_column_view(variable)
        self.min = bn.nanmin(col)
        self.max = bn.nanmax(col)

# TODO sparse data
def get_stats(data):
    return [BasicStats(i, data) for i, var in enumerate(data.domain)
            if isinstance(var, ContinuousVariable)
           ] + [
            BasicStats(-1 - i, data) for i, var in enumerate(data.domain.metas)
            if isinstance(var, ContinuousVariable)
           ]

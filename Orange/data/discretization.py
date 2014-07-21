import Orange
import Orange.feature.discretization

from Orange.data import ContinuousVariable, Domain

class DiscretizeTable:
    """Discretizes all continuous features in the data.

    .. attribute:: method

        Feature discretization method (instance of
        :obj:`Orange.feature.discretization.Discretization`). If left `None`,
        :class:`Orange.feature.discretization.EqualFreq` with 4 intervals is
        used.

    .. attribute:: clean
        
        If `True`, features discretized into a single interval constant are
        removed. This is useful for discretization methods that infer the
        number of intervals from the data, such as
        :class:`Orange.feature.discretization.Entropy` (default: `True`).

    .. attribute:: discretize_class
    
        Determines whether a target is also discretized if it is continuous.
        (default: `False`)
    """
    def __new__(cls, data=None,
                discretize_class=False, method=None, clean=True, fixed=None):
        self = super().__new__(cls)
        self.discretize_class = discretize_class
        self.method = method
        self.clean = clean
        if data is None:
            return self
        else:
            return self(data, fixed)


    def __call__(self, data, fixed=None):
        """
        Return the discretized data set.

        :param data: Data to discretize.
        """

        def transform_list(s, fixed=None):
            new_vars = []
            for var in s:
                if isinstance(var, ContinuousVariable):
                    if fixed and var.name in fixed.keys():
                        nv = method(data, var, fixed)
                    else:
                        nv = method(data, var)
                    if not self.clean or len(nv.values) > 1:
                        new_vars.append(nv)
                else:
                    new_vars.append(var)
            return new_vars

        if self.method is None:
            method = Orange.feature.discretization.EqualFreq(n=4)
        else:
            method = self.method
        domain = data.domain
        new_attrs = transform_list(domain.attributes, fixed)
        if self.discretize_class:
            new_classes = transform_list(domain.class_vars)
        else:
            new_classes = domain.class_vars
        nd = Domain(new_attrs, new_classes)
        return data.from_table(nd, data)

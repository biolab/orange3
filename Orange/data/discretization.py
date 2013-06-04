import Orange
import Orange.feature.discretization

from Orange.data import ContinuousVariable, Domain

class DiscretizeTable(object):
    """Discretizes all continuous features in the data.

    .. attribute:: method

        Feature discretization method (from 
        :obj:`Orange.feature.discretization`).
        Default: :class:`Orange.feature.discretization.EqualFreq`
        with 4 intervals.

    .. attribute:: clean
        
        If `True`, features discretized to a constant will
        be removed. Useful only for methods that infer the
        number of discretization intervals from the data, like
        :class:`Orange.feature.discretization.Entropy` (default: `True`).

    .. attribute:: include_class
    
        If `True`, discretize class values too (default: `False`).
        

    """
    def __new__(cls, data=None, discretize_class=False,
                method=Orange.feature.discretization.EqualFreq(n=4), 
                clean=True, include_class=False):
        self = super().__new__(cls)
        self.discretize_class = discretize_class
        self.method = method
        self.clean = clean
        self.include_class = include_class
        return self if data is None else self(data)

    def __call__(self, data):
        """
        Return the discretized data set.

        :param data: Data to discretize.
        """
        
        domain = data.domain

        def transform_list(s):
            new_vars = []
            for var in s:
                if isinstance(var, ContinuousVariable):
                    nv = self.method(data, var)
                    if not self.clean or len(nv.values) > 1:
                        new_vars.append(nv)
                else:
                    new_vars.append(var)
            return new_vars

        new_attrs = transform_list(domain.attributes)
        if self.include_class:
            new_classes = transform_list(domain.class_vars)
        else:
            new_classes = domain.class_vars

        nd = Domain(new_attrs, new_classes)
        return Orange.data.Table(nd, data)

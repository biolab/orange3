import sklearn.cluster as skl_cluster
from Orange.data import Table, DiscreteVariable, Domain, Instance
from Orange.projection import SklProjector, Projection
from numpy import atleast_2d, ndarray, where


__all__ = ["DBSCAN"]

class DBSCAN(SklProjector):
    __wraps__ = skl_cluster.DBSCAN

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = skl_cluster.DBSCAN(**self.params)
        self.X = X
        if isinstance(X, Table):
            proj = proj.fit(X.X,)
        else:
            proj = proj.fit(X, )
        return DBSCANModel(proj)


class DBSCANModel(Projection):
    def __init__(self, proj):
        super().__init__(proj=proj)

    def __call__(self, data):
        if isinstance(data, ndarray):
            return self.proj.fit_predict(data).reshape((len(data), 1))

        if isinstance(data, Table):
            if data.domain is not self.pre_domain:
                data = Table(self.pre_domain, data)
            y = self.proj.fit_predict(data.X)
            vals = [-1] + list(self.proj.core_sample_indices_)
            c = DiscreteVariable(name='Core sample index', values=vals)
            domain = Domain([c])
            return Table(domain, y.reshape(len(y), 1))

        elif isinstance(data, Instance):
            if data.domain is not self.pre_domain:
                data = Instance(self.pre_domain, data)
            # Instances-by-Instance classification is not defined;
            raise Exception("Core sample assignment is not supported "
                            "for single instances.")

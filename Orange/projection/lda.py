import sklearn.discriminant_analysis as skl_da

from Orange.projection import SklProjector, DomainProjection

__all__ = ["LDA"]


class LDAModel(DomainProjection):
    var_prefix = "LD"


class LDA(SklProjector):
    __wraps__ = skl_da.LinearDiscriminantAnalysis
    name = "LDA"
    supports_sparse = False

    def __init__(self, solver="svd", shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        params = self.params.copy()
        if params["n_components"] is not None:
            params["n_components"] = min(min(X.shape), params["n_components"])
        proj = self.__wraps__(**params)
        proj = proj.fit(X, Y)
        proj.components_ = proj.scalings_.T[:params["n_components"]]
        return LDAModel(proj, self.domain, len(proj.components_))

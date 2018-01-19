from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import Orange.data
from Orange.classification.logistic_regression import _FeatureScorerMixin
from Orange.data.util import SharedComputeValue
from Orange.projection import SklProjector, Projection

__all__ = ["LDA"]


class LDA(SklProjector, _FeatureScorerMixin):
    name = "LDA"
    supports_sparse = False

    def __init__(self, n_components=2, solver='eigen', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.n_components = n_components
        self.solver = solver

    def fit(self, X, Y=None):
        if self.n_components is not None:
            self.n_components = min(min(X.shape), self.n_components)
        proj = LinearDiscriminantAnalysis(solver='eigen', n_components=2)
        proj = proj.fit(X, Y)
        return LDAModel(proj, self.domain)


class _LDATransformDomain:
    """Computation common for all LDA variables."""
    def __init__(self, lda):
        self.lda = lda

    def __call__(self, data):
        if data.domain != self.lda.pre_domain:
            data = data.transform(self.lda.pre_domain)
        return self.lda.transform(data.X)


class LDAModel(Projection):
    name = "LDAModel"

    def __init__(self, proj, domain):
        lda_transform = _LDATransformDomain(self)
        self.components_ = proj.scalings_.T

        def lda_variable(i):
            return Orange.data.ContinuousVariable(
                'LD%d' % (i + 1), compute_value=LDAProjector(self, i, lda_transform))

        super().__init__(proj=proj)
        self.orig_domain = domain
        self.n_components = self.components_.shape[0]
        self.domain = Orange.data.Domain(
            [lda_variable(i) for i in range(proj.n_components)],
            domain.class_vars, domain.metas)


class LDAProjector(SharedComputeValue):
    """Transform into a given LDA component."""
    def __init__(self, projection, feature, lda_transform):
        super().__init__(lda_transform)
        self.feature = feature

    def compute(self, data, lda_space):
        return lda_space[:, self.feature]

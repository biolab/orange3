import numpy as np
import scipy.stats as ss
import sklearn.cross_decomposition as skl_pls
from sklearn.preprocessing import StandardScaler

from Orange.base import Learner
from Orange.data import Table, Domain, Variable, \
    ContinuousVariable, StringVariable
from Orange.data.util import get_unique_names, SharedComputeValue
from Orange.preprocess.score import LearnerScorer
from Orange.regression.base_regression import SklLearnerRegression
from Orange.regression.linear import LinearModel

__all__ = ["PLSRegressionLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        model = self(data)
        return np.abs(model.coefficients), model.domain.attributes


class _PLSCommonTransform:

    def __init__(self, pls_model):
        self.pls_model = pls_model

    def _transform_with_numpy_output(self, X, Y):
        """
        # the next command does the following
        x_center = X - pls._x_mean
        y_center = Y - pls._y_mean
        t = x_center @ pls.x_rotations_
        u = y_center @ pls.y_rotations_
        """
        pls = self.pls_model.skl_model
        mask = np.isnan(Y).any(axis=1)
        n_comp = pls.n_components
        t = np.full((len(X), n_comp), np.nan, dtype=float)
        u = np.full((len(X), n_comp), np.nan, dtype=float)
        if (~mask).sum() > 0:
            t_, u_ = pls.transform(X[~mask], Y[~mask])
            t[~mask] = t_
            u[~mask] = u_
        if mask.sum() > 0:
            t[mask] = pls.transform(X[mask])
        return np.hstack((t, u))

    def __call__(self, data):
        if data.domain != self.pls_model.domain:
            data = data.transform(self.pls_model.domain)
        if len(data.Y.shape) == 1:
            Y = data.Y.reshape(-1, 1)
        else:
            Y = data.Y
        return self._transform_with_numpy_output(data.X, Y)

    def __eq__(self, other):
        if self is other:
            return True
        return type(self) is type(other) \
            and self.pls_model == other.pls_model

    def __hash__(self):
        return hash(self.pls_model)


class PLSProjector(SharedComputeValue):
    def __init__(self, transform, feature):
        super().__init__(transform)
        self.feature = feature

    def compute(self, _, shared_data):
        return shared_data[:, self.feature]

    def __eq__(self, other):
        if self is other:
            return True
        return super().__eq__(other) and self.feature == other.feature

    def __hash__(self):
        return hash((super().__hash__(), self.feature))


class PLSModel(LinearModel):
    var_prefix_X = "PLS T"
    var_prefix_Y = "PLS U"

    def predict(self, X):
        vals = self.skl_model.predict(X)
        if len(self.domain.class_vars) == 1:
            vals = vals.ravel()
        return vals

    def __str__(self):
        return f"PLSModel {self.skl_model}"

    def _get_var_names(self, n, prefix):
        proposed = [f"{prefix}{postfix}" for postfix in range(1, n + 1)]
        names = [var.name for var in self.domain.metas + self.domain.variables]
        return get_unique_names(names, proposed)

    def project(self, data):
        if not isinstance(data, Table):
            raise RuntimeError("PLSModel can only project tables")

        transformer = _PLSCommonTransform(self)

        def trvar(i, name):
            return ContinuousVariable(
                name, compute_value=PLSProjector(transformer, i))

        n_components = self.skl_model.x_loadings_.shape[1]

        var_names_X = self._get_var_names(n_components, self.var_prefix_X)
        var_names_Y = self._get_var_names(n_components, self.var_prefix_Y)

        domain = Domain(
            [trvar(i, var_names_X[i]) for i in range(n_components)],
            data.domain.class_vars,
            [trvar(n_components + i, var_names_Y[i]) for i in
             range(n_components)]
        )

        return data.transform(domain)

    def components(self):
        orig_domain = self.domain
        names = [a.name for a in
                 orig_domain.attributes + orig_domain.class_vars]
        meta_name = get_unique_names(names, 'components')

        n_components = self.skl_model.x_loadings_.shape[1]

        meta_vars = [StringVariable(name=meta_name)]
        metas = np.array(
            [[f"Component {i + 1}" for i in range(n_components)]], dtype=object
        ).T
        dom = Domain(
            [ContinuousVariable(a.name) for a in orig_domain.attributes],
            [ContinuousVariable(a.name) for a in orig_domain.class_vars],
            metas=meta_vars)
        components = Table(dom,
                           self.skl_model.x_loadings_.T,
                           Y=self.skl_model.y_loadings_.T,
                           metas=metas)
        components.name = 'components'
        return components

    def coefficients_table(self):
        coeffs = self.coefficients.T
        domain = Domain(
            [ContinuousVariable(f"coef {i}") for i in range(coeffs.shape[1])],
            metas=[StringVariable("name")]
        )
        waves = [[attr.name] for attr in self.domain.attributes]
        coef_table = Table.from_numpy(domain, X=coeffs, metas=waves)
        coef_table.name = "coefficients"
        return coef_table

    @property
    def rotations(self) -> tuple[np.ndarray, np.ndarray]:
        return self.skl_model.x_rotations_, self.skl_model.y_rotations_

    @property
    def loadings(self) -> tuple[np.ndarray, np.ndarray]:
        return self.skl_model.x_loadings_, self.skl_model.y_loadings_

    def residuals_normal_probability(self, data: Table) -> Table:
        pred = self(data)
        n = len(data)
        m = len(data.domain.class_vars)

        err = data.Y - pred
        if m == 1:
            err = err[:, None]

        theoretical_percentiles = (np.arange(1.0, n + 1)) / (n + 1)
        quantiles = ss.norm.ppf(theoretical_percentiles)
        ind = np.argsort(err, axis=0)
        theoretical_quantiles = np.zeros((n, m), dtype=float)
        for i in range(m):
            theoretical_quantiles[ind[:, i], i] = quantiles

        # check names so that tables could later be merged
        proposed = [f"{name} ({var.name})" for var in data.domain.class_vars
                    for name in ("Sample Quantiles", "Theoretical Quantiles")]
        names = get_unique_names(data.domain, proposed)
        domain = Domain([ContinuousVariable(name) for name in names])
        X = np.zeros((n, m * 2), dtype=float)
        X[:, 0::2] = err
        X[:, 1::2] = theoretical_quantiles
        res_table = Table.from_numpy(domain, X)
        res_table.name = "residuals normal probability"
        return res_table

    def dmodx(self, data: Table) -> Table:
        data = self.data_to_model_domain(data)

        n_comp = self.skl_model.n_components
        resids_ssx = self._residual_ssx(data.X)
        s = np.sqrt(resids_ssx / (self.skl_model.x_loadings_.shape[0] - n_comp))
        s0 = np.sqrt(resids_ssx.sum() / (
                (self.skl_model.x_scores_.shape[0] - n_comp - 1) *
                (data.X.shape[1] - n_comp)))
        dist = np.sqrt((s / s0) ** 2)

        name = get_unique_names(data.domain, ["DModX"])[0]
        domain = Domain([ContinuousVariable(name)])
        dist_table = Table.from_numpy(domain, dist[:, None])
        dist_table.name = "DMod"
        return dist_table

    def _residual_ssx(self, X: np.ndarray) -> np.ndarray:
        pred_scores = self.skl_model.transform(X)
        inv_pred_scores = self.skl_model.inverse_transform(pred_scores)

        scaler = StandardScaler()
        scaler.fit(X)
        x_recons = scaler.transform(inv_pred_scores)
        x_scaled = scaler.transform(X)
        return np.sum((x_scaled - x_recons) ** 2, axis=1)


class PLSRegressionLearner(SklLearnerRegression, _FeatureScorerMixin):
    __wraps__ = skl_pls.PLSRegression
    __returns__ = PLSModel
    supports_multiclass = True
    preprocessors = SklLearnerRegression.preprocessors

    def fit(self, X, Y, W=None):
        params = self.params.copy()
        params["n_components"] = min(X.shape[1] - 1,
                                     X.shape[0] - 1,
                                     params["n_components"])
        clf = self.__wraps__(**params)
        return self.__returns__(clf.fit(X, Y))

    # pylint: disable=unused-argument
    def __init__(self, n_components=2, scale=True,
                 max_iter=500, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def incompatibility_reason(self, domain):
        reason = None
        if not domain.class_vars:
            reason = "Numeric targets expected."
        else:
            for cv in domain.class_vars:
                if not cv.is_continuous:
                    reason = "Only numeric target variables expected."
        return reason

    @property
    def fitted_parameters(self) -> list[Learner.FittedParameter]:
        return [self.FittedParameter("n_components", "Components",
                                     int, 1, None)]


if __name__ == '__main__':
    import Orange

    housing = Orange.data.Table('housing')
    learners = [PLSRegressionLearner(n_components=2, max_iter=100)]
    res = Orange.evaluation.CrossValidation()(housing, learners)
    for learner, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print(f"learner: {learner}\nRMSE: {ca}\n")

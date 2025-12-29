import numpy as np
from AnyQt.QtCore import Qt
import scipy.sparse as sp

from Orange.data import Table, Domain, ContinuousVariable, StringVariable, \
    DiscreteVariable
from Orange.regression import PLSRegressionLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg


class OWPLS(OWBaseLearner):
    name = 'PLS'
    description = "Partial Least Squares Regression widget for multivariate data analysis"
    icon = "icons/PLS.svg"
    priority = 85
    keywords = "partial least squares"

    LEARNER = PLSRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        coefsdata = Output("Coefficients and Loadings", Table, explicit=True)
        data = Output("Data with Scores", Table)
        components = Output("Components", Table)

    class Warning(OWBaseLearner.Warning):
        sparse_data = Msg(
            'Sparse input data: default preprocessing is to scale it.')

    n_components = Setting(2)
    max_iter = Setting(500)
    scale = Setting(True)

    def add_main_layout(self):
        optimization_box = gui.vBox(
            self.controlArea, "Optimization Parameters")
        gui.spin(
            optimization_box, self, "n_components", 1, 50, 1,
            label="Components: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed)
        gui.spin(
            optimization_box, self, "max_iter", 5, 1000000, 50,
            label="Iteration limit: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed,
            checkCallback=self.settings_changed)
        gui.checkBox(optimization_box, self, "scale",
                     "Scale features and target",
                     callback=self.settings_changed)

    def update_model(self):
        super().update_model()
        coef_table = None
        data = None
        components = None
        if self.model is not None:
            coef_table = self._create_output_coeffs_loadings()
            data = self._create_output_data()
            components = self.model.components()
        self.Outputs.coefsdata.send(coef_table)
        self.Outputs.data.send(data)
        self.Outputs.components.send(components)

    def _create_output_coeffs_loadings(self) -> Table:
        intercept = self.model.intercept.T[None, :]
        coefficients = self.model.coefficients
        _, y_loadings = self.model.loadings
        x_rotations, _ = self.model.rotations

        n_targets, n_features = coefficients.shape
        n_components = x_rotations.shape[1]

        names = [f"coef ({v.name})" for v in self.model.domain.class_vars]
        names += [f"coef * X_sd ({v.name})" for v in self.model.domain.class_vars]
        names += [f"w*c {i + 1}" for i in range(n_components)]
        domain = Domain(
            [ContinuousVariable(n) for n in names],
            metas=[StringVariable("Variable name"),
                   DiscreteVariable("Variable role", ("Feature", "Target"))]
        )

        data = self.model.data_to_model_domain(self.data)
        X_features = np.hstack((coefficients.T,
                                (coefficients * np.std(data.X, axis=0)).T,
                                x_rotations))
        X_targets = np.hstack((np.full((n_targets, n_targets), np.nan),
                               np.full((n_targets, n_targets), np.nan),
                               y_loadings))

        coeffs = coefficients * np.mean(data.X, axis=0)
        X_intercepts = np.hstack((intercept - coeffs.sum(),
                                  intercept,
                                  np.full((1, n_components), np.nan)))
        X = np.vstack((X_features, X_targets, X_intercepts))

        variables = self.model.domain.variables
        M = np.array([[v.name for v in variables] + ["intercept"],
                      [0] * n_features + [1] * n_targets + [np.nan]],
                     dtype=object).T

        table = Table.from_numpy(domain, X=X, metas=M)
        table.name = "Coefficients and Loadings"
        return table

    def _create_output_data(self) -> Table:
        projection = self.model.project(self.data)
        normal_probs = self.model.residuals_normal_probability(self.data)
        dmodx = self.model.dmodx(self.data)
        data_domain = self.data.domain
        proj_domain = projection.domain
        nprobs_domain = normal_probs.domain
        dmodx_domain = dmodx.domain
        metas = data_domain.metas + proj_domain.attributes + proj_domain.metas + \
            nprobs_domain.attributes + dmodx_domain.attributes
        domain = Domain(data_domain.attributes, data_domain.class_vars, metas)
        data: Table = self.data.transform(domain)
        with data.unlocked(data.metas):
            data.metas[:, -2 * len(self.data.domain.class_vars) - 1: -1] = \
                normal_probs.X
            data.metas[:, -1] = dmodx.X[:, 0]
        return data

    @OWBaseLearner.Inputs.data
    def set_data(self, data):
        # reimplemented completely because the base learner does not
        # allow multiclass

        self.Warning.sparse_data.clear()

        self.Error.data_error.clear()
        self.data = data

        if data is not None and data.domain.class_var is None and not data.domain.class_vars:
            self.Error.data_error(
                "Data has no target variable.\n"
                "Select one with the Select Columns widget.")
            self.data = None

        # invalidate the model so that handleNewSignals will update it
        self.model = None

        if self.data and sp.issparse(self.data.X):
            self.Warning.sparse_data()

    def create_learner(self):
        common_args = {'preprocessors': self.preprocessors}
        return PLSRegressionLearner(n_components=self.n_components,
                                    scale=self.scale,
                                    max_iter=self.max_iter,
                                    **common_args)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPLS).run(Table("housing"))

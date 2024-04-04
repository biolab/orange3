import numpy as np
import scipy.sparse as sp
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
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
    keywords = ["partial least squares"]

    LEARNER = PLSRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        coefsdata = Output("Coefficients", Table, explicit=True)
        data = Output("Data", Table)
        components = Output("Components", Table)
        loadings = Output("Loadings", Table)

    class Warning(OWBaseLearner.Warning):
        sparse_data = Msg(
            'Sparse input data: default preprocessing is to scale it.')

    n_components = Setting(2)
    max_iter = Setting(500)

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

    def update_model(self):
        super().update_model()
        coef_table = data = components = loadings = None
        if self.model is not None:
            coef_table = self.model.coefficients_table()
            data = self._create_output_data()
            components = self.model.components()
            loadings = self._create_output_loadings()
        self.Outputs.coefsdata.send(coef_table)
        self.Outputs.data.send(data)
        self.Outputs.components.send(components)
        self.Outputs.loadings.send(loadings)

    def _create_output_data(self) -> Table:
        projection = self.model.project(self.data)
        normal_probs = self.model.residuals_normal_probability(self.data)
        dmodx = self.model.dmodx(self.data)
        data_domain = self.data.domain
        proj_domain = projection.domain
        nprobs_domain = normal_probs.domain
        dmodx_domain = dmodx.domain
        metas = proj_domain.metas + proj_domain.attributes + \
            nprobs_domain.attributes + dmodx_domain.attributes
        domain = Domain(data_domain.attributes, data_domain.class_vars, metas)
        data: Table = self.data.transform(domain)
        with data.unlocked(data.metas):
            data.metas[:, -2 * len(self.data.domain.class_vars) - 1: -1] = \
                normal_probs.X
            data.metas[:, -1] = dmodx.X[:, 0]
        return data

    def _create_output_loadings(self) -> Table:
        components = self.model.components()
        n_comp = len(components) // 2

        names = [f"Loading {i + 1}" for i in range(n_comp)]
        domain = Domain([ContinuousVariable(n) for n in names],
                        metas=[StringVariable("Feature name")])

        rotations = components[n_comp: n_comp * 2].X.T
        loadings = components[:n_comp].Y.T
        X = np.vstack([rotations, loadings])

        M = np.array([[v.name] for v in components.domain.variables])

        table = Table.from_numpy(domain, X, metas=M)
        table.name = "Loadings"
        return table

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
                                    max_iter=self.max_iter,
                                    **common_args)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPLS).run(Table("housing"))

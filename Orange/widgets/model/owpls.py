from AnyQt.QtCore import Qt
import scipy.sparse as sp

from Orange.data import Table
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
    keywords = ["partial least squares"]

    LEARNER = PLSRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        coefsdata = Output("Coefficients", Table, explicit=True)
        transformed = Output("Transformed Data", Table)
        components = Output("Components", Table)

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
        coef_table = None
        projection = None
        components = None
        if self.model is not None:
            coef_table = self.model.coefficients_table()
            projection = self.model.project(self.data)
            components = self.model.components()
        self.Outputs.coefsdata.send(coef_table)
        self.Outputs.transformed.send(projection)
        self.Outputs.components.send(components)

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

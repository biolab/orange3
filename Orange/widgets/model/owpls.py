from collections import OrderedDict

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel, QGridLayout
import scipy.sparse as sp

from Orange.data import Table
from Orange.modelling import PLSRegressionLearner
from Orange.widgets import gui
from Orange.widgets.widget import Msg
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

class OWPLS(OWBaseLearner):
    name = 'PLS'
    description = "Partial Least Squares algorithm for multivariate data analysis"
    icon = "icons/SVM.svg"
    priority = 3100
    keywords = ["partial least squares"]

    LEARNER = PLSRegressionLearner

    class Outputs(OWBaseLearner.Outputs):
        coefsdata = Output("Coefficients", Table, explicit=True)

    class Warning(OWBaseLearner.Warning):
        sparse_data = Msg('Input data is sparse, default preprocessing is to scale it.')

    #: Optimization Types
    simple, varsel, movwin  = range(3)
    #: Selected kernel type
    optimize_type = Setting(simple)

    #: number of components
    n_components = Setting(2)
    #: whether or not to limit number of iterations
    iters = Setting(500)

    def add_main_layout(self):
        
        self.optimization_box = gui.vBox(
            self.controlArea, "Optimization Parameters")
        self.ncomps_spin = gui.doubleSpin(
            self.optimization_box, self, "n_components", 1, 50, 1,
            label="Number of Components: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed)
        self.n_iters = gui.spin(
            self.optimization_box, self, "iters", 5, 1e6, 50,
            label="Iteration limit: ",
            alignment=Qt.AlignRight, controlWidth=100,
            callback=self.settings_changed,
            checkCallback=self.settings_changed)
        
    '''
    def update_model(self):
        super().update_model()
        coef_table = None
        if self.model is not None:
            domain = Domain(
                [ContinuousVariable("coef")], metas=[StringVariable("name")])
            coefs = [self.model.intercept] + list(self.model.coefficients)
            names = ["intercept"] + \
                    [attr.name for attr in self.model.domain.attributes]
            coef_table = Table.from_list(domain, list(zip(coefs, names)))
            coef_table.name = "coefficients"
        self.Outputs.coefsdata.send(coef_table)
    '''
    
    def set_data(self, data):
        self.Warning.sparse_data.clear()
        super().set_data(data)
        if self.data and sp.issparse(self.data.X):
            self.Warning.sparse_data()

    def handleNewSignals(self):
        self.settings_changed()
            
    def create_learner(self):
        common_args = {'preprocessors': self.preprocessors}
        return PLSRegressionLearner(n_components=int(self.n_components), max_iter=int(self.iters), **common_args)

if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPLS).run(Table("housing"))

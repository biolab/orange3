from PyQt4.QtGui import QSizePolicy, QLayout
import Orange.data
from Orange.regression import linear
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, settings, gui


class OWLinearRegression(widget.OWWidget):
    name = "Linear Regression"
    description = "A linear regression algorithm with optional L1 and L2 " \
                  "regularization."
    icon = "icons/LinearRegression.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", linear.RidgeRegressionLearner),
               ("Predictor", linear.LinearModel)]

    #: Types
    OLS, Ridge, Lasso = 0, 1, 2

    learner_name = settings.Setting("Linear Regression")
    ridge = settings.Setting(False)
    reg_type = settings.Setting(OLS)
    ridgealpha = settings.Setting(1.0)
    lassoalpha = settings.Setting(1.0)

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None

        box = gui.widgetBox(self.controlArea, "Learner/Predictor Name")
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, "Options")
        box = gui.radioButtons(box, self, "reg_type",
                               callback=self._reg_type_changed)

        gui.appendRadioButton(box, "Ordinary linear regression")
        gui.appendRadioButton(box, "Ridge regression")
        ibox = gui.indentedBox(box)
        gui.doubleSpin(ibox, self, "ridgealpha", 0.0, 1000.0, label="alpha:")
        self.ridge_box = ibox
        gui.appendRadioButton(box, "Lasso regression")
        ibox = gui.indentedBox(box)
        gui.doubleSpin(ibox, self, "lassoalpha", 0.0, 1000.0, label="alpha")
        self.lasso_box = ibox

        gui.button(self.controlArea, self, "Apply", callback=self.apply,
                   default=True)

        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        self.ridge_box.setEnabled(self.reg_type == self.Ridge)
        self.lasso_box.setEnabled(self.reg_type == self.Lasso)

        self.apply()

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        args = {"preprocessors": self.preprocessors}
        if self.reg_type == OWLinearRegression.OLS:
            learner = linear.LinearRegressionLearner(**args)
        elif self.reg_type == OWLinearRegression.Ridge:
            learner = linear.RidgeRegressionLearner(
                alpha=self.ridgealpha, **args)
        elif self.reg_type == OWLinearRegression.Lasso:
            learner = linear.RidgeRegressionLearner(
                alpha=self.lassoalpha, **args)
        else:
            assert False

        learner.name = self.learner_name
        predictor = None
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = self.learner_name

        self.send("Learner", learner)
        self.send("Predictor", predictor)

    def _reg_type_changed(self):
        self.ridge_box.setEnabled(self.reg_type == self.Ridge)
        self.lasso_box.setEnabled(self.reg_type == self.Lasso)

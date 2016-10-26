from itertools import chain
from PyQt4.QtGui import QLayout
from PyQt4.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.regression.linear import (
    LassoRegressionLearner, LinearRegressionLearner,
    RidgeRegressionLearner, ElasticNetLearner)
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWLinearRegression(OWBaseLearner):
    name = "Linear Regression"
    description = "A linear regression algorithm with optional L1 (LASSO), " \
                  "L2 (ridge) or L1L2 (elastic net) regularization."
    icon = "icons/LinearRegression.svg"
    priority = 60

    LEARNER = LinearRegressionLearner

    outputs = [("Coefficients", Table)]

    #: Types
    REGULARIZATION_TYPES = ["No regularization", "Ridge regression (L2)",
                            "Lasso regression (L1)", "Elastic net regression"]
    OLS, Ridge, Lasso, Elastic = 0, 1, 2, 3

    ridge = settings.Setting(False)
    reg_type = settings.Setting(OLS)
    alpha_index = settings.Setting(0)
    l1_ratio = settings.Setting(0.5)
    autosend = settings.Setting(True)

    want_main_area = False

    alphas = list(chain([x / 10000 for x in range(1, 10)],
                        [x / 1000 for x in range(1, 20)],
                        [x / 100 for x in range(2, 20)],
                        [x / 10 for x in range(2, 9)],
                        range(1, 20),
                        range(20, 100, 5),
                        range(100, 1001, 100)))

    def add_main_layout(self):
        box = gui.hBox(self.controlArea, "Regularization")
        gui.radioButtons(box, self, "reg_type",
                         btnLabels=self.REGULARIZATION_TYPES,
                         callback=self._reg_type_changed)

        gui.separator(box, 20, 20)
        self.alpha_box = box2 = gui.vBox(box, margin=10)
        gui.widgetLabel(box2, "Regularization strength:")
        self.alpha_slider = gui.hSlider(
            box2, self, "alpha_index",
            minValue=0, maxValue=len(self.alphas) - 1,
            callback=self._alpha_changed, createLabel=False)
        box3 = gui.hBox(box2)
        box3.layout().setAlignment(Qt.AlignCenter)
        self.alpha_label = gui.widgetLabel(box3, "")
        self._set_alpha_label()

        gui.separator(box2, 10, 10)
        box4 = gui.vBox(box2, margin=0)
        gui.widgetLabel(box4, "Elastic net mixing:")
        box5 = gui.hBox(box4)
        gui.widgetLabel(box5, "L1")
        self.l1_ratio_slider = gui.hSlider(
            box5, self, "l1_ratio", minValue=0.01, maxValue=1,
            intOnly=False, ticks=0.1, createLabel=False,
            step=0.01, callback=self._l1_ratio_changed)
        gui.widgetLabel(box5, "L2")

    def add_bottom_buttons(self):
        box5 = gui.hBox(self.controlArea)
        box5.layout().setAlignment(Qt.AlignCenter)
        self.l1_ratio_label = gui.widgetLabel(box5, "")
        self._set_l1_ratio_label()

        auto_commit = gui.auto_commit(
                self.controlArea, self, "autosend", "Apply")
        auto_commit.layout().insertWidget(0, self.report_button)
        auto_commit.layout().insertSpacing(1, 20)
        self.report_button.setMinimumWidth(150)

        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        self.alpha_slider.setEnabled(self.reg_type != self.OLS)
        self.l1_ratio_slider.setEnabled(self.reg_type == self.Elastic)

    def handleNewSignals(self):
        self.commit()

    def _reg_type_changed(self):
        self.alpha_slider.setEnabled(self.reg_type != self.OLS)
        self.l1_ratio_slider.setEnabled(self.reg_type == self.Elastic)
        self.commit()

    def _set_alpha_label(self):
        self.alpha_label.setText("Alpha: {}".format(self.alphas[self.alpha_index]))

    def _alpha_changed(self):
        self._set_alpha_label()
        self.commit()

    def _set_l1_ratio_label(self):
        self.l1_ratio_label.setText(
            "{:.{}f} : {:.{}f}".format(self.l1_ratio, 2, 1 - self.l1_ratio, 2))

    def _l1_ratio_changed(self):
        self._set_l1_ratio_label()
        self.commit()

    def commit(self):
        self.apply()

    def create_learner(self):
        alpha = self.alphas[self.alpha_index]
        preprocessors = self.preprocessors
        args = {"preprocessors": preprocessors}
        if self.reg_type == OWLinearRegression.OLS:
            learner = LinearRegressionLearner(**args)
        elif self.reg_type == OWLinearRegression.Ridge:
            learner = RidgeRegressionLearner(alpha=alpha, **args)
        elif self.reg_type == OWLinearRegression.Lasso:
            learner = LassoRegressionLearner(alpha=alpha, **args)
        elif self.reg_type == OWLinearRegression.Elastic:
            learner = ElasticNetLearner(alpha=alpha,
                                        l1_ratio=self.l1_ratio, **args)
        return learner

    def update_model(self):
        super().update_model()
        coef_table = None
        if self.valid_data:
            domain = Domain(
                    [ContinuousVariable("coef", number_of_decimals=7)],
                    metas=[StringVariable("name")])
            coefs = [self.model.intercept] + list(self.model.coefficients)
            names = ["intercept"] + \
                    [attr.name for attr in self.model.domain.attributes]
            coef_table = Table(domain, list(zip(coefs, names)))
            coef_table.name = "coefficients"
        self.send("Coefficients", coef_table)

    def get_learner_parameters(self):
        regularization = "No Regularization"
        if self.reg_type == OWLinearRegression.Ridge:
            regularization = ("Ridge Regression (L2) with α={}"
                              .format(self.alphas[self.alpha_index]))
        elif self.reg_type == OWLinearRegression.Lasso:
            regularization = ("Lasso Regression (L1) with α={}"
                              .format(self.alphas[self.alpha_index]))
        elif self.reg_type == OWLinearRegression.Elastic:
            regularization = ("Elastic Net Regression with α={}"
                              " and L1:L2 ratio of {}:{}"
                              .format(self.alphas[self.alpha_index],
                                      self.l1_ratio,
                                      1 - self.l1_ratio))
        return ("Regularization", regularization),


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWLinearRegression()
    d = Table('housing')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

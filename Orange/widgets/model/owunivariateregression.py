from AnyQt.QtWidgets import QSizePolicy
from AnyQt.QtGui import QColor, QPen, QFont, QPalette
from AnyQt.QtCore import Qt, QRectF

import pyqtgraph as pg
import numpy as np

from Orange.data import Table, Domain
from Orange.data.variable import ContinuousVariable, StringVariable
from Orange.regression.linear import (RidgeRegressionLearner, PolynomialLearner,
                                      LinearRegressionLearner, LinearModel)
from Orange.regression import Learner
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import settings, gui
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.sql import check_sql_input
from Orange.canvas import report


class OWUnivariateRegression(OWBaseLearner):
    name = "Univariate Polynomial Regression"
    description = "Univariate regression with polynomial expansion."
    icon = "icons/UnivariateRegression.svg"

    inputs = [("Learner", Learner, "set_learner")]

    outputs = [("Coefficients", Table)]

    LEARNER = PolynomialLearner

    learner_name = settings.Setting("Univariate Regression")

    polynomialexpansion = settings.Setting(1)

    x_var_index = settings.ContextSetting(0)
    y_var_index = settings.ContextSetting(1)

    want_main_area = True

    def add_main_layout(self):

        self.data = None
        self.preprocessors = None
        self.learner = None

        self.scatterplot_item = None
        self.plot_item = None

        self.x_label = 'x'
        self.y_label = 'y'

        box = gui.vBox(self.controlArea, "Variables")

        self.x_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesX = gui.comboBox(
            box, self, value='x_var_index', label="Input: ",
            orientation=Qt.Horizontal, callback=self.apply, contentsLength=12)
        self.comboBoxAttributesX.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.comboBoxAttributesX.setModel(self.x_var_model)
        gui.doubleSpin(
            gui.indentedBox(box),
            self, "polynomialexpansion", 0, 10,
            label="Polynomial expansion:", callback=self.apply)

        gui.separator(box, height=8)
        self.y_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesY = gui.comboBox(
            box, self, value="y_var_index", label="Target: ",
            orientation=Qt.Horizontal, callback=self.apply, contentsLength=12)
        self.comboBoxAttributesY.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.comboBoxAttributesY.setModel(self.y_var_model)

        gui.rubber(self.controlArea)

        # main area GUI
        self.plotview = pg.PlotWidget(background="w")
        self.plot = self.plotview.getPlotItem()

        axis_color = self.palette().color(QPalette.Text)
        axis_pen = QPen(axis_color)

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setLabel(self.x_label)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        axis = self.plot.getAxis("left")
        axis.setLabel(self.y_label)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0),
                           disableAutoRange=True)

        self.mainArea.layout().addWidget(self.plotview)

    def send_report(self):
        if self.data is None:
            return
        caption = report.render_items_vert((
             ("Polynomial Expansion: ", self.polynomialexpansion),
        ))
        self.report_plot(self.plot)
        if caption:
            self.report_caption(caption)



    def clear(self):
        self.data = None
        self.clear_plot()

    def clear_plot(self):
        if self.plot_item is not None:
            self.plot_item.setParentItem(None)
            self.plotview.removeItem(self.plot_item)
            self.plot_item = None

        if self.scatterplot_item is not None:
            self.scatterplot_item.setParentItem(None)
            self.plotview.removeItem(self.scatterplot_item)
            self.scatterplot_item = None

        self.plotview.clear()

    @check_sql_input
    def set_data(self, data):
        self.clear()
        self.data = data
        if data is not None:
            cvars = [var for var in data.domain.variables if var.is_continuous]
            class_cvars = [var for var in data.domain.class_vars if var.is_continuous]

            self.x_var_model[:] = cvars
            self.y_var_model[:] = cvars

            nvars = len(cvars)
            nclass = len(class_cvars)
            self.x_var_index = min(max(0, self.x_var_index), nvars - 1)
            if nclass > 0:
                self.y_var_index = min(max(0, nvars-nclass), nvars - 1)
            else:
                self.y_var_index = min(max(0, nvars-1), nvars - 1)

    def set_learner(self, learner):
        self.learner = learner

    def handleNewSignals(self):
        self.apply()

    def plot_scatter_points(self, x_data, y_data):
        if self.scatterplot_item:
            self.plotview.removeItem(self.scatterplot_item)
        self.n_points = len(x_data)
        self.scatterplot_item = pg.ScatterPlotItem(
            x=x_data, y=y_data, data=np.arange(self.n_points),
            symbol="o", size=10, pen=pg.mkPen(0.2), brush=pg.mkBrush(0.7),
            antialias=True)
        self.scatterplot_item.opts["useCache"] = False
        self.plotview.addItem(self.scatterplot_item)
        self.plotview.replot()

    def set_range(self, x_data, y_data):
        min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
        min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
        self.plotview.setRange(
            QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
            padding=0.025)
        self.plotview.replot()

    def plot_regression_line(self, x_data, y_data):
        if self.plot_item:
            self.plotview.removeItem(self.plot_item)
        self.plot_item = pg.PlotCurveItem(
            x=x_data, y=y_data,
            pen=pg.mkPen(QColor(255, 0, 0), width=3),
            antialias=True
        )
        self.plotview.addItem(self.plot_item)
        self.plotview.replot()

    def apply(self):
        learner = self.learner
        predictor = None

        if self.data is not None:

            degree = int(self.polynomialexpansion)
            learner = self.LEARNER(preprocessors=self.preprocessors,
                                   degree=degree,
                                   learner=LinearRegressionLearner() if self.learner is None
                                    else learner)

            attributes = self.x_var_model[self.x_var_index]
            class_var = self.y_var_model[self.y_var_index]
            data_table = Table(Domain([attributes], class_vars=[class_var]), self.data)

            learner.name = self.learner_name
            predictor = learner(data_table)

            preprocessed_data = data_table
            if self.preprocessors is not None:
                for preprocessor in self.preprocessors:
                    preprocessed_data = preprocessor(preprocessed_data)

            x = preprocessed_data.X.ravel()
            y = preprocessed_data.Y.ravel()

            linspace = np.linspace(min(x), max(x), 1000).reshape(-1,1)
            values = predictor(linspace, predictor.Value)

            self.plot_scatter_points(x, y)

            self.plot_regression_line(linspace.ravel(), values.ravel())

            x_label = self.x_var_model[self.x_var_index]
            axis = self.plot.getAxis("bottom")
            axis.setLabel(x_label)

            y_label = self.y_var_model[self.y_var_index]
            axis = self.plot.getAxis("left")
            axis.setLabel(y_label)

            self.set_range(x, y)

        self.send("Learner", learner)
        self.send("Predictor", predictor)

        # Send model coefficents
        model = None
        if predictor is not None:
            model = predictor.model
            if hasattr(model, "model"):
                model = model.model
            elif hasattr(model, "skl_model"):
                model = model.skl_model
        if model is not None and hasattr(model, "coef_"):
            domain = Domain([ContinuousVariable("coef", number_of_decimals=7)],
                            metas=[StringVariable("name")])
            coefs = [model.intercept_ + model.coef_[0]] + list(model.coef_[1:])
            names = ["1", x_label] + \
                    ["{}^{}".format(x_label, i) for i in range(2, degree + 1)]
            coef_table = Table(domain, list(zip(coefs, names)))
            self.send("Coefficients", coef_table)
        else:
            self.send("Coefficients", None)



if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWUnivariateRegression()
    learner = RidgeRegressionLearner(alpha=1.0)
    polylearner = PolynomialLearner(learner, degree=2)
    d = Table('iris')
    ow.set_data(d)
    ow.set_learner(learner)
    ow.show()
    a.exec_()
    ow.saveSettings()



from PyQt4.QtGui import QLayout

from Orange.data import Table, Domain
from Orange.regression.linear import (RidgeRegressionLearner, LinearModel,
                                      LinearRegressionLearner, PolynomialLearner)
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import widget, settings, gui
from Orange.widgets.utils import itemmodels

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4.QtGui import QColor, QPen
from PyQt4.QtCore import QRectF

import pyqtgraph as pg
import numpy as np


class OWUnivariateRegression(widget.OWWidget):
    name = "Univariate Regression"
    description = "Univariate regression algorithm with optional Linear " \
                  "regression widget and polynomial degree setting."
    icon = "icons/UnivariateRegression.svg"

    inputs = [("Data", Table, "set_data", widget.Default),
              ("Preprocessor", Preprocess, "set_preprocessor"),
              ("Learner", RidgeRegressionLearner, "set_learner")]
    outputs = [("Learner", RidgeRegressionLearner),
               ("Predictor", LinearModel)]
    
    learner_name = settings.Setting("Univariate Regression")
    
    polynomialexpansion = settings.Setting(1)
    
    x_var_index = settings.ContextSetting(0)
    y_var_index = settings.ContextSetting(1)
    
    want_main_area = True

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.preprocessors = None
        self.learner = None
        self.scatterplot_item = None
        self.plot_item = None
        
        self.x_label = 'x'
        self.y_label = 'y'
        
        box = gui.widgetBox(self.controlArea, "Learner/Predictor Name")
        gui.lineEdit(box, self, "learner_name")

        box = gui.widgetBox(self.controlArea, "Input")

        self.x_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesX = gui.comboBox(
            box, self, value='x_var_index',
            callback=self.apply)
        self.comboBoxAttributesX.setModel(self.x_var_model)

        gui.doubleSpin(box, self, "polynomialexpansion", 0, 10, 
            label="Polynomial expansion:", callback=self.apply)

        self.y_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesY = gui.comboBox(
            self.controlArea, self, value='y_var_index', box='Target',
            callback=self.apply)
        self.comboBoxAttributesY.setModel(self.y_var_model)

        self.layout().setSizeConstraint(QLayout.SetFixedSize)
        
        gui.rubber(self.controlArea)
        
        # main area GUI
        self.plotview = pg.PlotWidget(background="w")
        self.plot = self.plotview.getPlotItem()

        axis_color = self.palette().color(QtGui.QPalette.Text)
        axis_pen = QtGui.QPen(axis_color)

        tickfont = QtGui.QFont(self.font())
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

        self.apply()

    def set_data(self, data):
        self.data = data
        if data is not None:
            cvars = [var for var in data.domain.variables if var.is_continuous]
            self.x_var_model[:] = cvars
            self.y_var_model[:] = cvars
            
            nvars = len(cvars)
            self.x_var_index = min(max(0, self.x_var_index), nvars - 1)
            self.y_var_index = min(max(0, self.y_var_index), nvars - 1)
            
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()
    
    def set_learner(self, learner):
        self.learner = learner
        if learner is not None:
            self.apply()

    def handleNewSignals(self):
        self.apply()

    def plot_scatter_points(self, x_data, y_data):
        if self.scatterplot_item:
            self.plotview.removeItem(self.scatterplot_item)
        self.n_points = len(x_data)
        self.scatterplot_item = pg.ScatterPlotItem(
            x=x_data, y=y_data, data=np.arange(self.n_points),
        )
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
            pen=pg.mkPen(QColor(255, 0, 0), width=2), 
            antialias=True
        )
        self.plotview.addItem(self.plot_item)
        self.plotview.replot()
    
    def apply(self):
        args = {"preprocessors": self.preprocessors}
        learner = self.learner
        predictor = None
        
        if len(self.x_var_model) != 0:
            self.x_label = self.x_var_model[self.x_var_index]
            axis = self.plot.getAxis("bottom")
            axis.setLabel(self.x_label)
        
        if len(self.y_var_model) != 0:
            self.y_label = self.y_var_model[self.y_var_index]
            axis = self.plot.getAxis("left")
            axis.setLabel(self.y_label)
        
        if self.data is not None:
            if self.learner is None:
                learner = LinearRegressionLearner(**args)

            Y = self.data.Y
            if len(self.data.Y.shape) == 1:
                Y = self.data.Y.reshape(-1,1)
            data = np.hstack([self.data.X, Y])
            x = data[:,self.x_var_index]
            y = data[:,self.y_var_index]

            data_table = Table(x.reshape(-1,1), y)
            attributes = self.data.domain[self.x_var_index]
            class_var = self.data.domain[self.y_var_index]
            data_table.domain = Domain([attributes], class_vars=[class_var])

            degree = int(self.polynomialexpansion)

            learner = PolynomialLearner(learner, degree=degree)
            learner.name = self.learner_name
            predictor = learner(data_table)

            linspace = np.linspace(min(x), max(x), 1000).reshape(-1,1)
            values = predictor(linspace, predictor.Value)

            self.plot_scatter_points(x, y)
            self.set_range(x, y)
            self.plot_regression_line(linspace.ravel(), values.ravel())

        self.send("Learner", learner)
        self.send("Predictor", predictor)


if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

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
    
    

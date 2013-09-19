# -*- coding: utf-8 -*-

import math
import itertools
import numpy as np

from PyQt4 import QtCore, QtGui
import scipy
from scipy import ndimage
import scipy.special


import Orange
from Orange import feature, statistics
from Orange.data import discretization, Table
from Orange.statistics import basic_stats
from Orange.statistics import contingency, distribution, tests

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, DomainContextHandler,
                                     ContextSetting)
from Orange.widgets.utils import datacaching, colorpalette
from Orange.widgets.utils.plot import owaxis


class OWScatterPlot(widget.OWWidget):
    _name = "Scatter plot"
    _description = "Shows a scatter plot, or a heatmap for big data"
    _long_description = """"""
    _icon = "icons/ScatterPlot.svg"
    _priority = 100
    _author = "Janez Demšar"
    inputs = [("Data", Table, "data")]

    settingsHandler = DomainContextHandler()

    def __init__(self):
        super().__init__()

        iris = Table("iris")

        disc = feature.discretization.EqualWidth(n=10)
        diris = discretization.DiscretizeTable(iris, method=disc)
        cont = statistics.contingency.get_contingency(diris, 2, 3)
        cx, cy = np.mgrid[0:(cont.shape[0]-1):256j, 0:(cont.shape[1]-1):256j]
        coords = np.array([cx, cy])
        image = ndimage.map_coordinates(cont, coords)
        image -= np.min(image)
        image /= np.max(image)
        print(np.max(image))
        image *= 255
        image = np.dstack((image, ) * 4)

        im255 = image.flatten().astype(np.uint8)

        im = QtGui.QImage(im255.data, 256, 256, QtGui.QImage.Format_RGB32)
        pim = QtGui.QPixmap.fromImage(im)

        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.view)
        self.scene.addPixmap(pim)

if __name__ == "__main__":
    a = QtGui.QApplication([])
    w = OWScatterPlot()
    w.show()
    a.exec_()

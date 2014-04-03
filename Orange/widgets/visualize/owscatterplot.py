# -*- coding: utf-8 -*-

import math
import itertools
import numpy as np

from PyQt4 import QtCore, QtGui
import scipy
from scipy import ndimage
from scipy.ndimage.morphology import black_tophat
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

        def insert_rows_cols_zeros(table, rowoffset = 1, coloffset = 1):
            if rowoffset not in [0, 1] or coloffset not in [0, 1]:
                return table

            res = table.copy()
            for row in range(table.shape[0]):
                res = np.insert(res, row+row+rowoffset, values=0, axis=0)
            for col in range(table.shape[1]):
                res = np.insert(res, col+col+coloffset, values=0, axis=1)

            return res

        def fill_neighbours(tables):
            # works for three arrays == three colors
            # arrays must have every other row and cols set to zeros
            table_count = len(tables)
            for row in range(0, tables[0].shape[0]-1, 2):
                for col in range(0, tables[0].shape[1]-1, 2):
                    if tables[0][row, col] != 0:
                        if tables[1][row, col+1] == 0 and tables[2][row+1, col] == 0:
                            # B0 --> BB
                            # 00     BB
                            tables[0][row, col+1] = tables[0][row, col]
                            tables[0][row+1, col] = tables[0][row, col]
                            tables[0][row+1, col+1] = tables[0][row, col]
                        elif tables[1][row, col+1] != 0 and tables[2][row+1, col] == 0:
                            # BG --> BG
                            # 00     BG
                            tables[0][row+1, col] = tables[0][row, col]
                            tables[1][row+1, col+1] = tables[1][row, col+1]
                        elif tables[1][row, col+1] == 0 and tables[2][row+1, col] != 0:
                            # B0 --> BB
                            # R0     RR
                            tables[0][row, col+1] = tables[0][row, col]
                            tables[2][row+1, col+1] = tables[2][row+1, col]
                    elif tables[1][row, col+1] != 0:
                        if tables[0][row, col] == 0 and tables[2][row+1, col] == 0:
                            # 0G --> GG
                            # 00     GG
                            tables[1][row, col] = tables[1][row, col+1]
                            tables[1][row+1, col] = tables[1][row, col+1]
                            tables[1][row+1, col+1] = tables[1][row, col+1]
                        elif tables[0][row, col] != 0 and tables[2][row+1, col] == 0:
                            # BG --> BG
                            # 00     BG
                            tables[0][row+1, col] = tables[0][row, col]
                            tables[1][row+1, col+1] = tables[1][row, col+1]
                        elif tables[0][row, col] == 0 and tables[2][row+1, col] != 0:
                            # 0G --> GG
                            # R0     RR
                            tables[1][row, col] = tables[1][row, col+1]
                            tables[2][row+1, col+1] = tables[2][row+1, col]
                    elif tables[2][row+1, col] != 0:
                        if tables[0][row, col] == 0 and tables[1][row, col+1] == 0:
                            # 00 --> RR
                            # R0     RR
                            tables[2][row, col] = tables[2][row+1, col]
                            tables[2][row, col+1] = tables[2][row+1, col]
                            tables[2][row+1, col+1] = tables[2][row+1, col]
                        elif tables[0][row, col] != 0 and tables[1][row, col+1] == 0:
                            # B0 --> BB
                            # R0     RR
                            tables[0][col+1, row] = tables[0][col, row]
                            tables[2][row+1, col+1] = tables[2][row+1, col]
                        elif tables[0][row, col] == 0 and tables[1][row, col+1] != 0:
                            # duplicate elif
                            # 0G --> GG
                            # R0     RR
                            tables[1][row, col] = tables[1][row, col+1]
                            tables[2][row+1, col+1] = tables[2][row+1, col]



        def combine_into_one(tables):
            # can combine min 2 tables or max 4 tables
            table_count = len(tables)
            if table_count < 2 or table_count > 4:
                return []

            # can combine only tables of same shape
            shape = tables[0].shape
            for arr in tables:
                if arr.shape != shape:
                    return []

            combined_table = np.zeros(tables[0].shape)
            for row in range(0, shape[0] - 1, 2):
                for col in range(0, shape[1] - 1, 2):
                    combined_table[row, col] = tables[0][row, col]
                    combined_table[row, col+1] = tables[1][row, col+1]
                    if table_count > 2:
                        combined_table[row+1, col] = tables[2][row+1, col]
                    if table_count > 3:
                        combined_table[row+1, col+1] = tables[3][row+1, col+1]

            return combined_table

        def max_contingency_diff_shape(contingencies):
            new_contingencies = []
            # check number of contingencies
            cont_count = len(contingencies)
            if cont_count == 0:
                return new_contingencies
            if cont_count == 1:
                return new_contingencies

            # contingencies must be of same shape
            new_contingencies = reshape_to_max(contingencies)

            # set all values to 0, except if max
            for row in range(0, new_contingencies[0].shape[0]):
                for col in range(0, new_contingencies[0].shape[1]):
                    max_value = 0
                    # get max value
                    for cont in new_contingencies:
                        if cont[row, col] > max_value:
                            max_value = cont[row, col]
                    # set value to 0, if not max
                    # assign_anyway = False
                    for cont in new_contingencies:
                        # what if two values are equal (max)
                        if cont[row, col] != max_value:# or assign_anyway:
                            cont[row, col] = 0
                        # else:
                        #     assign_anyway = True

            return new_contingencies

        def max_contingency_same_shape(contingencies):
            # check number of contingencies
            cont_count = len(contingencies)
            if cont_count == 0:
                return
            if cont_count == 1:
                return

            # contingencies must be of same shape
            shape = contingencies[0].shape

            # set all values to 0, except if max
            for row in range(0, shape[0]):
                for col in range(0, shape[1]):
                    max_value = 0
                    # get max value
                    for cont in contingencies:
                        if cont[row, col] > max_value:
                            max_value = cont[row, col]
                    # set value to 0, if not max
                    # assign_anyway = False
                    for cont in contingencies:
                        # what if two values are equal (max)
                        if cont[row, col] != max_value:# or assign_anyway:
                            cont[row, col] = 0
                        # else:
                        #     assign_anyway = True

            return

        def reshape_to_max(tables):
            new_tables = []
            max_shape = tables[0].shape
            for cont in tables:
                if cont.shape > max_shape:
                    max_shape = cont.shape
            for cont in tables:
                if cont.shape != max_shape:
                    z = np.zeros(max_shape)
                    z[0:cont.shape[0], 0:cont.shape[1]] = cont
                    new_tables.append(z.copy())
                else:
                    new_tables.append(cont)
            return new_tables


        def set_values(table, value):
            for row in range(0, table.shape[0]):
                for col in range(0, table.shape[1]):
                    table[row, col] = value

        def set_values_list(tables, value):
            for table in tables:
                for row in range(0, table.shape[0]):
                    for col in range(0, table.shape[1]):
                        table[row, col] = value

        #iris = Table("iris")
        iris = Orange.data.sql.table.SqlTable(host='localhost', database='test', table='iris')

        disc = feature.discretization.EqualFreq(n=10)
        diris = discretization.DiscretizeTable(iris, method=disc)
        # cont = np.flipud(statistics.contingency.get_contingency(diris, 2, 3))

        filt = Orange.data.filter.Values()
        irisclass = iris.domain['iris']
        filt.domain = iris.domain
        fd = Orange.data.filter.FilterDiscrete(
                column = iris.domain.attributes.index(irisclass),
                values = ["Iris-setosa"])
        filt.conditions.append(fd)
        iris_setosa = filt(iris)
        # diris_setosa = filt(diris)


        filt = Orange.data.filter.Values()
        irisclass = iris.domain['iris']
        filt.domain = iris.domain
        fd = Orange.data.filter.FilterDiscrete(
                column = iris.domain.attributes.index(irisclass),
                values = ["Iris-versicolor"])
        filt.conditions.append(fd)
        iris_versicolor = filt(iris)
        # diris_versicolor = filt(diris)

        filt = Orange.data.filter.Values()
        irisclass = iris.domain['iris']
        filt.domain = iris.domain
        fd = Orange.data.filter.FilterDiscrete(
                column = iris.domain.attributes.index(irisclass),
                values = ["Iris-virginica"])
        filt.conditions.append(fd)
        iris_virginica = filt(iris)
        # diris_virginica = filt(diris)

        disc = feature.discretization.EqualFreq(n=10)
        diris_setosa = discretization.DiscretizeTable(iris_setosa, method=disc)
        diris_versicolor = discretization.DiscretizeTable(iris_versicolor, method=disc)
        diris_virginica = discretization.DiscretizeTable(iris_virginica, method=disc)

        cont_setosa = np.flipud(statistics.contingency.get_contingency(diris_setosa, 2, 3))
        cont_versicolor = np.flipud(statistics.contingency.get_contingency(diris_versicolor, 2, 3))
        cont_virginica = np.flipud(statistics.contingency.get_contingency(diris_virginica, 2, 3))
        cont_zeros = np.zeros(cont_setosa.shape)

        print("*" * 40)
        print("cont_setosa: " + str(cont_setosa.shape))
        print(cont_setosa)
        print("cont_versicolor: " + str(cont_versicolor.shape))
        print(cont_versicolor)
        print("cont_virginica: " + str(cont_virginica.shape))
        print(cont_virginica)

        cont_setosa, cont_versicolor, cont_virginica = reshape_to_max([cont_setosa, cont_versicolor, cont_virginica])
        max_contingency_same_shape([cont_setosa, cont_versicolor, cont_virginica])

        cont_setosa = insert_rows_cols_zeros(cont_setosa)
        cont_versicolor = insert_rows_cols_zeros(cont_versicolor, coloffset=0)
        cont_virginica = insert_rows_cols_zeros(cont_virginica, rowoffset=0)
        cont_zeros = np.zeros(cont_setosa.shape)
        #
        fill_neighbours([cont_setosa, cont_versicolor, cont_virginica])

        # cont = combine_into_one([cont_setosa_z, cont_versicolor_z, cont_virginica_z])

        # set_values_list([cont_setosa, cont_versicolor, cont_virginica], 0.)
        # cont_setosa, cont_versicolor, cont_virginica = max_contingency_diff_shape([cont_setosa, cont_versicolor, cont_virginica])
        # max_contingency_diff_shape([cont_setosa, cont_versicolor, cont_virginica])
        # max_contingency_same_shape([cont_setosa, cont_versicolor, cont_virginica])

        print("cont_setosa: " + str(cont_setosa.shape))
        print(cont_setosa)
        print("cont_versicolor: " + str(cont_versicolor.shape))
        print(cont_versicolor)
        print("cont_virginica: " + str(cont_virginica.shape))
        print(cont_virginica)
        print("*" * 40)

        # print(cont_setosa.shape)
        # print(cont_setosa)
        # print(cont_setosa_z)
        #
        # print(cont_versicolor.shape)
        # print(cont_versicolor)
        # print(cont_versicolor_z)
        #
        # print(cont_virginica.shape)
        # print(cont_virginica)
        # print(cont_virginica_z)


        # cont1 = insert_rows_cols_zeros(cont)
        # cont2 = insert_rows_cols_zeros(cont, coloffset=0)
        # cont3 = insert_rows_cols_zeros(cont, rowoffset=0)
        # cont = combine_into_one([cont1, cont2, cont3])

        # blue = np.array([1.] * 100)
        # blue = blue.reshape((10, 10))
        # green = np.array([1.] * 100)
        # green = green.reshape((10, 10))
        # red = np.array([1.] * 100)
        # red = red.reshape((10, 10))
        # black = np.array([0] * 100)
        # black = black.reshape((10, 10))
        # print("blue: " + str(blue.shape))
        # print(blue)
        # print("green: " + str(green.shape))
        # print(green)
        # print("red: " + str(red.shape))
        # print(red)
        # blue = insert_rows_cols_zeros(blue)
        # green = insert_rows_cols_zeros(green, coloffset=0)
        # red = insert_rows_cols_zeros(red, rowoffset=0)
        # black = insert_rows_cols_zeros(black, rowoffset=0, coloffset=0)
        # cont = combine_into_one([blue, green, red])


        # cx, cy = np.mgrid[0:(cont.shape[0]-1):256j, 0:(cont.shape[1]-1):256j]
        # coords = np.array([cx, cy])
        # image = ndimage.map_coordinates(cont, coords)
        # image -= np.min(image)
        # image /= np.max(image)
        # print(np.max(image))
        # image *= 255
        # print("cont: " + str(cont.shape))
        # print(cont)

        # cx, cy = np.mgrid[0:(blue.shape[0]-1):256j, 0:(blue.shape[1]-1):256j]
        # coords = np.array([cx, cy])
        # blue_image = ndimage.map_coordinates(blue, coords)
        # blue_image -= np.min(blue_image)
        # blue_image /= np.max(blue_image)
        # print(np.max(blue_image))
        # blue_image *= 255
        # print("blue: " + str(blue.shape))
        # print(blue)
        # print("blue coords: " + str(coords.shape))
        # print(coords)
        # print("blueimage: " +str(blue_image.shape))
        # print(blue_image)
        #
        #
        # cx, cy = np.mgrid[0:(green.shape[0]-1):256j, 0:(green.shape[1]-1):256j]
        # coords = np.array([cx, cy])
        # green_image = ndimage.map_coordinates(green, coords)
        # green_image -= np.min(green_image)
        # green_image /= np.max(green_image)
        # print(np.max(green_image))
        # green_image *= 255
        #
        # cx, cy = np.mgrid[0:(red.shape[0]-1):256j, 0:(red.shape[1]-1):256j]
        # coords = np.array([cx, cy])
        # red_image = ndimage.map_coordinates(red, coords)
        # red_image -= np.min(red_image)
        # red_image /= np.max(red_image)
        # print(np.max(red_image))
        # red_image *= 255
        # print("red: " + str(red.shape))
        # print(red)
        #
        # cx, cy = np.mgrid[0:(black.shape[0]-1):256j, 0:(black.shape[1]-1):256j]
        # coords = np.array([cx, cy])
        # black_image = ndimage.map_coordinates(black, coords)
        # black_image -= np.min(black_image)
        # black_image /= np.max(black_image)
        # print(np.max(black_image))
        # black_image *= 255
        # print("black: " + str(black.shape))
        # print(black)

        cx, cy = np.mgrid[0:(cont_setosa.shape[0]-1):256j, 0:(cont_setosa.shape[1]-1):256j]
        coords = np.array([cx, cy])
        setosa_image = ndimage.map_coordinates(cont_setosa, coords)
        setosa_image -= np.min(setosa_image)
        setosa_image /= np.max(setosa_image)
        print(np.max(setosa_image))
        setosa_image *= 255
        # print("cont_setosa: " + str(cont_setosa.shape))
        # print(cont_setosa)
        # print("cont_setosa coords: " + str(coords.shape))
        # print(coords)
        # print("setosa_image: " +str(setosa_image.shape))
        # print(setosa_image)

        cx, cy = np.mgrid[0:(cont_versicolor.shape[0]-1):256j, 0:(cont_versicolor.shape[1]-1):256j]
        coords = np.array([cx, cy])
        versicolor_image = ndimage.map_coordinates(cont_versicolor, coords)
        versicolor_image -= np.min(versicolor_image)
        versicolor_image /= np.max(versicolor_image)
        print(np.max(versicolor_image))
        versicolor_image *= 255
        # print("cont_versicolor: " + str(cont_versicolor.shape))
        # print(cont_versicolor)
        # print("cont_versicolor coords: " + str(coords.shape))
        # print(coords)
        # print("versicolor_image: " +str(versicolor_image.shape))
        # print(versicolor_image)
        #
        cx, cy = np.mgrid[0:(cont_virginica.shape[0]-1):256j, 0:(cont_virginica.shape[1]-1):256j]
        coords = np.array([cx, cy])
        virginica_image = ndimage.map_coordinates(cont_virginica, coords)
        virginica_image -= np.min(virginica_image)
        virginica_image /= np.max(virginica_image)
        print(np.max(virginica_image))
        virginica_image *= 255
        # print("cont_virginica: " + str(cont_virginica.shape))
        # print(cont_virginica)
        # print("cont_virginica coords: " + str(coords.shape))
        # print(coords)
        # print("virginica_image: " + str(virginica_image.shape))
        # print(virginica_image)

        cx, cy = np.mgrid[0:(cont_zeros.shape[0]-1):256j, 0:(cont_zeros.shape[1]-1):256j]
        coords = np.array([cx, cy])
        zeros_image = ndimage.map_coordinates(cont_zeros, coords)
        zeros_image -= np.min(zeros_image)
        zeros_image /= np.max(zeros_image)
        print(np.max(zeros_image))
        zeros_image *= 255

        # cx, cy = np.mgrid[0:(black.shape[0]-1):256j, 0:(black.shape[1]-1):256j]
        # coords = np.array([cx, cy])
        # blackimage = ndimage.map_coordinates(black, coords)
        # blackimage -= np.min(blackimage)
        # blackimage /= np.max(blackimage)
        # print(np.max(blackimage))
        # blackimage *= 255

        # setosa - blue, versicolor - green, virginica - red
        # image = np.dstack((setosa_image, zeros_image, zeros_image, zeros_image))
        # image = np.dstack((zeros_image, versicolor_image, zeros_image, zeros_image))
        # image = np.dstack((zeros_image, zeros_image, virginica_image, zeros_image))
        image = np.dstack((setosa_image, versicolor_image, virginica_image, zeros_image))
        # image = np.dstack((blue_image, green_image, red_image, black_image))
        # image = np.dstack((image, image, image, image ))

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

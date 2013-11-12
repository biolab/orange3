import Orange
from Orange import feature, statistics
from Orange.data import discretization, Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import contingency
from Orange.widgets import widget, gui

from PyQt4 import QtGui, QtCore

import numpy as np

from scipy import ndimage

import time


class OWHeatmap(widget.OWWidget):
    _name = "Heat map"
    _description = "Shows heat maps"
    #_long_description = """Shows itself to see if added correctly."""
    # _icon = "../icons/Dlg_zoom.png"
    _priority = 100

    inputs = [("Data", Table, "data")]
    outputs = [("Sampled data", Table)]
    # inputs = [("Data", SqlTable, "data")]
    # outputs = [("Sampled data", SqlTable)]

    X_attributes_select = 0
    Y_attributes_select = 1
    classvar_select = 0
    discretization_select = 0
    n_discretization_intervals = 10
    check_insert_rows_cols = 0
    color_array = np.array([[  0,   0, 255],   # blue
                            [  0, 255,   0],   # green
                            [255,   0,   0],   # red
                            [255, 255,   0],   # yellow
                            [255,  10, 255],   # magenta
                            [  0, 255, 255],   # aqua
                            [128,  10, 203],   # violet
                            [255, 107,   0],   # orange
                            [223, 224, 249]])  # lavender

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(self, parent, signalManager, settings, "Heat map")
        print("***OWHeatmap.__init__()")

        controlBox = gui.widgetBox(self.controlArea)
        self.labelDataInput = gui.widgetLabel(controlBox, 'No data input yet')
        self.labelOutput = gui.widgetLabel(controlBox, '')

        mainBox = gui.widgetBox(self.mainArea)
        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.view)

    def data(self, dataset):
        print("***OWHeatmap.data()")
        if dataset:
            self.dataset = dataset

            self.comboBoxAttributesX = gui.comboBox(self.controlArea, self, value='X_attributes_select', box='X Attribute',
                                                    items=[a.name for a in dataset.domain.attributes],
                                                    callback=self.attr_changed)
            self.comboBoxAttributesY = gui.comboBox(self.controlArea, self, value='Y_attributes_select', box='Y Attribute',
                                                    items=[a.name for a in dataset.domain.attributes],
                                                    callback=self.attr_changed)
            self.comboBoxClassvars = gui.comboBox(self.controlArea, self, value='classvar_select', box='Color by',
                                                    items=[a.name for a in dataset.domain.class_vars],
                                                    callback=self.attr_changed)
            self.checkBoxesColorsShownBox = gui.widgetBox(self.controlArea, box='Colors shown')
            self.checkBoxesColorsShown = []
            self.checkBoxesColorsShownValueList = []
            for (index, value) in enumerate(dataset.domain.class_vars[self.classvar_select].values):
                setattr(self, 'check_%s' % value, True)
                self.checkBoxesColorsShownValueList.append('check_%s' % value)
                self.checkBoxesColorsShown.append(gui.checkBox(self.checkBoxesColorsShownBox, self, value='check_%s' % value,
                                                               label=value, callback=self.attr_changed))
                self.checkBoxesColorsShown[index].setStyleSheet('QCheckBox {color: rgb(%i, %i, %i)}' % (self.color_array[index][0],
                                                                                                        self.color_array[index][1],
                                                                                                        self.color_array[index][2]))

            self.comboBoxDiscretization = gui.comboBox(self.controlArea, self, value='discretization_select',
                                                       box='Discretization method', items=['EqualFreq', 'EqualWidth'],
                                                       callback=self.attr_changed)
            self.lineEditDiscretizationIntervals = gui.lineEdit(self.controlArea, self, value='n_discretization_intervals',
                                                                box='Number of discretization intervals',
                                                                orientation='horizontal', valueType=int,
                                                                callback=self.attr_changed, enterPlaceholder=True)
            self.checkBoxInsertRowsCols = gui.checkBox(self.controlArea, self, value='check_insert_rows_cols',
                                                       label='Insert rows and cols', callback=self.attr_changed)
            self.X_attributes_select = 0
            self.Y_attributes_select = 1
            self.classvar_select = 0
            self.discretization_select = 0
            self.n_discretization_intervals = 10
            self.check_insert_rows_cols = 0
            self.labelDataInput.setText('Data set: %s\nInstances: %d' % (self.dataset.name, len(dataset)))
            sample = dataset
            self.send("Sampled data", sample)
        else:
            self.labelDataInput.setText('No data input anymore')
            self.send("Sampled data", None)
        self.display_changed()

    def attr_changed(self):
        print("***OWHeatmap.attr_changed()")
        self.display_changed()

    def display_changed(self):
        print("***OWHeatmap.display_changed()")
        if self.dataset == None:
            return

        if self.n_discretization_intervals < 2:
            return

        if self.discretization_select == 0:
            disc = feature.discretization.EqualFreq(n=self.n_discretization_intervals)
        else:
            disc = feature.discretization.EqualWidth(n=self.n_discretization_intervals)
        disc_data = discretization.DiscretizeTable(self.dataset, method=disc)

        filt_data_by_class_var = []
        for cv_value in self.dataset.domain.class_var.values:
            filt = Orange.data.filter.Values()
            filt.domain = self.dataset.domain
            fd = Orange.data.filter.FilterDiscrete(
                column = self.dataset.domain.class_var,
                values = [cv_value])
            filt.conditions.append(fd)
            filt_data_by_class_var.append((cv_value, filt(disc_data)))

        checkedindices = []
        for i in range(len(self.checkBoxesColorsShownValueList)):
            checkedindices.append(getattr(self, self.checkBoxesColorsShownValueList[i]))

        contingencies = []
        for (index, (value, filt_data)) in enumerate(filt_data_by_class_var):
            cont = np.flipud(statistics.contingency.get_contingency(
                            filt_data, self.X_attributes_select, self.Y_attributes_select))
            if checkedindices[index]:
                contingencies.append(cont)
            else:
                contingencies.append(np.zeros(cont.shape))

        if self.check_insert_rows_cols:
            if len(contingencies) >= 3:
                contingencies[0] = self.insert_rows_cols_zeros(contingencies[0])
                contingencies[1] = self.insert_rows_cols_zeros(contingencies[1], coloffset=0)
                contingencies[2] = self.insert_rows_cols_zeros(contingencies[2], rowoffset=0)

                self.fill_neighbours(contingencies[0:3])

        image_width = 512
        image_height = 512
        images = np.zeros((len(contingencies), image_width, image_height))
        for (i, cont) in enumerate(contingencies):
            if checkedindices[i]:
                cx, cy = np.mgrid[0:(cont.shape[0]-1):512j, 0:(cont.shape[1]-1):512j]
                coords = np.array([cx, cy])
                image = ndimage.map_coordinates(1-0.3**cont, coords)
                image -= np.min(image)
                image /= np.max(image)
                # image *= 255
                images[i] = image
            print(np.max(images[i]))

        images_argmax = images.argmax(axis=0)
        rows, cols = np.indices(images_argmax.shape)
        images_valmax = images[images_argmax, rows, cols]
        colors = self.color_array[images_argmax] * images_valmax[:, :, None]

        # when creating the image, blue and red are swapped; the color order is:
        # image = np.dstack(("blue", "green", "red", "nothing"))
        image = np.dstack((colors[:, :, 2],
                           colors[:, :, 1],
                           colors[:, :, 0],
                           np.zeros((image_width, image_height))))
        im255 = image.flatten().astype(np.uint8)
        im = QtGui.QImage(im255.data, image_width, image_height, QtGui.QImage.Format_RGB32)
        pim = QtGui.QPixmap.fromImage(im)
        self.scene.addPixmap(pim)

    def resizeEvent(self, event):
        print("***OWHeatmap.resizeEvent()")
        super().resizeEvent(event)

    # def max_contingency(self, contingencies):
    #     # check number of contingencies
    #     cont_count = len(contingencies)
    #     if cont_count == 0:
    #         return
    #     if cont_count == 1:
    #         return
    #
    #     # contingencies must be of same shape
    #     shape = contingencies[0].shape
    #
    #     # set all values to 0, except if max
    #     for row in range(0, shape[0]):
    #         for col in range(0, shape[1]):
    #             max_value = 0
    #             max_index = -1
    #             # get max value
    #             for i in range(len(contingencies)):
    #                 if contingencies[i][row, col] > max_value:
    #                     max_value = contingencies[i][row, col]
    #                     max_index = i
    #             # set value to 0, if not max
    #
    #             print("max_value=%.1f" % max_value, end="    ")
    #             for i in range(len(contingencies)):
    #                 print(contingencies[i][row, col], end=" ")
    #             print("  -->  ", end="")
    #
    #             if max_index != -1:
    #                 indices = list(range(len(contingencies)))
    #                 indices.remove(max_index)
    #                 for i in indices:
    #                     contingencies[i][row, col] = 0
    #
    #             for i in range(len(contingencies)):
    #                 print(contingencies[i][row, col], end=" ")
    #             print()
    #
    #     return
    #
    # def select_single_contingency(self, contingencies, colorPreferences):
    #     # check number of contingencies
    #     cont_count = len(contingencies)
    #     if cont_count == 0:
    #         return
    #     if cont_count == 1:
    #         return
    #
    #     # contingencies must be of same shape
    #     shape = contingencies[0].shape
    #
    #     # set all values to 0, except one;
    #     # when single max value, select that one
    #     # when multiple max values, select preferred one
    #     for row in range(0, shape[0]):
    #         for col in range(0, shape[1]):
    #             max_value = 0
    #             max_indices = []
    #             value_set = []
    #             # get max value and value set for this row&col
    #             for i in range(len(contingencies)):
    #                 if contingencies[i][row, col] > max_value:
    #                     max_value = contingencies[i][row, col]
    #                 if contingencies[i][row, col] not in value_set:
    #                     value_set.append(contingencies[i][row, col])
    #
    #             for i in range(len(contingencies)):
    #                 if contingencies[i][row, col] == max_value:
    #                     max_indices.append(i)
    #
    #             print("max_value=%.1f" % max_value, end="    ")
    #             for i in range(len(contingencies)):
    #                 print(contingencies[i][row, col], end=" ")
    #             print("  -->  ", end="")
    #
    #             for i in range(len(contingencies)):
    #                 if len(value_set) != len(contingencies):
    #                     # some values are the same --> select preferred
    #                     if i in max_indices and len(max_indices) == 1:
    #                         continue
    #                     elif i in max_indices:
    #                         nonzero_coloPreferences = []
    #                         for j in range(len(contingencies)):
    #                             if contingencies[j][row, col] == max_value:
    #                                 nonzero_coloPreferences.append(colorPreferences[j])
    #                             else:
    #                                 nonzero_coloPreferences.append(-1)
    #                         if nonzero_coloPreferences[i] != max(nonzero_coloPreferences):
    #                             contingencies[i][row, col] = 0
    #                     elif i not in max_indices:
    #                         if contingencies[i][row, col] < max_value:
    #                             contingencies[i][row, col] = 0
    #                             continue
    #                         indices_with_same_values = []
    #                         for j in range(len(contingencies)):
    #                             if j == i:
    #                                 continue
    #                             if contingencies[i][row, col] == contingencies[j][row, col]:
    #                                 indices_with_same_values.append(j)
    #                         if len(indices_with_same_values) != 0:
    #                             for isw in indices_with_same_values:
    #                                 if colorPreferences[i] > colorPreferences[isw]:
    #                                     contingencies[i][row, col] = 0
    #                     else:
    #                         contingencies[i][row, col] = 0
    #                 else:
    #                     # all values are different --> select max
    #                     if contingencies[i][row, col] != max_value:
    #                         contingencies[i][row, col] = 0
    #
    #             value_set2 = []
    #             for i in range(len(contingencies)):
    #                 print(contingencies[i][row, col], end=" ")
    #                 if contingencies[i][row, col] not in value_set2:
    #                     value_set2.append(contingencies[i][row, col])
    #             value_set2.remove(0.0)
    #             if len(value_set2) > 1:
    #                 print(" *", end="")
    #             print()
    #
    #     return
    #
    # def insert_rows_cols_zeros(self, table, rowoffset = 1, coloffset = 1):
    #     if rowoffset not in [0, 1] or coloffset not in [0, 1]:
    #         return table
    #
    #     res = table.copy()
    #     for row in range(table.shape[0]):
    #         res = np.insert(res, row+row+rowoffset, values=0, axis=0)
    #     for col in range(table.shape[1]):
    #         res = np.insert(res, col+col+coloffset, values=0, axis=1)
    #
    #     return res
    #
    # def fill_neighbours(self, tables):
    #     # works for three arrays == three colors
    #     # arrays must have every other row and cols set to zeros
    #     table_count = len(tables)
    #     for row in range(0, tables[0].shape[0]-1, 2):
    #         for col in range(0, tables[0].shape[1]-1, 2):
    #             if tables[0][row, col] != 0:
    #                 if tables[1][row, col+1] == 0 and tables[2][row+1, col] == 0:
    #                     # B0 --> BB
    #                     # 00     BB
    #                     tables[0][row, col+1] = tables[0][row, col]
    #                     tables[0][row+1, col] = tables[0][row, col]
    #                     tables[0][row+1, col+1] = tables[0][row, col]
    #                 elif tables[1][row, col+1] != 0 and tables[2][row+1, col] == 0:
    #                     # BG --> BG
    #                     # 00     BG
    #                     tables[0][row+1, col] = tables[0][row, col]
    #                     tables[1][row+1, col+1] = tables[1][row, col+1]
    #                 elif tables[1][row, col+1] == 0 and tables[2][row+1, col] != 0:
    #                     # B0 --> BB
    #                     # R0     RR
    #                     tables[0][row, col+1] = tables[0][row, col]
    #                     tables[2][row+1, col+1] = tables[2][row+1, col]
    #             elif tables[1][row, col+1] != 0:
    #                 if tables[0][row, col] == 0 and tables[2][row+1, col] == 0:
    #                     # 0G --> GG
    #                     # 00     GG
    #                     tables[1][row, col] = tables[1][row, col+1]
    #                     tables[1][row+1, col] = tables[1][row, col+1]
    #                     tables[1][row+1, col+1] = tables[1][row, col+1]
    #                 elif tables[0][row, col] != 0 and tables[2][row+1, col] == 0:
    #                     # BG --> BG
    #                     # 00     BG
    #                     tables[0][row+1, col] = tables[0][row, col]
    #                     tables[1][row+1, col+1] = tables[1][row, col+1]
    #                 elif tables[0][row, col] == 0 and tables[2][row+1, col] != 0:
    #                     # 0G --> GG
    #                     # R0     RR
    #                     tables[1][row, col] = tables[1][row, col+1]
    #                     tables[2][row+1, col+1] = tables[2][row+1, col]
    #             elif tables[2][row+1, col] != 0:
    #                 if tables[0][row, col] == 0 and tables[1][row, col+1] == 0:
    #                     # 00 --> RR
    #                     # R0     RR
    #                     tables[2][row, col] = tables[2][row+1, col]
    #                     tables[2][row, col+1] = tables[2][row+1, col]
    #                     tables[2][row+1, col+1] = tables[2][row+1, col]
    #                 elif tables[0][row, col] != 0 and tables[1][row, col+1] == 0:
    #                     # B0 --> BB
    #                     # R0     RR
    #                     tables[0][col+1, row] = tables[0][col, row]
    #                     tables[2][row+1, col+1] = tables[2][row+1, col]
    #                 elif tables[0][row, col] == 0 and tables[1][row, col+1] != 0:
    #                     # duplicate elif
    #                     # 0G --> GG
    #                     # R0     RR
    #                     tables[1][row, col] = tables[1][row, col+1]
    #                     tables[2][row+1, col+1] = tables[2][row+1, col]


from PyQt4.QtGui import QApplication

__author__ = 'jurre'
import pyqtgraph as pg
import Orange
from Orange import feature, statistics
from Orange.data import discretization, Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import contingency
from Orange.widgets import widget, gui

from PyQt4 import QtGui, QtCore

import numpy as np

import re
import heapq

from datetime import datetime


class Heatmap(pg.ImageItem):
    def __init__(self, image=None):
        if image is not None:
            self.image = image
        else:
            self.image = np.zeros((500, 500))

        pg.ImageItem.__init__(self, self.image)

    def getImage_(self):
        return self.image

    def setImage_(self, image):
        self.image = image
        self.render()
        self.update()

    def updateImage_(self, image, image_rect):
        if image_rect.x() + image.shape[1] > self.image.shape[1]:
            image_width = self.image.shape[1] - int(image_rect.x())
        else:
            image_width = image.shape[1]
        if image_rect.y() + image.shape[0] > self.image.shape[0]:
            image_height = self.image.shape[0] - int(image_rect.y())
        else:
            image_height = image.shape[0]

        if image_width <= 0 or image_height <= 0:
            # this is the case where user selected a part of region that is outside of X and Y values
            return

        self.image[image_rect.x() : image_rect.x()+image_width, image_rect.y() : image_rect.y()+image_height] = image[:image_width, :image_height]

        self.render()
        self.update()

    def drawRect(self, rect):
        self.image[rect.x()                        , rect.y() : rect.y()+rect.height()] = 0
        self.image[rect.x()+rect.width()-1         , rect.y() : rect.y()+rect.height()] = 0
        self.image[rect.x() : rect.x()+rect.width(), rect.y()] = 0
        self.image[rect.x() : rect.x()+rect.width(), rect.y()+rect.height()-1] = 0

    def mapRectFromView_(self):
        vb = self.getViewBox()
        return self.mapRectFromView(vb.viewRect())

class OWHeatmap(widget.OWWidget):
    """
    OWHeatmap draws a heatmap.

    Data is first drawn with less precision (big rects) and gets updated to more detail (smaller rects).
    This takes some time, so the heatmap gets updated, when more detail is calculated.
    """
    name = "Heat map"
    description = "Draws a heat map."
    # long_description = """Long description"""
    # icon = "../icons/Dlg_zoom.png"
    author = "Jure Bergant"
    priority = 100

    inputs = [("Data", Table, "data")]
    outputs = [("Sampled data", Table)]

    X_attributes_select = 2
    Y_attributes_select = 3
    classvar_select = 0
    check_use_cache = 1
    n_discretization_intervals = 10
    radio_mouse_behaviour = 0
    color_array = np.array([[  0,   0, 255],   # blue
                            [  0, 255,   0],   # green
                            [255,   0,   0],   # red
                            [255, 255,   0],   # yellow
                            [255,   0, 255],   # magenta
                            [  0, 255, 255],   # aqua
                            [128,  10, 203],   # violet
                            [255, 107,   0],   # orange
                            [223, 224, 249]])  # lavender
    default_image_width = 500
    default_image_height = 500

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(self, parent, signalManager, settings, "Heat map")

        controlBox = gui.widgetBox(self.controlArea)
        self.labelDataInput = gui.widgetLabel(controlBox, 'No data input yet')
        self.labelOutput = gui.widgetLabel(controlBox, '')

        self.icons = gui.attributeIconDict

        self.comboBoxAttributesX = gui.comboBox(self.controlArea, self, value='X_attributes_select', box='X Attribute',
                                                callback=self.attrChanged)
        self.comboBoxAttributesY = gui.comboBox(self.controlArea, self, value='Y_attributes_select', box='Y Attribute',
                                                callback=self.attrChanged)
        self.comboBoxClassvars = gui.comboBox(self.controlArea, self, value='classvar_select', box='Color by',
                                                callback=self.attrChanged)
        self.checkBoxesColorsShownBox = gui.widgetBox(self.controlArea, box='Colors shown')
        self.checkBoxesColorsShown = []
        self.checkBoxesColorsShownAttributeList = []

        self.mouseBehaviourBox = gui.radioButtons(self.controlArea, self, value='radio_mouse_behaviour',
                                                  btnLabels=('Drag', 'Select'),
                                                  box='Mouse left button behaviour', callback=self.mouseBehaviourChanged)

        self.displayBox = gui.widgetBox(self.controlArea, box='Display')
        self.checkBoxUseCache = gui.checkBox(self.displayBox, self, label='Use cache', value='check_use_cache')
        self.buttonDisplay = gui.button(self.displayBox, self, label='Display heatmap', callback=self.buttonDisplayClick)
        gui.rubber(self.controlArea)

        self.image_width = self.image_height = self.default_image_width

        self.hmi = None
        self.plot = pg.PlotWidget()
        self.plot.setBackground((255, 255, 255))
        pg.setConfigOption('foreground', (0, 0, 0))
        self.mainArea.layout().addWidget(self.plot)
        self.mainArea.setMinimumWidth(self.image_width+100)
        self.mainArea.setMinimumHeight(self.image_height+100)

        self.sharpeningRegion = False # flag is set when sharpening a region, not the whole heatmap
        self.regionSharpened = False  # flag is set when first region has been sharpened

        self.cachedHeatmaps = {}

        np.set_printoptions(precision=2)

    def data(self, dataset):
        if dataset:




            ##TODO: remove the lines that "repair" the tables
            # "repair" SqlTable
            if type(dataset) == SqlTable or not dataset.domain.class_var:
                classvars = []
                for attr in dataset.domain.attributes:
                    if isinstance(attr, Orange.data.DiscreteVariable):
                        classvars.append(attr)
                dataset.domain.class_vars = tuple(classvars)
                dataset.domain.class_var = classvars[0] if len(classvars) > 0 else None

                if not dataset.domain.class_var and "glass" in dataset.name:
                    glass = Orange.data.Table("Glass")
                    dataset.domain = glass.domain
                    dataset._create_domain()





            self.dataset = dataset

            self.clearControls()

            for attr in dataset.domain.attributes:
                self.comboBoxAttributesX.addItem(self.icons[attr.var_type], attr.name)
                self.comboBoxAttributesY.addItem(self.icons[attr.var_type], attr.name)
            for var in dataset.domain.class_vars:
                self.comboBoxClassvars.addItem(self.icons[var.var_type], var.name)
            self.X_attributes_select = 2
            self.Y_attributes_select = 3
            self.classvar_select = 0

            for (index, value) in enumerate(dataset.domain.class_vars[self.classvar_select].values):
                setattr(self, 'check_%s' % str(value), True)
                self.checkBoxesColorsShownAttributeList.append('check_%s' % str(value))
                self.checkBoxesColorsShown.append(gui.checkBox(self.checkBoxesColorsShownBox, self, value='check_%s' % str(value),
                                                               label=str(value), callback=self.attrChanged))
                self.checkBoxesColorsShown[index].setStyleSheet('QCheckBox {color: rgb(%i, %i, %i)}' % (self.color_array[index][0],
                                                                                                        self.color_array[index][1],
                                                                                                        self.color_array[index][2]))
            self.labelDataInput.setText('Data set: %s\nInstances: %d' % (self.dataset.name, len(dataset)))

            sample = dataset
            self.send("Sampled data", sample)
        else:
            self.labelDataInput.setText('No data input anymore')
            self.send("Sampled data", None)
        self.attrChanged()

    def mouseBehaviourChanged(self):
        if self.radio_mouse_behaviour == 0:
            self.hmi.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        else:
            self.hmi.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def buttonDisplayClick(self):
        self.attrChanged(True)

    def attrChanged(self, callSharpen=False):
        if not callSharpen:
            self.regionSharpened = False

        if self.dataset == None:
            return

        if self.n_discretization_intervals < 2:
            return

        self.checkedindices = []
        for i in range(len(self.checkBoxesColorsShownAttributeList)):
            self.checkedindices.append(getattr(self, self.checkBoxesColorsShownAttributeList[i]))

        tstart = datetime.now()
        self.changeDisplay(callSharpen)
        tend = datetime.now()
        print(tend - tstart)

    def addToCache(self):
        ind = ''.join([str(i) for i in self.checkedindices])
        self.cachedHeatmaps[(self.dataset.name, self.X_attributes_select, self.Y_attributes_select, ind)] = self.hmi.getImage_()

    def getCachedImage(self):
        ind = ''.join([str(i) for i in self.checkedindices])
        if (self.dataset.name, self.X_attributes_select, self.Y_attributes_select, ind) in self.cachedHeatmaps:
            image = self.cachedHeatmaps[(self.dataset.name, self.X_attributes_select, self.Y_attributes_select, ind)]
            return image
        return None

    def changeDisplay(self, callSharpen=False):
        if self.check_use_cache:
            image = self.getCachedImage()
            if image is not None:
                self.hmi.setImage_(image)
                return

        self.progress = gui.ProgressBar(self, 100) # iterations are set to arbitrary number, since it will soon get updated
        self.progress.advance(1)    # advance right away, so that user can see something is happening
        self.countHeappush = 0

        if not self.regionSharpened:
            d = Orange.statistics.distribution.get_distribution(self.dataset, self.dataset.domain.attributes[self.X_attributes_select])
            self.X_min = d[0, 0]
            self.X_max = d[0, -1]
            d = Orange.statistics.distribution.get_distribution(self.dataset, self.dataset.domain.attributes[self.Y_attributes_select])
            self.Y_min = d[0, 0]
            self.Y_max = d[0, -1]

            self.plot.clear()
            self.plot.getAxis('bottom').setLabel(self.dataset.domain.attributes[self.X_attributes_select].name)
            self.plot.getAxis('left').setLabel(self.dataset.domain.attributes[self.Y_attributes_select].name)

            disc = feature.discretization.EqualWidth(n=self.n_discretization_intervals)
            self.disc_dataset = discretization.DiscretizeTable(self.dataset, method=disc)
            self.contingencies = self.computeContingencies(self.disc_dataset)

            # calculate new image width, with interval width as integer
            self.interval_width = int(self.image_width / self.contingencies.shape[2])
            self.interval_height = int(self.image_height / self.contingencies.shape[1])
            self.image_width = self.interval_width * self.contingencies.shape[2]
            self.image_height = self.interval_height * self.contingencies.shape[1]
        # compute main rect
        rect = np.empty((1, 1), dtype=QtCore.QRectF)
        rect[0, 0] = QtCore.QRectF(0, 0, self.image_width, self.image_height)

        if not self.regionSharpened:
            # if the image is not updated when region is sharpened, otherwise some error occurs
            self.contingencies_valmax = self.updateImage(self.contingencies, rect[0, 0], None)

        if callSharpen:
            vr = self.plot.viewRect()

            if vr.x() < self.X_min:
                vr.setLeft(self.X_min)
            if vr.x() + vr.width() > self.X_max:
                vr.setRight(self.X_max)
            if vr.y() < self.Y_min:
                vr.setTop(self.Y_min)
            if vr.y() + vr.height() > self.Y_max:
                vr.setBottom(self.Y_max)

            if self.hmi is not None:
                pr = self.hmi.mapRectFromView_()

            self.h = []
            self.updated_fields = []
            self.count = 0
            self.disc_dataset.domain.X_min = self.X_min
            self.disc_dataset.domain.X_max = self.X_max
            self.disc_dataset.domain.Y_min = self.Y_min
            self.disc_dataset.domain.Y_max = self.Y_max

            if rect[0, 0].width() > pr.width() and rect[0, 0].height() > pr.height():
                self.sharpeningRegion = True
                self.sharpenHeatMapRegion(rect[0, 0], pr, self.contingencies,
                                          self.dataset, self.disc_dataset.domain,
                                          int(self.n_discretization_intervals/2),
                                          valmax_array=self.contingencies_valmax)
                self.sharpeningRegion = False
                self.regionSharpened = True
            else:
                contingencies = self.contingencies
                self.sharpenHeatMap(rect[0, 0], contingencies,
                                    self.dataset, self.disc_dataset.domain,
                                    int(self.n_discretization_intervals/2),
                                    valmax_array=self.contingencies_valmax)
                self.addToCache()
        self.progress.finish()

    def sharpenHeatMapRegion(self, wholerect, pixelrect, contingencies, dataset, domain, n_discretization_intervals, valmax_array):
        """
        This function is called when user selects only a region of heatmap.
        """
        grid = self.computeGrid(contingencies, frameRect=wholerect)
        rects = self.computeRects(grid)

        rects_in_region = []
        for row in range(rects.shape[0]):
            for col in range(rects.shape[1]):
                if rects[row, col].intersects(pixelrect):
                    rects_in_region.append((rects[row, col], row, col))
        self.progress.iter = len(rects_in_region) * 100

        for (rect, row, col) in rects_in_region:
            sub_contingencies, subdataset, subdomain = self.computeSubContingencies(dataset, domain,
                                                                                    n_discretization_intervals,
                                                                                    col, row)
            if sub_contingencies.max():
                self.sharpenHeatMap(rect, sub_contingencies, subdataset, subdomain, n_discretization_intervals, valmax_array=None, valmax_scalar=valmax_array[row, col])

    def sharpenHeatMap(self, rect, contingencies, dataset, domain, n_discretization_intervals, valmax_array, valmax_scalar=None):
        """
        Called recursively to draw the image in more detail.

        Image is divided into smaller rects. The chi squares of all neighbours are calculated, then the rect with highest
        chi square value is drawn first. When the rect becomes to small to draw it in more detail, the recursion stops.

        rect: the region which is to be sharpened
        contingencies: contingencies for region in rect
        dataset: dataset for region in rect
        domain: domain for region in rect
        n_discretization_intervals: this is the number of discretization intervalas for the next time the contingencies are calcuted
        valmax_array: valmax values for rect - calculated from contingencies parameter
        valmax_scalar: used when the valmax value is just one
        """
        grid = self.computeGrid(contingencies, frameRect=rect)
        rects = self.computeRects(grid)
        if rects[0, 0].width() < contingencies.shape[2]: # stop when rects become too small
            return

        estimates = self.getEstimates(contingencies)
        chi_squares_lr, chi_squares_ud = self.computeChiSquares(contingencies, estimates)

        for row in range(chi_squares_lr.shape[0]):
            for col in range(chi_squares_lr.shape[1]):
                heapq.heappush(self.h, (-chi_squares_lr[row, col], self.count, row, col, rects[row, col], dataset, domain, valmax_scalar if valmax_scalar != None else valmax_array[row, col]))
                heapq.heappush(self.h, (-chi_squares_lr[row, col], self.count, row, col+1, rects[row, col+1], dataset, domain, valmax_scalar if valmax_scalar != None else valmax_array[row, col+1]))
                self.countHeappush += 2

        self.count += 1

        for row in range(chi_squares_ud.shape[0]):
            for col in range(chi_squares_ud.shape[1]):
                heapq.heappush(self.h, (-chi_squares_ud[row, col], self.count, row, col, rects[row, col], dataset, domain, valmax_scalar if valmax_scalar != None else valmax_array[row, col]))
                heapq.heappush(self.h, (-chi_squares_ud[row, col], self.count, row+1, col, rects[row+1, col], dataset, domain, valmax_scalar if valmax_scalar != None else valmax_array[row+1, col]))
                self.countHeappush += 2

        self.count += 1
        if not self.sharpeningRegion:
            self.progress.iter = self.countHeappush

        while self.h:
            self.progress.advance(1)
            chi, count, r, c, rct, ds, dom, vm = heapq.heappop(self.h)
            if (rct, r, c) not in self.updated_fields:
                if rct.width() // n_discretization_intervals < n_discretization_intervals:
                    n_discretization_intervals = int(rct.width())
                if n_discretization_intervals > 1:
                    sub_contingencies, subdataset, subdomain = self.computeSubContingencies(ds, dom,
                                                                                            n_discretization_intervals,
                                                                                            c, r)
                    if sub_contingencies.max(): # == if sub_contingencies not empty
                        self.updateImage(sub_contingencies, rct, vm)
                        self.updated_fields.append((rct, r, c))
                        self.sharpenHeatMap(rct, sub_contingencies, subdataset, subdomain, n_discretization_intervals, None, valmax_scalar=vm)
                    else:
                        if self.sharpeningRegion:
                            self.updateImage(sub_contingencies, rct, vm)
                        self.updated_fields.append((rct, r, c))
                else:
                     return

    def updateImage(self, contingencies, rect, sup_valmax):
        """
        Makes an image of size rect from contingencies. The image is used to update a rect inside the heatmap.
        """
        interval_width = int(rect.width() / contingencies.shape[2])
        interval_height = int(rect.height() / contingencies.shape[1])

        contingencies -= np.min(contingencies)
        contingencies /= np.max(contingencies)
        contingencies = np.nan_to_num(contingencies)
        contingencies_argmax = contingencies.argmax(axis=0)
        rows, cols = np.indices(contingencies_argmax.shape)
        contingencies_valmax = contingencies[contingencies_argmax, rows, cols]

        colors_argmax = np.repeat(np.repeat(contingencies_argmax, interval_width, axis=0),
                                  interval_height, axis=1)
        colors_valmax = np.repeat(np.repeat(contingencies_valmax, interval_width, axis=0),
                                  interval_height, axis=1)

        colors = self.color_array[colors_argmax] + ((255-self.color_array[colors_argmax]) * (1-colors_valmax[:, :, None]))
        if sup_valmax:
            colors += ((255-colors) * (1-sup_valmax))

        if rect.width() == self.image_width and rect.height() == self.image_height:
            self.hmi = Heatmap(colors)
            self.plot.addItem(self.hmi)
            self.hmi.setRect(QtCore.QRectF(self.X_min, self.Y_min, self.X_max-self.X_min, self.Y_max-self.Y_min))
        else:
            self.hmi.updateImage_(colors, rect)

        return contingencies_valmax

    def computeSubContingencies(self, dataset, domain, n_discretization_intervals, x_index, y_index, rect=None):
        # if rect is given, the values are extracted from rect
        # otherwise, the values are computed from domain
        if rect:
            X_min = rect.x()
            X_max = rect.x() + rect.width()
            Y_min = rect.y()
            Y_max = rect.y() + rect.height()
        else:
            X_min, X_max, Y_min, Y_max = self.getValuesLimits(domain, x_index, y_index)

        filt = Orange.data.filter.Values()
        filt.domain = dataset.domain
        fd = Orange.data.filter.FilterContinuous(
            position=dataset.domain.attributes[self.X_attributes_select].name,
            min=X_min,
            max=X_max,
            oper=Orange.data.filter.FilterContinuous.Between
        )
        filt.conditions.append(fd)
        fd = Orange.data.filter.FilterContinuous(
            position=dataset.domain.attributes[self.Y_attributes_select].name,
            min=Y_min,
            max=Y_max,
            oper=Orange.data.filter.FilterContinuous.Between
        )
        filt.conditions.append(fd)
        subdataset = filt(dataset)

        if len(subdataset) == 0:
            return np.zeros( (len(domain.class_var.values), n_discretization_intervals, n_discretization_intervals) ), None, None

        disc = feature.discretization.EqualWidth(n=n_discretization_intervals)
        disc_subdataset = discretization.DiscretizeTable(subdataset, method=disc,
                                                         fixed=( {dataset.domain.attributes[self.X_attributes_select].name: (X_min, X_max),
                                                                  dataset.domain.attributes[self.Y_attributes_select].name: (Y_min, Y_max)} ))
        disc_subdataset.domain.X_min = X_min
        disc_subdataset.domain.X_max = X_max
        disc_subdataset.domain.Y_min = Y_min
        disc_subdataset.domain.Y_max = Y_max
        return self.computeContingencies(disc_subdataset), subdataset, disc_subdataset.domain

    def computeContingencies(self, disc_dataset):
        filt_data_by_class_var = []
        for cv_value in self.dataset.domain.class_var.values:
            filt = Orange.data.filter.Values()
            filt.domain = disc_dataset.domain
            fd = Orange.data.filter.FilterDiscrete(
                column=disc_dataset.domain.class_var,
                values=[cv_value])
            filt.conditions.append(fd)
            filt_data_by_class_var.append((cv_value, filt(disc_dataset)))

        contingencies = []
        for (index, (value, filt_data)) in enumerate(filt_data_by_class_var):
            cont = statistics.contingency.get_contingency(
                            filt_data, self.X_attributes_select, self.Y_attributes_select)
            if self.checkedindices[index]:
                contingencies.append(cont.T)
            else:
                contingencies.append(np.zeros(cont.shape))
        return np.array(contingencies)

    def getValuesLimits(self, domain, x_attr_index, y_attr_index):
        x_values = domain.attributes[self.X_attributes_select].values
        y_values = domain.attributes[self.Y_attributes_select].values

        if x_attr_index == 0:
            split = re.split(r'<', x_values[x_attr_index])
            X_min = domain.X_min
            X_max, = filter(None, split)
            X_max = float(X_max)
        elif x_attr_index == len(x_values) - 1:
            split = re.split(r'>=', x_values[x_attr_index])
            X_min, = filter(None, split)
            X_min = float(X_min)
            X_max = domain.X_max
        else:
            splits = re.split(r'[\[\]\(\), ]', x_values[x_attr_index])
            X_min, X_max = filter(None, splits)
            X_min, X_max = float(X_min), float(X_max)

        if y_attr_index == 0:
            split = re.split(r'<', y_values[y_attr_index])
            Y_min = domain.Y_min
            Y_max, = filter(None, split)
            Y_max = float(Y_max)
        elif y_attr_index == len(y_values) - 1:
            split = re.split(r'>=', y_values[y_attr_index])
            Y_min, = filter(None, split)
            Y_min = float(Y_min)
            Y_max = domain.Y_max
        else:
            splits = re.split(r'[\[\]\(\), ]', y_values[y_attr_index])
            Y_min, Y_max = filter(None, splits)
            Y_min, Y_max = float(Y_min), float(Y_max)

        return X_min, X_max, Y_min, Y_max

    def computeGrid(self, contingencies, frameRect):
        self.widths = np.zeros(contingencies.shape[2])
        self.heights = np.zeros(contingencies.shape[1])
        self.widths[:] = frameRect.width() / contingencies.shape[2]
        self.heights[:] = frameRect.height() / contingencies.shape[1]

        # grid[0, :] is the x-axis ( columns have widths )
        # grid[1, :] is the y-axis ( rows have heights )
        grid = np.array( [np.zeros(len(self.widths)), np.zeros(len(self.heights))] )
        for i in range(len(self.widths)):
            grid[0, i] = frameRect.x() + self.widths[:i+1].sum()
        for i in range(len(self.heights)):
            grid[1, i] = frameRect.y() + self.heights[:i+1].sum()
        return grid

    def computeRects(self, grid):
        rects = np.empty((grid[1].shape[0], grid[0].shape[0]), dtype=QtCore.QRectF)
        for row in range(grid[1].shape[0]):
            for col in range(grid[0].shape[0]):
                rects[row, col] = QtCore.QRectF(grid[0, col] - self.widths[col],
                                                grid[1, row] - self.heights[row],
                                                self.widths[col], self.heights[row])
        return rects

    def getEstimates(self, observes):
        estimates = []
        for obs in observes:
            n = obs.sum()
            sum_rows = obs.sum(1)
            sum_cols = obs.sum(0)
            prob_rows = sum_rows / n
            prob_cols = sum_cols / n
            rows, cols = np.indices(obs.shape)
            est = np.zeros(obs.shape)
            est[rows, cols] = n * prob_rows[rows] * prob_cols[cols]
            estimates.append(est)
        return np.nan_to_num(np.array(estimates))

    def computeChiSquares(self, observes, estimates):
        # compute chi squares for left-right neighbours
        depth, rows, coll = np.indices(( observes.shape[0], observes.shape[1], observes.shape[2]-1 ))
        colr = coll + 1
        obs_dblstack = np.array([ observes[depth, rows, coll], observes[depth, rows, colr] ])
        obs_pairs = np.zeros(( obs_dblstack.shape[1], obs_dblstack.shape[2], obs_dblstack.shape[3], obs_dblstack.shape[0] ))
        depth, rows, coll, pairs = np.indices(obs_pairs.shape)
        obs_pairs[depth, rows, coll, pairs] = obs_dblstack[pairs, depth, rows, coll]

        depth, rows, coll = np.indices(( estimates.shape[0], estimates.shape[1], estimates.shape[2]-1 ))
        colr = coll + 1
        est_dblstack = np.array([ estimates[depth, rows, coll], estimates[depth, rows, colr] ])
        est_pairs = np.zeros(( est_dblstack.shape[1], est_dblstack.shape[2], est_dblstack.shape[3], est_dblstack.shape[0] ))
        depth, rows, coll, pairs = np.indices(est_pairs.shape)
        est_pairs[depth, rows, coll, pairs] = est_dblstack[pairs, depth, rows, coll]

        oe2e = (obs_pairs - est_pairs)**2 / est_pairs
        chi_squares_lr = np.nan_to_num(np.nansum(np.nansum(oe2e, axis=3), axis=0))

        # compute chi squares for up-down neighbours
        depth, rowu, cols = np.indices(( observes.shape[0], observes.shape[1]-1, observes.shape[2] ))
        rowd = rowu + 1
        obs_dblstack = np.array([ observes[depth, rowu, cols], observes[depth, rowd, cols] ])
        obs_pairs = np.zeros(( obs_dblstack.shape[1], obs_dblstack.shape[2], obs_dblstack.shape[3], obs_dblstack.shape[0] ))
        depth, rowu, cols, pairs = np.indices(obs_pairs.shape)
        obs_pairs[depth, rowu, cols, pairs] = obs_dblstack[pairs, depth, rowu, cols]

        depth, rowu, cols = np.indices(( estimates.shape[0], estimates.shape[1]-1, estimates.shape[2] ))
        rowd = rowu + 1
        est_dblstack = np.array([ estimates[depth, rowu, cols], estimates[depth, rowd, cols] ])
        est_pairs = np.zeros(( est_dblstack.shape[1], est_dblstack.shape[2], est_dblstack.shape[3], est_dblstack.shape[0] ))
        depth, rowu, cols, pairs = np.indices(est_pairs.shape)
        est_pairs[depth, rowu, cols, pairs] = est_dblstack[pairs, depth, rowu, cols]

        oe2e = (obs_pairs - est_pairs)**2 / est_pairs
        chi_squares_ud = np.nan_to_num(np.nansum(np.nansum(oe2e, axis=3), axis=0))

        return (chi_squares_lr, chi_squares_ud)

    def clearControls(self):
        self.comboBoxAttributesX.clear()
        self.comboBoxAttributesY.clear()
        self.comboBoxClassvars.clear()

        if self.checkBoxesColorsShown:
            while self.checkBoxesColorsShown:
                chkbox = self.checkBoxesColorsShown.pop()
                chkbox.setParent(None)

        if self.checkBoxesColorsShownAttributeList:
            while self.checkBoxesColorsShownAttributeList:
                attr = self.checkBoxesColorsShownAttributeList.pop()
                delattr(self, attr)

        self.X_attributes_select = 2
        self.Y_attributes_select = 3
        self.classvar_select = 0
        self.n_discretization_intervals = 10
        self.radio_mouse_behaviour = 0
        self.image_width = self.image_height = self.default_image_width

        self.plot.clear()
        self.regionSharpened = False
        self.sharpeningRegion = False
        # self.hmi = None


if __name__ == "__main__":
    a = QtGui.QApplication([])
    w = OWHeatmap()
    w.show()
    d = Orange.data.Table('iris')
    w.data(d)
    a.exec_()
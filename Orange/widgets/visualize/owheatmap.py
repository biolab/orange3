__author__ = 'jurre'
import Orange
from Orange import feature, statistics
from Orange.data import discretization, Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import contingency
from Orange.widgets import widget, gui
from Orange.widgets.utils.plot import owplot, owconstants, owaxis, owcurve

from PyQt4 import QtGui, QtCore

import numpy as np

import re
import heapq

from datetime import datetime


class HeatMapCurve(owcurve.OWCurve):
    def __init__(self, image, image_rect, heatmap_height, xData=[], yData=[], x_axis_key=owconstants.xBottom, y_axis_key=owconstants.yLeft, tooltip=None):
        super().__init__(xData, yData, x_axis_key, y_axis_key, tooltip)

        # flip the image up-down (y becomes -y)
        self.t = QtGui.QTransform().scale(1, -1)
        self.image = image.transformed(self.t)

        self.heatmap_height = heatmap_height
        self.image_rect = image_rect

        self.setFlag(QtGui.QGraphicsItem.ItemHasNoContents, False)


    def updateImage(self, rect, image):
        image = image.transformed(self.t)
        image_rect = QtCore.QRectF(rect.x(), self.heatmap_height - rect.y() - rect.height(), rect.width(), rect.height())
        painter = QtGui.QPainter(self.image)
        painter.drawImage(image_rect, image)
        self.update()

    def paint(self, painter, options, widget):
        painter.setRenderHints(QtGui.QPainter.Antialiasing |
                                    QtGui.QPainter.TextAntialiasing |
                                    QtGui.QPainter.SmoothPixmapTransform)

        # move the rect of image, so that it is drawn inside axes
        rect = QtCore.QRectF(self.image_rect.x() + self.plot().heatmap_rect.x() + 1,
                             self.image_rect.y() + self.plot().heatmap_rect.y() + 1,
                             self.image_rect.width(), self.image_rect.height())
        painter.drawImage(rect, self.image)

    def boundingRect(self, *args, **kwargs):
        return self.image_rect


class HeatMapPlot(owplot.OWPlot):
    def __init__(self, parent=None, name="None", show_legend=1, axes=None,
                 widget=None, width=0, height=0):
        super().__init__(parent, name, show_legend,
                         axes or [owconstants.xBottom, owconstants.yLeft],
                         widget)
        self.state = owconstants.NOTHING
        self.graph_margin = 20
        self.y_axis_extra_margin = -10
        self.animate_plot = False
        self.animate_points = False
        self.show_grid = False
        self.tool = None

        # default values, call set_graph_size() to set
        self.heatmap_width = width if width else 512
        self.heatmap_height = height if height else 512

    def setGraphSize(self, width, height):
        self.heatmap_width = width
        self.heatmap_height = height

    def set_graph_rect(self, rect):
        self.heatmap_rect = rect
        rect = QtCore.QRectF(rect.x(), rect.y(), self.heatmap_width, self.heatmap_height)
        self.graph_area = rect
        super().set_graph_rect(rect)

    def mousePressEvent(self, event):
        if self.state == owconstants.NOTHING and self.tool:
            self.tool.mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.state == owconstants.NOTHING and self.tool:
            self.tool.mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.state == owconstants.NOTHING and self.tool:
            self.tool.mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

class OWHeatmap(widget.OWWidget):
    _name = "Heat map"
    _description = "Draws a heat map."
    #_long_description = """Shows itself to see if added correctly."""
    # _icon = "../icons/Dlg_zoom.png"
    _priority = 100

    inputs = [("Data", Table, "data")]
    outputs = [("Sampled data", Table)]

    X_attributes_select = 0
    Y_attributes_select = 1
    classvar_select = 0
    n_discretization_intervals = 10
    check_commit_on_change = 0
    color_array = np.array([[  0,   0, 255],   # blue
                            [  0, 255,   0],   # green
                            [255,   0,   0],   # red
                            [255, 255,   0],   # yellow
                            [255,   0, 255],   # magenta
                            [  0, 255, 255],   # aqua
                            [128,  10, 203],   # violet
                            [255, 107,   0],   # orange
                            [223, 224, 249]])  # lavender

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

        self.lineEditDiscretizationIntervals = gui.lineEdit(self.controlArea, self, value='n_discretization_intervals',
                                                            box='Number of discretization intervals',
                                                            orientation='horizontal', valueType=int,
                                                            callback=self.attrChanged, enterPlaceholder=True)
        self.commitBox = gui.widgetBox(self.controlArea, box='Commit')
        self.checkBoxCommit = gui.checkBox(self.commitBox, self, value='check_commit_on_change',
                                           label='Commit on change')
        self.buttonCommit = gui.button(self.commitBox, self, label='Commit', callback=self.buttonCommit)
        gui.rubber(self.controlArea)

        self.image_width = self.image_height = 512
        self.plot = HeatMapPlot(self.mainArea, "Heatmap plot", widget=self, width=self.image_width, height=self.image_height)
        self.mainArea.layout().addWidget(self.plot)

        self.mainArea.setMinimumWidth(self.image_width + self.plot.graph_margin + self.plot.axis_margin + self.plot.title_margin + self.plot.y_axis_extra_margin)
        self.mainArea.setMinimumHeight(self.image_height + self.plot.graph_margin + self.plot.axis_margin + self.plot.title_margin + self.plot.y_axis_extra_margin)

        np.set_printoptions(precision=2)

    def data(self, dataset):
        if dataset:

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
            self.plot.clear()

            for attr in dataset.domain.attributes:
                self.comboBoxAttributesX.addItem(self.icons[attr.var_type], attr.name)
                self.comboBoxAttributesY.addItem(self.icons[attr.var_type], attr.name)
            for var in dataset.domain.class_vars:
                self.comboBoxClassvars.addItem(self.icons[var.var_type], var.name)
            self.X_attributes_select = 0
            self.Y_attributes_select = 1
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

    def buttonCommit(self):
        self.attrChanged(True)

    def attrChanged(self, callDisplay=False):
        if self.dataset == None:
            return

        if self.n_discretization_intervals < 2:
            return

        self.checkedindices = []
        for i in range(len(self.checkBoxesColorsShownAttributeList)):
            self.checkedindices.append(getattr(self, self.checkBoxesColorsShownAttributeList[i]))

        if self.check_commit_on_change or callDisplay:
            tstart = datetime.now()
            self.displayChanged()
            tend = datetime.now()
            print(tend - tstart)

    def displayChanged(self):
        d = Orange.statistics.distribution.get_distribution(self.dataset, self.dataset.domain.attributes[self.X_attributes_select])
        self.X_min = d[0, 0]
        self.X_max = d[0, -1]
        d = Orange.statistics.distribution.get_distribution(self.dataset, self.dataset.domain.attributes[self.Y_attributes_select])
        self.Y_min = d[0, 0]
        self.Y_max = d[0, -1]

        disc = feature.discretization.EqualWidth(n=self.n_discretization_intervals)
        self.disc_dataset = discretization.DiscretizeTable(self.dataset, method=disc)
        self.contingencies = self.computeContingencies(self.disc_dataset)

        # calculate new image width, with interval width as integer
        self.interval_width = int(self.image_width / self.contingencies.shape[2])
        self.interval_height = int(self.image_height / self.contingencies.shape[1])
        self.image_width = self.interval_width * self.contingencies.shape[2]
        self.image_height = self.interval_height * self.contingencies.shape[1]
        self.initPlot()
        QtCore.QCoreApplication.processEvents()
        # compute main rect
        rect = np.empty((1, 1), dtype=QtCore.QRectF)
        rect[0, 0] = QtCore.QRectF(0, 0, self.image_width, self.image_height)

        contingencies_valmax = self.updateImage(self.contingencies, rect[0, 0], None)

        self.progress = gui.ProgressBar(self, 3240)
        self.countHeappush = 0

        self.h = []
        self.updated_fields = []
        self.count = 0
        self.disc_dataset.domain.X_min = self.X_min
        self.disc_dataset.domain.X_max = self.X_max
        self.disc_dataset.domain.Y_min = self.Y_min
        self.disc_dataset.domain.Y_max = self.Y_max
        self.sharpenHeatMap(rect[0, 0], self.contingencies,
                            self.dataset, self.disc_dataset.domain,
                            int(self.n_discretization_intervals/2),
                            valmax_array=contingencies_valmax)
        self.progress.finish()

    def sharpenHeatMap(self, rect, contingencies, dataset, domain, n_discretization_intervals, valmax_array, valmax_scalar=None):
        grid = self.computeGrid(contingencies, frameRect=rect)
        rects = self.computeRects(grid)
        if rects[0, 0].width() < n_discretization_intervals: # stop when rects become too small
            return

        # self.drawRects(rects, self.pim)

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
        self.progress.iter = self.countHeappush

        while self.h:
            self.progress.advance(1)
            chi, count, r, c, rct, ds, dom, vm = heapq.heappop(self.h)
            if (rct, r, c) not in self.updated_fields:
                sub_contingencies, subdataset, subdomain = self.computeSubContingencies(ds, dom,
                                                                                        c, r,
                                                                                        n_discretization_intervals)
                if sub_contingencies.max(): # == if sub_contingencies not empty
                    self.updateImage(sub_contingencies, rct, vm)
                    self.updated_fields.append((rct, r, c))
                    self.sharpenHeatMap(rct, sub_contingencies, subdataset, subdomain, n_discretization_intervals, None, valmax_scalar=vm)
                else:
                    self.updated_fields.append((rct, r, c))

    def updateImage(self, contingencies, rect, sup_valmax):
        interval_width = int(rect.width() / contingencies.shape[2])
        interval_height = int(rect.height() / contingencies.shape[1])
        image_width = interval_width * contingencies.shape[2]
        image_height = interval_height * contingencies.shape[1]

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

        # when creating the image, blue and red are swapped; the color order is:
        # image = np.dstack(("blue", "green", "red", "whatever"))
        image = np.dstack((colors[:, :, 2],
                           colors[:, :, 1],
                           colors[:, :, 0],
                           np.zeros((image_width, image_height))))
        im255 = image.flatten().astype(np.uint8)
        im = QtGui.QImage(im255.data, image_width, image_height, QtGui.QImage.Format_RGB32)

        if rect.width() == self.image_width and rect.height() == self.image_height:
            self.hm_curve = HeatMapCurve(im, rect, self.image_height)
            self.plot.add_custom_curve(self.hm_curve)
        else:
            self.hm_curve.updateImage(rect, im)

        QtCore.QCoreApplication.processEvents() # slows everything down

        return contingencies_valmax

    def computeSubContingencies(self, dataset, domain, x_index, y_index, n_discretization_intervals):
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
                contingencies.append(cont)
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

    def drawGrid(self, grid, pim):
        painter = QtGui.QPainter(pim)
        painter.setPen(QtCore.Qt.black)
        for i in range(grid.shape[1]-1):
            painter.drawLine(grid[0, i], 0, grid[0, i], grid[0, -1])
            painter.drawLine(0, grid[1, i], grid[1, -1], grid[1, i])
        self.scene.addPixmap(pim)
        QtCore.QCoreApplication.processEvents()

    def computeRects(self, grid):
        rects = np.empty((grid[1].shape[0], grid[0].shape[0]), dtype=QtCore.QRectF)
        for row in range(grid[1].shape[0]):
            for col in range(grid[0].shape[0]):
                rects[row, col] = QtCore.QRectF(grid[0, col] - self.widths[col],
                                                grid[1, row] - self.heights[row],
                                                self.widths[col], self.heights[row])
        return rects

    def drawRects(self, rects, pim):
        painter = QtGui.QPainter(self.pim)
        painter.setPen(QtCore.Qt.black)
        for row in range(rects[1].shape[0]):
            for col in range(rects[0].shape[0]):
                painter.drawRect(rects[row, col])
                # painter.drawText(rects[row, col].x(), rects[row, col].y() + rects[row, col].height()/2,
                #                  "[%s, %s]" % (col, row))
        self.scene.addPixmap(pim)
        QtCore.QCoreApplication.processEvents()

    def initPlot(self):
        self.plot.set_axis_title(owconstants.xBottom, self.dataset.domain.attributes[self.X_attributes_select].name)
        self.plot.set_axis_title(owconstants.yLeft, self.dataset.domain.attributes[self.Y_attributes_select].name)
        self.plot.set_show_axis_title(owconstants.xBottom, True)
        self.plot.set_show_axis_title(owconstants.yLeft, True)
        self.plot.set_axis_scale(owconstants.xBottom, self.X_min, self.X_max)
        self.plot.set_axis_scale(owconstants.yLeft, self.Y_min, self.Y_max)

        self.plot.setGraphSize(self.image_width, self.image_height)

        self.mainArea.setMinimumWidth(self.image_width + self.plot.graph_margin + self.plot.axis_margin + self.plot.title_margin + self.plot.y_axis_extra_margin)
        self.mainArea.setMinimumHeight(self.image_height + self.plot.graph_margin + self.plot.axis_margin + self.plot.title_margin + self.plot.y_axis_extra_margin)

        self.plot.update()

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

        self.X_attributes_select = 0
        self.Y_attributes_select = 1
        self.classvar_select = 0
        self.n_discretization_intervals = 10
        self.image_width = self.image_height = 512
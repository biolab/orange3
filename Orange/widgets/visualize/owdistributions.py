"""
Distributions
-------------

A widget for plotting attribute distributions.

"""
from math import sqrt
import sys
import collections
from xml.sax.saxutils import escape

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import numpy
import pyqtgraph as pg

import Orange.data
from Orange.statistics import distribution, contingency
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette
from Orange.widgets.widget import InputSignal
from Orange.widgets.visualize.owlinearprojection import LegendItem, ScatterPlotItem
from Orange.widgets.io import FileFormats

def selected_index(view):
    """Return the selected integer `index` (row) in the view.

    If no index is selected return -1

    `view` must be in single selection mode.
    """
    indices = view.selectedIndexes()
    assert len(indices) < 2, "View must be in single selection mode"
    if indices:
        return indices[0].row()
    else:
        return -1


class DistributionBarItem(pg.GraphicsObject):
    def __init__(self, geometry, dist, colors):
        super().__init__()
        self.geometry = geometry
        self.dist = dist
        self.colors = colors
        self.__picture = None

    def paint(self, painter, options, widget):
        if self.__picture is None:
            self.__paint()
        painter.drawPicture(0, 0, self.__picture)

    def boundingRect(self):
        return self.geometry

    def __paint(self):
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        pen = QtGui.QPen(QtGui.QBrush(Qt.white), 0.5)
        pen.setCosmetic(True)
        painter.setPen(pen)

        geom = self.geometry
        x, y = geom.x(), geom.y()
        w, h = geom.width(), geom.height()
        wsingle = w / len(self.dist)
        for d, c in zip(self.dist, self.colors):
            painter.setBrush(QtGui.QBrush(c))
            painter.drawRect(QtCore.QRectF(x, y, wsingle, d * h))
            x += wsingle
        painter.end()

        self.__picture = picture


class OWDistributions(widget.OWWidget):
    name = "Distributions"
    description = "Display value distributions of a data feature in a graph."
    icon = "icons/Distribution.svg"
    priority = 100
    inputs = [InputSignal("Data", Orange.data.Table, "set_data",
                          doc="Set the input data set")]

    settingsHandler = settings.DomainContextHandler(
        match_values=settings.DomainContextHandler.MATCH_VALUES_ALL)
    #: Selected variable index
    variable_idx = settings.ContextSetting(-1)
    #: Selected group variable
    groupvar_idx = settings.ContextSetting(0)

    relative_freq = settings.Setting(False)
    disc_cont = settings.Setting(False)

    smoothing_index = settings.Setting(5)
    show_prob = settings.ContextSetting(0)

    want_graph = True
    ASH_HIST = 50

    bins = [ 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 50 ]
    smoothing_facs = list(reversed([ 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 4, 6, 10 ]))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.distributions = None
        self.contingencies = None
        self.var = self.cvar = None
        varbox = gui.widgetBox(self.controlArea, "Variable")

        self.varmodel = itemmodels.VariableListModel()
        self.groupvarmodel = []

        self.varview = QtGui.QListView(
            selectionMode=QtGui.QListView.SingleSelection)
        self.varview.setSizePolicy(
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.varview.setModel(self.varmodel)
        self.varview.setSelectionModel(
            itemmodels.ListSingleSelectionModel(self.varmodel))
        self.varview.selectionModel().selectionChanged.connect(
            self._on_variable_idx_changed)
        varbox.layout().addWidget(self.varview)

        box = gui.widgetBox(self.controlArea, "Precision")

        gui.separator(self.controlArea, 4, 4)

        box2 = gui.widgetBox(box, orientation="horizontal")
        self.l_smoothing_l = gui.widgetLabel(box2, "Smooth")
        gui.hSlider(box2, self, "smoothing_index",
                    minValue=0, maxValue=len(self.smoothing_facs) - 1,
                    callback=self._on_set_smoothing, createLabel=False)
        self.l_smoothing_r = gui.widgetLabel(box2, "Precise")

        self.cb_disc_cont = gui.checkBox(
            gui.indentedBox(box, sep=4),
            self, "disc_cont", "Bin continuous variables",
            callback=self._on_groupvar_idx_changed)

        box = gui.widgetBox(self.controlArea, "Group by")
        self.icons = gui.attributeIconDict
        self.groupvarview = gui.comboBox(box, self, "groupvar_idx",
             callback=self._on_groupvar_idx_changed, valueType=str,
             contentsLength=12)
        box2 = gui.indentedBox(box, sep=4)
        self.cb_rel_freq = gui.checkBox(
            box2, self, "relative_freq", "Show relative frequencies",
            callback=self._on_relative_freq_changed)
        gui.separator(box2)
        self.cb_prob = gui.comboBox(
            box2, self, "show_prob", label="Show probabilities",
            orientation="horizontal",
            callback=self._on_relative_freq_changed)

        plotview = pg.PlotWidget(background=None)
        self.mainArea.layout().addWidget(plotview)
        w = QtGui.QLabel()
        w.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        self.mainArea.layout().addWidget(w, Qt.AlignCenter)

        self.plot = pg.PlotItem()
        self.plot.hideButtons()
        plotview.setCentralItem(self.plot)

        self.plot_prob = pg.ViewBox()
        self.plot.hideAxis('right')
        self.plot.scene().addItem(self.plot_prob)
        self.plot.getAxis("right").linkToView(self.plot_prob)
        self.plot.getAxis("right").setLabel("Probability")
        self.plot_prob.setZValue(10)
        self.plot_prob.setXLink(self.plot)
        self.update_views()
        self.plot.vb.sigResized.connect(self.update_views)
        self.plot_prob.setRange(yRange=[0,1])

        def disable_mouse(plot):
            plot.setMouseEnabled(False, False)
            plot.setMenuEnabled(False)

        disable_mouse(self.plot.getViewBox())
        disable_mouse(self.plot_prob)

        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(pen)

        self._legend = LegendItem()
        self._legend.setParentItem(self.plot.getViewBox())
        self._legend.hide()
        self._legend.anchor((1, 0), (1, 0))
        self.graphButton.clicked.connect(self.save_graph)

    def update_views(self):
        self.plot_prob.setGeometry(self.plot.vb.sceneBoundingRect())
        self.plot_prob.linkedViewChanged(self.plot.vb, self.plot_prob.XAxis)

    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.data = data
        if self.data is not None:
            domain = self.data.domain
            self.varmodel[:] = list(domain)

            self.groupvarview.clear()
            self.groupvarmodel = \
                ["(None)"] + [var for var in domain if var.is_discrete]
            self.groupvarview.addItem("(None)")
            for var in self.groupvarmodel[1:]:
                self.groupvarview.addItem(self.icons[var], var.name)
            if domain.has_discrete_class:
                self.groupvar_idx = \
                    self.groupvarmodel.index(domain.class_var)
            self.openContext(domain)
            self.variable_idx = min(max(self.variable_idx, 0),
                                    len(self.varmodel) - 1)
            self.groupvar_idx = min(max(self.groupvar_idx, 0),
                                    len(self.groupvarmodel) - 1)
            itemmodels.select_row(self.varview, self.variable_idx)
            self._setup()

    def clear(self):
        self.plot.clear()
        self.plot_prob.clear()
        self.varmodel[:] = []
        self.groupvarmodel = []
        self.variable_idx = -1
        self.groupvar_idx = 0
        self._legend.clear()
        self._legend.hide()

    def _setup_smoothing(self):
        if not self.disc_cont and self.var and self.var.is_continuous:
            self.cb_disc_cont.setText("Bin continuous variables")
            self.l_smoothing_l.setText("Smooth")
            self.l_smoothing_r.setText("Precise")
        else:
            self.cb_disc_cont.setText("Bin continuous variables into {} bins".
                                      format(self.bins[self.smoothing_index]))
            self.l_smoothing_l.setText(" " + str(self.bins[0]))
            self.l_smoothing_r.setText(" " + str(self.bins[-1]))

    def _setup(self):
        self.plot.clear()
        self.plot_prob.clear()
        self._legend.clear()
        self._legend.hide()

        varidx = self.variable_idx
        self.var = self.cvar = None
        if varidx >= 0:
            self.var = self.varmodel[varidx]
        if self.groupvar_idx > 0:
            self.cvar = self.groupvarmodel[self.groupvar_idx]
            self.cb_prob.clear()
            self.cb_prob.addItem("(None)")
            self.cb_prob.addItems(self.cvar.values)
            self.cb_prob.addItem("(All)")
            self.show_prob = min(max(self.show_prob, 0),
                    len(self.cvar.values) + 1)
        data = self.data
        self._setup_smoothing()
        if self.var is None:
            return
        if self.disc_cont:
            data = self.data[:, (self.var, self.cvar) if self.cvar else self.var ]
            disc = Orange.preprocess.discretize.EqualWidth(n=self.bins[self.smoothing_index])
            data = Orange.preprocess.Discretize(data, method=disc)
            self.var = data.domain.variables[0]
        self.set_left_axis_name()
        self.enable_disable_rel_freq()
        if self.cvar:
            self.contingencies = \
                contingency.get_contingency(data, self.var, self.cvar)
            self.display_contingency()
        else:
            self.distributions = \
                distribution.get_distribution(data, self.var)
            self.display_distribution()
        self.plot.autoRange()

    def display_distribution(self):
        dist = self.distributions
        var = self.var
        assert len(dist) > 0
        self.plot.clear()
        self.plot_prob.clear()
        self.plot.hideAxis('right')

        bottomaxis = self.plot.getAxis("bottom")
        bottomaxis.setLabel(var.name)

        self.set_left_axis_name()
        if var and var.is_continuous:
            bottomaxis.setTicks(None)
            edges, curve = ash_curve(dist, None, m=OWDistributions.ASH_HIST,
                smoothing_factor=self.smoothing_facs[self.smoothing_index])
            edges = edges + (edges[1] - edges[0])/2
            edges = edges[:-1]
            item = pg.PlotCurveItem()
            pen = QtGui.QPen(QtGui.QBrush(Qt.white), 3)
            pen.setCosmetic(True)
            item.setData(edges, curve, antialias=True, stepMode=False,
                         fillLevel=0, brush=QtGui.QBrush(Qt.gray),
                         pen=pen)
            self.plot.addItem(item)
        else:
            bottomaxis.setTicks([list(enumerate(var.values))])
            for i, w in enumerate(dist):
                geom = QtCore.QRectF(i - 0.33, 0, 0.66, w)
                item = DistributionBarItem(geom, [1.0],
                                           [QtGui.QColor(128, 128, 128)])
                self.plot.addItem(item)

    def _on_relative_freq_changed(self):
        self.set_left_axis_name()
        if self.cvar and self.cvar.is_discrete:
            self.display_contingency()
        else:
            self.display_distribution()
        self.plot.autoRange()

    def display_contingency(self):
        """
        Set the contingency to display.
        """
        cont = self.contingencies
        var, cvar = self.var, self.cvar
        assert len(cont) > 0
        self.plot.clear()
        self.plot_prob.clear()
        self._legend.clear()

        if self.show_prob:
            self.plot.showAxis('right')
        else:
            self.plot.hideAxis('right')

        bottomaxis = self.plot.getAxis("bottom")
        bottomaxis.setLabel(var.name)

        cvar_values = cvar.values
        palette = colorpalette.ColorPaletteGenerator(len(cvar_values))
        colors = [palette[i].lighter() for i in range(len(cvar_values))]

        if var and var.is_continuous:
            bottomaxis.setTicks(None)

            weights, cols, cvar_values, curves = [], [], [], []
            for i, dist in enumerate(cont):
                v, W = dist
                if len(v):
                    weights.append(numpy.sum(W))
                    cols.append(colors[i])
                    cvar_values.append(cvar.values[i])
                    curves.append(ash_curve(dist, cont,  m=OWDistributions.ASH_HIST,
                        smoothing_factor=self.smoothing_facs[self.smoothing_index]))
            weights = numpy.array(weights)
            sumw = numpy.sum(weights)
            weights /= sumw
            colors = cols
            curves = [(X, Y * w) for (X, Y), w in zip(curves, weights)]
            ncval = len(cvar_values)

            curvesline = [] #from histograms to lines
            for (X,Y) in curves:
                X = X + (X[1] - X[0])/2
                X = X[:-1]
                X = numpy.array(X)
                Y = numpy.array(Y)
                curvesline.append((X,Y))

            for t in [ "fill", "line" ]:
                for (X, Y), color, w in reversed(list(zip(curvesline, colors, weights))):
                    item = pg.PlotCurveItem()
                    pen = QtGui.QPen(QtGui.QBrush(color), 3)
                    pen.setCosmetic(True)
                    color = QtGui.QColor(color)
                    color.setAlphaF(0.2)
                    item.setData(X, Y/(w if self.relative_freq else 1), antialias=True, stepMode=False,
                         fillLevel=0 if t == "fill" else None,
                         brush=QtGui.QBrush(color), pen=pen)
                    self.plot.addItem(item)

            if self.show_prob:
                M_EST = 5 #for M estimate
                all_X = numpy.array(numpy.unique(numpy.hstack([X for X,_ in curvesline])))
                inter_X = numpy.array(numpy.linspace(all_X[0], all_X[-1], len(all_X)*2))
                curvesinterp = [ numpy.interp(inter_X, X, Y) for (X,Y) in curvesline ]
                sumprob = numpy.sum(curvesinterp, axis=0)
                # allcorrection = M_EST/sumw*numpy.sum(sumprob)/len(inter_X)
                legal = sumprob > 0.05 * numpy.max(sumprob)

                i = len(curvesinterp) + 1
                show_all = self.show_prob == i
                for Y, color in reversed(list(zip(curvesinterp, colors))):
                    i -= 1
                    if show_all or self.show_prob == i:
                        item = pg.PlotCurveItem()
                        pen = QtGui.QPen(QtGui.QBrush(color), 3, style=QtCore.Qt.DotLine)
                        pen.setCosmetic(True)
                        #prob = (Y+allcorrection/ncval)/(sumprob+allcorrection)
                        prob = Y[legal] / sumprob[legal]
                        item.setData(inter_X[legal], prob, antialias=True, stepMode=False,
                             fillLevel=None, brush=None, pen=pen)
                        self.plot_prob.addItem(item)

        elif var and var.is_discrete:
            bottomaxis.setTicks([list(enumerate(var.values))])

            cont = numpy.array(cont)
            ncval = len(cvar_values)

            maxh = 0 #maximal column height
            maxrh = 0 #maximal relative column height
            scvar = cont.sum(axis=1)
            #a cvar with sum=0 with allways have distribution counts 0,
            #therefore we can divide it by anything
            scvar[scvar==0] = 1
            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                maxh = max(maxh, max(dist))
                maxrh = max(maxrh, max(dist/scvar))

            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                dsum = sum(dist)
                geom = QtCore.QRectF(i - 0.333, 0, 0.666, maxrh
                                     if self.relative_freq else maxh)
                if self.show_prob:
                    prob = dist / dsum
                    ci = 1.96 * numpy.sqrt(prob * (1 - prob) / dsum)
                else:
                    ci = None
                item = DistributionBarItem(geom, dist/scvar/maxrh
                                           if self.relative_freq
                                           else dist/maxh, colors)
                self.plot.addItem(item)

                if self.show_prob:
                    for ic, a in enumerate(dist):
                        if self.show_prob - 1 != ic and \
                                self.show_prob - 1 != len(dist):
                            continue
                        position = -0.333 + ((ic+0.5)*0.666/len(dist))
                        if dsum < 1e-6:
                            continue
                        prob = a / dsum
                        if not 1e-6 < prob < 1 - 1e-6:
                            continue
                        ci = 1.96 * sqrt(prob * (1 - prob) / dsum)
                        mark = pg.ScatterPlotItem()
                        bar = pg.ErrorBarItem()
                        pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(0)), 1)
                        pen.setCosmetic(True)
                        bar.setData(x=[i+position], y=[prob],
                                    bottom=min(numpy.array([ci]), prob),
                                    top=min(numpy.array([ci]), 1 - prob),
                                     beam=numpy.array([0.05]),
                                     brush=QtGui.QColor(1), pen=pen)
                        mark.setData([i+position], [prob], antialias=True, symbol="o",
                                 fillLevel=None, pxMode=True, size=10,
                                 brush=QtGui.QColor(colors[ic]), pen=pen)
                        self.plot_prob.addItem(bar)
                        self.plot_prob.addItem(mark)

        for color, name in zip(colors, cvar_values):
            self._legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10, shape="s"),
                escape(name)
            )
        self._legend.show()

    def set_left_axis_name(self):
        set_label = self.plot.getAxis("left").setLabel
        if self.var and self.var.is_continuous:
            set_label(["Density", "Relative density"]
                      [self.cvar is not None and self.relative_freq])
        else:
            set_label(["Frequency", "Relative frequency"]
                      [self.cvar is not None and self.relative_freq])

    def enable_disable_rel_freq(self):
        self.cb_prob.setDisabled(self.var is None or self.cvar is None)
        self.cb_rel_freq.setDisabled(
            self.var is None or self.cvar is None)

    def _on_variable_idx_changed(self):
        self.variable_idx = selected_index(self.varview)
        self._setup()

    def _on_groupvar_idx_changed(self):
        self._setup()

    def _on_set_smoothing(self):
        self._setup()

    def onDeleteWidget(self):
        self.plot.clear()
        super().onDeleteWidget()

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.plot,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


def dist_sum(D1, D2):
    """
    A sum of two continuous distributions.
    """
    X1, W1 = D1
    X2, W2 = D2
    X = numpy.r_[X1, X2]
    W = numpy.r_[W1, W2]
    sort_ind = numpy.argsort(X)
    X, W = X[sort_ind], W[sort_ind]

    unique, uniq_index = numpy.unique(X, return_index=True)
    spans = numpy.diff(numpy.r_[uniq_index, len(X)])
    W = [numpy.sum(W[start:start + span])
         for start, span in zip(uniq_index, spans)]
    W = numpy.array(W)
    assert W.shape[0] == unique.shape[0]
    return unique, W


def ash_curve(dist, cont=None, bandwidth=None, m=3, smoothing_factor=1):
    dist = numpy.asarray(dist)
    X, W = dist
    if bandwidth is None:
        std = weighted_std(X, weights=W)
        size = X.size
        # if only one sample in the class
        if std == 0 and cont is not None:
            std = weighted_std(cont.values, weights=numpy.sum(cont.counts, axis=0))
            size = cont.values.size
        # if attr is constant or contingencies is None (no class variable)
        if std == 0:
            std = 0.1
            size = X.size
        bandwidth = 3.5 * std * (size ** (-1 / 3))

    hist, edges = average_shifted_histogram(X, bandwidth, m, weights=W,
                                            smoothing=smoothing_factor)
    return edges, hist


def average_shifted_histogram(a, h, m=3, weights=None, smoothing=1):
    """
    Compute the average shifted histogram.

    Parameters
    ----------
    a : array-like
        Input data.
    h : float
        Base bin width.
    m : int
        Number of shifted histograms.
    weights : array-like
        An array of weights of the same shape as `a`
    """
    a = numpy.asarray(a)

    if weights is not None:
        weights = numpy.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a")
        weights = weights.ravel()

    a = a.ravel()

    amin, amax = a.min(), a.max()
    h = h * 0.5 * smoothing
    delta = h / m
    wfac = 4 #extended windows for gaussian smoothing
    offset = (wfac * m - 1) * delta
    nbins = max(numpy.ceil((amax - amin + 2 * offset) / delta), 2 * m * wfac - 1)

    bins = numpy.linspace(amin - offset, amax + offset, nbins + 1,
                          endpoint=True)
    hist, edges = numpy.histogram(a, bins, weights=weights, density=True)

    kernel = gaussian_kernel((numpy.arange(2 * wfac * m - 1) - (wfac * m - 1)) / (wfac * m), wfac)
    kernel = kernel / numpy.sum(kernel)
    ash = numpy.convolve(hist, kernel, mode="same")

    ash = ash / numpy.diff(edges) / ash.sum()
#     assert abs((numpy.diff(edges) * ash).sum()) <= 1e-6
    return ash, edges


def triangular_kernel(x):
    return numpy.clip(1, 0, 1 - numpy.abs(x))


def gaussian_kernel(x, k):
    #fit k standard deviations into available space from [-1 .. 1]
    return 1/(numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (x*k)**2 / (2))


def weighted_std(a, axis=None, weights=None, ddof=0):
    mean = numpy.average(a, axis=axis, weights=weights)

    if axis is not None:
        shape = shape_reduce_keep_dims(a.shape, axis)
        mean = mean.reshape(shape)

    sq_diff = numpy.power(a - mean, 2)
    mean_sq_diff, wsum = numpy.average(
        sq_diff, axis=axis, weights=weights, returned=True
    )

    if ddof != 0:
        mean_sq_diff *= wsum / (wsum - ddof)

    return numpy.sqrt(mean_sq_diff)


def weighted_quantiles(a, prob=[0.25, 0.5, 0.75], alphap=0.4, betap=0.4,
                       axis=None, weights=None):
    a = numpy.asarray(a)
    prob = numpy.asarray(prob)

    sort_ind = numpy.argsort(a, axis)
    a = a[sort_ind]

    if weights is None:
        weights = numpy.ones_like(a)
    else:
        weights = numpy.asarray(weights)
        weights = weights[sort_ind]

    n = numpy.sum(weights)
    k = numpy.cumsum(weights, axis)

    # plotting positions for the known n knots
    pk = (k - alphap * weights) / (n + 1 - alphap * weights - betap * weights)

#     m = alphap + prob * (1 - alphap - betap)

    return numpy.interp(prob, pk, a, left=a[0], right=a[-1])


def shape_reduce_keep_dims(shape, axis):
    if shape is None:
        return ()

    shape = list(shape)
    if isinstance(axis, collections.Sequence):
        for ax in axis:
            shape[ax] = 1
    else:
        shape[axis] = 1
    return tuple(shape)


def main(argv=None):
    import gc
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWDistributions()
    w.show()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"
    data = Orange.data.Table(filename)
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    sys.exit(main())

"""
Distributions
-------------

A widget for plotting attribute distributions.

"""
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import numpy
import pyqtgraph as pg

import Orange.data
from Orange.statistics import distribution, contingency
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette
from Orange.widgets.widget import InputSignal


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
        for d, c in zip(self.dist, self.colors):
            if d == 0:
                continue
            painter.setBrush(QtGui.QBrush(c.lighter()))
            painter.drawRect(QtCore.QRectF(x, y, w, d * h))
            y += d * h
        painter.end()

        self.__picture = picture


class OWDistributions(widget.OWWidget):
    name = "Distributions"
    description = "Display value distributions of a data feature in a graph."
    icon = "icons/Distribution.svg"
    priority = 100
    inputs = [InputSignal("Data", Orange.data.Table, "set_data",
                          doc="Set the input data set")]

    settingsHandler = settings.DomainContextHandler()
    #: Selected variable index
    variable_idx = settings.ContextSetting(-1)
    #: Selected group variable
    groupvar_idx = settings.ContextSetting(0)

    Hist, ASH, Kernel = 0, 1, 2
    #: Continuous variable density estimation method
    cont_est_type = settings.Setting(ASH)
    relative_freq = settings.Setting(False)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.distributions = None
        self.contingencies = None
        self.var = self.cvar = None
        varbox = gui.widgetBox(self.controlArea, "Variable")

        self.varmodel = itemmodels.VariableListModel()
        self.groupvarmodel = itemmodels.VariableListModel()

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
        gui.separator(varbox, 8, 8)
        gui.comboBox(
            varbox, self, "cont_est_type", label="Show continuous variables by",
            valueType=int,
            items=["Histograms", "Average shifted histograms",
                   "Kernel density estimators"],
            callback=self._on_cont_est_type_changed)

        box = gui.widgetBox(self.controlArea, "Group by")
        self.groupvarview = QtGui.QListView(
            selectionMode=QtGui.QListView.SingleSelection)
        self.groupvarview.setFixedHeight(100)
        self.groupvarview.setSizePolicy(
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.groupvarview.setModel(self.groupvarmodel)
        self.groupvarview.selectionModel().selectionChanged.connect(
            self._on_groupvar_idx_changed)
        box.layout().addWidget(self.groupvarview)
        self.cb_rel_freq = gui.checkBox(
            box, self, "relative_freq", "Show relative frequencies",
            callback=self._on_relative_freq_changed)

        plotview = pg.PlotWidget(background=None)
        self.mainArea.layout().addWidget(plotview)
        w = QtGui.QLabel()
        w.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        self.mainArea.layout().addWidget(w, Qt.AlignCenter)

        self.plot = pg.PlotItem()
#         self.plot.getViewBox().setMouseEnabled(False, False)
        self.plot.getViewBox().setMenuEnabled(False)
        plotview.setCentralItem(self.plot)

        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(pen)

    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.data = data
        if self.data is not None:
            domain = self.data.domain
            self.varmodel[:] = list(domain)
            self.groupvarmodel[:] = \
                ["(None)"] + [var for var in domain if var.is_discrete]
            if domain.has_discrete_class:
                self.groupvar_idx = \
                    list(self.groupvarmodel).index(domain.class_var)
            self.openContext(domain)
            self.variable_idx = min(max(self.variable_idx, 0),
                                    len(self.varmodel) - 1)
            self.groupvar_idx = min(max(self.groupvar_idx, 0),
                                    len(self.groupvarmodel) - 1)
            itemmodels.select_row(self.groupvarview, self.groupvar_idx)
            itemmodels.select_row(self.varview, self.variable_idx)
            self._setup()

    def clear(self):
        self.plot.clear()
        self.varmodel[:] = []
        self.groupvarmodel[:] = []
        self.variable_idx = -1
        self.groupvar_idx = 0

    def _setup(self):
        self.plot.clear()
        varidx = self.variable_idx
        self.var = self.cvar = None
        if varidx >= 0:
            self.var = self.varmodel[varidx]
        if self.groupvar_idx > 0:
            self.cvar = self.groupvarmodel[self.groupvar_idx]
        self.set_left_axis_name()
        self.enable_disable_rel_freq()
        if self.var is None:
            return
        if self.cvar:
            self.contingencies = \
                contingency.get_contingency(self.data, self.var, self.cvar)
            self.display_contingency()
        else:
            self.distributions = \
                distribution.get_distribution(self.data, self.var)
            self.display_distribution()

    def _density_estimator(self):
        if self.cont_est_type == OWDistributions.Hist:
            def hist(dist):
                h, edges = numpy.histogram(dist[0, :], bins=10,
                                           weights=dist[1, :])
                return edges, h
            return hist
        elif self.cont_est_type == OWDistributions.ASH:
            return lambda dist: ash_curve(dist, m=5)
        elif self.cont_est_type == OWDistributions.Kernel:
            return rect_kernel_curve

    def display_distribution(self):
        dist = self.distributions
        var = self.var
        assert len(dist) > 0
        self.plot.clear()

        bottomaxis = self.plot.getAxis("bottom")
        bottomaxis.setLabel(var.name)

        self.set_left_axis_name()
        if var and var.is_continuous:
            bottomaxis.setTicks(None)
            curve_est = self._density_estimator()
            edges, curve = curve_est(dist)
            item = pg.PlotCurveItem()
            item.setData(edges, curve, antialias=True, stepMode=True,
                         fillLevel=0, brush=QtGui.QBrush(Qt.gray),
                         pen=QtGui.QColor(Qt.white))
            self.plot.addItem(item)
        else:
            bottomaxis.setTicks([list(enumerate(var.values))])
            for i, w in enumerate(dist):
                geom = QtCore.QRectF(i - 0.33, 0, 0.66, w)
                print(w, list(enumerate(var.values)))
                item = DistributionBarItem(geom, [1.0],
                                           [QtGui.QColor(128, 128, 128)])
                self.plot.addItem(item)

    def _on_relative_freq_changed(self):
        self.set_left_axis_name()
        if self.cvar and self.cvar.is_discrete:
            self.display_contingency()
        else:
            self.display_distribution()

    def display_contingency(self):
        """
        Set the contingency to display.
        """
        cont = self.contingencies
        var, cvar = self.var, self.cvar
        assert len(cont) > 0
        self.plot.clear()

        bottomaxis = self.plot.getAxis("bottom")
        bottomaxis.setLabel(var.name)

        palette = colorpalette.ColorPaletteGenerator(len(cvar.values))
        colors = [palette[i] for i in range(len(cvar.values))]

        if var and var.is_continuous:
            bottomaxis.setTicks(None)

            weights = numpy.array([numpy.sum(W) for _, W in cont])
            weights /= numpy.sum(weights)

            curve_est = self._density_estimator()
            curves = [curve_est(dist) for dist in cont]
            curves = [(X, Y * w) for (X, Y), w in zip(curves, weights)]

            cum_curves = [curves[0]]
            for X, Y in curves[1:]:
                cum_curves.append(sum_rect_curve(X, Y, *cum_curves[-1]))

            for (X, Y), color in reversed(list(zip(cum_curves, colors))):
                item = pg.PlotCurveItem()
                pen = QtGui.QPen(QtGui.QBrush(Qt.white), 0.5)
                pen.setCosmetic(True)
                item.setData(X, Y, antialias=True, stepMode=True,
                             fillLevel=0, brush=QtGui.QBrush(color.lighter()),
                             pen=pen)
                self.plot.addItem(item)

#             # XXX: sum the individual curves and not the distributions.
#             # The conditional distributions might be 'smoother' than
#             # the cumulative one
#             cum_dist = [cont[0]]
#             for dist in cont[1:]:
#                 cum_dist.append(dist_sum(dist, cum_dist[-1]))
#
#             curves = [rect_kernel_curve(dist) for dist in cum_dist]
#             colors = [Qt.blue, Qt.red, Qt.magenta]
#             for (X, Y), color in reversed(list(zip(curves, colors))):
#                 item = pg.PlotCurveItem()
#                 item.setData(X, Y, antialias=True, stepMode=True,
#                              fillLevel=0, brush=QtGui.QBrush(color))
#                 item.setPen(QtGui.QPen(color))
#                 self.plot.addItem(item)
        elif var and var.is_discrete:
            bottomaxis.setTicks([list(enumerate(var.values))])

            cont = numpy.array(cont)
            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                dsum = sum(dist)
                geom = QtCore.QRectF(i - 0.333, 0, 0.666, 100
                                     if self.relative_freq else dsum)
                item = DistributionBarItem(geom, dist / dsum, colors)
                self.plot.addItem(item)

    def set_left_axis_name(self):
        set_label = self.plot.getAxis("left").setLabel
        if (self.var and
            self.var.is_continuous and
            self.cont_est_type != OWDistributions.Hist):
            set_label("Density")
        else:
            set_label(["Frequency", "Relative frequency"]
                      [self.cvar is not None and self.relative_freq])

    def enable_disable_rel_freq(self):
        self.cb_rel_freq.setDisabled(
            self.var is None or self.cvar is None or self.var.is_continuous)

    def _on_variable_idx_changed(self):
        self.variable_idx = selected_index(self.varview)
        self._setup()

    def _on_groupvar_idx_changed(self):
        self.groupvar_idx = selected_index(self.groupvarview)
        self._setup()

    def _on_cont_est_type_changed(self):
        self.set_left_axis_name()
        if self.data is not None:
            self._setup()

    def onDeleteWidget(self):
        self.plot.clear()
        super().onDeleteWidget()


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


def rect_kernel_curve(dist, bandwidth=None):
    """
    Return a rectangular kernel density curve for `dist`.

    `dist` must not be empty.
    """
    # XXX: what to do with unknowns
    #      Distribute uniformly between the all points?

    dist = numpy.array(dist)
    if dist.size == 0:
        raise ValueError("'dist' is empty.")

    X = dist[0, :]
    W = dist[1, :]

    def IQR(a, weights=None):
        """Interquartile range of `a`."""
        q1, q3 = weighted_quantiles(a, [0.25, 0.75], weights=weights)
        return q3 - q1

    if bandwidth is None:
        # Silverman's rule of thumb.
        A = weighted_std(X, weights=W)
        iqr = IQR(X, weights=W)
        if iqr > 0:
            A = min(A, iqr / 1.34)

        bandwidth = 0.9 * A * (X.size ** -0.2)

    bottom_edges = X - bandwidth / 2
    top_edges = X + bandwidth / 2

    edges = numpy.hstack((bottom_edges, top_edges))
    edge_weights = numpy.hstack((W, -W))

    sort_ind = numpy.argsort(edges)
    edges = edges[sort_ind]
    edge_weights = edge_weights[sort_ind]

    # NOTE: The final cumulative sum element is 0
    curve = numpy.cumsum(edge_weights)[:-1]
    curve /= numpy.sum(W) * bandwidth
    return edges, curve


def sum_rect_curve(Xa, Ya, Xb, Yb):
    """
    Sum two curves (i.e. stack one over the other).
    """
    X = numpy.r_[Xa, Xb]
    Y = numpy.r_[Ya, 0, Yb, 0]
    assert X.shape == Y.shape

    dY = numpy.r_[Y[0], numpy.diff(Y)]
    sort_ind = numpy.argsort(X)
    X = X[sort_ind]
    dY = dY[sort_ind]

    unique, uniq_index = numpy.unique(X, return_index=True)
    spans = numpy.diff(numpy.r_[uniq_index, len(X)])
    dY = [numpy.sum(dY[start:start + span])
          for start, span in zip(uniq_index, spans)]
    dY = numpy.array(dY)
    assert dY.shape[0] == unique.shape[0]
    # NOTE: The final cumulative sum element is 0
    Y = numpy.cumsum(dY)[:-1]

    return unique, Y


def ash_curve(dist, bandwidth=None, m=3, weights=None):
    dist = numpy.asarray(dist)
    X, W = dist
    if bandwidth is None:
        bandwidth = 3.5 * weighted_std(X, weights=W) * (X.size ** (-1 / 3))

    hist, edges = average_shifted_histogram(X, bandwidth, m, weights=W)
    return edges, hist


def average_shifted_histogram(a, h, m=3, weights=None):
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
    delta = h / m
    offset = (m - 1) * delta
    nbins = numpy.ceil((amax - amin + 2 * offset) / delta)
    bins = numpy.linspace(amin - offset, amax + offset, nbins + 1,
                          endpoint=True)
    hist, edges = numpy.histogram(a, bins, weights=weights, density=True)

    kernel = triangular_kernel((numpy.arange(2 * m - 1) - (m - 1)) / m)
    kernel = kernel / numpy.sum(kernel)
    ash = numpy.convolve(hist, kernel, mode="same")

    ash = ash / numpy.diff(edges) / ash.sum()
#     assert abs((numpy.diff(edges) * ash).sum()) <= 1e-6
    return ash, edges


def triangular_kernel(x):
    return numpy.clip(1, 0, 1 - numpy.abs(x))


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


def main():
    import gc
    app = QtGui.QApplication([])
    w = OWDistributions()
    w.show()
#     data = Orange.data.Table("brown-selected")
#     data = Orange.data.Table("lenses")
#     data = Orange.data.Table("housing")
    data = Orange.data.Table("heart_disease")
    w.set_data(data)
    rval = app.exec_()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    main()

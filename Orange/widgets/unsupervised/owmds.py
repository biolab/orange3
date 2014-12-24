import numpy

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import sklearn.manifold

import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette

from Orange.widgets.utils import itemmodels

import Orange.data
import Orange.distance
import Orange.misc


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


def torgerson(distances, n_components=2):
    distances = numpy.asarray(distances)
    assert distances.shape[0] == distances.shape[1]
    N = distances.shape[0]
    # O ^ 2
    D_sq = distances ** 2

    # double center the D_sq
    rsum = numpy.sum(D_sq, axis=1, keepdims=True)
    csum = numpy.sum(D_sq, axis=0, keepdims=True)
    total = numpy.sum(csum)
    D_sq -= rsum / N
    D_sq -= csum / N
    D_sq += total / (N ** 2)
    B = numpy.multiply(D_sq, -0.5, out=D_sq)

    U, L, _ = numpy.linalg.svd(B)
    U = U[:, :n_components]
    L = L[:n_components]
    D = numpy.diag(numpy.sqrt(L))
    return numpy.dot(U, D)


class OWMDS(widget.OWWidget):
    name = "MDS"
    description = "Multidimensional scaling"
    icon = "icons/MDS.svg"

    inputs = (
        {"name": "Data",
         "type": Orange.data.Table,
         "handler": "set_data"},
        {"name": "Distances",
         "type": Orange.misc.DistMatrix,
         "handler": "set_disimilarity"}
    )

    outputs = ({"name": "Data", "type": Orange.data.Table},)

    #: Initialization type
    PCA, Random = 0, 1

    settingsHandler = settings.DomainContextHandler()

    max_iter = settings.Setting(300)
    eps = settings.Setting(1e-3)
    initialization = settings.Setting(PCA)
    n_init = settings.Setting(1)

    output_embeding_role = settings.Setting(1)
    autocommit = settings.Setting(True)

    color_var = settings.ContextSetting(0, not_variable=True)
    shape_var = settings.ContextSetting(0, not_variable=True)
    size_var = settings.ContextSetting(0, not_variable=True)

    # output embeding role.
    NoRole, AttrRole, MetaRole = 0, 1, 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix = None
        self.data = None

        self._pen_data = None
        self._shape_data = None
        self._size_data = None

        self._invalidated = False
        self._effective_matrix = None
        self._output_changed = False

        box = gui.widgetBox(self.controlArea, "MDS Optimization")
        form = QtGui.QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
        )

        form.addRow("Max iterations:",
                    gui.spin(box, self, "max_iter", 10, 10 ** 4, step=1))

#         form.addRow("Eps:",
#                     gui.spin(box, self, "eps", 1e-9, 1e-3, step=1e-9,
#                              spinType=float))

        form.addRow("Initialization",
                    gui.comboBox(box, self, "initialization",
                                 items=["PCA (Torgerson)", "Random"]))

#         form.addRow("N Restarts:",
#                     gui.spin(box, self, "n_init", 1, 10, step=1))

        box.layout().addLayout(form)
        gui.button(box, self, "Apply", callback=self._invalidate_embeding)

        box = gui.widgetBox(self.controlArea, "Graph")
        self.colorvar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "color_var", box="Color",
                          callback=self._on_color_var_changed)
        cb.setModel(self.colorvar_model)
        cb.box.setFlat(True)

        self.shapevar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "shape_var", box="Shape",
                          callback=self._on_shape_var_changed)
        cb.setModel(self.shapevar_model)
        cb.box.setFlat(True)

        self.sizevar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "size_var", "Size",
                          callback=self._on_size_var_changed)
        cb.setModel(self.sizevar_model)
        cb.box.setFlat(True)

        gui.rubber(self.controlArea)
        box = gui.widgetBox(self.controlArea, "Output")
        cb = gui.comboBox(box, self, "output_embeding_role",
                          box="Append coordinates",
                          items=["Do not append", "As attributes", "As metas"],
                          callback=self._invalidate_output)
        cb.box.setFlat(True)

        cb = gui.checkBox(box, self, "autocommit", "Auto commit")
        b = gui.button(box, self, "Commit", callback=self.commit, default=True)
        gui.setStopper(self, b, cb, "_output_changed", callback=self.commit)

        self.plot = pg.PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, data):
        self.closeContext()
        self._clear()
        self.data = data
        if data is not None:
            self._initialize(data)
            self.openContext(data)

        if self.matrix is None:
            self._effective_matrix = None
            self._invalidated = True

    def set_disimilarity(self, matrix):
        self.matrix = matrix
        self._effective_matrix = matrix
        self._invalidated = True

    def _clear(self):
        self._pen_data = None
        self._shape_data = None
        self._size_data = None
        self.colorvar_model[:] = ["Same color"]
        self.shapevar_model[:] = ["Same shape"]
        self.sizevar_model[:] = ["Same size"]

        self.color_var = 0
        self.shape_var = 0
        self.size_var = 0

    def _initialize(self, data):
        # initialize the graph state from data
        domain = data.domain
        all_vars = list(domain.variables + domain.metas)
        disc_vars = list(filter(is_discrete, all_vars))
        cont_vars = list(filter(is_continuous, all_vars))

        def set_separator(model, index):
            index = model.index(index, 0)
            model.setData(index, "separator", Qt.AccessibleDescriptionRole)
            model.setData(index, Qt.NoItemFlags, role="flags")

        self.colorvar_model[:] = ["Same color", ""] + all_vars
        set_separator(self.colorvar_model, 1)

        self.shapevar_model[:] = ["Same shape", ""] + disc_vars
        set_separator(self.shapevar_model, 1)

        self.sizevar_model[:] = ["Same size", ""] + cont_vars
        set_separator(self.sizevar_model, 1)

        if domain.class_var is not None:
            self.color_var = list(self.colorvar_model).index(domain.class_var)

    def apply(self):
        if self.data is None and self.matrix is None:
            self.embeding = None
            self._update_plot()
            return

        if self._effective_matrix is None:
            if self.matrix is not None:
                self._effective_matrix = self.matrix
            elif self.data is not None:
                self._effective_matrix = Orange.distance.Euclidean(self.data)

        X = self._effective_matrix.X

        if self.initialization == OWMDS.PCA:
            init = torgerson(X, n_components=2)
            n_init = 1
        else:
            init = None
            n_init = self.n_init

        dissim = "precomputed"

        mds = sklearn.manifold.MDS(
            dissimilarity=dissim, n_components=2,
            n_init=n_init, max_iter=self.max_iter
        )
        embeding = mds.fit_transform(X, init=init)
        self.embeding = embeding
        self.stress = mds.stress_

    def handleNewSignals(self):
        if self._invalidated:
            self._invalidated = False
            self.apply()

        self._update_plot()
        self.commit()

    def _invalidate_embeding(self):
        self.apply()
        self._update_plot()
        self._invalidate_output()

    def _invalidate_output(self):
        if self.autocommit:
            self.commit()
        else:
            self._output_changed = True

    def _on_color_var_changed(self):
        self._pen_data = None
        self._update_plot()

    def _on_shape_var_changed(self):
        self._shape_data = None
        self._update_plot()

    def _on_size_var_changed(self):
        self._size_data = None
        self._update_plot()

    def _update_plot(self):
        self.plot.clear()
        if self.embeding is not None:
            self._setup_plot()

    def _setup_plot(self):
        have_data = self.data is not None

        if self._pen_data is None:
            if have_data and self.color_var > 0:
                color_var = self.colorvar_model[self.color_var]
                if is_discrete(color_var):
                    palette = colorpalette.ColorPaletteGenerator(
                        len(color_var.values)
                    )
                else:
                    palette = None

                color_data = colors(self.data, color_var, palette)
                pen_data = [QtGui.QPen(QtGui.QColor(r, g, b))
                            for r, g, b in color_data]
            else:
                pen_data = QtGui.QPen(Qt.black)
            self._pen_data = pen_data

        if self._shape_data is None:
            if have_data and self.shape_var > 0:
                Symbols = pg.graphicsItems.ScatterPlotItem.Symbols
                symbols = numpy.array(list(Symbols.keys()))

                shape_var = self.shapevar_model[self.shape_var]
                data = numpy.array(self.data[:, shape_var]).ravel()
                data = data % (len(Symbols) - 1)
                data[numpy.isnan(data)] = len(Symbols) - 1
                shape_data = symbols[data.astype(int)]
            else:
                shape_data = "o"
            self._shape_data = shape_data

        if self._size_data is None:
            MinPointSize = 1
            point_size = 8 + MinPointSize
            if have_data and self.size_var > 0:
                size_var = self.sizevar_model[self.size_var]
                size_data = numpy.array(self.data[:, size_var]).ravel()
                dmin, dmax = numpy.nanmin(size_data), numpy.nanmax(size_data)
                if dmax - dmin > 0:
                    size_data = (size_data - dmin) / (dmax - dmin)

                size_data = MinPointSize + size_data * point_size
            else:
                size_data = point_size

        item = pg.ScatterPlotItem(
            x=self.embeding[:, 0], y=self.embeding[:, 1],
            pen=self._pen_data, symbol=self._shape_data,
            brush=QtGui.QBrush(Qt.transparent),
            size=size_data,
            antialias=True
        )
        # plot(x, y, colors=plot.colors(data[:, color_var]),
        #      point_size=data[:, size_var],
        #      symbol=data[:, symbol_var])

        self.plot.addItem(item)

    def commit(self):
        if self.embeding is not None:
            output = embeding = Orange.data.Table.from_numpy(
                Orange.data.Domain([Orange.data.ContinuousVariable("X"),
                                    Orange.data.ContinuousVariable("Y")]),
                self.embeding
            )
        else:
            output = embeding = None

        if self.embeding is not None and self.data is not None:
            X, Y, M = self.data.X, self.data.Y, self.data.metas
            domain = self.data.domain
            attrs = domain.attributes
            class_vars = domain.class_vars
            metas = domain.metas

            if self.output_embeding_role == OWMDS.NoRole:
                pass
            elif self.output_embeding_role == OWMDS.AttrRole:
                attrs = attrs + embeding.domain.attributes
                X = numpy.c_[X, embeding.X]
            elif self.output_embeding_role == OWMDS.MetaRole:
                metas = metas + embeding.domain.attributes
                M = numpy.c_[M, embeding.X]

            domain = Orange.data.Domain(attrs, class_vars, metas)
            output = Orange.data.Table.from_numpy(domain, X, Y, M)

        self.send("Data", output)
        self._output_changed = False

    def onDeleteWidget(self):
        self.plot.clear()
        super().onDeleteWidget()


def colors(data, variable, palette=None):
    if palette is None:
        if is_discrete(variable):
            palette = colorpalette.ColorPaletteGenerator(len(variable.values))
        elif is_continuous(variable):
            palette = colorpalette.ColorPaletteBW()
            palette = colorpalette.ContinuousPaletteGenerator(
                QtGui.QColor(220, 220, 220),
                QtGui.QColor(0, 0, 0),
                False
            )
        else:
            raise TypeError()

    x = numpy.array(data[:, variable]).ravel()

    if is_discrete(variable):
        nvalues = len(variable.values)
        x[numpy.isnan(x)] = nvalues
        color_index = palette.getRGB(numpy.arange(nvalues + 1))
        # Unknown values as gray
        # TODO: This should already be a part of palette
        color_index[nvalues] = (128, 128, 128)
        colors = color_index[x.astype(int)]
    else:
        x, _ = scaled(x)
        mask = numpy.isnan(x)
        colors = numpy.empty((len(x), 3))
        colors[mask] = (128, 128, 128)
        colors[~mask] = [palette.getRGB(v) for v in x[~mask]]
#         colors[~mask] = interpolate(palette, x[~mask], left=Qt.gray)

    return colors


def scaled(a):
    amin, amax = numpy.nanmin(a), numpy.nanmax(a)
    span = amax - amin
    return (a - amin) / (span or 1), (amin, amax)


def main_test():
    import gc
    app = QtGui.QApplication([])
    w = OWMDS()
    w.set_data(Orange.data.Table("iris"))
#     w.set_data(Orange.data.Table("wine"))
    w.handleNewSignals()

    w.show()
    w.raise_()
    rval = app.exec_()

    w.saveSettings()
    w.onDeleteWidget()
    w.deleteLater()
    del w
    gc.collect()
    app.processEvents()
    return rval

if __name__ == "__main__":
    import sys
    sys.exit(main_test())

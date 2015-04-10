import numpy
import scipy.spatial.distance

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette

from Orange.widgets.utils import itemmodels

import Orange.data
import Orange.projection
import Orange.distance
import Orange.misc


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


def torgerson(distances, n_components=2):
    """
    Perform classical mds (Torgerson scaling).

    ..note ::
        If the distances are euclidean then this is equivalent to projecting
        the original data points to the first `n` principal components.

    """
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


def stress(X, D):
    assert X.shape[0] == D.shape[0] == D.shape[1]
    D1_c = scipy.spatial.distance.pdist(X, metric="euclidean")
    D1 = scipy.spatial.distance.squareform(D1_c, checks=False)
    delta = D1 - D
    delta_sq = numpy.square(delta, out=delta)
    return delta_sq.sum(axis=0) / 2


def make_pen(color, width=1.5, style=Qt.SolidLine, cosmetic=False):
    pen = QtGui.QPen(color, width, style)
    pen.setCosmetic(cosmetic)
    return pen


class ScatterPlotItem(pg.ScatterPlotItem):
    Symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, option, widget=None):
        if self.opts["pxMode"]:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        if self.opts["antialias"]:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        super().paint(painter, option, widget)


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
    initialization = settings.Setting(PCA)

    # output embedding role.
    NoRole, AttrRole, MetaRole = 0, 1, 2

    output_embedding_role = settings.Setting(1)
    autocommit = settings.Setting(True)

    color_index = settings.ContextSetting(0, not_attribute=True)
    shape_index = settings.ContextSetting(0, not_attribute=True)
    size_index = settings.ContextSetting(0, not_attribute=True)
    label_index = settings.ContextSetting(0, not_attribute=True)

    symbol_size = settings.Setting(8)
    symbol_opacity = settings.Setting(230)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix = None
        self.data = None
        self.matrix_data = None
        self.signal_data = None

        self._pen_data = None
        self._shape_data = None
        self._size_data = None
        self._label_data = None

        self._invalidated = False
        self._effective_matrix = None

        box = gui.widgetBox(self.controlArea, "MDS Optimization")
        form = QtGui.QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
        )

        form.addRow("Max iterations:",
                    gui.spin(box, self, "max_iter", 10, 10 ** 4, step=1))

        form.addRow("Initialization",
                    gui.comboBox(box, self, "initialization",
                                 items=["PCA (Torgerson)", "Random"]))

        box.layout().addLayout(form)
        gui.button(box, self, "Apply", callback=self._invalidate_embedding)

        box = gui.widgetBox(self.controlArea, "Graph")
        self.colorvar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "color_index", box="Color",
                          callback=self._on_color_index_changed)
        cb.setModel(self.colorvar_model)
        cb.box.setFlat(True)

        self.shapevar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "shape_index", box="Shape",
                          callback=self._on_shape_index_changed)
        cb.setModel(self.shapevar_model)
        cb.box.setFlat(True)

        self.sizevar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "size_index", "Size",
                          callback=self._on_size_index_changed)
        cb.setModel(self.sizevar_model)
        cb.box.setFlat(True)

        self.labelvar_model = itemmodels.VariableListModel()
        cb = gui.comboBox(box, self, "label_index", "Label",
                          callback=self._on_label_index_changed)
        cb.setModel(self.labelvar_model)
        cb.box.setFlat(True)

        form = QtGui.QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
        )
        form.addRow("Symbol size",
                    gui.hSlider(box, self, "symbol_size",
                                minValue=1, maxValue=20,
                                callback=self._on_size_index_changed,
                                createLabel=False))
        form.addRow("Symbol opacity",
                    gui.hSlider(box, self, "symbol_opacity",
                                minValue=100, maxValue=255, step=100,
                                callback=self._on_color_index_changed,
                                createLabel=False))
        box.layout().addLayout(form)
        gui.rubber(self.controlArea)
        box = gui.widgetBox(self.controlArea, "Output")
        cb = gui.comboBox(box, self, "output_embedding_role",
                          box="Append coordinates",
                          items=["Do not append", "As attributes", "As metas"],
                          callback=self._invalidate_output)
        cb.box.setFlat(True)

        gui.auto_commit(box, self, "autocommit", "Send data",
                        checkbox_label="Send after any change",
                        box=None)

        self.plot = pg.PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, data):
        self.signal_data = data

        if self.matrix and data is not None and len(self.matrix.X) == len(data):
            self.closeContext()
            self.data = data
            self.update_controls()
            self.openContext(data)
        else:
            self._invalidated = True

    def set_disimilarity(self, matrix):
        self.matrix = matrix
        if matrix and matrix.row_items:
            self.matrix_data = matrix.row_items
        if matrix is None:
            self.matrix_data = None
        self._invalidated = True

    def _clear(self):
        self._pen_data = None
        self._shape_data = None
        self._size_data = None
        self._label_data = None

        self.colorvar_model[:] = ["Same color"]
        self.shapevar_model[:] = ["Same shape"]
        self.sizevar_model[:] = ["Same size"]
        self.labelvar_model[:] = ["No labels"]

        self.color_index = 0
        self.shape_index = 0
        self.size_index = 0
        self.label_index = 0

    def update_controls(self):
        if getattr(self.matrix, 'axis', 1) == 0:
            # Column-wise distances
            attr = "Attribute names"
            self.labelvar_model[:] = ["No labels", attr]
            self.shapevar_model[:] = ["Same shape", attr]
            self.colorvar_model[:] = ["Same color", attr]

            self.color_index = list(self.colorvar_model).index(attr)
            self.shape_index = list(self.shapevar_model).index(attr)
        else:
            # initialize the graph state from data
            domain = self.data.domain
            all_vars = list(domain.variables + domain.metas)
            disc_vars = list(filter(is_discrete, all_vars))
            cont_vars = list(filter(is_continuous, all_vars))
            str_vars = [var for var in all_vars
                        if isinstance(var, (Orange.data.DiscreteVariable,
                                            Orange.data.StringVariable))]

            def set_separator(model, index):
                index = model.index(index, 0)
                model.setData(index, "separator", Qt.AccessibleDescriptionRole)
                model.setData(index, Qt.NoItemFlags, role="flags")

            self.colorvar_model[:] = ["Same color", ""] + all_vars
            set_separator(self.colorvar_model, 1)

            self.shapevar_model[:] = ["Same shape", ""] + disc_vars
            set_separator(self.shapevar_model, 1)

            self.sizevar_model[:] = ["Same size", "Stress", ""] + cont_vars
            set_separator(self.sizevar_model, 2)

            self.labelvar_model[:] = ["No labels", ""] + str_vars
            set_separator(self.labelvar_model, 1)

            if domain.class_var is not None:
                self.color_index = list(self.colorvar_model).index(domain.class_var)

    def apply(self):
        # clear everything
        self.closeContext()
        self._clear()
        self.data = None
        self._effective_matrix = None
        self.embedding = None

        # if no data nor matrix is present reset plot
        if self.signal_data is None and self.matrix is None:
            return self._update_plot()

        if self.signal_data and self.matrix_data and len(self.signal_data) != len(self.matrix_data):
            self.error(1, "Data and distances dimensions do not match.")
            return self._update_plot()
        self.error(1)

        if self.signal_data:
            self.data = self.signal_data
        elif self.matrix_data:
            self.data = self.matrix_data

        if self.matrix:
            self._effective_matrix = self.matrix
            if self.matrix.axis == 0:
                self.data = None
        else:
            self._effective_matrix = Orange.distance.Euclidean(self.data)

        self.update_controls()
        self.openContext(self.data)

        X = self._effective_matrix.X

        if self.initialization == OWMDS.PCA:
            init = torgerson(X, n_components=2)
            n_init = 1
        else:
            init = None
            n_init = 4

        dissim = "precomputed"

        mds = Orange.projection.MDS(
            dissimilarity=dissim, n_components=2,
            n_init=n_init, max_iter=self.max_iter
        )
        mds_fit = mds.fit(X, init=init)
        self.embedding = mds_fit.embedding_

    def handleNewSignals(self):
        if self._invalidated:
            self._invalidated = False
            self.apply()

        self._update_plot()
        self.unconditional_commit()

    def _invalidate_embedding(self):
        self.apply()
        self._update_plot()
        self._invalidate_output()

    def _invalidate_output(self):
        self.commit()

    def _on_color_index_changed(self):
        self._pen_data = None
        self._update_plot()

    def _on_shape_index_changed(self):
        self._shape_data = None
        self._update_plot()

    def _on_size_index_changed(self):
        self._size_data = None
        self._update_plot()

    def _on_label_index_changed(self):
        self._label_data = None
        self._update_plot()

    def _update_plot(self):
        self.plot.clear()
        if self.embedding is not None:
            self._setup_plot()

    def _setup_plot(self):
        have_data = self.data is not None
        have_matrix_transposed = self.matrix is not None and not self.matrix.axis

        def column(data, variable):
            a, _ = data.get_column_view(variable)
            return a.ravel()

        def attributes(matrix):
            return matrix.row_items.domain.attributes

        def scale(a):
            dmin, dmax = numpy.nanmin(a), numpy.nanmax(a)
            if dmax - dmin > 0:
                return (a - dmin) / (dmax - dmin)
            else:
                return numpy.zeros_like(a)

        if self._pen_data is None:
            if have_data and self.color_index > 0:
                color_var = self.colorvar_model[self.color_index]
                if is_discrete(color_var):
                    palette = colorpalette.ColorPaletteGenerator(
                        len(color_var.values)
                    )
                else:
                    palette = None

                color_data = colors(self.data, color_var, palette)
                pen_data = [make_pen(QtGui.QColor(r, g, b, self.symbol_opacity),
                                     cosmetic=True)
                            for r, g, b in color_data]
            elif have_matrix_transposed and self.colorvar_model[self.color_index] == 'Attribute names':
                attr = attributes(self.matrix)
                palette = colorpalette.ColorPaletteGenerator(len(attr))
                color_data = [palette.getRGB(i) for i in range(len(attr))]
                pen_data = [make_pen(QtGui.QColor(r, g, b, self.symbol_opacity),
                                     cosmetic=True)
                            for r, g, b in color_data]
            else:
                pen_data = make_pen(QtGui.QColor(Qt.darkGray), cosmetic=True)
            self._pen_data = pen_data

        if self._shape_data is None:
            if have_data and self.shape_index > 0:
                Symbols = ScatterPlotItem.Symbols
                symbols = numpy.array(list(Symbols.keys()))

                shape_var = self.shapevar_model[self.shape_index]
                data = column(self.data, shape_var)
                data = data % (len(Symbols) - 1)
                data[numpy.isnan(data)] = len(Symbols) - 1
                shape_data = symbols[data.astype(int)]
            elif have_matrix_transposed and self.shapevar_model[self.shape_index] == 'Attribute names':
                Symbols = ScatterPlotItem.Symbols
                symbols = numpy.array(list(Symbols.keys()))
                attr = [i % (len(Symbols) - 1) for i, _ in enumerate(attributes(self.matrix))]
                shape_data = symbols[attr]
            else:
                shape_data = "o"
            self._shape_data = shape_data

        if self._size_data is None:
            MinPointSize = 3
            point_size = self.symbol_size + MinPointSize
            if have_data and self.size_index == 1:
                # size by stress
                size_data = stress(self.embedding, self._effective_matrix.X)
                size_data = scale(size_data)
                size_data = MinPointSize + size_data * point_size
            elif have_data and self.size_index > 0:
                size_var = self.sizevar_model[self.size_index]
                size_data = column(self.data, size_var)
                size_data = scale(size_data)
                size_data = MinPointSize + size_data * point_size
            else:
                size_data = point_size

        if self._label_data is None:
            if have_data and self.label_index > 0:
                label_var = self.labelvar_model[self.label_index]
                label_data = column(self.data, label_var)
                label_data = [label_var.repr_val(val) for val in label_data]
                label_items = [pg.TextItem(text, anchor=(0.5, 0))
                               for text in label_data]
            elif have_matrix_transposed and self.labelvar_model[self.label_index] == 'Attribute names':
                attr = attributes(self.matrix)
                label_items = [pg.TextItem(str(text), anchor=(0.5, 0))
                               for text in attr]
            else:
                label_items = None
            self._label_data = label_items

        item = ScatterPlotItem(
            x=self.embedding[:, 0], y=self.embedding[:, 1],
            pen=self._pen_data, symbol=self._shape_data,
            brush=QtGui.QBrush(Qt.transparent),
            size=size_data,
            antialias=True
        )
        self.plot.addItem(item)

        if self._label_data is not None:
            for (x, y), text_item in zip(self.embedding, self._label_data):
                self.plot.addItem(text_item)
                text_item.setPos(x, y)

    def commit(self):
        if self.embedding is not None:
            output = embedding = Orange.data.Table.from_numpy(
                Orange.data.Domain([Orange.data.ContinuousVariable("X"),
                                    Orange.data.ContinuousVariable("Y")]),
                self.embedding
            )
        else:
            output = embedding = None

        if self.embedding is not None and self.data is not None:
            X, Y, M = self.data.X, self.data.Y, self.data.metas
            domain = self.data.domain
            attrs = domain.attributes
            class_vars = domain.class_vars
            metas = domain.metas

            if self.output_embedding_role == OWMDS.NoRole:
                pass
            elif self.output_embedding_role == OWMDS.AttrRole:
                attrs = attrs + embedding.domain.attributes
                X = numpy.c_[X, embedding.X]
            elif self.output_embedding_role == OWMDS.MetaRole:
                metas = metas + embedding.domain.attributes
                M = numpy.c_[M, embedding.X]

            domain = Orange.data.Domain(attrs, class_vars, metas)
            output = Orange.data.Table.from_numpy(domain, X, Y, M)

        self.send("Data", output)

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
#     w.set_data(Orange.data.Table("iris"))
    w.set_data(Orange.data.Table("wine"))
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

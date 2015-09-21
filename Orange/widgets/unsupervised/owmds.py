import sys
import warnings
from xml.sax.saxutils import escape

import pkg_resources

import numpy
import scipy.spatial.distance
from itertools import chain

from PyQt4 import QtGui
from PyQt4.QtCore import Qt, QEvent

import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette

from Orange.widgets.utils import itemmodels

import Orange.data
import Orange.projection
import Orange.distance
import Orange.misc
from Orange.widgets.io import FileFormats


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
    if n_components > N:
        U = numpy.hstack((U, numpy.zeros((N, n_components - N))))
        L = numpy.hstack((L, numpy.zeros((n_components - N))))
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
    description = "Two-dimensional data projection by multidimensional " \
                  "scaling constructed from a distance matrix."
    icon = "icons/MDS.svg"
    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Distances", Orange.misc.DistMatrix, "set_disimilarity")]
    outputs = [("Data", Orange.data.Table, widget.Default),
               ("Selected Data", Orange.data.Table)]

    #: Initialization type
    PCA, Random = 0, 1

    #: Refresh rate
    RefreshRate = [
        ("Every iteration", 1),
        ("Every 5 steps", 5),
        ("Every 10 steps", 10),
        ("Every 25 steps", 25),
        ("Every 50 steps", 50),
        ("None", -1)
    ]

    #: Runtime state
    Running, Finished, Waiting = 1, 2, 3

    settingsHandler = settings.DomainContextHandler()

    max_iter = settings.Setting(300)
    initialization = settings.Setting(PCA)
    refresh_rate = settings.Setting(3)

    # output embedding role.
    NoRole, AttrRole, AddAttrRole, MetaRole = 0, 1, 2, 3

    output_embedding_role = settings.Setting(2)
    autocommit = settings.Setting(True)

    color_value = settings.ContextSetting("")
    shape_value = settings.ContextSetting("")
    size_value = settings.ContextSetting("")
    label_value = settings.ContextSetting("")

    symbol_size = settings.Setting(8)
    symbol_opacity = settings.Setting(230)
    connected_pairs = settings.Setting(5)
    spread_equal_points = settings.Setting(False)

    legend_anchor = settings.Setting(((1, 0), (1, 0)))

    want_graph = True

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
        self._similar_pairs = None
        self._scatter_item = None
        self._legend_item = None
        self._selection_mask = None
        self._invalidated = False
        self._effective_matrix = None

        self.__update_loop = None
        self.__state = OWMDS.Waiting
        self.__in_next_step = False
        self.__draw_similar_pairs = False

        box = gui.widgetBox(self.controlArea, "MDS Optimization")
        form = QtGui.QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )

        form.addRow("Max iterations:",
                    gui.spin(box, self, "max_iter", 10, 10 ** 4, step=1))

        form.addRow("Initialization",
                    gui.comboBox(box, self, "initialization",
                                 items=["PCA (Torgerson)", "Random"],
                                 callback=self.__invalidate_embedding))

        box.layout().addLayout(form)
        form.addRow("Refresh",
                    gui.comboBox(
                        box, self, "refresh_rate",
                        items=[t for t, _ in OWMDS.RefreshRate],
                        callback=self.__invalidate_refresh))
        gui.separator(box, 10)
        gui.checkBox(box, self, "spread_equal_points",
                     "Spread points at zero-distances",
                     callback=self.__invalidate_embedding)
        gui.separator(box, 10)
        self.runbutton = gui.button(
            box, self, "Run", callback=self._toggle_run)

        box = gui.widgetBox(self.controlArea, "Graph")
        self.colorvar_model = itemmodels.VariableListModel()

        common_options = {"sendSelectedValue": True, "valueType": str,
                          "orientation": "horizontal", "labelWidth": 50, }

        self.cb_color_value = gui.comboBox(
            box, self, "color_value", label="Color",
            callback=self._on_color_index_changed, **common_options)
        self.cb_color_value.setModel(self.colorvar_model)

        self.shapevar_model = itemmodels.VariableListModel()
        self.cb_shape_value = gui.comboBox(
            box, self, "shape_value", label="Shape",
            callback=self._on_shape_index_changed, **common_options)
        self.cb_shape_value.setModel(self.shapevar_model)

        self.sizevar_model = itemmodels.VariableListModel()
        self.cb_size_value = gui.comboBox(
            box, self, "size_value", label="Size",
            callback=self._on_size_index_changed, **common_options)
        self.cb_size_value.setModel(self.sizevar_model)

        self.labelvar_model = itemmodels.VariableListModel()
        self.cb_label_value = gui.comboBox(
            box, self, "label_value", label="Label",
            callback=self._on_label_index_changed, **common_options)
        self.cb_label_value.setModel(self.labelvar_model)

        form = QtGui.QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
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
        form.addRow("Show similar pairs",
                    gui.hSlider(
                        gui.widgetBox(self.controlArea,
                                      orientation="horizontal"),
                        self, "connected_pairs", minValue=0, maxValue=20,
                        createLabel=False,
                        callback=self._on_connected_changed))
        box.layout().addLayout(form)

        gui.rubber(self.controlArea)

        box = QtGui.QGroupBox("Zoom/Select", )
        box.setLayout(QtGui.QHBoxLayout())
        box.layout().setMargin(2)

        group = QtGui.QActionGroup(self, exclusive=True)

        def icon(name):
            path = "icons/Dlg_{}.png".format(name)
            path = pkg_resources.resource_filename(widget.__name__, path)
            return QtGui.QIcon(path)

        action_select = QtGui.QAction(
            "Select", self, checkable=True, checked=True, icon=icon("arrow"),
            shortcut=QtGui.QKeySequence(Qt.ControlModifier + Qt.Key_1))
        action_zoom = QtGui.QAction(
            "Zoom", self, checkable=True, checked=False, icon=icon("zoom"),
            shortcut=QtGui.QKeySequence(Qt.ControlModifier + Qt.Key_2))
        action_pan = QtGui.QAction(
            "Pan", self, checkable=True, checked=False, icon=icon("pan_hand"),
            shortcut=QtGui.QKeySequence(Qt.ControlModifier + Qt.Key_3))

        action_reset_zoom = QtGui.QAction(
            "Zoom to fit", self, icon=icon("zoom_reset"),
            shortcut=QtGui.QKeySequence(Qt.ControlModifier + Qt.Key_0))
        action_reset_zoom.triggered.connect(
            lambda: self.plot.autoRange(padding=0.1,
                                        items=[self._scatter_item]))
        group.addAction(action_select)
        group.addAction(action_zoom)
        group.addAction(action_pan)
        self.addActions(group.actions() + [action_reset_zoom])
        action_select.setChecked(True)

        def button(action):
            b = QtGui.QToolButton()
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setDefaultAction(action)
            return b

        box.layout().addWidget(button(action_select))
        box.layout().addWidget(button(action_zoom))
        box.layout().addWidget(button(action_pan))
        box.layout().addSpacing(4)
        box.layout().addWidget(button(action_reset_zoom))
        box.layout().addStretch()

        self.controlArea.layout().addWidget(box)

        box = gui.widgetBox(self.controlArea, "Output")
        gui.comboBox(box, self, "output_embedding_role",
                     items=["Original features only",
                            "Coordinates only",
                            "Coordinates as features",
                            "Coordinates as meta attributes"],
                     callback=self._invalidate_output, addSpace=4)
        gui.auto_commit(box, self, "autocommit", "Send data",
                        checkbox_label="Send after any change",
                        box=None)

        self.plot = pg.PlotWidget(background="w", enableMenu=False)
        self.plot.getPlotItem().hideAxis("bottom")
        self.plot.getPlotItem().hideAxis("left")
        self.plot.getPlotItem().hideButtons()
        self.plot.setRenderHint(QtGui.QPainter.Antialiasing)
        self.mainArea.layout().addWidget(self.plot)

        self.selection_tool = PlotSelectionTool(parent=self)
        self.zoom_tool = PlotZoomTool(parent=self)
        self.pan_tool = PlotPanTool(parent=self)
        self.pinch_tool = PlotPinchZoomTool(parent=self)
        self.pinch_tool.setViewBox(self.plot.getViewBox())
        self.selection_tool.setViewBox(self.plot.getViewBox())
        self.selection_tool.selectionFinished.connect(self.__selection_end)
        self.current_tool = self.selection_tool

        def activate_tool(action):
            self.current_tool.setViewBox(None)

            if action is action_select:
                active, cur = self.selection_tool, Qt.ArrowCursor
            elif action is action_zoom:
                active, cur = self.zoom_tool, Qt.ArrowCursor
            elif action is action_pan:
                active, cur = self.pan_tool, Qt.OpenHandCursor
            self.current_tool = active
            self.current_tool.setViewBox(self.plot.getViewBox())
            self.plot.getViewBox().setCursor(QtGui.QCursor(cur))

        group.triggered[QtGui.QAction].connect(activate_tool)
        self.graphButton.clicked.connect(self.save_graph)

        self._initialize()

    def set_data(self, data):
        self.signal_data = data

        if self.matrix is not None and data is not None and len(self.matrix) == len(data):
            self.closeContext()
            self.data = data
            self.update_controls()
            self.openContext(data)
        else:
            self._invalidated = True
        self._selection_mask = None

    def set_disimilarity(self, matrix):
        self.matrix = matrix
        if matrix is not None and matrix.row_items:
            self.matrix_data = matrix.row_items
        if matrix is None:
            self.matrix_data = None
        self._invalidated = True
        self._selection_mask = None

    def _clear(self):
        self._pen_data = None
        self._shape_data = None
        self._size_data = None
        self._label_data = None
        self._similar_pairs = None

        self.colorvar_model[:] = ["Same color"]
        self.shapevar_model[:] = ["Same shape"]
        self.sizevar_model[:] = ["Same size"]
        self.labelvar_model[:] = ["No labels"]

        self.color_value = self.colorvar_model[0]
        self.shape_value = self.shapevar_model[0]
        self.size_value = self.sizevar_model[0]
        self.label_value = self.labelvar_model[0]

        self.__set_update_loop(None)
        self.__state = OWMDS.Waiting

    def _clear_plot(self):
        self.plot.clear()
        self._scatter_item = None
        if self._legend_item is not None:
            anchor = legend_anchor_pos(self._legend_item)
            if anchor is not None:
                self.legend_anchor = anchor
            if self._legend_item.scene() is not None:
                self._legend_item.scene().removeItem(self._legend_item)
            self._legend_item = None

    def update_controls(self):
        if getattr(self.matrix, 'axis', 1) == 0:
            # Column-wise distances
            attr = "Attribute names"
            self.labelvar_model[:] = ["No labels", attr]
            self.shapevar_model[:] = ["Same shape", attr]
            self.colorvar_model[:] = ["Same color", attr]

            self.color_value = attr
            self.shape_value = attr
        else:
            # initialize the graph state from data
            domain = self.data.domain
            all_vars = list(domain.variables + domain.metas)
            cd_vars = [var for var in all_vars if var.is_primitive()]
            disc_vars = [var for var in all_vars if var.is_discrete]
            cont_vars = [var for var in all_vars if var.is_continuous]
            shape_vars = [var for var in disc_vars
                          if len(var.values) <= len(ScatterPlotItem.Symbols) - 1]
            self.colorvar_model[:] = chain(["Same color"],
                                           [self.colorvar_model.Separator],
                                           cd_vars)
            self.shapevar_model[:] = chain(["Same shape"],
                                           [self.shapevar_model.Separator],
                                           shape_vars)
            self.sizevar_model[:] = chain(["Same size", "Stress"],
                                          [self.sizevar_model.Separator],
                                          cont_vars)
            self.labelvar_model[:] = chain(["No labels"],
                                           [self.labelvar_model.Separator],
                                           all_vars)

            if domain.class_var is not None:
                self.color_value = domain.class_var.name

    def _initialize(self):
        # clear everything
        self.closeContext()
        self._clear()
        self.data = None
        self._effective_matrix = None
        self.embedding = None

        # if no data nor matrix is present reset plot
        if self.signal_data is None and self.matrix is None:
            return

        if self.signal_data and self.matrix_data and len(self.signal_data) != len(self.matrix_data):
            self.error(1, "Data and distances dimensions do not match.")
            self._update_plot()
            return

        self.error(1)

        if self.signal_data:
            self.data = self.signal_data
        elif self.matrix_data:
            self.data = self.matrix_data

        if self.matrix is not None:
            self._effective_matrix = self.matrix
            if self.matrix.axis == 0:
                self.data = None
        else:
            preprocessed_data = Orange.projection.MDS().preprocess(self.data)
            self._effective_matrix = Orange.distance.Euclidean(preprocessed_data)

        self.update_controls()
        self.openContext(self.data)

    def _toggle_run(self):
        if self.__state == OWMDS.Running:
            self.stop()
            self._invalidate_output()
        else:
            self.start()

    def start(self):
        if self.__state == OWMDS.Running:
            return
        elif self.__state == OWMDS.Finished:
            # Resume/continue from a previous run
            self.__start()
        elif self.__state == OWMDS.Waiting and \
                self._effective_matrix is not None:
            self.__start()

    def stop(self):
        if self.__state == OWMDS.Running:
            self.__set_update_loop(None)

    def __start(self):
        self.__draw_similar_pairs = False
        X = self._effective_matrix
        if self.spread_equal_points:
            maxval = numpy.max(X)
            X = numpy.clip(X, maxval / 10, maxval)

        if self.embedding is not None:
            init = self.embedding
        elif self.initialization == OWMDS.PCA:
            init = torgerson(X, n_components=2)
        else:
            init = None

        # number of iterations per single GUI update step
        _, step_size = OWMDS.RefreshRate[self.refresh_rate]
        if step_size == -1:
            step_size = self.max_iter

        def update_loop(X, max_iter, step, init):
            """
            return an iterator over successive improved MDS point embeddings.
            """
            # NOTE: this code MUST NOT call into QApplication.processEvents
            done = False
            iterations_done = 0
            oldstress = numpy.finfo(numpy.float).max

            while not done:
                step_iter = min(max_iter - iterations_done, step)
                mds = Orange.projection.MDS(
                    dissimilarity="precomputed", n_components=2,
                    n_init=1, max_iter=step_iter)

                mdsfit = mds.fit(X, init=init)
                iterations_done += step_iter

                embedding, stress = mdsfit.embedding_, mdsfit.stress_
                stress /= numpy.sqrt(numpy.sum(embedding ** 2, axis=1)).sum()

                if iterations_done >= max_iter:
                    done = True
                elif (oldstress - stress) < mds.params["eps"]:
                    done = True
                init = embedding
                oldstress = stress

                yield embedding, mdsfit.stress_, iterations_done / max_iter

        self.__set_update_loop(update_loop(X, self.max_iter, step_size, init))
        self.progressBarInit(processEvents=None)

    def __set_update_loop(self, loop):
        """
        Set the update `loop` coroutine.

        The `loop` is a generator yielding `(embedding, stress, progress)`
        tuples where `embedding` is a `(N, 2) ndarray` of current updated
        MDS points, `stress` is the current stress and `progress` a float
        ratio (0 <= progress <= 1)

        If an existing update loop is already in palace it is interrupted
        (closed).

        .. note::
            The `loop` must not explicitly yield control flow to the event
            loop (i.e. call `QApplication.processEvents`)

        """
        if self.__update_loop is not None:
            self.__update_loop.close()
            self.__update_loop = None
            self.progressBarFinished(processEvents=None)

        self.__update_loop = loop

        if loop is not None:
            self.progressBarInit(processEvents=None)
            self.setStatusMessage("Running")
            self.runbutton.setText("Stop")
            self.__state = OWMDS.Running
            QtGui.QApplication.postEvent(self, QEvent(QEvent.User))
        else:
            self.setStatusMessage("")
            self.runbutton.setText("Start")
            self.__state = OWMDS.Finished

    def __next_step(self):
        if self.__update_loop is None:
            return

        loop = self.__update_loop
        try:
            embedding, stress, progress = next(self.__update_loop)
            assert self.__update_loop is loop
        except StopIteration:
            self.__set_update_loop(None)
            self.unconditional_commit()
            self.__draw_similar_pairs = True
            self._update_plot()
            self.plot.autoRange(padding=0.1, items=[self._scatter_item])
        else:
            self.progressBarSet(100.0 * progress, processEvents=None)
            self.embedding = embedding
            self._update_plot()
            self.plot.autoRange(padding=0.1, items=[self._scatter_item])
            # schedule next update
            QtGui.QApplication.postEvent(
                self, QEvent(QEvent.User), Qt.LowEventPriority)

    def customEvent(self, event):
        if event.type() == QEvent.User and self.__update_loop is not None:
            if not self.__in_next_step:
                self.__in_next_step = True
                try:
                    self.__next_step()
                finally:
                    self.__in_next_step = False
            else:
                warnings.warn(
                    "Re-entry in update loop detected. "
                    "A rogue `proccessEvents` is on the loose.",
                    RuntimeWarning)
                # re-schedule the update iteration.
                QtGui.QApplication.postEvent(self, QEvent(QEvent.User))
        return super().customEvent(event)

    def __invalidate_embedding(self):
        # reset/invalidate the MDS embedding, to the default initialization
        # (Random or PCA), restarting the optimization if necessary.
        if self.embedding is None:
            return
        state = self.__state
        if self.__update_loop is not None:
            self.__set_update_loop(None)

        X = self._effective_matrix

        if self.initialization == OWMDS.PCA:
            self.embedding = torgerson(X)
        else:
            self.embedding = numpy.random.rand(len(X), 2)

        self._update_plot()
        self.plot.autoRange(padding=0.1, items=[self._scatter_item])

        # restart the optimization if it was interrupted.
        if state == OWMDS.Running:
            self.__start()

    def __invalidate_refresh(self):
        state = self.__state

        if self.__update_loop is not None:
            self.__set_update_loop(None)

        # restart the optimization if it was interrupted.
        # TODO: decrease the max iteration count by the already
        # completed iterations count.
        if state == OWMDS.Running:
            self.__start()

    def handleNewSignals(self):
        if self._invalidated:
            self._invalidated = False
            self._initialize()
            self.start()
        self.__draw_similar_pairs = False
        self._update_plot()
        self.plot.autoRange(padding=0.1)
        self.unconditional_commit()

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

    def _on_connected_changed(self):
        self._similar_pairs = None
        self._update_plot()

    def _update_plot(self):
        self._clear_plot()

        if self.embedding is not None:
            self._setup_plot()

    def _setup_plot(self):
        have_data = self.data is not None
        have_matrix_transposed = self.matrix is not None and not self.matrix.axis
        plotstyle = mdsplotutils.plotstyle

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
            if self._selection_mask is not None:
                pointflags = numpy.where(
                    self._selection_mask,
                    mdsplotutils.Selected, mdsplotutils.NoFlags)
            else:
                pointflags = None

            color_index = self.cb_color_value.currentIndex()
            if have_data and color_index > 0:
                color_var = self.colorvar_model[color_index]
                if color_var.is_discrete:
                    palette = colorpalette.ColorPaletteGenerator(
                        len(color_var.values)
                    )
                    plotstyle = plotstyle.updated(discrete_palette=palette)
                else:
                    palette = None

                color_data = mdsplotutils.color_data(
                    self.data, color_var, plotstyle=plotstyle)
                color_data = numpy.hstack(
                    (color_data,
                     numpy.full((len(color_data), 1), self.symbol_opacity))
                )
                pen_data = mdsplotutils.pen_data(color_data * 0.8, pointflags)
                brush_data = mdsplotutils.brush_data(color_data)
            elif have_matrix_transposed and \
                    self.colorvar_model[color_index] == 'Attribute names':
                attr = attributes(self.matrix)
                palette = colorpalette.ColorPaletteGenerator(len(attr))
                color_data = [palette.getRGB(i) for i in range(len(attr))]
                color_data = numpy.hstack((
                    color_data,
                    numpy.full((len(color_data), 1), self.symbol_opacity))
                )
                pen_data = mdsplotutils.pen_data(color_data * 0.8, pointflags)
                brush_data = mdsplotutils.brush_data(color_data)
            else:
                pen_data = make_pen(QtGui.QColor(Qt.darkGray), cosmetic=True)
                if self._selection_mask is not None:
                    pen_data = numpy.array(
                        [pen_data, plotstyle.selected_pen])
                    pen_data = pen_data[self._selection_mask.astype(int)]
                else:
                    pen_data = numpy.full(len(self.data), pen_data,
                                          dtype=object)
                brush_data = numpy.full(
                    len(self.data),
                    pg.mkColor((192, 192, 192, self.symbol_opacity)),
                    dtype=object)

            self._pen_data = pen_data
            self._brush_data = brush_data

        if self._shape_data is None:
            shape_index = self.cb_shape_value.currentIndex()
            if have_data and shape_index > 0:
                Symbols = ScatterPlotItem.Symbols
                symbols = numpy.array(list(Symbols.keys()))

                shape_var = self.shapevar_model[shape_index]
                data = column(self.data, shape_var)
                data = data % (len(Symbols) - 1)
                data[numpy.isnan(data)] = len(Symbols) - 1
                shape_data = symbols[data.astype(int)]
            elif have_matrix_transposed and \
                    self.shapevar_model[shape_index] == 'Attribute names':
                Symbols = ScatterPlotItem.Symbols
                symbols = numpy.array(list(Symbols.keys()))
                attr = [i % (len(Symbols) - 1)
                        for i, _ in enumerate(attributes(self.matrix))]
                shape_data = symbols[attr]
            else:
                shape_data = "o"
            self._shape_data = shape_data

        if self._size_data is None:
            MinPointSize = 3
            point_size = self.symbol_size + MinPointSize
            size_index = self.cb_size_value.currentIndex()
            if have_data and size_index == 1:
                # size by stress
                size_data = stress(self.embedding, self._effective_matrix)
                size_data = scale(size_data)
                size_data = MinPointSize + size_data * point_size
            elif have_data and size_index > 0:
                size_var = self.sizevar_model[size_index]
                size_data = column(self.data, size_var)
                size_data = scale(size_data)
                size_data = MinPointSize + size_data * point_size
            else:
                size_data = point_size
            self._size_data = size_data

        if self._label_data is None:
            label_index = self.cb_label_value.currentIndex()
            if have_data and label_index > 0:
                label_var = self.labelvar_model[label_index]
                label_data = column(self.data, label_var)
                label_data = [label_var.str_val(val) for val in label_data]
                label_items = [pg.TextItem(text, anchor=(0.5, 0), color=0.0)
                               for text in label_data]
            elif have_matrix_transposed and \
                    self.labelvar_model[label_index] == 'Attribute names':
                attr = attributes(self.matrix)
                label_items = [pg.TextItem(str(text), anchor=(0.5, 0))
                               for text in attr]
            else:
                label_items = None
            self._label_data = label_items

        emb_x, emb_y = self.embedding[:, 0], self.embedding[:, 1]

        if self.connected_pairs and self.__draw_similar_pairs:
            if self._similar_pairs is None:
                # This code requires storing lower triangle of X (n x n / 2
                # doubles), n x n / 2 * 2 indices to X, n x n / 2 indices for
                # argsort result. If this becomes an issue, it can be reduced to
                # n x n argsort indices by argsorting the entire X. Then we
                # take the first n + 2 * p indices. We compute their coordinates
                # i, j in the original matrix. We keep those for which i < j.
                # n + 2 * p will suffice to exclude the diagonal (i = j). If the
                # number of those for which i < j is smaller than p, we instead
                # take i > j. Among those that remain, we take the first p.
                # Assuming that MDS can't show so many points that memory could
                # become an issue, I preferred using simpler code.
                m = self._effective_matrix
                n = len(m)
                p = (n * (n - 1) // 2 * self.connected_pairs) // 100
                indcs = numpy.triu_indices(n, 1)
                sorted = numpy.argsort(m[indcs])[:p]
                self._similar_pairs = fpairs = numpy.empty(2 * p, dtype=int)
                fpairs[::2] = indcs[0][sorted]
                fpairs[1::2] = indcs[1][sorted]
            for i in range(int(len(emb_x[self._similar_pairs]) / 2)):
                item = QtGui.QGraphicsLineItem(
                    emb_x[self._similar_pairs][i * 2],
                    emb_y[self._similar_pairs][i * 2],
                    emb_x[self._similar_pairs][i * 2 + 1],
                    emb_y[self._similar_pairs][i * 2 + 1]
                )
                pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(204, 204, 204)), 2)
                pen.setCosmetic(True)
                item.setPen(pen)
                self.plot.addItem(item)

        data = numpy.arange(len(self.data if have_data else self.matrix))
        self._scatter_item = item = ScatterPlotItem(
            x=emb_x, y=emb_y,
            pen=self._pen_data, brush=self._brush_data, symbol=self._shape_data,
            size=self._size_data, data=data,
            antialias=True
        )
        self.plot.addItem(item)

        if self._label_data is not None:
            for (x, y), text_item in zip(self.embedding, self._label_data):
                self.plot.addItem(text_item)
                text_item.setPos(x, y)

        self._legend_item = LegendItem()
        self._legend_item.setParentItem(self.plot.getViewBox())
        self._legend_item.anchor(*self.legend_anchor)

        color_var = shape_var = None
        color_index = self.cb_color_value.currentIndex()
        if have_data and 1 <= color_index < len(self.colorvar_model):
            color_var = self.colorvar_model[color_index]
            assert isinstance(color_var, Orange.data.Variable)
        shape_index = self.cb_shape_value.currentIndex()
        if have_data and 1 <= shape_index < len(self.shapevar_model):
            shape_var = self.shapevar_model[shape_index]
            assert isinstance(shape_var, Orange.data.Variable)

        if shape_var is not None or \
                (color_var is not None and color_var.is_discrete):

            legend_data = mdsplotutils.legend_data(
                color_var, shape_var, plotstyle=plotstyle)

            for color, symbol, text in legend_data:
                self._legend_item.addItem(
                    ScatterPlotItem(pen=color, brush=color, symbol=symbol,
                                    size=10),
                    escape(text)
                )
        else:
            self._legend_item.hide()

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
            domain = self.data.domain
            attrs = domain.attributes
            class_vars = domain.class_vars
            metas = domain.metas

            if self.output_embedding_role == OWMDS.AttrRole:
                attrs = embedding.domain.attributes
            elif self.output_embedding_role == OWMDS.AddAttrRole:
                attrs = domain.attributes + embedding.domain.attributes
            elif self.output_embedding_role == OWMDS.MetaRole:
                metas += embedding.domain.attributes

            domain = Orange.data.Domain(attrs, class_vars, metas)
            output = Orange.data.Table.from_table(domain, self.data)

            if self.output_embedding_role == OWMDS.AttrRole:
                output.X[:] = embedding.X
            if self.output_embedding_role == OWMDS.AddAttrRole:
                output.X[:, -2:] = embedding.X
            elif self.output_embedding_role == OWMDS.MetaRole:
                output.metas[:, -2:] = embedding.X

        self.send("Data", output)
        if output is not None and self._selection_mask is not None and \
                numpy.any(self._selection_mask):
            subset = output[self._selection_mask]
        else:
            subset = None
        self.send("Selected Data", subset)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._clear_plot()
        self._clear()

    def __selection_end(self, path):
        self.select(path)
        self._pen_data = None
        self._update_plot()
        self._invalidate_output()

    def select(self, region):
        item = self._scatter_item
        if item is None:
            return

        indices = numpy.array(
            [spot.data() for spot in item.points()
             if region.contains(spot.pos())],
            dtype=int)

        if not QtGui.QApplication.keyboardModifiers():
            self._selection_mask = None

        self.select_indices(indices, QtGui.QApplication.keyboardModifiers())

    def select_indices(self, indices, modifiers=Qt.NoModifier):
        if self.data is None:
            return

        if self._selection_mask is None or \
                not modifiers & (Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier):
            self._selection_mask = numpy.zeros(len(self.data), dtype=bool)

        if modifiers & Qt.AltModifier:
            self._selection_mask[indices] = False
        elif modifiers & Qt.ControlModifier:
            self._selection_mask[indices] = ~self._selection_mask[indices]
        else:
            self._selection_mask[indices] = True

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.plot.plotItem,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


def colors(data, variable, palette=None):
    if palette is None:
        if variable.is_discrete:
            palette = colorpalette.ColorPaletteGenerator(len(variable.values))
        elif variable.is_continuous:
            palette = colorpalette.ColorPaletteBW()
            palette = colorpalette.ContinuousPaletteGenerator(
                QtGui.QColor(220, 220, 220),
                QtGui.QColor(0, 0, 0),
                False
            )
        else:
            raise TypeError()

    x = data[:, variable]
    if variable in data.domain.metas:
        x = numpy.array(x.metas, dtype='float').ravel()
    else:
        x = numpy.array(x).ravel()

    if variable.is_discrete:
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

from types import SimpleNamespace as namespace

from Orange.widgets.visualize.owlinearprojection import \
    PlotSelectionTool, PlotZoomTool, PlotPanTool, PlotPinchZoomTool, \
    LegendItem, legend_anchor_pos
from Orange.widgets.visualize.owlinearprojection import plotutils


class namespace(namespace):
    def updated(self, **kwargs):
        ns = self.__dict__.copy()
        ns.update(**kwargs)
        return namespace(**ns)


class mdsplotutils(plotutils):
    NoFlags, Selected, Highlight = 0, 1, 2
    NoFill, Filled = 0, 1

    plotstyle = namespace(
        selected_pen=make_pen(Qt.yellow, width=3, cosmetic=True),
        highligh_pen=QtGui.QPen(Qt.blue, 1),
        selected_brush=None,
        default_color=QtGui.QColor(Qt.darkGray).rgba(),
        discrete_palette=colorpalette.ColorPaletteGenerator(),
        continuous_palette=colorpalette.ContinuousPaletteGenerator(
            QtGui.QColor(220, 220, 220),
            QtGui.QColor(0, 0, 0),
            False
        ),
        symbols=ScatterPlotItem.Symbols,
        point_size=10,
        min_point_size=5,
    )

    @staticmethod
    def column_data(table, var, mask=None):
        col, _ = table.get_column_view(var)
        dtype = float if var.is_primitive() else object
        col = numpy.asarray(col, dtype=dtype)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=bool)
            return col[mask]
        else:
            return col

    @staticmethod
    def color_data(table, var=None, mask=None, plotstyle=None):
        N = len(table)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=bool)
            N = numpy.count_nonzero(mask)

        if plotstyle is None:
            plotstyle = mdsplotutils.plotstyle

        if var is None:
            col = numpy.zeros(N, dtype=float)
            color_data = numpy.full(N, plotstyle.default_color, dtype=object)
        elif var.is_primitive():
            col = mdsplotutils.column_data(table, var, mask)
            if var.is_discrete:
                palette = plotstyle.discrete_palette
                if len(var.values) >= palette.number_of_colors:
                    palette = colorpalette.ColorPaletteGenerator(len(var.values))

                color_data = plotutils.discrete_colors(
                    col, nvalues=len(var.values), palette=palette)
            elif var.is_continuous:
                color_data = plotutils.continuous_colors(
                    col, palette=plotstyle.continuous_palette)
        else:
            raise TypeError("Discrete/Continuous variable or None expected.")

        return color_data

    @staticmethod
    def pen_data(basecolors, flags=None, plotstyle=None):
        if plotstyle is None:
            plotstyle = mdsplotutils.plotstyle

        pens = numpy.array(
            [mdsplotutils.make_pen(QtGui.QColor(*rgba), width=1)
             for rgba in basecolors],
            dtype=object)

        if flags is None:
            return pens

        selected_mask = flags & mdsplotutils.Selected
        if numpy.any(selected_mask):
            pens[selected_mask.astype(bool)] = plotstyle.selected_pen

        highlight_mask = flags & mdsplotutils.Highlight
        if numpy.any(highlight_mask):
            pens[highlight_mask.astype(bool)] = plotstyle.hightlight_pen

        return pens

    @staticmethod
    def brush_data(basecolors, flags=None, plotstyle=None):
        if plotstyle is None:
            plotstyle = mdsplotutils.plotstyle

        brush = numpy.array(
            [mdsplotutils.make_brush(QtGui.QColor(*c))
             for c in basecolors],
            dtype=object)

        if flags is None:
            return brush

        fill_mask = flags & mdsplotutils.Filled

        if not numpy.all(fill_mask):
            brush[~fill_mask] = QtGui.QBrush(Qt.NoBrush)
        return brush

    @staticmethod
    def shape_data(table, var, mask=None, plotstyle=None):
        if plotstyle is None:
            plotstyle = mdsplotutils.plotstyle

        N = len(table)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=bool)
            N = numpy.nonzero(mask)

        if var is None:
            return numpy.full(N, "o", dtype=object)
        elif var.is_discrete:
            shape_data = mdsplotutils.column_data(table, var, mask)
            maxsymbols = len(plotstyle.symbols) - 1
            validmask = numpy.isfinite(shape_data)
            shape = shape_data % (maxsymbols - 1)
            shape[~validmask] = maxsymbols  # Special symbol for unknown values
            symbols = numpy.array(list(plotstyle.symbols))
            shape_data = symbols[numpy.asarray(shape, dtype=int)]

            if mask is None:
                return shape_data
            else:
                return shape_data[mask]
        else:
            raise TypeError()

    @staticmethod
    def size_data(table, var, mask=None, plotstyle=None):
        if plotstyle is None:
            plotstyle = mdsplotutils.plotstyle

        N = len(table)
        if mask is not None:
            mask = numpy.asarray(mask, dtype=bool)
            N = numpy.nonzero(mask)

        if var is None:
            return numpy.full(N, plotstyle.point_size, dtype=float)
        else:
            size_data = mdsplotutils.column_data(table, var, mask)
            size_data = mdsplotutils.normalized(size_data)
            size_mask = numpy.isnan(size_data)
            size_data = size_data * plotstyle.point_size + \
                        plotstyle.min_point_size
            size_data[size_mask] = plotstyle.min_point_size - 2

            if mask is None:
                return size_data
            else:
                return size_data[mask]

    @staticmethod
    def legend_data(color_var=None, shape_var=None, plotstyle=None):
        if plotstyle is None:
            plotstyle = mdsplotutils.plotstyle

        if color_var is not None and not color_var.is_discrete:
            color_var = None
        assert shape_var is None or shape_var.is_discrete
        if color_var is None and shape_var is None:
            return []

        if color_var is not None:
            palette = plotstyle.discrete_palette
            if len(color_var.values) >= palette.number_of_colors:
                palette = colorpalette.ColorPaletteGenerator(len(color_var.values))
        else:
            palette = None

        symbols = list(plotstyle.symbols)

        if shape_var is color_var:
            items = [(palette[i], symbols[i], name)
                     for i, name in enumerate(color_var.values)]
        else:
            colors = shapes = []
            if color_var is not None:
                colors = [(palette[i], "o", name)
                          for i, name in enumerate(color_var.values)]
            if shape_var is not None:
                shapes = [(QtGui.QColor(Qt.gray),
                           symbols[i % (len(symbols) - 1)], name)
                          for i, name in enumerate(shape_var.values)]
            items = colors + shapes

        return items

    @staticmethod
    def make_pen(color, width=1, cosmetic=True):
        pen = QtGui.QPen(color)
        pen.setWidthF(width)
        pen.setCosmetic(cosmetic)
        return pen

    @staticmethod
    def make_brush(color, ):
        return QtGui.QBrush(color, )


def main_test(argv=sys.argv):
    import gc
    argv = list(argv)
    app = QtGui.QApplication(argv)

    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Orange.data.Table(filename)
    w = OWMDS()
    w.set_data(data)
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
    sys.exit(main_test())

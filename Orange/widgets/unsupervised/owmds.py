import sys
import warnings

from xml.sax.saxutils import escape
from itertools import chain

import pkg_resources

import numpy
import scipy.spatial.distance

from AnyQt.QtWidgets import (
    QFormLayout, QHBoxLayout, QGroupBox, QToolButton, QActionGroup, QAction, QApplication,
    QGraphicsLineItem
)
from AnyQt.QtGui import QColor, QPen, QBrush, QPainter, QKeySequence, QCursor, QIcon
from AnyQt.QtCore import Qt, QEvent

import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem

import Orange.data
import Orange.projection
from Orange.projection.manifold import torgerson
import Orange.distance
from Orange.data.domain import filter_visible
import Orange.misc
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette, itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.canvas import report
from Orange.widgets.widget import Msg, OWWidget
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)


def stress(X, D):
    assert X.shape[0] == D.shape[0] == D.shape[1]
    D1_c = scipy.spatial.distance.pdist(X, metric="euclidean")
    D1 = scipy.spatial.distance.squareform(D1_c, checks=False)
    delta = D1 - D
    delta_sq = numpy.square(delta, out=delta)
    return delta_sq.sum(axis=0) / 2


def make_pen(color, width=1.5, style=Qt.SolidLine, cosmetic=False):
    pen = QPen(color, width, style)
    pen.setCosmetic(cosmetic)
    return pen


class ScatterPlotItem(pg.ScatterPlotItem):
    Symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, option, widget=None):
        if self.opts["pxMode"]:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self.opts["antialias"]:
            painter.setRenderHint(QPainter.Antialiasing, True)

        super().paint(painter, option, widget)


class OWMDS(OWWidget):
    name = "MDS"
    description = "Two-dimensional data projection by multidimensional " \
                  "scaling constructed from a distance matrix."
    icon = "icons/MDS.svg"
    inputs = [("Data", Orange.data.Table, "set_data", widget.Default),
              ("Distances", Orange.misc.DistMatrix, "set_disimilarity"),
              ("Data Subset", Orange.data.Table, "set_subset_data")]

    outputs = [("Selected Data", Orange.data.Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Orange.data.Table)]

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

    JitterAmount = [
        ("None", 0),
        ("0.1 %", 0.1),
        ("0.5 %", 0.5),
        ("1 %", 1.0),
        ("2 %", 2.0)
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
    label_only_selected = settings.Setting(False)

    symbol_size = settings.Setting(8)
    symbol_opacity = settings.Setting(230)
    connected_pairs = settings.Setting(5)
    jitter = settings.Setting(0)

    legend_anchor = settings.Setting(((1, 0), (1, 0)))

    graph_name = "plot.plotItem"

    class Error(OWWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        matrix_too_small = Msg("Input matrix must be at least 2x2")
        no_attributes = Msg("Data has no attributes")
        mismatching_dimensions = \
            Msg("Data and distances dimensions do not match.")

    def __init__(self):
        super().__init__()
        #: Input dissimilarity matrix
        self.matrix = None  # type: Optional[Orange.misc.DistMatrix]
        #: Effective data used for plot styling/annotations. Can be from the
        #: input signal (`self.signal_data`) or the input matrix
        #: (`self.matrix.data`)
        self.data = None  # type: Optional[Orange.data.Table]
        #: Input subset data table
        self.subset_data = None  # type: Optional[Orange.data.Table]
        #: Data table from the `self.matrix.row_items` (if present)
        self.matrix_data = None  # type: Optional[Orange.data.Table]
        #: Input data table
        self.signal_data = None

        self._pen_data = None
        self._brush_data = None
        self._shape_data = None
        self._size_data = None
        self._label_data = None
        self._similar_pairs = None
        self._scatter_item = None
        self._legend_item = None
        self._selection_mask = None
        self._subset_mask = None  # type: Optional[numpy.ndarray]
        self._invalidated = False
        self._effective_matrix = None

        self.__update_loop = None
        self.__state = OWMDS.Waiting
        self.__in_next_step = False
        self.__draw_similar_pairs = False

        box = gui.vBox(self.controlArea, "MDS Optimization")
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )

        form.addRow("Max iterations:",
                    gui.spin(box, self, "max_iter", 10, 10 ** 4, step=1))

        form.addRow("Initialization:",
                    gui.radioButtons(box, self, "initialization",
                                     btnLabels=("PCA (Torgerson)", "Random"),
                                     callback=self.__invalidate_embedding))

        box.layout().addLayout(form)
        form.addRow("Refresh:",
                    gui.comboBox(
                        box, self, "refresh_rate",
                        items=[t for t, _ in OWMDS.RefreshRate],
                        callback=self.__invalidate_refresh))
        gui.separator(box, 10)
        self.runbutton = gui.button(
            box, self, "Run", callback=self._toggle_run)

        box = gui.vBox(self.controlArea, "Graph")
        self.colorvar_model = itemmodels.VariableListModel()

        common_options = dict(
            sendSelectedValue=True, valueType=str, orientation=Qt.Horizontal,
            labelWidth=50, contentsLength=12)

        self.cb_color_value = gui.comboBox(
            box, self, "color_value", label="Color:",
            callback=self._on_color_index_changed, **common_options)
        self.cb_color_value.setModel(self.colorvar_model)

        self.shapevar_model = itemmodels.VariableListModel()
        self.cb_shape_value = gui.comboBox(
            box, self, "shape_value", label="Shape:",
            callback=self._on_shape_index_changed, **common_options)
        self.cb_shape_value.setModel(self.shapevar_model)

        self.sizevar_model = itemmodels.VariableListModel()
        self.cb_size_value = gui.comboBox(
            box, self, "size_value", label="Size:",
            callback=self._on_size_index_changed, **common_options)
        self.cb_size_value.setModel(self.sizevar_model)

        self.labelvar_model = itemmodels.VariableListModel()
        self.cb_label_value = gui.comboBox(
            box, self, "label_value", label="Label:",
            callback=self._on_label_index_changed, **common_options)
        self.cb_label_value.setModel(self.labelvar_model)
        gui.checkBox(
            gui.indentedBox(box), self, 'label_only_selected',
            'Label only selected points', callback=self._on_label_index_changed)

        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )
        form.addRow("Symbol size:",
                    gui.hSlider(box, self, "symbol_size",
                                minValue=1, maxValue=20,
                                callback=self._on_size_index_changed,
                                createLabel=False))
        form.addRow("Symbol opacity:",
                    gui.hSlider(box, self, "symbol_opacity",
                                minValue=100, maxValue=255, step=100,
                                callback=self._on_color_index_changed,
                                createLabel=False))
        form.addRow("Show similar pairs:",
                    gui.hSlider(
                        gui.hBox(self.controlArea),
                        self, "connected_pairs", minValue=0, maxValue=20,
                        createLabel=False,
                        callback=self._on_connected_changed))
        form.addRow("Jitter:",
                    gui.comboBox(
                        box, self, "jitter",
                        items=[text for text, _ in self.JitterAmount],
                        callback=self._update_plot))

        box.layout().addLayout(form)

        gui.rubber(self.controlArea)

        box = QGroupBox("Zoom/Select", )
        box.setLayout(QHBoxLayout())
        box.layout().setContentsMargins(2, 2, 2, 2)

        group = QActionGroup(self, exclusive=True)

        def icon(name):
            path = "icons/Dlg_{}.png".format(name)
            path = pkg_resources.resource_filename(widget.__name__, path)
            return QIcon(path)

        action_select = QAction(
            "Select", self, checkable=True, checked=True, icon=icon("arrow"),
            shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_1))
        action_zoom = QAction(
            "Zoom", self, checkable=True, checked=False, icon=icon("zoom"),
            shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_2))
        action_pan = QAction(
            "Pan", self, checkable=True, checked=False, icon=icon("pan_hand"),
            shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_3))

        action_reset_zoom = QAction(
            "Zoom to fit", self, icon=icon("zoom_reset"),
            shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_0))
        action_reset_zoom.triggered.connect(
            lambda: self.plot.autoRange(padding=0.1,
                                        items=[self._scatter_item]))
        group.addAction(action_select)
        group.addAction(action_zoom)
        group.addAction(action_pan)
        self.addActions(group.actions() + [action_reset_zoom])
        action_select.setChecked(True)

        def button(action):
            b = QToolButton()
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

        box = gui.vBox(self.controlArea, "Output")
        self.output_combo = gui.comboBox(
            box, self, "output_embedding_role",
            items=["Original features only",
                   "Coordinates only",
                   "Coordinates as features",
                   "Coordinates as meta attributes"],
            callback=self._invalidate_output, addSpace=4)
        gui.auto_commit(box, self, "autocommit", "Send Selected",
                        checkbox_label="Send selected automatically",
                        box=None)

        self.plot = pg.PlotWidget(background="w", enableMenu=False)
        self.plot.setAspectLocked(True)
        self.plot.getPlotItem().hideAxis("bottom")
        self.plot.getPlotItem().hideAxis("left")
        self.plot.getPlotItem().hideButtons()
        self.plot.setRenderHint(QPainter.Antialiasing)
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
            self.plot.getViewBox().setCursor(QCursor(cur))

        group.triggered[QAction].connect(activate_tool)

        self._initialize()

    @check_sql_input
    def set_data(self, data):
        """Set the input data set.

        Parameters
        ----------
        data : Optional[Orange.data.Table]
        """
        if data is not None and len(data) < 2:
            self.Error.not_enough_rows()
            data = None
        else:
            self.Error.not_enough_rows.clear()

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
        """Set the dissimilarity (distance) matrix.

        Parameters
        ----------
        matrix : Optional[Orange.misc.DistMatrix]
        """

        if matrix is not None and len(matrix) < 2:
            self.Error.matrix_too_small()
            matrix = None
        else:
            self.Error.matrix_too_small.clear()

        self.matrix = matrix
        if matrix is not None and matrix.row_items:
            self.matrix_data = matrix.row_items
        if matrix is None:
            self.matrix_data = None
        self._invalidated = True
        self._selection_mask = None

    def set_subset_data(self, subset_data):
        """Set a subset of `data` input to highlight in the plot.

        Parameters
        ----------
        subset_data: Optional[Orange.data.Table]
        """
        self.subset_data = subset_data
        # invalidate the pen/brush when the subset is changed
        self._pen_data = self._brush_data = None
        self._subset_mask = None  # type: Optional[numpy.ndarray]

    def _clear(self):
        self._pen_data = None
        self._brush_data = None
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
        if self.data is None and getattr(self.matrix, 'axis', 1) == 0:
            # Column-wise distances
            attr = "Attribute names"
            self.labelvar_model[:] = ["No labels", attr]
            self.shapevar_model[:] = ["Same shape", attr]
            self.colorvar_model[:] = ["Same solor", attr]

            self.color_value = attr
            self.shape_value = attr
        else:
            domain = self.data.domain
            all_vars = list(filter_visible(domain.variables + domain.metas))
            cd_vars = [var for var in all_vars if var.is_primitive()]
            disc_vars = [var for var in all_vars if var.is_discrete]
            cont_vars = [var for var in all_vars if var.is_continuous]
            shape_vars = [var for var in disc_vars
                          if len(var.values) <= len(ScatterPlotItem.Symbols) - 1]
            self.colorvar_model[:] = chain(["Same color"],
                                           [self.colorvar_model.Separator] if cd_vars else [],
                                           cd_vars)
            self.shapevar_model[:] = chain(["Same shape"],
                                           [self.shapevar_model.Separator] if shape_vars else [],
                                           shape_vars)
            self.sizevar_model[:] = chain(["Same size", "Stress"],
                                          [self.sizevar_model.Separator] if cont_vars else [],
                                          cont_vars)
            self.labelvar_model[:] = chain(["No labels"],
                                           [self.labelvar_model.Separator] if all_vars else [],
                                           all_vars)

            if domain.class_var is not None:
                self.color_value = domain.class_var.name

    def _initialize(self):
        # clear everything
        self.closeContext()
        self._clear()
        self.Error.clear()
        self.data = None
        self._effective_matrix = None
        self.embedding = None

        # if no data nor matrix is present reset plot
        if self.signal_data is None and self.matrix is None:
            return

        if self.signal_data is not None and self.matrix is not None and \
                len(self.signal_data) != len(self.matrix):
            self.Error.mismatching_dimensions()
            self._update_plot()
            return

        if self.signal_data is not None:
            self.data = self.signal_data
        elif self.matrix_data is not None:
            self.data = self.matrix_data

        if self.matrix is not None:
            self._effective_matrix = self.matrix
            if self.matrix.axis == 0 and self.data is self.matrix_data:
                self.data = None
        elif self.data.domain.attributes:
            preprocessed_data = Orange.projection.MDS().preprocess(self.data)
            self._effective_matrix = Orange.distance.Euclidean(preprocessed_data)
        else:
            self.Error.no_attributes()
            return

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
        init = self.embedding

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
            init_type = "PCA" if self.initialization == OWMDS.PCA else "random"

            while not done:
                step_iter = min(max_iter - iterations_done, step)
                mds = Orange.projection.MDS(
                    dissimilarity="precomputed", n_components=2,
                    n_init=1, max_iter=step_iter,
                    init_type=init_type, init_data=init)

                mdsfit = mds(X)
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
            QApplication.postEvent(self, QEvent(QEvent.User))
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
            QApplication.postEvent(
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
                QApplication.postEvent(self, QEvent(QEvent.User))
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

        if self._subset_mask is None and self.subset_data is not None and \
                self.data is not None:
            self._subset_mask = numpy.in1d(self.data.ids, self.subset_data.ids)

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

        size = self._effective_matrix.shape[0]

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

        def jitter(x, factor=1, rstate=None):
            if rstate is None:
                rstate = numpy.random.RandomState()
            elif not isinstance(rstate, numpy.random.RandomState):
                rstate = numpy.random.RandomState(rstate)
            span = numpy.nanmax(x) - numpy.nanmin(x)
            if span < numpy.finfo(x.dtype).eps * 100:
                span = 1
            a = factor * span / 100.
            return x + (rstate.random_sample(x.shape) - 0.5) * a

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
                     numpy.full((len(color_data), 1), self.symbol_opacity,
                                dtype=float))
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
                    numpy.full((len(color_data), 1), self.symbol_opacity,
                               dtype=float))
                )
                pen_data = mdsplotutils.pen_data(color_data * 0.8, pointflags)
                brush_data = mdsplotutils.brush_data(color_data)
            else:
                pen_data = make_pen(QColor(Qt.darkGray), cosmetic=True)
                if self._selection_mask is not None:
                    pen_data = numpy.array(
                        [pen_data, plotstyle.selected_pen])
                    pen_data = pen_data[self._selection_mask.astype(int)]
                else:
                    pen_data = numpy.full(self._effective_matrix.dim, pen_data,
                                          dtype=object)
                brush_data = numpy.full(
                    size, pg.mkColor((192, 192, 192, self.symbol_opacity)),
                    dtype=object)

            if self._subset_mask is not None and have_data and \
                    self._subset_mask.shape == (size, ):
                # clear brush fill for non subset data
                brush_data[~self._subset_mask] = QBrush(Qt.NoBrush)

            self._pen_data = pen_data
            self._brush_data = brush_data

        if self._shape_data is None:
            shape_index = self.cb_shape_value.currentIndex()
            if have_data and shape_index > 0:
                Symbols = ScatterPlotItem.Symbols
                symbols = numpy.array(list(Symbols.keys()))

                shape_var = self.shapevar_model[shape_index]
                data = column(self.data, shape_var).astype(numpy.float)
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
        if self.jitter > 0:
            _, jitter_factor = self.JitterAmount[self.jitter]
            emb_x = jitter(emb_x, jitter_factor, rstate=42)
            emb_y = jitter(emb_y, jitter_factor, rstate=667)

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
                item = QGraphicsLineItem(
                    emb_x[self._similar_pairs][i * 2],
                    emb_y[self._similar_pairs][i * 2],
                    emb_x[self._similar_pairs][i * 2 + 1],
                    emb_y[self._similar_pairs][i * 2 + 1]
                )
                if item.line().isNull():
                    # Null (zero length) line causes bad rendering artifacts
                    # in Qt when using the raster graphics system
                    # (see gh-issue: 1668).
                    continue
                pen = QPen(QBrush(QColor(204, 204, 204)), 2)
                pen.setCosmetic(True)
                item.setPen(pen)
                self.plot.addItem(item)

        data = numpy.arange(size)
        self._scatter_item = item = ScatterPlotItem(
            x=emb_x, y=emb_y,
            pen=self._pen_data, brush=self._brush_data, symbol=self._shape_data,
            size=self._size_data, data=data,
            antialias=True
        )
        self.plot.addItem(item)

        if self._label_data is not None:
            if self.label_only_selected:
                if self._selection_mask is not None:
                    for (x, y), text_item, selected \
                            in zip(self.embedding, self._label_data,
                                   self._selection_mask):
                        if selected:
                            self.plot.addItem(text_item)
                            text_item.setPos(x, y)
            else:
                for (x, y), text_item in zip(self.embedding, self._label_data):
                    self.plot.addItem(text_item)
                    text_item.setPos(x, y)

        self._legend_item = LegendItem()
        viewbox = self.plot.getViewBox()
        self._legend_item.setParentItem(self.plot.getViewBox())
        self._legend_item.setZValue(viewbox.zValue() + 10)
        self._legend_item.restoreAnchor(self.legend_anchor)

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

        if output is not None and self._selection_mask is not None and \
                numpy.any(self._selection_mask):
            subset = output[self._selection_mask]
        else:
            subset = None
        self.send("Selected Data", subset)
        self.send(ANNOTATED_DATA_SIGNAL_NAME,
                  create_annotated_table(output, self._selection_mask))

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

        if not QApplication.keyboardModifiers():
            self._selection_mask = None

        self.select_indices(indices, QApplication.keyboardModifiers())

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

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()
        caption = report.render_items_vert((
            ("Color", self.color_value != "Same color" and self.color_value),
            ("Shape", self.shape_value != "Same shape" and self.shape_value),
            ("Size", self.size_value != "Same size" and self.size_value),
            ("Labels", self.label_value != "No labels" and self.label_value)))
        if caption:
            self.report_caption(caption)
        self.report_items((("Output", self.output_combo.currentText()),))



def colors(data, variable, palette=None):
    if palette is None:
        if variable.is_discrete:
            palette = colorpalette.ColorPaletteGenerator(len(variable.values))
        elif variable.is_continuous:
            palette = colorpalette.ColorPaletteBW()
            palette = colorpalette.ContinuousPaletteGenerator(
                QColor(220, 220, 220),
                QColor(0, 0, 0),
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
        highligh_pen=QPen(Qt.blue, 1),
        selected_brush=None,
        default_color=QColor(Qt.darkGray).rgba(),
        discrete_palette=colorpalette.ColorPaletteGenerator(),
        continuous_palette=colorpalette.ContinuousPaletteGenerator(
            QColor(220, 220, 220),
            QColor(0, 0, 0),
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
            [mdsplotutils.make_pen(QColor(*rgba), width=1)
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
            [mdsplotutils.make_brush(QColor(*c))
             for c in basecolors],
            dtype=object)

        if flags is None:
            return brush

        fill_mask = flags & mdsplotutils.Filled

        if not numpy.all(fill_mask):
            brush[~fill_mask] = QBrush(Qt.NoBrush)
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
                shapes = [(QColor(Qt.gray),
                           symbols[i % (len(symbols) - 1)], name)
                          for i, name in enumerate(shape_var.values)]
            items = colors + shapes

        return items

    @staticmethod
    def make_pen(color, width=1, cosmetic=True):
        pen = QPen(color)
        pen.setWidthF(width)
        pen.setCosmetic(cosmetic)
        return pen

    @staticmethod
    def make_brush(color, ):
        return QBrush(color, )


def main_test(argv=sys.argv):
    import gc
    app = QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Orange.data.Table(filename)
    w = OWMDS()
    w.set_data(data)
    w.set_subset_data(data[numpy.random.choice(len(data), 10)])
    w.handleNewSignals()

    w.show()
    w.raise_()
    rval = app.exec_()

    w.set_subset_data(None)
    w.set_data(None)
    w.handleNewSignals()

    w.saveSettings()
    w.onDeleteWidget()
    w.deleteLater()
    del w
    gc.collect()
    app.processEvents()
    return rval

if __name__ == "__main__":
    sys.exit(main_test())

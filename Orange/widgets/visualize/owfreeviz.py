from itertools import chain
import sys
from types import SimpleNamespace as namespace
from xml.sax.saxutils import escape

from scipy.spatial import distance
import numpy as np

from AnyQt.QtWidgets import (
    QFormLayout, QApplication, QGraphicsEllipseItem, QGraphicsSceneMouseEvent, QToolTip
)
from AnyQt.QtGui import QPen
from AnyQt.QtCore import Qt, QObject, QEvent, QSize, QRectF, QLineF, QTimer, QPoint
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph as pg

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.projection.freeviz import FreeViz
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table
)
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph, InteractiveViewBox, \
    HelpEventDelegate
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.widget import Input, Output
from Orange.widgets import report


class AsyncUpdateLoop(QObject):
    """
    Run/drive an coroutine from the event loop.

    This is a utility class which can be used for implementing
    asynchronous update loops. I.e. coroutines which periodically yield
    control back to the Qt event loop.

    """
    Next = QEvent.registerEventType()

    #: State flags
    Idle, Running, Cancelled, Finished = 0, 1, 2, 3
    #: The coroutine has yielded control to the caller (with `object`)
    yielded = Signal(object)
    #: The coroutine has finished/exited (either with an exception
    #: or with a return statement)
    finished = Signal()

    #: The coroutine has returned (normal return statement / StopIteration)
    returned = Signal(object)
    #: The coroutine has exited with with an exception.
    raised = Signal(object)
    #: The coroutine was cancelled/closed.
    cancelled = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__coroutine = None
        self.__next_pending = False  # Flag for compressing scheduled events
        self.__in_next = False
        self.__state = AsyncUpdateLoop.Idle

    @Slot(object)
    def setCoroutine(self, loop):
        """
        Set the coroutine.

        The coroutine will be resumed (repeatedly) from the event queue.
        If there is an existing coroutine set it is first closed/cancelled.

        Raises an RuntimeError if the current coroutine is running.
        """
        if self.__coroutine is not None:
            self.__coroutine.close()
            self.__coroutine = None
            self.__state = AsyncUpdateLoop.Cancelled

            self.cancelled.emit()
            self.finished.emit()

        if loop is not None:
            self.__coroutine = loop
            self.__state = AsyncUpdateLoop.Running
            self.__schedule_next()

    @Slot()
    def cancel(self):
        """
        Cancel/close the current coroutine.

        Raises an RuntimeError if the current coroutine is running.
        """
        self.setCoroutine(None)

    def state(self):
        """
        Return the current state.
        """
        return self.__state

    def isRunning(self):
        return self.__state == AsyncUpdateLoop.Running

    def __schedule_next(self):
        if not self.__next_pending:
            self.__next_pending = True
            QTimer.singleShot(10, self.__on_timeout)

    def __next(self):
        if self.__coroutine is not None:
            try:
                rval = next(self.__coroutine)
            except StopIteration as stop:
                self.__state = AsyncUpdateLoop.Finished
                self.returned.emit(stop.value)
                self.finished.emit()
                self.__coroutine = None
            except BaseException as er:
                self.__state = AsyncUpdateLoop.Finished
                self.raised.emit(er)
                self.finished.emit()
                self.__coroutine = None
            else:
                self.yielded.emit(rval)
                self.__schedule_next()

    @Slot()
    def __on_timeout(self):
        assert self.__next_pending
        self.__next_pending = False
        if not self.__in_next:
            self.__in_next = True
            try:
                self.__next()
            finally:
                self.__in_next = False
        else:
            # warn
            self.__schedule_next()

    def customEvent(self, event):
        if event.type() == AsyncUpdateLoop.Next:
            self.__on_timeout()
        else:
            super().customEvent(event)


class FreeVizInteractiveViewBox(InteractiveViewBox):
    def __init__(self, graph, enable_menu=False):
        self.mousestate = 0
        self.point_i = None
        super().__init__(graph, enable_menu)

    def _dragtip_pos(self):
        return 10, 10

    def mouseDragEvent(self, ev, axis=None):
        master = self.graph.master
        if master.data is None:
            super().mouseDragEvent(ev, axis)
            return
        pos = self.childGroup.mapFromParent(ev.pos())
        minradius = master.radius / 100 + 1e-5
        points = master.plotdata.anchors
        mask = np.zeros((len(points)), dtype=bool)
        for i, point in enumerate(points):
            if np.linalg.norm(point) > minradius:
                mask[i] = True
        np_pos = np.array([[pos.x(), pos.y()]])
        distances = distance.cdist(np_pos, points[:, :2])[0]
        is_near = False if not len(distances[mask]) else np.min(distances[mask]) < 0.1

        if ev.button() != Qt.LeftButton or (ev.start and not is_near):
            self.mousestate = 2  # finished
        if self.mousestate == 2:
            if ev.finish:
                self.mousestate = 0  # ready for new task
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        if ev.start:
            self.setCursor(Qt.ClosedHandCursor)
            self.mousestate = 1  # working
            self.point_i = np.flatnonzero(mask)[np.argmin(distances[mask])]
            master.randomize_indices()
        is_moving = True
        if self.mousestate == 1:
            if ev.finish:
                self.setCursor(Qt.OpenHandCursor)
                self.mousestate = 0
                is_moving = False
            points[self.point_i][0] = pos.x()
            points[self.point_i][1] = pos.y()
            if is_moving:
                master.manual_move_anchor()
            else:
                master.setup_plot(reset_view=False)
            self.graph.show_indicator(point_i=self.point_i)


class EventDelegate(HelpEventDelegate):
    def __init__(self, delegate, delegate2, parent=None):
        self.delegate2 = delegate2
        super().__init__(delegate, parent=parent)

    def eventFilter(self, obj, ev):
        if isinstance(ev, QGraphicsSceneMouseEvent):
            self.delegate2(ev)
        return super().eventFilter(obj, ev)


SELECTION_WIDTH = 5
RANGE = QRectF(-1.05, -1.05, 2.1, 2.1)

class OWFreeVizGraph(OWScatterPlotGraph):
    jitter_size = settings.Setting(0)

    def __init__(self, scatter_widget, parent=None, name="None", view_box=None):
        super().__init__(scatter_widget, parent=parent, _=name, view_box=view_box)
        self._tooltip_delegate = EventDelegate(self.help_event, self._show_indicator)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)
        self.master = scatter_widget
        for axis_loc in ["left", "bottom"]:
            self.plot_widget.hideAxis(axis_loc)

    def update_data(self, attr_x, attr_y, reset_view=True):
        super().update_data(attr_x, attr_y, reset_view=reset_view)
        for axis in ["left", "bottom"]:
            self.plot_widget.hideAxis(axis)

        if reset_view:
            self.view_box.setRange(
                RANGE,
                padding=0.025)
            self.master.viewbox.setAspectLocked(True, 1)
            self.master.viewbox.init_history()
            self.master.viewbox.tag_history()

    def _show_indicator(self, ev):
        scene = self.plot_widget.scene()
        if self.scatterplot_item is None or scene.drag_tooltip.isVisible():
            return False

        for indicator in self.master.plotdata.indicators:
            self.plot_widget.removeItem(indicator)
        self.master.plotdata.indicators = []
        pos = self.scatterplot_item.mapFromScene(ev.scenePos())
        x = pos.x()
        y = pos.y()
        master = self.master
        minradius = master.radius / 100 + 1e-5
        points = master.plotdata.anchors
        mask = np.zeros((len(points)), dtype=bool)
        for i, point in enumerate(points):
            if np.linalg.norm(point) > minradius:
                mask[i] = True
        np_pos = np.array([[x, y]])
        distances = distance.cdist(np_pos, points[:, :2])[0]
        if len(distances[mask]) and np.min(distances[mask]) < 0.08:
            if self.view_box.mousestate == 0:
                self.view_box.setCursor(Qt.OpenHandCursor)
            self.show_indicator(point_i=np.flatnonzero(mask)[np.argmin(distances[mask])])
        else:
            self.view_box.setCursor(Qt.ArrowCursor)
        return True

    def show_indicator(self, point_i):
        points = self.master.plotdata.anchors
        func = self.view_box.childGroup.mapToDevice
        dx = (func(QPoint(1, 0)) - func(QPoint(-1, 0))).x()
        scene_size = 600 / dx
        self.master.plotdata.indicators.append(
            MoveIndicator(points[point_i][0], points[point_i][1], scene_size=scene_size)
        )
        self.plot_widget.addItem(self.master.plotdata.indicators[0])

    def help_event(self, event):
        if self.scatterplot_item is None:
            return False

        act_pos = self.scatterplot_item.mapFromScene(event.scenePos())
        points = self.scatterplot_item.pointsAt(act_pos)
        text = ""
        attr = lambda i: self.domain.attributes[i]
        if len(points):
            for i, p in enumerate(points):
                index = p.data()
                text += "Attributes:\n"
                text += "".join(
                    "   {} = {}\n".format(attr(i).name,
                                          self.data[index][attr(i)])
                    for i in self.master.plotdata.topattrs[index])
                if len(self.domain.attributes) > 10:
                    text += "   ... and {} others\n\n".format(len(self.domain.attributes) - 12)
                #  class_var is always:
                text += "Class:\n   {} = {}\n".format(self.domain.class_var.name,
                                                      self.data[index][self.data.domain.class_var])
                if i < len(points) - 1:
                    text += '------------------\n'
            text = ('<span style="white-space:pre">{}</span>'.format(escape(text)))

            QToolTip.showText(event.screenPos(), text, widget=self.plot_widget)
            return True
        else:
            return False

MAX_ITERATIONS = 1000
MAX_ANCHORS = 20
MAX_POINTS = 300
MAX_INSTANCES = 10000


class OWFreeViz(widget.OWWidget):
    name = "FreeViz"
    description = "Displays FreeViz projection"
    icon = "icons/Freeviz.svg"
    priority = 240

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        components = Output("Components", Table)

    #: Initialization type
    Circular, Random = 0, 1

    jitter_sizes = [0, 0.1, 0.5, 1, 2]

    settings_version = 2
    settingsHandler = settings.DomainContextHandler()

    radius = settings.Setting(0)
    initialization = settings.Setting(Circular)
    auto_commit = settings.Setting(True)

    resolution = 256
    graph = settings.SettingProvider(OWFreeVizGraph)

    ReplotRequest = QEvent.registerEventType()

    graph_name = "graph.plot_widget.plotItem"


    class Warning(widget.OWWidget.Warning):
        sparse_not_supported = widget.Msg("Sparse data is ignored.")

    class Error(widget.OWWidget.Error):
        no_class_var = widget.Msg("Need a class variable")
        not_enough_class_vars = widget.Msg("Needs discrete class variable " \
                                          "with at lest 2 values")
        features_exceeds_instances = widget.Msg("Algorithm should not be used when " \
                                                "number of features exceeds the number " \
                                                "of instances.")
        too_many_data_instances = widget.Msg("Cannot handle so large data.")
        no_valid_data = widget.Msg("No valid data.")


    def __init__(self):
        super().__init__()

        self.data = None
        self.subset_data = None
        self._subset_mask = None
        self._validmask = None
        self._X = None
        self._Y = None
        self._selection = None
        self.__replot_requested = False

        self.variable_x = ContinuousVariable("freeviz-x")
        self.variable_y = ContinuousVariable("freeviz-y")

        box0 = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWFreeVizGraph(self, box0, "Plot", view_box=FreeVizInteractiveViewBox)
        box0.layout().addWidget(self.graph.plot_widget)
        plot = self.graph.plot_widget

        box = gui.widgetBox(self.controlArea, "Optimization", spacing=10)
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )
        form.addRow(
            "Initialization",
            gui.comboBox(box, self, "initialization",
                         items=["Circular", "Random"],
                         callback=self.reset_initialization)
        )
        box.layout().addLayout(form)

        self.btn_start = gui.button(widget=box, master=self, label="Optimize",
                                    callback=self.toogle_start, enabled=False)

        self.viewbox = plot.getViewBox()
        self.replot = None

        g = self.graph.gui
        g.point_properties_box(self.controlArea)
        self.models = g.points_models

        box = gui.widgetBox(self.controlArea, "Show anchors")
        self.rslider = gui.hSlider(
            box, self, "radius", minValue=0, maxValue=100,
            step=5, label="Radius", createLabel=False, ticks=True,
            callback=self.update_radius)
        self.rslider.setTickInterval(0)
        self.rslider.setPageStep(10)

        box = gui.vBox(self.controlArea, "Plot Properties")

        g.add_widgets([g.JitterSizeSlider], box)
        g.add_widgets([g.ShowLegend,
                       g.ClassDensity,
                       g.LabelOnlySelected],
                      box)

        self.graph.box_zoom_select(self.controlArea)
        self.controlArea.layout().addStretch(100)
        self.icons = gui.attributeIconDict

        p = self.graph.plot_widget.palette()
        self.graph.set_palette(p)

        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")
        self.graph.zoom_actions(self)
        # FreeViz
        self._loop = AsyncUpdateLoop(parent=self)
        self._loop.yielded.connect(self.__set_projection)
        self._loop.finished.connect(self.__freeviz_finished)
        self._loop.raised.connect(self.__on_error)

        self._new_plotdata()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def update_radius(self):
        # Update the anchor/axes visibility
        assert not self.plotdata is None
        if self.plotdata.hidecircle is None:
            return

        minradius = self.radius / 100 + 1e-5
        for anchor, item in zip(self.plotdata.anchors,
                                self.plotdata.anchoritem):
            item.setVisible(np.linalg.norm(anchor) > minradius)
        self.plotdata.hidecircle.setRect(
            QRectF(-minradius, -minradius,
                   2 * minradius, 2 * minradius))

    def toogle_start(self):
        if self._loop.isRunning():
            self._loop.cancel()
            if isinstance(self, OWFreeViz):
                self.btn_start.setText("Optimize")
            self.progressBarFinished(processEvents=False)
        else:
            self._start()

    def _start(self):
        """
        Start the projection optimization.
        """
        assert not self.plotdata is None

        X, Y = self.plotdata.X, self.plotdata.Y
        anchors = self.plotdata.anchors

        def update_freeviz(interval, initial):
            anchors = initial
            while True:
                res = FreeViz.freeviz(X, Y, scale=False, center=False,
                                      initial=anchors, maxiter=interval)
                _, anchors_new = res[:2]
                yield res[:2]
                if np.allclose(anchors, anchors_new, rtol=1e-5, atol=1e-4):
                    return

                anchors = anchors_new

        interval = 10  # TODO

        self._loop.setCoroutine(
            update_freeviz(interval, anchors))
        self.btn_start.setText("Stop")
        self.progressBarInit(processEvents=False)
        self.setBlocking(True)
        self.setStatusMessage("Optimizing")

    def reset_initialization(self):
        """
        Reset the current 'anchor' initialization, and restart the
        optimization if necessary.
        """
        running = self._loop.isRunning()

        if running:
            self._loop.cancel()

        if self.data is not None:
            self._clear_plot()
            self.setup_plot()

        if running:
            self._start()

    def __set_projection(self, res):
        # Set/update the projection matrix and coordinate embeddings
        # assert self.plotdata is not None, "__set_projection call unexpected"
        assert not self.plotdata is None
        increment = 1  # TODO
        self.progressBarAdvance(
            increment * 100. / MAX_ITERATIONS, processEvents=False)  # TODO
        embedding_coords, projection = res
        self.plotdata.embedding_coords = embedding_coords
        self.plotdata.anchors = projection
        self._update_xy()
        self.update_radius()
        self.update_density()

    def __freeviz_finished(self):
        # Projection optimization has finished
        self.btn_start.setText("Optimize")
        self.setStatusMessage("")
        self.setBlocking(False)
        self.progressBarFinished(processEvents=False)
        self.commit()

    def __on_error(self, err):
        sys.excepthook(type(err), err, getattr(err, "__traceback__"))

    def _update_xy(self):
        # Update the plotted embedding coordinates
        self.graph.plot_widget.clear()
        coords = self.plotdata.embedding_coords
        radius = np.max(np.linalg.norm(coords, axis=1))
        self.plotdata.embedding_coords = coords / radius
        self.plot(show_anchors=(len(self.data.domain.attributes) < MAX_ANCHORS))

    def _new_plotdata(self):
        self.plotdata = namespace(
            validmask=None,
            embedding_coords=None,
            anchors=[],
            anchoritem=[],
            X=None,
            Y=None,
            indicators=[],
            hidecircle=None,
            data=None,
            items=[],
            topattrs=None,
            rand=None,
            selection=None,  # np.array
        )

    def _anchor_circle(self):
        # minimum visible anchor radius (radius)
        minradius = self.radius / 100 + 1e-5
        for item in chain(self.plotdata.anchoritem, self.plotdata.items):
            self.viewbox.removeItem(item)
        self.plotdata.anchoritem = []
        self.plotdata.items = []
        for anchor, var in zip(self.plotdata.anchors, self.data.domain.attributes):
            if True or np.linalg.norm(anchor) > minradius:
                axitem = AnchorItem(
                    line=QLineF(0, 0, *anchor), text=var.name,)
                axitem.setVisible(np.linalg.norm(anchor) > minradius)
                axitem.setPen(pg.mkPen((100, 100, 100)))
                axitem.setArrowVisible(True)
                self.plotdata.anchoritem.append(axitem)
                self.viewbox.addItem(axitem)

        hidecircle = QGraphicsEllipseItem()
        hidecircle.setRect(
            QRectF(-minradius, -minradius,
                   2 * minradius, 2 * minradius))

        _pen = QPen(Qt.lightGray, 1)
        _pen.setCosmetic(True)
        hidecircle.setPen(_pen)
        self.viewbox.addItem(hidecircle)
        self.plotdata.items.append(hidecircle)
        self.plotdata.hidecircle = hidecircle

    def update_colors(self):
        pass

    def sizeHint(self):
        return QSize(800, 500)

    def _clear(self):
        """
        Clear/reset the widget state
        """
        self._loop.cancel()
        self.data = None
        self._selection = None
        self._clear_plot()

    def _clear_plot(self):
        for item in chain(self.plotdata.anchoritem, self.plotdata.items):
            self.viewbox.removeItem(item)
        self.graph.plot_widget.clear()
        self._new_plotdata()

    def init_attr_values(self):
        self.graph.set_domain(self.data)

    @Inputs.data
    def set_data(self, data):
        self.clear_messages()
        self._clear()
        self.closeContext()
        if data is not None:
            if data and data.is_sparse():
                self.Warning.sparse_not_supported()
                data = None
            elif data.domain.class_var is None:
                self.Error.no_class_var()
                data = None
            elif data.domain.class_var.is_discrete and \
                            len(data.domain.class_var.values) < 2:
                self.Error.not_enough_class_vars()
                data = None
            if data and len(data.domain.attributes) > data.X.shape[0]:
                self.Error.features_exceeds_instances()
                data = None
        if data is not None:
            valid_instances_count = self._prepare_freeviz_data(data)
            if valid_instances_count > MAX_INSTANCES:
                self.Error.too_many_data_instances()
                data = None
            elif valid_instances_count == 0:
                self.Error.no_valid_data()
                data = None
        self.data = data
        self.init_attr_values()
        if data is not None:
            self.cb_class_density.setEnabled(data.domain.has_discrete_class)
            self.openContext(data)
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(False)
            self._X = self._Y = None
            self.graph.new_data(None, None)

    @Inputs.data_subset
    def set_subset_data(self, subset):
        self.subset_data = subset
        self.plotdata.subset_mask = None
        self.controls.graph.alpha_value.setEnabled(subset is None)

    def handleNewSignals(self):
        if all(v is not None for v in [self.data, self.subset_data]):
            dataids = self.data.ids.ravel()
            subsetids = np.unique(self.subset_data.ids)
            self._subset_mask = np.in1d(dataids, subsetids, assume_unique=True)
        if self._X is not None:
            self.setup_plot(True)
        self.commit()

    def customEvent(self, event):
        if event.type() == OWFreeViz.ReplotRequest:
            self.__replot_requested = False
            self.setup_plot()
        else:
            super().customEvent(event)

    def _prepare_freeviz_data(self, data):
        X = data.X
        Y = data.Y
        mask = np.bitwise_or.reduce(np.isnan(X), axis=1)
        mask |= np.isnan(Y)
        validmask = ~mask
        X = X[validmask, :]
        Y = Y[validmask]

        if not len(X):
            self._X = None
            return 0

        if data.domain.class_var.is_discrete:
            Y = Y.astype(int)
        X = (X - np.mean(X, axis=0))
        span = np.ptp(X, axis=0)
        X[:, span > 0] /= span[span > 0].reshape(1, -1)
        self._X = X
        self._Y = Y
        self._validmask = validmask
        return len(X)

    def setup_plot(self, reset_view=True):
        assert not self._X is None

        self.graph.jitter_continuous = True
        self.__replot_requested = False

        X = self.plotdata.X = self._X
        self.plotdata.Y = self._Y
        self.plotdata.validmask = self._validmask
        self.plotdata.selection = self._selection if self._selection is not None else \
            np.zeros(len(self._validmask), dtype=np.uint8)
        anchors = self.plotdata.anchors
        if len(anchors) == 0:
            if self.initialization == self.Circular:
                anchors = FreeViz.init_radial(X.shape[1])
            else:
                anchors = FreeViz.init_random(X.shape[1], 2)

        EX = np.dot(X, anchors)
        c = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            c[i] = np.argsort((np.power(X[i] * anchors[:, 0], 2) +
                               np.power(X[i] * anchors[:, 1], 2)))[::-1]
        self.plotdata.topattrs = np.array(c, dtype=int)[:, :10]
        radius = np.max(np.linalg.norm(EX, axis=1))

        self.plotdata.anchors = anchors

        coords = (EX / radius)
        self.plotdata.embedding_coords = coords
        if reset_view:
            self.viewbox.setRange(RANGE)
            self.viewbox.setAspectLocked(True, 1)
        self.plot(reset_view=reset_view)

    def randomize_indices(self):
        X = self._X
        self.plotdata.rand = np.random.choice(len(X), MAX_POINTS, replace=False) \
            if len(X) > MAX_POINTS else None

    def manual_move_anchor(self, show_anchors=True):
        self.__replot_requested = False
        X = self.plotdata.X = self._X
        anchors = self.plotdata.anchors
        validmask = self.plotdata.validmask
        EX = np.dot(X, anchors)
        data_x = self.data.X[validmask]
        data_y = self.data.Y[validmask]
        radius = np.max(np.linalg.norm(EX, axis=1))
        if self.plotdata.rand is not None:
            rand = self.plotdata.rand
            EX = EX[rand]
            data_x = data_x[rand]
            data_y = data_y[rand]
            selection = self.plotdata.selection[validmask]
            selection = selection[rand]
        else:
            selection = self.plotdata.selection[validmask]
        coords = (EX / radius)

        if show_anchors:
            self._anchor_circle()
        attributes = () + self.data.domain.attributes + (self.variable_x, self.variable_y)
        domain = Domain(attributes=attributes,
                        class_vars=self.data.domain.class_vars)
        data = Table.from_numpy(domain, X=np.hstack((data_x, coords)),
                                Y=data_y)
        self.graph.new_data(data, None)
        self.graph.selection = selection
        self.graph.update_data(self.variable_x, self.variable_y, reset_view=False)

    def plot(self, reset_view=False, show_anchors=True):
        if show_anchors:
            self._anchor_circle()
        attributes = () + self.data.domain.attributes + (self.variable_x, self.variable_y)
        domain = Domain(attributes=attributes,
                        class_vars=self.data.domain.class_vars,
                        metas=self.data.domain.metas)
        mask = self.plotdata.validmask
        array = np.zeros((len(self.data), 2), dtype=np.float)
        array[mask] = self.plotdata.embedding_coords
        data = self.data.transform(domain)
        data[:, self.variable_x] = array[:, 0].reshape(-1, 1)
        data[:, self.variable_y] = array[:, 1].reshape(-1, 1)
        subset_data = data[self._subset_mask & mask]\
            if self._subset_mask is not None and len(self._subset_mask) else None
        self.plotdata.data = data
        self.graph.new_data(data[mask], subset_data)
        if self.plotdata.selection is not None:
            self.graph.selection = self.plotdata.selection[self.plotdata.validmask]
        self.graph.update_data(self.variable_x, self.variable_y, reset_view=reset_view)

    def reset_graph_data(self, *_):
        if self.data is not None:
            self.graph.rescale_data()
            self._update_graph()

    def _update_graph(self, reset_view=True, **_):
        self.graph.zoomStack = []
        assert not self.graph.data is None
        self.graph.update_data(self.variable_x, self.variable_y, reset_view)

    def update_density(self):
        if self.graph.data is None:
            return
        self._update_graph(reset_view=False)

    def selection_changed(self):
        if self.graph.selection is not None:
            pd = self.plotdata
            pd.selection[pd.validmask] = self.graph.selection
            self._selection = pd.selection
        self.commit()

    def prepare_data(self):
        pass

    def commit(self):
        selected = annotated = components = None
        graph = self.graph
        if self.data is not None and self.plotdata.validmask is not None:
            name = self.data.name
            metas = () + self.data.domain.metas + (self.variable_x, self.variable_y)
            domain = Domain(attributes=self.data.domain.attributes,
                            class_vars=self.data.domain.class_vars,
                            metas=metas)
            data = self.plotdata.data.transform(domain)
            validmask = self.plotdata.validmask
            mask = np.array(validmask, dtype=int)
            mask[mask == 1] = graph.selection if graph.selection is not None \
                else [False * len(mask)]
            selection = np.array([], dtype=np.uint8) if mask is None else np.flatnonzero(mask)
            if len(selection):
                selected = data[selection]
                selected.name = name + ": selected"
                selected.attributes = self.data.attributes
            if graph.selection is not None and np.max(graph.selection) > 1:
                annotated = create_groups_table(data, mask)
            else:
                annotated = create_annotated_table(data, selection)
            annotated.attributes = self.data.attributes
            annotated.name = name + ": annotated"

            comp_domain = Domain(
                self.data.domain.attributes,
                metas=[StringVariable(name='component')])

            metas = np.array([["FreeViz 1"], ["FreeViz 2"]])
            components = Table.from_numpy(
                comp_domain,
                X=self.plotdata.anchors.T,
                metas=metas)

            components.name = name + ": components"

        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)
        self.Outputs.components.send(components)

    def send_report(self):
        if self.data is None:
            return

        def name(var):
            return var and var.name

        caption = report.render_items_vert((
            ("Color", name(self.graph.attr_color)),
            ("Label", name(self.graph.attr_label)),
            ("Shape", name(self.graph.attr_shape)),
            ("Size", name(self.graph.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and "{} %".format(self.graph.jitter_size))))
        self.report_plot()
        if caption:
            self.report_caption(caption)


class MoveIndicator(pg.GraphicsObject):
    def __init__(self, x, y, parent=None, line=QLineF(), scene_size=1, text="", **kwargs):
        super().__init__(parent, **kwargs)
        self.arrows = [
            pg.ArrowItem(pos=(x - scene_size * 0.07 * np.cos(np.radians(angle)),
                              y + scene_size * 0.07 * np.sin(np.radians(angle))),
                         parent=self, angle=angle,
                         headLen=13, tipAngle=45,
                         brush=pg.mkColor(128, 128, 128))
            for angle in (0, 90, 180, 270)]

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()


def main(argv=None):
    import sip

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        filename = argv[0]
    else:
        filename = "zoo"

    data = Table(filename)

    app = QApplication([])
    w = OWFreeViz()
    w.set_data(data)
    w.set_subset_data(data[::10])
    w.handleNewSignals()
    w.show()
    w.raise_()
    r = app.exec()
    w.set_data(None)
    w.saveSettings()
    sip.delete(w)
    del w
    return r


if __name__ == "__main__":
    sys.exit(main())

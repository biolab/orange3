from enum import IntEnum
import sys

import numpy as np
from scipy.spatial import distance

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QRectF, QLineF, QTimer, QPoint,
    pyqtSignal as Signal, pyqtSlot as Slot
)
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QApplication, QGraphicsEllipseItem

import pyqtgraph as pg

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.projection.freeviz import FreeViz
from Orange.widgets import widget, gui, settings, report
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table
)
from Orange.widgets.visualize.owscatterplotgraph import OWProjectionWidget
from Orange.widgets.visualize.utils.component import OWVizGraph
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.widget import Input, Output


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


class OWFreeVizGraph(OWVizGraph):
    radius = settings.Setting(0)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self._points = []
        self._point_items = []

    def update_radius(self):
        if self._circle_item is None:
            return

        r = self.radius / 100 + 1e-5
        for point, axitem in zip(self._points, self._point_items):
            axitem.setVisible(np.linalg.norm(point) > r)
        self._circle_item.setRect(QRectF(-r, -r, 2 * r, 2 * r))

    def set_view_box_range(self):
        self.view_box.setRange(RANGE)

    def can_show_indicator(self, pos):
        if not len(self._points):
            return False, None

        r = self.radius / 100 + 1e-5
        mask = np.zeros((len(self._points)), dtype=bool)
        mask[np.linalg.norm(self._points, axis=1) > r] = True
        distances = distance.cdist([[pos.x(), pos.y()]], self._points)[0]
        distances = distances[mask]
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF:
            return True, np.flatnonzero(mask)[np.argmin(distances)]
        return False, None

    def _remove_point_items(self):
        for item in self._point_items:
            self.plot_widget.removeItem(item)
        self._point_items = []

    def _add_point_items(self):
        r = self.radius / 100 + 1e-5
        for point, var in zip(self._points, self._attributes):
            axitem = AnchorItem(line=QLineF(0, 0, *point), text=var.name)
            axitem.setVisible(np.linalg.norm(point) > r)
            axitem.setPen(pg.mkPen((100, 100, 100)))
            self.plot_widget.addItem(axitem)
            self._point_items.append(axitem)

    def _add_circle_item(self):
        if not len(self._points):
            return
        r = self.radius / 100 + 1e-5
        pen = pg.mkPen(QColor(Qt.lightGray), width=1, cosmetic=True)
        self._circle_item = QGraphicsEllipseItem()
        self._circle_item.setRect(QRectF(-r, -r, 2 * r, 2 * r))
        self._circle_item.setPen(pen)
        self.plot_widget.addItem(self._circle_item)

    def _add_indicator_item(self, point_i):
        x, y = self._points[point_i]
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self._indicator_item = MoveIndicator(x, y, 600 / dx)
        self.plot_widget.addItem(self._indicator_item)


MAX_ITERATIONS = 1000
MAX_POINTS = 300
MAX_INSTANCES = 10000
RANGE = QRectF(-1.05, -1.05, 2.1, 2.1)


class InitType(IntEnum):
    Circular, Random = 0, 1

    @staticmethod
    def items():
        return ["Circular", "Random"]


class OWFreeViz(OWProjectionWidget):
    name = "FreeViz"
    description = "Displays FreeViz projection"
    icon = "icons/Freeviz.svg"
    priority = 240
    keywords = ["viz"]

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        components = Output("Components", Table)

    settings_version = 3
    settingsHandler = settings.DomainContextHandler()

    initialization = settings.Setting(InitType.Circular)
    auto_commit = settings.Setting(True)

    graph = settings.SettingProvider(OWFreeVizGraph)
    graph_name = "graph.plot_widget.plotItem"

    class Error(OWProjectionWidget.Error):
        sparse_data = widget.Msg("Sparse data is not supported")
        no_class_var = widget.Msg("Need a class variable")
        not_enough_class_vars = widget.Msg(
            "Needs discrete class variable with at lest 2 values"
        )
        features_exceeds_instances = widget.Msg(
            "Algorithm should not be used when number of features "
            "exceeds the number of instances."
        )
        too_many_data_instances = widget.Msg("Cannot handle so large data.")
        no_valid_data = widget.Msg("No valid data.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.subset_data = None
        self.subset_indices = None
        self._embedding_coords = None
        self._X = None
        self._Y = None
        self._rand_indices = None
        self.variable_x = ContinuousVariable("freeviz-x")
        self.variable_y = ContinuousVariable("freeviz-y")

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWFreeVizGraph(self, box)
        box.layout().addWidget(self.graph.plot_widget)

        box = gui.vBox(self.controlArea, box=True)
        gui.comboBox(box, self, "initialization", label="Initialization:",
                     items=InitType.items(), orientation=Qt.Horizontal,
                     labelWidth=90, callback=self.__init_combo_changed)
        self.btn_start = gui.button(box, self, "Optimize", self.__toggle_start,
                                    enabled=False)

        g = self.graph.gui
        g.point_properties_box(self.controlArea)
        box = g.effects_box(self.controlArea)
        g.add_control(box, gui.hSlider, "Hide radius:",
            master=self.graph, value="radius",
            minValue=0, maxValue=100,
            step=10, createLabel=False,
            callback=self.__radius_slider_changed)
        g.plot_properties_box(self.controlArea)

        self.controlArea.layout().addStretch(100)
        self.graph.box_zoom_select(self.controlArea)

        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

        # FreeViz
        self._loop = AsyncUpdateLoop(parent=self)
        self._loop.yielded.connect(self.__set_projection)
        self._loop.finished.connect(self.__freeviz_finished)
        self._loop.raised.connect(self.__on_error)

        self.graph.view_box.started.connect(self._randomize_indices)
        self.graph.view_box.moved.connect(self._manual_move)
        self.graph.view_box.finished.connect(self._finish_manual_move)

    def __radius_slider_changed(self):
        self.graph.update_radius()

    def __toggle_start(self):
        if self._loop.isRunning():
            self._loop.cancel()
            self.btn_start.setText("Optimize")
            self.progressBarFinished(processEvents=False)
        else:
            self._start()

    def __init_combo_changed(self):
        running = self._loop.isRunning()
        if running:
            self._loop.cancel()
        if self.data is not None:
            self.setup_plot()
        if running:
            self._start()

    def _start(self):
        """
        Start the projection optimization.
        """
        def update_freeviz(anchors):
            while True:
                projection = FreeViz.freeviz(
                    self._X, self._Y, scale=False, center=False,
                    initial=anchors, maxiter=10
                )
                yield projection[0], projection[1]
                if np.allclose(anchors, projection[1], rtol=1e-5, atol=1e-4):
                    return
                anchors = projection[1]

        self._loop.setCoroutine(update_freeviz(self.graph.get_points()))
        self.btn_start.setText("Stop")
        self.progressBarInit()
        self.setBlocking(True)
        self.setStatusMessage("Optimizing")

    def __set_projection(self, projection):
        # Set/update the projection matrix and coordinate embeddings
        self.progressBarAdvance(100. / MAX_ITERATIONS)
        self._embedding_coords = projection[0]
        self.graph.set_points(projection[1])
        self._update_xy()

    def __freeviz_finished(self):
        # Projection optimization has finished
        self.btn_start.setText("Optimize")
        self.setStatusMessage("")
        self.setBlocking(False)
        self.progressBarFinished()
        self.commit()

    def __on_error(self, err):
        sys.excepthook(type(err), err, getattr(err, "__traceback__"))

    def _update_xy(self):
        coords = self._embedding_coords
        self._embedding_coords /= np.max(np.linalg.norm(coords, axis=1))
        self.graph.update_coordinates()

    def clear(self):
        self._loop.cancel()
        self.data = None
        self.valid_data = None
        self._embedding_coords = None
        self._X = None
        self._Y = None
        self._rand_indices = None

        self.graph.set_attributes(())
        self.graph.set_points([])
        self.graph.update_coordinates()
        self.graph.clear()

    @Inputs.data
    def set_data(self, data):
        self.clear_messages()
        self.closeContext()
        self.clear()
        self.data = data
        self._check_data()
        self.init_attr_values()
        self.openContext(data)
        self.btn_start.setEnabled(self.data is not None)
        self.cb_class_density.setEnabled(self.can_draw_density())

    def _check_data(self):
        if self.data is not None:
            if self.data.is_sparse():
                self.Error.sparse_data()
                self.data = None
            elif self.data.domain.class_var is None:
                self.Error.no_class_var()
                self.data = None
            elif self.data.domain.class_var.is_discrete and \
                    len(self.data.domain.class_var.values) < 2:
                self.Error.not_enough_class_vars()
                self.data = None
            elif len(self.data.domain.attributes) > self.data.X.shape[0]:
                self.Error.features_exceeds_instances()
                self.data = None
            else:
                self._prepare_freeviz_data()
                if self._X is not None:
                    if len(self._X) > MAX_INSTANCES:
                        self.Error.too_many_data_instances()
                        self.data = None
                    elif np.allclose(np.nan_to_num(self._X - self._X[0]), 0) \
                            or not len(self._X):
                        self.Error.no_valid_data()
                        self.data = None
                else:
                    self.Error.no_valid_data()
                    self.data = None

    def _prepare_freeviz_data(self):
        valid_mask = np.all(np.isfinite(self.data.X), axis=1) & \
                     np.isfinite(self.data.Y)
        X, Y = self.data.X[valid_mask], self.data.Y[valid_mask]
        if not len(X):
            self.valid_data = None
            return

        if self.data.domain.class_var.is_discrete:
            Y = Y.astype(int)
        X = (X - np.mean(X, axis=0))
        span = np.ptp(X, axis=0)
        X[:, span > 0] /= span[span > 0].reshape(1, -1)
        self._X, self._Y, self.valid_data = X, Y, valid_mask

    @Inputs.data_subset
    def set_subset_data(self, subset):
        self.subset_data = subset
        self.subset_indices = {e.id for e in subset} \
            if subset is not None else {}
        self.controls.graph.alpha_value.setEnabled(subset is None)

    def handleNewSignals(self):
        if self.data is not None and self.valid_data is not None:
            self.setup_plot()
        self.commit()

    def get_coordinates_data(self):
        return (self._embedding_coords[:, 0], self._embedding_coords[:, 1]) \
            if self._embedding_coords is not None else (None, None)

    def get_subset_mask(self):
        if self.subset_indices:
            return np.array([ex.id in self.subset_indices
                             for ex in self.data[self.valid_data]])

    def setup_plot(self):
        points = FreeViz.init_radial(self._X.shape[1]) \
            if self.initialization == InitType.Circular \
            else FreeViz.init_random(self._X.shape[1], 2)
        self.graph.set_points(points)
        self.__set_embedding_coords()
        self.graph.set_attributes(self.data.domain.attributes)
        self.graph.reset_graph()

    def _randomize_indices(self):
        n = len(self._X)
        if n > MAX_POINTS:
            self._rand_indices = np.random.choice(n, MAX_POINTS, replace=False)
            self._rand_indices = sorted(self._rand_indices)

    def _manual_move(self):
        self.__set_embedding_coords()
        if self._rand_indices is not None:
            # save widget state
            selection = self.graph.selection
            valid_data = self.valid_data.copy()
            data = self.data.copy()
            ec = self._embedding_coords.copy()

            # plot subset
            self.__plot_random_subset(selection)

            # restore widget state
            self.graph.selection = selection
            self.valid_data = valid_data
            self.data = data
            self._embedding_coords = ec
        else:
            self.graph.update_coordinates()

    def __plot_random_subset(self, selection):
        self._embedding_coords = self._embedding_coords[self._rand_indices]
        self.data = self.data[self._rand_indices]
        self.valid_data = self.valid_data[self._rand_indices]
        self.graph.reset_graph()
        if selection is not None:
            self.graph.selection = selection[self._rand_indices]
            self.graph.update_selection_colors()

    def _finish_manual_move(self):
        if self._rand_indices is not None:
            selection = self.graph.selection
            self.graph.reset_graph()
            if selection is not None:
                self.graph.selection = selection
                self.graph.select_by_index(self.graph.get_selection())

    def __set_embedding_coords(self):
        points = self.graph.get_points()
        ex = np.dot(self._X, points)
        self._embedding_coords = (ex / np.max(np.linalg.norm(ex, axis=1)))

    def selection_changed(self):
        self.commit()

    def commit(self):
        selected = annotated = components = None
        if self.data is not None and self.valid_data is not None:
            name = self.data.name
            domain = self.data.domain
            metas = domain.metas + (self.variable_x, self.variable_y)
            domain = Domain(domain.attributes, domain.class_vars, metas)
            embedding_coords = np.zeros((len(self.data), 2), dtype=np.float)
            embedding_coords[self.valid_data] = self._embedding_coords

            data = self.data.transform(domain)
            data[:, self.variable_x] = embedding_coords[:, 0][:, None]
            data[:, self.variable_y] = embedding_coords[:, 1][:, None]

            selection = self.graph.get_selection()
            if len(selection):
                selected = data[selection]
                selected.name = name + ": selected"
                selected.attributes = self.data.attributes
            if self.graph.selection is not None and \
                    np.max(self.graph.selection) > 1:
                annotated = create_groups_table(data, self.graph.selection)
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
                X=self.graph.get_points().T,
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
            ("Color", name(self.attr_color)),
            ("Label", name(self.attr_label)),
            ("Shape", name(self.attr_shape)),
            ("Size", name(self.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and
             "{} %".format(self.graph.jitter_size))))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    @classmethod
    def migrate_settings(cls, _settings, version):
        if version < 3:
            if "radius" in _settings:
                _settings["graph"]["radius"] = _settings["radius"]

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


class MoveIndicator(pg.GraphicsObject):
    def __init__(self, x, y, scene_size, parent=None):
        super().__init__(parent)
        self.arrows = [
            pg.ArrowItem(pos=(x - scene_size * 0.07 * np.cos(np.radians(ang)),
                              y + scene_size * 0.07 * np.sin(np.radians(ang))),
                         parent=self, angle=ang,
                         headLen=13, tipAngle=45,
                         brush=pg.mkColor(128, 128, 128))
            for ang in (0, 90, 180, 270)]

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

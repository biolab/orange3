from enum import IntEnum
import sys

import numpy as np

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QRectF, QLineF, QTimer, QPoint,
    pyqtSignal as Signal, pyqtSlot as Slot
)
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QApplication

import pyqtgraph as pg

from Orange.data import Table, Domain, StringVariable
from Orange.projection.freeviz import FreeViz
from Orange.widgets import widget, gui, settings
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.visualize.utils.widget import OWAnchorProjectionWidget


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


class OWFreeVizGraph(OWGraphWithAnchors):
    hide_radius = settings.Setting(0)

    @property
    def scaled_radius(self):
        return self.hide_radius / 100 + 1e-5

    def update_radius(self):
        self.update_circle()
        self.update_anchors()

    def set_view_box_range(self):
        self.view_box.setRange(QRectF(-1.05, -1.05, 2.1, 2.1))

    def closest_draggable_item(self, pos):
        points, *_ = self.master.get_anchors()
        if points is None or not len(points):
            return None
        mask = np.linalg.norm(points, axis=1) > self.scaled_radius
        xi, yi = points[mask].T
        distances = (xi - pos.x()) ** 2 + (yi - pos.y()) ** 2
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF ** 2:
            return np.flatnonzero(mask)[np.argmin(distances)]
        return None

    def update_anchors(self):
        points, labels = self.master.get_anchors()
        if points is None:
            return
        r = self.scaled_radius
        if self.anchor_items is None:
            self.anchor_items = []
            for point, label in zip(points, labels):
                anchor = AnchorItem(line=QLineF(0, 0, *point), text=label)
                anchor.setVisible(np.linalg.norm(point) > r)
                anchor.setPen(pg.mkPen((100, 100, 100)))
                self.plot_widget.addItem(anchor)
                self.anchor_items.append(anchor)
        else:
            for anchor, point, label in zip(self.anchor_items, points, labels):
                anchor.setLine(QLineF(0, 0, *point))
                anchor.setText(label)
                anchor.setVisible(np.linalg.norm(point) > r)

    def update_circle(self):
        super().update_circle()
        if self.circle_item is not None:
            r = self.scaled_radius
            self.circle_item.setRect(QRectF(-r, -r, 2 * r, 2 * r))
            pen = pg.mkPen(QColor(Qt.lightGray), width=1, cosmetic=True)
            self.circle_item.setPen(pen)

    def _add_indicator_item(self, anchor_idx):
        x, y = self.anchor_items[anchor_idx].get_xy()
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self.indicator_item = MoveIndicator(x, y, 600 / dx)
        self.plot_widget.addItem(self.indicator_item)


class InitType(IntEnum):
    Circular, Random = 0, 1

    @staticmethod
    def items():
        return ["Circular", "Random"]


class OWFreeViz(OWAnchorProjectionWidget):
    MAX_ITERATIONS = 1000
    MAX_INSTANCES = 10000

    name = "FreeViz"
    description = "Displays FreeViz projection"
    icon = "icons/Freeviz.svg"
    priority = 240
    keywords = ["viz"]

    settings_version = 3
    initialization = settings.Setting(InitType.Circular)
    GRAPH_CLASS = OWFreeVizGraph
    graph = settings.SettingProvider(OWFreeVizGraph)
    embedding_variables_names = ("freeviz-x", "freeviz-y")

    class Error(OWAnchorProjectionWidget.Error):
        no_class_var = widget.Msg("Data has no target variable")
        not_enough_class_vars = widget.Msg(
            "Target variable is not at least binary")
        features_exceeds_instances = widget.Msg(
            "Number of features exceeds the number of instances.")
        too_many_data_instances = widget.Msg("Data is too large.")

    def __init__(self):
        super().__init__()
        self._X = None
        self._Y = None

        # FreeViz
        self._loop = AsyncUpdateLoop(parent=self)
        self._loop.yielded.connect(self.__set_projection)
        self._loop.finished.connect(self.__freeviz_finished)
        self._loop.raised.connect(self.__on_error)

    def _add_controls(self):
        self.__add_controls_start_box()
        super()._add_controls()
        self.graph.gui.add_control(
            self._effects_box, gui.hSlider, "Hide radius:", master=self.graph,
            value="hide_radius", minValue=0, maxValue=100, step=10,
            createLabel=False, callback=self.__radius_slider_changed
        )

    def __add_controls_start_box(self):
        box = gui.vBox(self.controlArea, box=True)
        gui.comboBox(
            box, self, "initialization", label="Initialization:",
            items=InitType.items(), orientation=Qt.Horizontal,
            labelWidth=90, callback=self.__init_combo_changed)
        self.btn_start = gui.button(
            box, self, "Optimize", self.__toggle_start, enabled=False)

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
        if self.data is None:
            return
        running = self._loop.isRunning()
        if running:
            self._loop.cancel()
        self.init_embedding_coords()
        self.graph.update_coordinates()
        if running:
            self._start()

    def _start(self):
        def update_freeviz(anchors):
            while True:
                _, projection, *_ = FreeViz.freeviz(
                    self._X, self._Y, scale=False, center=False,
                    initial=anchors, maxiter=10)
                yield projection
                if np.allclose(anchors, projection, rtol=1e-5, atol=1e-4):
                    return
                anchors = projection

        self.graph.set_sample_size(self.SAMPLE_SIZE)
        self._loop.setCoroutine(update_freeviz(self.projection))
        self.btn_start.setText("Stop")
        self.progressBarInit()
        self.setBlocking(True)
        self.setStatusMessage("Optimizing")

    def __set_projection(self, projection):
        # Set/update the projection matrix and coordinate embeddings
        self.progressBarAdvance(100. / self.MAX_ITERATIONS)
        self.projection = projection
        self.graph.update_coordinates()

    def __freeviz_finished(self):
        self.graph.set_sample_size(None)
        self.btn_start.setText("Optimize")
        self.setStatusMessage("")
        self.setBlocking(False)
        self.progressBarFinished()
        self.commit()

    def __on_error(self, err):
        sys.excepthook(type(err), err, getattr(err, "__traceback__"))

    def check_data(self):
        def error(err):
            err()
            self.data = None

        super().check_data()
        if self.data is not None:
            class_var = self.data.domain.class_var
            if class_var is None:
                error(self.Error.no_class_var)
            elif class_var.is_discrete and len(np.unique(self.data.Y)) < 2:
                error(self.Error.not_enough_class_vars)
            elif len(self.data.domain.attributes) < 2:
                error(self.Error.not_enough_features)
            elif len(self.data.domain.attributes) > self.data.X.shape[0]:
                error(self.Error.features_exceeds_instances)
            else:
                self.valid_data = np.all(np.isfinite(self.data.X), axis=1) & \
                                  np.isfinite(self.data.Y)
                n_valid = np.sum(self.valid_data)
                if n_valid > self.MAX_INSTANCES:
                    error(self.Error.too_many_data_instances)
                elif n_valid == 0:
                    error(self.Error.no_valid_data)
        self.btn_start.setEnabled(self.data is not None)

    def set_data(self, data):
        super().set_data(data)
        if self.data is not None:
            self.prepare_projection_data()
            self.init_embedding_coords()

    def prepare_projection_data(self):
        if not np.any(self.valid_data):
            self._X = self._Y = self.valid_data = None
            return

        self._X = self.data.X.copy()
        self._X -= np.nanmean(self._X, axis=0)
        span = np.ptp(self._X[self.valid_data], axis=0)
        self._X[:, span > 0] /= span[span > 0].reshape(1, -1)

        self._Y = self.data.Y
        if self.data.domain.class_var.is_discrete:
            self._Y = self._Y.astype(int)

    def init_embedding_coords(self):
        self.projection = FreeViz.init_radial(self._X.shape[1]) \
            if self.initialization == InitType.Circular \
            else FreeViz.init_random(self._X.shape[1], 2)

    def get_embedding(self):
        if self.data is None:
            return None
        embedding = np.dot(self._X, self.projection)
        embedding /= \
            np.max(np.linalg.norm(embedding[self.valid_data], axis=1)) or 1
        return embedding

    def get_anchors(self):
        if self.projection is None:
            return None, None
        return self.projection, [a.name for a in self.data.domain.attributes]

    def send_components(self):
        components = None
        if self.data is not None and self.valid_data is not None:
            meta_attrs = [StringVariable(name='component')]
            domain = Domain(self.data.domain.attributes, metas=meta_attrs)
            metas = np.array([["FreeViz 1"], ["FreeViz 2"]])
            components = Table(domain, self.projection.T, metas=metas)
            components.name = self.data.name
        self.Outputs.components.send(components)

    def clear(self):
        super().clear()
        self._loop.cancel()
        self._X = None
        self._Y = None

    @classmethod
    def migrate_settings(cls, _settings, version):
        if version < 3:
            if "radius" in _settings:
                _settings["graph"]["hide_radius"] = _settings["radius"]

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

    del w
    return r


if __name__ == "__main__":
    sys.exit(main())

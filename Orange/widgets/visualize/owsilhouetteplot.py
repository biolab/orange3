import sys
import itertools
import enum

from xml.sax.saxutils import escape
from types import SimpleNamespace as namespace

import numpy
import sklearn.metrics

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QEvent, QRectF, QSizeF, pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data
import Orange.distance

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.unsupervised.owhierarchicalclustering import \
    WrapperLayoutItem
from Orange.widgets.widget import Msg


class OWSilhouettePlot(widget.OWWidget):
    name = "Silhouette Plot"
    description = "Visually assess cluster quality and " \
                  "the degree of cluster membership."

    icon = "icons/SilhouettePlot.svg"
    priority = 300

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Selected Data", Orange.data.Table, widget.Default),
               ("Other Data", Orange.data.Table)]

    replaces = [
        "orangecontrib.prototypes.widgets.owsilhouetteplot.OWSilhouettePlot",
        "Orange.widgets.unsupervised.owsilhouetteplot.OWSilhouettePlot"
    ]

    settingsHandler = settings.PerfectDomainContextHandler()

    #: Distance metric index
    distance_idx = settings.Setting(0)
    #: Group/cluster variable index
    cluster_var_idx = settings.ContextSetting(0)
    #: Annotation variable index
    annotation_var_idx = settings.ContextSetting(0)
    #: Group the silhouettes by cluster
    group_by_cluster = settings.Setting(True)
    #: A fixed size for an instance bar
    bar_size = settings.Setting(3)
    #: Add silhouette scores to output data
    add_scores = settings.Setting(False)
    auto_commit = settings.Setting(False)

    Distances = [("Euclidean", Orange.distance.Euclidean),
                 ("Manhattan", Orange.distance.Manhattan)]

    graph_name = "scene"
    buttons_area_orientation = Qt.Vertical

    class Error(widget.OWWidget.Error):
        need_two_clusters = Msg("Need at least two non-empty clusters")

    def __init__(self):
        super().__init__()

        self.data = None
        self._effective_data = None
        self._matrix = None
        self._silhouette = None
        self._labels = None
        self._silplot = None

        gui.comboBox(
            self.controlArea, self, "distance_idx", box="Distance",
            items=[name for name, _ in OWSilhouettePlot.Distances],
            orientation=Qt.Horizontal, callback=self._invalidate_distances)

        box = gui.vBox(self.controlArea, "Cluster Label")
        self.cluster_var_cb = gui.comboBox(
            box, self, "cluster_var_idx", addSpace=4,
            callback=self._invalidate_scores)
        gui.checkBox(
            box, self, "group_by_cluster", "Group by cluster",
            callback=self._replot)
        self.cluster_var_model = itemmodels.VariableListModel(parent=self)
        self.cluster_var_cb.setModel(self.cluster_var_model)

        box = gui.vBox(self.controlArea, "Bars")
        gui.widgetLabel(box, "Bar width:")
        gui.hSlider(
            box, self, "bar_size", minValue=1, maxValue=10, step=1,
            callback=self._update_bar_size, addSpace=6)
        gui.widgetLabel(box, "Annotations:")
        self.annotation_cb = gui.comboBox(
            box, self, "annotation_var_idx", callback=self._update_annotations)
        self.annotation_var_model = itemmodels.VariableListModel(parent=self)
        self.annotation_var_model[:] = ["None"]
        self.annotation_cb.setModel(self.annotation_var_model)
        ibox = gui.indentedBox(box, 5)
        self.ann_hidden_warning = warning = gui.widgetLabel(
            ibox, "(increase the width to show)")
        ibox.setFixedWidth(ibox.sizeHint().width())
        warning.setVisible(False)

        gui.rubber(self.controlArea)

        gui.separator(self.buttonsArea)
        box = gui.vBox(self.buttonsArea, "Output")
        # Thunk the call to commit to call conditional commit
        gui.checkBox(box, self, "add_scores", "Add silhouette scores",
                     callback=lambda: self.commit())
        gui.auto_commit(
            box, self, "auto_commit", "Commit",
            auto_label="Auto commit", box=False)
        # Ensure that the controlArea is not narrower than buttonsArea
        self.controlArea.layout().addWidget(self.buttonsArea)

        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.mainArea.layout().addWidget(self.view)

    def sizeHint(self):
        sh = self.controlArea.sizeHint()
        return sh.expandedTo(QtCore.QSize(600, 720))

    @check_sql_input
    def set_data(self, data):
        """
        Set the input data set.
        """
        self.closeContext()
        self.clear()
        error_msg = ""
        warning_msg = ""
        candidatevars = []
        if data is not None:
            candidatevars = [
                v for v in data.domain.variables + data.domain.metas
                if v.is_discrete and len(v.values) >= 2]
            if not candidatevars:
                error_msg = "Input does not have any suitable cluster labels."
                data = None

        if data is not None:
            ncont = sum(v.is_continuous for v in data.domain.attributes)
            ndiscrete = len(data.domain.attributes) - ncont
            if ncont == 0:
                data = None
                error_msg = "No continuous columns"
            elif ncont < len(data.domain.attributes):
                warning_msg = "{0} discrete columns will not be used for " \
                              "distance computation".format(ndiscrete)

        self.data = data
        if data is not None:
            self.cluster_var_model[:] = candidatevars
            if data.domain.class_var in candidatevars:
                self.cluster_var_idx = \
                    candidatevars.index(data.domain.class_var)
            else:
                self.cluster_var_idx = 0

            annotvars = [var for var in data.domain.metas if var.is_string]
            self.annotation_var_model[:] = ["None"] + annotvars
            self.annotation_var_idx = 1 if len(annotvars) else 0
            self._effective_data = Orange.distance._preprocess(data)
            self.openContext(Orange.data.Domain(candidatevars))

        self.error(error_msg)
        self.warning(warning_msg)

    def handleNewSignals(self):
        if self._effective_data is not None:
            self._update()
            self._replot()

        self.unconditional_commit()

    def clear(self):
        """
        Clear the widget state.
        """
        self.data = None
        self._effective_data = None
        self._matrix = None
        self._silhouette = None
        self._labels = None
        self.cluster_var_model[:] = []
        self.annotation_var_model[:] = ["None"]
        self._clear_scene()

    def _clear_scene(self):
        # Clear the graphics scene and associated objects
        self.scene.clear()
        self.scene.setSceneRect(QRectF())
        self._silplot = None

    def _invalidate_distances(self):
        # Invalidate the computed distance matrix and recompute the silhouette.
        self._matrix = None
        self._invalidate_scores()

    def _invalidate_scores(self):
        # Invalidate and recompute the current silhouette scores.
        self._labels = self._silhouette = None
        self._update()
        self._replot()
        if self.data is not None:
            self.commit()

    def _update(self):
        # Update/recompute the distances/scores as required
        if self.data is None:
            self._silhouette = None
            self._labels = None
            self._matrix = None
            self._clear_scene()
            return

        if self._matrix is None and self._effective_data is not None:
            _, metric = self.Distances[self.distance_idx]
            self._matrix = numpy.asarray(metric(self._effective_data))

        labelvar = self.cluster_var_model[self.cluster_var_idx]
        labels, _ = self.data.get_column_view(labelvar)
        labels = labels.astype(int)
        _, counts = numpy.unique(labels, return_counts=True)
        if numpy.count_nonzero(counts) >= 2:
            self.Error.need_two_clusters.clear()
            silhouette = sklearn.metrics.silhouette_samples(
                self._matrix, labels, metric="precomputed")
        else:
            self.Error.need_two_clusters()
            labels = silhouette = None

        self._labels = labels
        self._silhouette = silhouette

    def _set_bar_height(self):
        visible = self.bar_size >= 5
        self._silplot.setBarHeight(self.bar_size)
        self._silplot.setRowNamesVisible(visible)
        self.ann_hidden_warning.setVisible(
            not visible and self.annotation_var_idx > 0)

    def _replot(self):
        # Clear and replot/initialize the scene
        self._clear_scene()
        if self._silhouette is not None and self._labels is not None:
            var = self.cluster_var_model[self.cluster_var_idx]
            self._silplot = silplot = SilhouettePlot()
            self._set_bar_height()

            if self.group_by_cluster:
                silplot.setScores(self._silhouette, self._labels, var.values)
            else:
                silplot.setScores(
                    self._silhouette,
                    numpy.zeros(len(self._silhouette), dtype=int),
                    [""]
                )

            self.scene.addItem(silplot)
            self._update_annotations()

            silplot.resize(silplot.effectiveSizeHint(Qt.PreferredSize))
            silplot.selectionChanged.connect(self.commit)

            self.scene.setSceneRect(
                QRectF(QtCore.QPointF(0, 0),
                       self._silplot.effectiveSizeHint(Qt.PreferredSize)))

    def _update_bar_size(self):
        if self._silplot is not None:
            self._set_bar_height()
            self.scene.setSceneRect(
                QRectF(QtCore.QPointF(0, 0),
                       self._silplot.effectiveSizeHint(Qt.PreferredSize)))

    def _update_annotations(self):
        if 0 < self.annotation_var_idx < len(self.annotation_var_model):
            annot_var = self.annotation_var_model[self.annotation_var_idx]
        else:
            annot_var = None
        self.ann_hidden_warning.setVisible(
            self.bar_size < 5 and annot_var is not None)

        if self._silplot is not None:
            if annot_var is not None:
                column, _ = self.data.get_column_view(annot_var)
                self._silplot.setRowNames(
                    [annot_var.str_val(value) for value in column])
            else:
                self._silplot.setRowNames(None)

    def commit(self):
        """
        Commit/send the current selection to the output.
        """
        selected = other = None
        if self.data is not None:
            selectedmask = numpy.full(len(self.data), False, dtype=bool)
            if self._silplot is not None:
                indices = self._silplot.selection()
                selectedmask[indices] = True
            scores = self._silhouette
            silhouette_var = None
            if self.add_scores:
                var = self.cluster_var_model[self.cluster_var_idx]
                silhouette_var = Orange.data.ContinuousVariable(
                    "Silhouette ({})".format(escape(var.name)))
                domain = Orange.data.Domain(
                    self.data.domain.attributes,
                    self.data.domain.class_vars,
                    self.data.domain.metas + (silhouette_var, ))
            else:
                domain = self.data.domain

            if numpy.count_nonzero(selectedmask):
                selected = self.data.from_table(
                    domain, self.data, numpy.flatnonzero(selectedmask))

            if numpy.count_nonzero(~selectedmask):
                other = self.data.from_table(
                    domain, self.data, numpy.flatnonzero(~selectedmask))

            if self.add_scores:
                if selected is not None:
                    selected[:, silhouette_var] = numpy.c_[scores[selectedmask]]
                if other is not None:
                    other[:, silhouette_var] = numpy.c_[scores[~selectedmask]]

        self.send("Selected Data", selected)
        self.send("Other Data", other)

    def send_report(self):
        if not len(self.cluster_var_model):
            return

        self.report_plot()
        caption = "Silhouette plot ({} distance), clustered by '{}'".format(
            self.Distances[self.distance_idx][0],
            self.cluster_var_model[self.cluster_var_idx])
        if self.annotation_var_idx and self._silplot.rowNamesVisible():
            caption += ", annotated with '{}'".format(
                self.annotation_var_model[self.annotation_var_idx])
        self.report_caption(caption)

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()


class SelectAction(enum.IntEnum):
    NoUpdate, Clear, Select, Deselect, Toogle, Current = 1, 2, 4, 8, 16, 32


class SilhouettePlot(QtGui.QGraphicsWidget):
    """
    A silhouette plot widget.
    """
    #: Emitted when the current selection has changed
    selectionChanged = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setAcceptHoverEvents(True)
        self.__groups = []
        self.__rowNamesVisible = True
        self.__barHeight = 3
        self.__selectionRect = None
        self.__selection = numpy.asarray([], dtype=int)
        self.__selstate = None
        self.__pen = QtGui.QPen(Qt.NoPen)
        self.__brush = QtGui.QBrush(QtGui.QColor("#3FCFCF"))
        self.__layout = QtGui.QGraphicsGridLayout()
        self.__hoveredItem = None
        self.setLayout(self.__layout)
        self.layout().setColumnSpacing(0, 1.)

    def setScores(self, scores, labels, values, rownames=None):
        """
        Set the silhouette scores/labels to for display.

        Arguments
        ---------
        scores : (N,) ndarray
            The silhouette scores.
        labels : (N,) ndarray
            A ndarray (dtype=int) of label/clusters indices.
        values : list of str
            A list of label/cluster names.
        rownames : list of str, optional
            A list (len == N) of row names.
        """
        scores = numpy.asarray(scores, dtype=float)
        labels = numpy.asarray(labels, dtype=int)
        if rownames is not None:
            rownames = numpy.asarray(rownames, dtype=object)

        if not scores.ndim == labels.ndim == 1:
            raise ValueError("scores and labels must be 1 dimensional")
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if rownames is not None and rownames.shape != scores.shape:
            raise ValueError("rownames must have the same size as scores")

        Ck = numpy.unique(labels)
        assert Ck[0] >= 0 and Ck[-1] < len(values)
        cluster_indices = [numpy.flatnonzero(labels == i)
                           for i in range(len(values))]
        cluster_indices = [indices[numpy.argsort(scores[indices])[::-1]]
                           for indices in cluster_indices]
        groups = [
            namespace(scores=scores[indices], indices=indices, label=label,
                      rownames=(rownames[indices] if rownames is not None
                                else None))
            for indices, label in zip(cluster_indices, values)
        ]
        self.clear()
        self.__groups = groups
        self.__setup()

    def setRowNames(self, names):
        if names is not None:
            names = numpy.asarray(names, dtype=object)

        layout = self.layout()

        font = self.font()
        font.setPixelSize(self.__barHeight)

        for i, grp in enumerate(self.__groups):
            grp.rownames = names[grp.indices] if names is not None else None
            item = layout.itemAt(i + 1, 3)

            if grp.rownames is not None:
                item.setItems(grp.rownames)
                item.setVisible(self.__rowNamesVisible)
            else:
                item.setItems([])
                item.setVisible(False)

            barplot = list(self.__plotItems())[i]
            baritems = barplot.items()

            if grp.rownames is None:
                tooltips = itertools.repeat("")
            else:
                tooltips = grp.rownames

            for bar, tooltip in zip(baritems, tooltips):
                bar.setToolTip(tooltip)

        self.layout().activate()

    def setRowNamesVisible(self, visible):
        if self.__rowNamesVisible != visible:
            self.__rowNamesVisible = visible
            for item in self.__textItems():
                item.setVisible(visible)

    def rowNamesVisible(self):
        return self.__rowNamesVisible

    def setBarHeight(self, height):
        """
        Set silhouette bar height (row height).
        """
        if height != self.__barHeight:
            self.__barHeight = height
            for item in self.__plotItems():
                item.setPreferredBarSize(height)
            font = self.font()
            font.setPixelSize(height)
            for item in self.__textItems():
                item.setFont(font)

    def barHeight(self):
        """
        Return the silhouette bar (row) height.
        """
        return self.__barHeight

    def clear(self):
        """
        Clear the widget state
        """
        scene = self.scene()
        for child in self.childItems():
            child.setParentItem(None)
            scene.removeItem(child)
        self.__groups = []

    def __setup(self):
        # Setup the subwidgets/groups/layout
        smax = max((numpy.max(g.scores) for g in self.__groups
                    if g.scores.size),
                   default=1)

        smin = min((numpy.min(g.scores) for g in self.__groups
                    if g.scores.size),
                   default=-1)
        smin = min(smin, 0)

        font = self.font()
        font.setPixelSize(self.__barHeight)
        axispen = QtGui.QPen(Qt.black)

        ax = pg.AxisItem(parent=self, orientation="top", maxTickLength=7,
                         pen=axispen)
        ax.setRange(smin, smax)
        self.layout().addItem(ax, 0, 2)

        for i, group in enumerate(self.__groups):
            silhouettegroup = BarPlotItem(parent=self)
            silhouettegroup.setBrush(self.__brush)
            silhouettegroup.setPen(self.__pen)
            silhouettegroup.setDataRange(smin, smax)
            silhouettegroup.setPlotData(group.scores)
            silhouettegroup.setPreferredBarSize(self.__barHeight)
            silhouettegroup.setData(0, group.indices)
            self.layout().addItem(silhouettegroup, i + 1, 2)

            if group.label:
                line = QtGui.QFrame(frameShape=QtGui.QFrame.VLine)
                proxy = QtGui.QGraphicsProxyWidget(self)
                proxy.setWidget(line)
                self.layout().addItem(proxy, i + 1, 1)
                label = QtGui.QGraphicsSimpleTextItem(self)
                label.setText("{} ({})".format(escape(group.label),
                                               len(group.scores)))
                item = WrapperLayoutItem(label, Qt.Vertical, parent=self)
                self.layout().addItem(item, i + 1, 0, Qt.AlignCenter)

            textlist = TextListWidget(self, font=font)
            sp = textlist.sizePolicy()
            sp.setVerticalPolicy(QtGui.QSizePolicy.Ignored)
            textlist.setSizePolicy(sp)
            textlist.setParent(self)
            if group.rownames is not None:
                textlist.setItems(group.items)
                textlist.setVisible(self.__rowNamesVisible)
            else:
                textlist.setVisible(False)

            self.layout().addItem(textlist, i + 1, 3)

        ax = pg.AxisItem(parent=self, orientation="bottom", maxTickLength=7,
                         pen=axispen)
        ax.setRange(smin, smax)
        self.layout().addItem(ax, len(self.__groups) + 1, 2)

    def __updateTextSizeConstraint(self):
        # set/update fixed height constraint on the text annotation items so
        # it matches the silhouette's height
        for silitem, textitem in zip(self.__plotItems(), self.__textItems()):
            height = silitem.effectiveSizeHint(Qt.PreferredSize).height()
            textitem.setMaximumHeight(height)
            textitem.setMinimumHeight(height)

    def event(self, event):
        # Reimplemented
        if event.type() == QEvent.LayoutRequest and \
                self.parentLayoutItem() is None:
            self.__updateTextSizeConstraint()
            self.resize(self.effectiveSizeHint(Qt.PreferredSize))
        return super().event(event)

    def __setHoveredItem(self, item):
        # Set the current hovered `item` (:class:`QGraphicsRectItem`)
        if self.__hoveredItem is not item:
            if self.__hoveredItem is not None:
                self.__hoveredItem.setPen(QtGui.QPen(Qt.NoPen))
            self.__hoveredItem = item
            if item is not None:
                item.setPen(QtGui.QPen(Qt.lightGray))

    def hoverEnterEvent(self, event):
        # Reimplemented
        event.accept()

    def hoverMoveEvent(self, event):
        # Reimplemented
        event.accept()
        item = self.itemAtPos(event.pos())
        self.__setHoveredItem(item)

    def hoverLeaveEvent(self, event):
        # Reimplemented
        self.__setHoveredItem(None)
        event.accept()

    def mousePressEvent(self, event):
        # Reimplemented
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                saction = SelectAction.Toogle
            elif event.modifiers() & Qt.AltModifier:
                saction = SelectAction.Deselect
            elif event.modifiers() & Qt.ShiftModifier:
                saction = SelectAction.Select
            else:
                saction = SelectAction.Clear | SelectAction.Select
            self.__selstate = namespace(
                modifiers=event.modifiers(),
                selection=self.__selection,
                action=saction,
                rect=None,
            )
            if saction & SelectAction.Clear:
                self.__selstate.selection = numpy.array([], dtype=int)
                self.setSelection(self.__selstate.selection)
            event.accept()

    def mouseMoveEvent(self, event):
        # Reimplemented
        if event.buttons() & Qt.LeftButton:
            assert self.__selstate is not None
            if self.__selectionRect is None:
                self.__selectionRect = QtGui.QGraphicsRectItem(self)

            rect = (QRectF(event.buttonDownPos(Qt.LeftButton),
                           event.pos()).normalized())

            if not rect.width():
                rect = rect.adjusted(-1e-7, -1e-7, 1e-7, 1e-7)

            rect = rect.intersected(self.contentsRect())
            self.__selectionRect.setRect(rect)
            self.__selstate.rect = rect
            self.__selstate.action |= SelectAction.Current

            self.__setSelectionRect(rect, self.__selstate.action)
            event.accept()

    def mouseReleaseEvent(self, event):
        # Reimplemented
        if event.button() == Qt.LeftButton:
            if self.__selectionRect is not None:
                self.__selectionRect.setParentItem(None)
                if self.scene() is not None:
                    self.scene().removeItem(self.__selectionRect)
                self.__selectionRect = None
            event.accept()

            rect = (QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                    .normalized())

            if not rect.isValid():
                rect = rect.adjusted(-1e-7, -1e-7, 1e-7, 1e-7)

            rect = rect.intersected(self.contentsRect())
            action = action = self.__selstate.action & ~SelectAction.Current
            self.__setSelectionRect(rect, action)
            self.__selstate = None

    def __setSelectionRect(self, rect, action):
        # Set the current mouse drag selection rectangle
        if not rect.isValid():
            rect = rect.adjusted(-0.01, -0.01, 0.01, 0.01)

        rect = rect.intersected(self.contentsRect())

        indices = self.__selectionIndices(rect)

        if action & SelectAction.Clear:
            selection = []
        elif self.__selstate is not None:
            # Mouse drag selection is in progress. Update only the current
            # selection
            selection = self.__selstate.selection
        else:
            selection = self.__selection

        if action & SelectAction.Toogle:
            selection = numpy.setxor1d(selection, indices)
        elif action & SelectAction.Deselect:
            selection = numpy.setdiff1d(selection, indices)
        elif action & SelectAction.Select:
            selection = numpy.union1d(selection, indices)

        self.setSelection(selection)

    def __selectionIndices(self, rect):
        items = [item for item in self.__plotItems()
                 if item.geometry().intersects(rect)]
        selection = [numpy.array([], dtype=int)]
        for item in items:
            indices = item.data(0)
            itemrect = item.geometry().intersected(rect)
            crect = item.contentsRect()
            itemrect = (item.mapFromParent(itemrect).boundingRect()
                        .intersected(crect))
            assert itemrect.top() >= 0
            rowh = crect.height() / item.count()
            indextop = numpy.floor(itemrect.top() / rowh)
            indexbottom = numpy.ceil(itemrect.bottom() / rowh)
            selection.append(indices[int(indextop): int(indexbottom)])
        return numpy.hstack(selection)

    def itemAtPos(self, pos):
        items = [item for item in self.__plotItems()
                 if item.geometry().contains(pos)]
        if not items:
            return None
        else:
            item = items[0]
        crect = item.contentsRect()
        pos = item.mapFromParent(pos)
        if not crect.contains(pos):
            return None

        assert pos.x() >= 0
        rowh = crect.height() / item.count()
        index = int(numpy.floor(pos.y() / rowh))
        index = min(index, item.count() - 1)
        if index >= 0:
            return item.items()[index]
        else:
            return None

    def indexAtPos(self, pos):
        items = [item for item in self.__plotItems()
                 if item.geometry().contains(pos)]
        if not items:
            return -1
        else:
            item = items[0]
        indices = item.data(0)
        assert (isinstance(indices, numpy.ndarray) and
                indices.shape == (item.count(),))
        crect = item.contentsRect()
        pos = item.mapFromParent(pos)
        if not crect.contains(pos):
            return -1

        assert pos.x() >= 0
        rowh = crect.height() / item.count()
        index = numpy.floor(pos.y() / rowh)
        index = min(index, indices.size - 1)

        if index >= 0:
            return indices[index]
        else:
            return -1

    def __selectionChanged(self, selected, deselected):
        for item, grp in zip(self.__plotItems(), self.__groups):
            select = numpy.flatnonzero(
                numpy.in1d(grp.indices, selected, assume_unique=True))
            items = item.items()
            if select.size:
                for i in select:
                    items[i].setBrush(Qt.red)

            deselect = numpy.flatnonzero(
                numpy.in1d(grp.indices, deselected, assume_unique=True))
            if deselect.size:
                for i in deselect:
                    items[i].setBrush(self.__brush)

    def __plotItems(self):
        for i in range(len(self.__groups)):
            item = self.layout().itemAt(i + 1, 2)
            if item is not None:
                assert isinstance(item, BarPlotItem)
                yield item

    def __textItems(self):
        for i in range(len(self.__groups)):
            item = self.layout().itemAt(i + 1, 3)
            if item is not None:
                assert isinstance(item, TextListWidget)
                yield item

    def setSelection(self, indices):
        indices = numpy.unique(numpy.asarray(indices, dtype=int))
        select = numpy.setdiff1d(indices, self.__selection)
        deselect = numpy.setdiff1d(self.__selection, indices)

        self.__selectionChanged(select, deselect)

        self.__selection = indices

        if deselect.size or select.size:
            self.selectionChanged.emit()

    def selection(self):
        return numpy.asarray(self.__selection, dtype=int)


class BarPlotItem(QtGui.QGraphicsWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__barsize = 5
        self.__spacing = 1
        self.__pen = QtGui.QPen(Qt.NoPen)
        self.__brush = QtGui.QBrush(QtGui.QColor("#3FCFCF"))
        self.__range = (0., 1.)
        self.__data = numpy.array([], dtype=float)
        self.__items = []

    def count(self):
        return self.__data.size

    def items(self):
        return list(self.__items)

    def setGeometry(self, geom):
        super().setGeometry(geom)
        self.__layout()

    def event(self, event):
        if event.type() == QEvent.GraphicsSceneResize:
            self.__layout()
        return super().event(event)

    def sizeHint(self, which, constraint=QSizeF()):
        spacing = max(self.__spacing * (self.count() - 1), 0)
        return QSizeF(300, self.__barsize * self.count() + spacing)

    def setPreferredBarSize(self, size):
        if self.__barsize != size:
            self.__barsize = size
            self.updateGeometry()

    def spacing(self):
        return self.__spacing

    def setSpacing(self, spacing):
        if self.__spacing != spacing:
            self.__spacing = spacing
            self.updateGeometry()

    def setPen(self, pen):
        pen = QtGui.QPen(pen)
        if self.__pen != pen:
            self.__pen = pen
            for item in self.__items:
                item.setPen(pen)

    def pen(self):
        return QtGui.QPen(self.__pen)

    def setBrush(self, brush):
        brush = QtGui.QBrush(brush)
        if self.__brush != brush:
            self.__brush = brush
            for item in self.__items:
                item.setBrush(brush)

    def brush(self):
        return QtGui.QBrush(self.__brush)

    def setPlotData(self, values):
        self.__data = numpy.array(values, copy=True)
        self.__update()
        self.updateGeometry()

    def setDataRange(self, rangemin, rangemax):
        if self.__range != (rangemin, rangemax):
            self.__range = (rangemin, rangemax)
            self.__layout()

    def __clear(self):
        for item in self.__items:
            item.setParentItem(None)
        scene = self.scene()
        if scene is not None:
            for item in self.__items:
                scene.removeItem(item)
        self.__items = []

    def __update(self):
        self.__clear()

        pen = self.pen()
        brush = self.brush()
        for i in range(self.count()):
            item = QtGui.QGraphicsRectItem(self)
            item.setPen(pen)
            item.setBrush(brush)
            self.__items.append(item)

        self.__layout()

    def __layout(self):
        (N, ) = self.__data.shape
        if not N:
            return

        spacing = self.__spacing
        rect = self.contentsRect()
        w = rect.width()
        if rect.height() - (spacing * (N - 1)) <= 0:
            spacing = 0

        h = (rect.height() - (spacing * (N - 1))) / N
        xmin, xmax = self.__range
        span = xmax - xmin
        if span < 1e-9:
            span = 1
        scalef = w * 1 / span

        base = 0
        base = (base - xmin) * scalef
        datascaled = (self.__data - xmin) * scalef

        for i, (v, item) in enumerate(zip(datascaled, self.__items)):
            item.setRect(QRectF(base, rect.top() + i * (h + spacing),
                                v - base, h).normalized())


from Orange.widgets.visualize.owheatmap import scaled


class TextListWidget(QtGui.QGraphicsWidget):
    def __init__(self, parent=None, items=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setFlag(QtGui.QGraphicsWidget.ItemClipsChildrenToShape, True)
        self.__items = []
        self.__textitems = []
        self.__group = None
        self.__spacing = 0

        sp = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                               QtGui.QSizePolicy.Preferred)
        sp.setWidthForHeight(True)
        self.setSizePolicy(sp)

        if items is not None:
            self.setItems(items)

    def setItems(self, items):
        self.__clear()
        self.__items = list(items)
        self.__setup()
        self.__layout()
        self.updateGeometry()

    def clear(self):
        self.__clear()
        self.__items = []
        self.updateGeometry()

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            sh = self.__naturalsh()
            if 0 < constraint.height() < sh.height():
                sh = scaled(sh, constraint, Qt.KeepAspectRatioByExpanding)
            return sh

        return super().sizeHint(which, constraint)

    def __naturalsh(self):
        fm = QtGui.QFontMetrics(self.font())
        spacing = self.__spacing
        N = len(self.__items)
        width = max((fm.width(text) for text in self.__items),
                    default=0)
        height = N * fm.height() + (N - 1) * spacing
        return QSizeF(width, height)

    def event(self, event):
        if event.type() == QEvent.LayoutRequest:
            self.__layout()
        elif event.type() == QEvent.GraphicsSceneResize:
            self.__layout()

        return super().event(event)

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.updateGeometry()
            font = self.font()
            for item in self.__textitems:
                item.setFont(font)

    def __setup(self):
        self.__clear()
        font = self.font()
        group = QtGui.QGraphicsItemGroup(self)

        for text in self.__items:
            t = QtGui.QGraphicsSimpleTextItem(text, group)
            t.setData(0, text)
            t.setFont(font)
            t.setToolTip(text)
            self.__textitems.append(t)

    def __layout(self):
        crect = self.contentsRect()
        spacing = self.__spacing
        N = len(self.__items)

        if not N:
            return

        fm = QtGui.QFontMetrics(self.font())
        naturalheight = fm.height()
        th = (crect.height() - (N - 1) * spacing) / N
        if th > naturalheight and N > 1:
            th = naturalheight
            spacing = (crect.height() - N * th) / (N - 1)

        for i, item in enumerate(self.__textitems):
            item.setPos(crect.left(), crect.top() + i * (th + spacing))

    def __clear(self):
        def remove(items, scene):
            for item in items:
                item.setParentItem(None)
                if scene is not None:
                    scene.removeItem(item)

        remove(self.__textitems, self.scene())
        if self.__group is not None:
            remove([self.__group], self.scene())

        self.__textitems = []


def main(argv=sys.argv):
    app = QtGui.QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"
    w = OWSilhouettePlot()
    w.show()
    w.raise_()
    w.set_data(Orange.data.Table(filename))
    w.handleNewSignals()
    app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

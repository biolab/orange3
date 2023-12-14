from itertools import islice, permutations, chain, combinations
from math import factorial, comb
import warnings

import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from AnyQt.QtGui import QColor, QPalette
from AnyQt.QtCore import Qt, QRectF, QPoint

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem

from Orange.data import Table, Domain, IsDefined
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import RadViz
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, SettingProvider
from Orange.widgets.utils.plot.owplotgui import VariableSelectionModel, \
    variables_selection
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils import vizrank
from Orange.widgets.visualize.utils.vizrank import VizRankDialogNAttrs, \
    VizRankMixin
from Orange.widgets.visualize.utils.component import OWGraphWithAnchors
from Orange.widgets.visualize.utils.plotutils import TextItem
from Orange.widgets.visualize.utils.widget import OWAnchorProjectionWidget


MAX_DISPLAYED_VARS = 20
MAX_LABEL_LEN = 16


class RadvizVizRank(VizRankDialogNAttrs):
    minK = 10

    def __init__(self, parent, data, attributes, color, n_attrs):
        super().__init__(parent, data, attributes, color, n_attrs,
                         spin_label="Maximum number of variables: ")

    def score_attributes(self):
        attrs = [v for v in self.attrs if v is not self.attr_color]
        data = self.data.transform(Domain(attrs, self.attr_color))
        relief = ReliefF if self.attr_color.is_discrete else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        attrs = sorted(zip(weights, attrs), key=lambda x: (-x[0], x[1].name))
        return [a for _, a in attrs]

    def state_count(self):
        n_all_attrs = self.max_attrs()
        if not n_all_attrs:
            return 0
        return sum(comb(n_all_attrs, n_attrs) * factorial(n_attrs - 1) // 2
                   for n_attrs in range(3, self.n_attrs + 1))

    def state_generator(self):
        return (
            (c[0], *p)
            for k in range(3, self.n_attrs + 1)
            for c in combinations(list(range(len(self.attr_order))), k)
            for p in islice(permutations(c[1:]), factorial(len(c) - 1) // 2))

    def compute_score(self, state):
        attrs = [self.attr_order[i] for i in state]
        domain = Domain(attributes=attrs, class_vars=[self.attr_color])
        data = self.data.transform(domain)
        valid_data = IsDefined()(data)
        projector = RadViz()
        projection = projector(valid_data)
        radviz_xy = projection(valid_data).X
        y = projector.preprocess(valid_data).Y

        neigh = (KNeighborsClassifier if self.attr_color.is_discrete else
                 KNeighborsRegressor)(n_neighbors=3)
        neigh.fit(radviz_xy, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scores = cross_val_score(neigh, radviz_xy, y, cv=3)
        return -scores.mean() * len(valid_data) / len(data)


class OWRadvizGraph(OWGraphWithAnchors):
    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.anchors_scatter_item = None
        self.padding = 0.025

    def clear(self):
        super().clear()
        self.anchors_scatter_item = None

    def set_view_box_range(self):
        self.view_box.setRange(QRectF(-1, -1, 2, 2), padding=self.padding)

    def closest_draggable_item(self, pos):
        points, _ = self.master.get_anchors()
        if points is None:
            return None
        np_pos = np.array([[pos.x(), pos.y()]])
        distances = distance.cdist(np_pos, points[:, :2])[0]
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF:
            return np.argmin(distances)
        return None

    def update_anchors(self):
        points, labels = self.master.get_anchors()
        if points is None:
            return
        if self.anchor_items is not None:
            for anchor in self.anchor_items:
                self.plot_widget.removeItem(anchor)

        self.anchor_items = []
        label_len = 1
        foreground = self.plot_widget.palette().color(QPalette.Text)
        for point, label in zip(points, labels):
            anchor = TextItem(color=foreground)
            anchor.textItem.setToolTip(f"<b>{label}</b>")

            if len(label) > MAX_LABEL_LEN:
                i = label.rfind(" ", 0, MAX_LABEL_LEN)
                if i != -1:
                    first_row = label[:i] + "\n"
                    second_row = label[i + 1:]
                    if len(second_row) > MAX_LABEL_LEN:
                        j = second_row.rfind(" ", 0, MAX_LABEL_LEN)
                        if j != -1:
                            second_row = second_row[:j + 1] + "..."
                        else:
                            second_row = second_row[:MAX_LABEL_LEN - 3] + "..."
                    label = first_row + second_row
                else:
                    label = label[:MAX_LABEL_LEN - 3] + "..."

            anchor.setText(label)
            anchor.setFont(self.parameter_setter.anchor_font)
            label_len = min(MAX_LABEL_LEN, len(label))

            x, y = point
            angle = np.rad2deg(np.arctan2(y, x))
            anchor.setPos(x * 1.025, y * 1.025)

            if abs(angle) < 90:
                anchor.setAngle(angle)
                anchor.setAnchor((0, 0.5))
            else:
                anchor.setAngle(angle + 180)
                anchor.setAnchor((1, 0.5))

                anchor.textItem.setTextWidth(anchor.textItem.boundingRect().width())
                option = anchor.textItem.document().defaultTextOption()
                option.setAlignment(Qt.AlignRight)
                anchor.textItem.document().setDefaultTextOption(option)

            self.plot_widget.addItem(anchor)
            self.anchor_items.append(anchor)

        self.padding = label_len * 0.0175
        self._update_anchors_scatter_item(points)

    def _update_anchors_scatter_item(self, points):
        if self.anchors_scatter_item is not None:
            self.plot_widget.removeItem(self.anchors_scatter_item)
            self.anchors_scatter_item = None
        self.anchors_scatter_item = ScatterPlotItem(x=points[:, 0],
                                                    y=points[:, 1])
        self.plot_widget.addItem(self.anchors_scatter_item)

    def _add_indicator_item(self, anchor_idx):
        if anchor_idx is None:
            return
        x, y = self.anchor_items[anchor_idx].get_xy()
        col = self.view_box.mouse_state
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self.indicator_item = MoveIndicator(np.arctan2(y, x), col, 6000 / dx)
        self.plot_widget.addItem(self.indicator_item)


class OWRadviz(OWAnchorProjectionWidget, VizRankMixin(RadvizVizRank)):
    name = "Radviz"
    description = "Display Radviz projection"
    icon = "icons/Radviz.svg"
    priority = 241
    keywords = "radviz, viz"

    settings_version = 3

    selected_vars = ContextSetting([])
    GRAPH_CLASS = OWRadvizGraph
    graph = SettingProvider(OWRadvizGraph)
    n_attrs_vizrank = Setting(3)

    class Warning(OWAnchorProjectionWidget.Warning):
        removed_vars = widget.Msg(
            "Categorical variables with more than two values are not shown.")
        max_vars_selected = widget.Msg(
            "Maximum number of selected variables reached.")

    def __init__(self):
        VizRankMixin.__init__(self)  # pylint: disable=non-parent-init-called
        OWAnchorProjectionWidget.__init__(self)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, box="Features")
        self.model_selected = VariableSelectionModel(self.selected_vars,
                                                     max_vars=20)
        variables_selection(box, self, self.model_selected)
        self.model_selected.selection_changed.connect(
            self.__model_selected_changed)
        box.layout().addWidget(self.vizrank_button("Suggest features"))
        self.vizrankSelectionChanged.connect(self.vizrank_set_attrs)
        self.vizrankRunStateChanged.connect(self.store_vizrank_n_attrs)
        super()._add_controls()

    def _add_buttons(self):
        self.gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    @property
    def primitive_variables(self):
        if self.data is None or self.data.domain is None:
            return []
        dom = self.data.domain
        return [v for v in chain(dom.variables, dom.metas)
                if v.is_continuous or v.is_discrete and len(v.values) == 2]

    @property
    def effective_variables(self):
        return self.selected_vars

    @property
    def effective_data(self):
        return self.data.transform(Domain(self.effective_variables))

    def store_vizrank_n_attrs(self, state, data):
        if state == vizrank.RunState.Running:
            self.n_attrs_vizrank = data["n_attrs"]

    def vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.selected_vars[:] = attrs
        # Ugly, but the alternative is to have yet another signal to which
        # the view will have to connect
        self.model_selected.selection_changed.emit()

    def __model_selected_changed(self):
        if self.model_selected.is_full():
            self.Warning.max_vars_selected()
        else:
            self.Warning.max_vars_selected.clear()
        self.init_projection()
        self.setup_plot()
        self.commit.deferred()

    def colors_changed(self):
        super().colors_changed()
        self.init_vizrank()

    @OWAnchorProjectionWidget.Inputs.data
    def set_data(self, data):
        super().set_data(data)
        self.init_vizrank()
        self.init_projection()

    def init_vizrank(self):
        msgerr = ""
        if self.data is None:
            msgerr = "No data"
        elif len(self.primitive_variables) <= 3:
            msgerr = "Not enough variables"
        elif self.attr_color is None:
            msgerr = "Color is not set."
        elif np.isnan(self.data.get_column(self.attr_color)).all():
            msgerr = "No rows with defined color variable"
        elif np.sum(np.all(np.isfinite(self.data.X), axis=1)) <= 1:
            msgerr = "Not enough rows without missing data"
        elif not np.all(np.nan_to_num(np.nanstd(self.data.X, 0)) != 0):
            msgerr = "Constant data"

        if not msgerr:
            super().init_vizrank(
                self.data, self.primitive_variables, self.attr_color,
                self.n_attrs_vizrank)
        else:
            self.disable_vizrank(msgerr)

    def check_data(self):
        super().check_data()
        if self.data is not None:
            domain = self.data.domain
            vars_ = chain(domain.variables, domain.metas)
            n_vars = sum(v.is_primitive() for v in vars_)
            if len(self.primitive_variables) < n_vars:
                self.Warning.removed_vars()

    def init_attr_values(self):
        super().init_attr_values()
        self.selected_vars[:] = self.primitive_variables[:5]
        self.model_selected[:] = self.primitive_variables

    def _manual_move(self, anchor_idx, x, y):
        angle = np.arctan2(y, x)
        super()._manual_move(anchor_idx, np.cos(angle), np.sin(angle))

    def _send_components_x(self):
        components_ = super()._send_components_x()
        angle = np.arctan2(*components_[::-1])
        return np.row_stack((components_, angle))

    def _send_components_metas(self):
        return np.vstack((super()._send_components_metas(), ["angle"]))

    def clear(self):
        super().clear()
        self.projector = RadViz()

    @classmethod
    def migrate_context(cls, context, version):
        values = context.values
        if version < 2:
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]
        if version < 3 and "selected_vars" in values:
            values["selected_vars"] = (values["selected_vars"], -3)


class MoveIndicator(pg.GraphicsObject):
    def __init__(self, angle, col, dangle=5, parent=None):
        super().__init__(parent)
        color = QColor(0, 0, 0) if col else QColor(128, 128, 128)
        angle_d = np.rad2deg(angle)
        angle_2 = 90 - angle_d - dangle
        angle_1 = 270 - angle_d + dangle
        dangle = np.deg2rad(dangle)
        arrow1 = pg.ArrowItem(
            parent=self, angle=angle_1, brush=color, pen=pg.mkPen(color)
        )
        arrow1.setPos(np.cos(angle - dangle), np.sin(angle - dangle))
        arrow2 = pg.ArrowItem(
            parent=self, angle=angle_2, brush=color, pen=pg.mkPen(color)
        )
        arrow2.setPos(np.cos(angle + dangle), np.sin(angle + dangle))
        arc_x = np.fromfunction(
            lambda i: np.cos((angle - dangle) + (2 * dangle) * i / 120.),
            (121,), dtype=int
        )
        arc_y = np.fromfunction(
            lambda i: np.sin((angle - dangle) + (2 * dangle) * i / 120.),
            (121,), dtype=int
        )
        pg.PlotCurveItem(
            parent=self, x=arc_x, y=arc_y, pen=pg.mkPen(color), antialias=False
        )

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()


if __name__ == "__main__":  # pragma: no cover
    brown = Table("brown-selected")
    WidgetPreview(OWRadviz).run(set_data=brown, set_subset_data=brown[::10])

import warnings
from collections import namedtuple, OrderedDict

import numpy as np

from AnyQt.QtWidgets import QListView, QApplication, QSizePolicy
from AnyQt.QtGui import QBrush, QColor, QPainter, QPalette
from AnyQt.QtCore import QEvent, Qt

import pyqtgraph as pg

from orangewidget.utils.listview import ListViewSearch

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.statistics import contingency
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalettes
from Orange.widgets.utils.itemmodels import select_rows
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.plotutils import PlotWidget
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting


class ScatterPlotItem(pg.ScatterPlotItem):
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)
        super().paint(painter, option, widget)


class OWCorrespondenceAnalysis(widget.OWWidget):
    name = "Correspondence Analysis"
    description = "Correspondence analysis for categorical multivariate data."
    icon = "icons/CorrespondenceAnalysis.svg"
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        coordinates = Output("Coordinates", Table)

    Invalidate = QEvent.registerEventType()

    settingsHandler = settings.DomainContextHandler()

    selected_var_indices = settings.ContextSetting([])
    auto_commit = Setting(True)

    graph_name = "plot.plotItem"

    class Error(widget.OWWidget.Error):
        empty_data = widget.Msg("Empty dataset")
        no_disc_vars = widget.Msg("No categorical data")

    def __init__(self):
        super().__init__()

        self.data = None
        self.component_x = 0
        self.component_y = 1

        box = gui.vBox(self.controlArea, "Variables")
        self.varlist = itemmodels.VariableListModel()
        self.varview = view = ListViewSearch(
            selectionMode=QListView.MultiSelection,
            uniformItemSizes=True
        )
        view.setModel(self.varlist)
        view.selectionModel().selectionChanged.connect(self._var_changed)

        box.layout().addWidget(view)

        axes_box = gui.vBox(self.controlArea, "Axes")
        self.axis_x_cb = gui.comboBox(
            axes_box, self, "component_x", label="X:",
            callback=self._component_changed, orientation=Qt.Horizontal,
            sizePolicy=(QSizePolicy.MinimumExpanding,
                        QSizePolicy.Preferred)
        )

        self.axis_y_cb = gui.comboBox(
            axes_box, self, "component_y", label="Y:",
            callback=self._component_changed, orientation=Qt.Horizontal,
            sizePolicy=(QSizePolicy.MinimumExpanding,
                        QSizePolicy.Preferred)
        )

        self.infotext = gui.widgetLabel(
            gui.vBox(self.controlArea, "Contribution to Inertia"), "\n"
        )

        gui.auto_send(self.buttonsArea, self, "auto_commit")

        self.plot = PlotWidget()
        self.plot.setMenuEnabled(False)
        self.mainArea.layout().addWidget(self.plot)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.Error.clear()

        if data is not None and not len(data):
            self.Error.empty_data()
            data = None

        self.data = data
        if data is not None:
            self.varlist[:] = [var for var in data.domain.variables
                               if var.is_discrete]
            if not len(self.varlist[:]):
                self.Error.no_disc_vars()
                self.data = None
            else:
                self.selected_var_indices = [0, 1][:len(self.varlist)]
                # This widget's update flow is broken in many ways, starting
                # from using context domain handler without having any valid
                # context settings. Getting rid of these warnings would require
                # rewriting large portins; @ales-erjavec is doing it and will
                # finish it eventually, so let us these warnings are
                # uninformative and would better be silenced.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "combo box 'component_[xy]' .*", UserWarning)
                    self.component_x = 0
                    self.component_y = int(len(self.varlist[self.selected_var_indices[-1]].values) > 1)
                self.openContext(data)
                self._restore_selection()
        self._update_CA()
        self.commit.now()

    @gui.deferred
    def commit(self):
        output_table = None
        if self.ca is not None:
            sel_vars = self.selected_vars()
            if len(sel_vars) == 2:
                rf = np.vstack((self.ca.row_factors, self.ca.col_factors))
            else:
                rf = self.ca.row_factors
            vars_data = [(val.name, var) for val in sel_vars for var in val.values]
            output_table = Table(
                Domain([ContinuousVariable(f"Component {i + 1}")
                        for i in range(rf.shape[1])],
                       metas=[StringVariable("Variable"),
                              StringVariable("Value")]),
                rf, metas=vars_data
            )
        self.Outputs.coordinates.send(output_table)

    def clear(self):
        self.data = None
        self.ca = None
        self.plot.clear()
        self.varlist[:] = []

    def selected_vars(self):
        rows = sorted(
            ind.row() for ind in self.varview.selectionModel().selectedRows())
        return [self.varlist[i] for i in rows]

    def _restore_selection(self):
        def restore(view, indices):
            with itemmodels.signal_blocking(view.selectionModel()):
                select_rows(view, indices)
        restore(self.varview, self.selected_var_indices)

    def _p_axes(self):
        return self.component_x, self.component_y

    def _var_changed(self):
        self.selected_var_indices = sorted(
            ind.row() for ind in self.varview.selectionModel().selectedRows())
        rfs = self.update_XY()
        if rfs is not None:
            if self.component_x >= rfs:
                self.component_x = rfs-1
            if self.component_y >= rfs:
                self.component_y = rfs-1
        self._invalidate()

    def _component_changed(self):
        if self.ca is not None:
            self._setup_plot()
            self._update_info()

    def _invalidate(self):
        self.__invalidated = True
        QApplication.postEvent(self, QEvent(self.Invalidate))

    def customEvent(self, event):
        if event.type() == self.Invalidate:
            self.ca = None
            self.plot.clear()
            self._update_CA()
            self.commit.deferred()
            return
        return super().customEvent(event)

    def _update_CA(self):
        self.update_XY()
        # See the comment about catch_warnings above.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "combo box 'component_[xy]' .*", UserWarning)
            self.component_x, self.component_y = \
                self.component_x, self.component_y

        self._setup_plot()
        self._update_info()

    def update_XY(self):
        self.axis_x_cb.clear()
        self.axis_y_cb.clear()
        ca_vars = self.selected_vars()
        if len(ca_vars) == 0:
            return

        multi = len(ca_vars) != 2
        if multi:
            _, ctable = burt_table(self.data, ca_vars)
        else:
            ctable = contingency.get_contingency(self.data, *ca_vars[::-1])

        self.ca = correspondence(ctable, )
        rfs = self.ca.row_factors.shape[1]
        axes = ["{}".format(i + 1)
                for i in range(rfs)]
        self.axis_x_cb.addItems(axes)
        self.axis_y_cb.addItems(axes)
        return rfs

    def _setup_plot(self):
        def get_minmax(points):
            minmax = [float('inf'),
                      float('-inf'),
                      float('inf'),
                      float('-inf')]
            for pp in points:
                for p in pp:
                    minmax[0] = min(p[0], minmax[0])
                    minmax[1] = max(p[0], minmax[1])
                    minmax[2] = min(p[1], minmax[2])
                    minmax[3] = max(p[1], minmax[3])
            return minmax

        self.plot.clear()
        points = self.ca
        variables = self.selected_vars()
        colors = colorpalettes.LimitedDiscretePalette(len(variables))

        p_axes = self._p_axes()

        if points is None:
            return

        if len(variables) == 2:
            row_points = self.ca.row_factors[:, p_axes]
            col_points = self.ca.col_factors[:, p_axes]
            points = [row_points, col_points]
        else:
            points = self.ca.row_factors[:, p_axes]
            counts = [len(var.values) for var in variables]
            range_indices = np.cumsum([0] + counts)
            ranges = zip(range_indices, range_indices[1:])
            points = [points[s:e] for s, e in ranges]

        minmax = get_minmax(points)

        margin = abs(minmax[0] - minmax[1])
        margin = margin * 0.05 if margin > 1e-10 else 1
        self.plot.setXRange(minmax[0] - margin, minmax[1] + margin)
        margin = abs(minmax[2] - minmax[3])
        margin = margin * 0.05 if margin > 1e-10 else 1
        self.plot.setYRange(minmax[2] - margin, minmax[3] + margin)

        foreground = self.palette().color(QPalette.Text)
        for i, (v, points) in enumerate(zip(variables, points)):
            color_outline = colors[i]
            color_outline.setAlpha(200)
            color = QColor(color_outline)
            color.setAlpha(120)
            item = ScatterPlotItem(
                x=points[:, 0], y=points[:, 1], brush=QBrush(color),
                pen=pg.mkPen(color_outline.darker(120), width=1.5),
                size=np.full((points.shape[0],), 10.1),
            )
            self.plot.addItem(item)

            for name, point in zip(v.values, points):
                item = pg.TextItem(name, anchor=(0.5, 0), color=foreground)
                self.plot.addItem(item)
                item.setPos(point[0], point[1])

        inertia = self.ca.inertia_of_axis()
        if np.sum(inertia) == 0:
            inertia = 100 * inertia
        else:
            inertia = 100 * inertia / np.sum(inertia)

        ax = self.plot.getAxis("bottom")
        ax.setLabel("Component {} ({:.1f}%)"
                    .format(p_axes[0] + 1, inertia[p_axes[0]]))
        ax = self.plot.getAxis("left")
        ax.setLabel("Component {} ({:.1f}%)"
                    .format(p_axes[1] + 1, inertia[p_axes[1]]))

    def _update_info(self):
        if self.ca is None:
            self.infotext.setText("\n\n")
        else:
            fmt = ("Axis 1: {:.2f}\n"
                   "Axis 2: {:.2f}")
            inertia = self.ca.inertia_of_axis()
            if np.sum(inertia) == 0:
                inertia = 100 * inertia
            else:
                inertia = 100 * inertia / np.sum(inertia)

            ax1, ax2 = self._p_axes()
            self.infotext.setText(fmt.format(inertia[ax1], inertia[ax2]))

    def send_report(self):
        if self.data is None:
            return

        vars = self.selected_vars()
        if not vars:
            return

        items = OrderedDict()
        items["Data instances"] = len(self.data)
        if len(vars) == 1:
            items["Selected variable"] = vars[0]
        else:
            items["Selected variables"] = "{} and {}".format(
                ", ".join(var.name for var in vars[:-1]), vars[-1].name)
        self.report_items(items)

        self.report_plot()


def burt_table(data, variables):
    """
    Construct a 'Burt table' (all values cross-tabulation) for variables.

    Return and ordered list of (variable, value) pairs and a
    numpy.ndarray contingency

    :param Orange.data.Table data: Data table.
    :param variables: List of variables (discrete).
    :type variables: list of Orange.data.DiscreteVariable

    """
    values = [(var, value) for var in variables for value in var.values]

    table = np.zeros((len(values), len(values)))
    counts = [len(attr.values) for attr in variables]
    offsets = np.r_[0, np.cumsum(counts)]

    for i in range(len(variables)):
        for j in range(i + 1):
            var1 = variables[i]
            var2 = variables[j]

            cm = contingency.get_contingency(data, var2, var1)

            start1, end1 = offsets[i], offsets[i] + counts[i]
            start2, end2 = offsets[j], offsets[j] + counts[j]

            table[start1: end1, start2: end2] += cm
            if i != j:
                table[start2: end2, start1: end1] += cm.T

    return values, table


def correspondence(A):
    """
    :param numpy.ndarray A:
    """
    A = np.asarray(A)

    total = np.sum(A)
    if total > 0:
        corr_mat = A / total
    else:
        # ???
        corr_mat = A

    col_sum = np.sum(corr_mat, axis=0, keepdims=True)
    row_sum = np.sum(corr_mat, axis=1, keepdims=True)
    E = row_sum * col_sum

    with np.errstate(divide="ignore"):
        D_r, D_c = row_sum.ravel() ** -1, col_sum.ravel() ** -1
    D_r, D_c = np.nan_to_num(D_r), np.nan_to_num(D_c)

    def gsvd(M, wu, wv):
        assert len(M.shape) == 2
        assert len(wu.shape) == 1 and len(wv.shape) == 1
        Wu_sqrt = np.sqrt(wu)
        Wv_sqrt = np.sqrt(wv)
        B = np.c_[Wu_sqrt] * M * np.r_[Wv_sqrt]
        Ub, D, Vb = np.linalg.svd(B, full_matrices=False)
        U = np.c_[Wu_sqrt ** -1] * Ub
        V = (np.c_[Wv_sqrt ** -1] * Vb.T).T
        return U, D, V

    U, D, V = gsvd(corr_mat - E, D_r, D_c)

    F = np.c_[D_r] * U * D
    G = np.c_[D_c] * V.T * D

    if F.shape == (1, 1) and F[0, 0] == 0:
        F[0, 0] = 1

    return CA(U, D, V, F, G, row_sum, col_sum)

CA = namedtuple("CA", ["U", "D", "V", "row_factors", "col_factors",
                       "row_sums", "column_sums"])


class CA(CA):
    def row_inertia(self):
        return self.row_sums * (self.row_factors ** 2)

    def column_inertia(self):
        return self.column_sums.T * (self.col_factors ** 2)

    def inertia_of_axis(self):
        return np.sum(self.row_inertia(), axis=0)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCorrespondenceAnalysis).run(Table("titanic"))

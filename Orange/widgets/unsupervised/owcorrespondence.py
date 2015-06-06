import sys

import numpy

from PyQt4.QtGui import (
    QListView, QItemSelectionModel, QItemSelection, QBrush, QColor, QPainter,
    QApplication
)
from PyQt4.QtCore import Qt, QEvent

import pyqtgraph as pg
import Orange.data
from Orange.statistics import contingency

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette

from Orange.widgets.visualize.owscatterplotgraph import ScatterPlotItem


class ScatterPlotItem(pg.ScatterPlotItem):
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)
        super().paint(painter, option, widget)


def select_rows(view, row_indices, command=QItemSelectionModel.ClearAndSelect):
    """
    Select rows in view.

    :param PyQt4.QtGui.QAbstractItemView view:
    :param row_indices: Integer indices of rows to select.
    :param command: QItemSelectionModel.SelectionFlags
    """
    selmodel = view.selectionModel()
    model = view.model()
    selection = QItemSelection()
    for row in row_indices:
        index = model.index(row, 0)
        selection.select(index, index)
    selmodel.select(selection, command | QItemSelectionModel.Rows)


class OWCorrespondenceAnalysis(widget.OWWidget):
    name = "Correspondence Analysis"
    description = "Correspondence analysis for categorical multivariate data."
    icon = "icons/CorrespondenceAnalysis.svg"

    inputs = [("Data", Orange.data.Table, "set_data")]

    Invalidate = QEvent.registerEventType()

    settingsHandler = settings.DomainContextHandler()

    selected_var_indices = settings.ContextSetting([])

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.component_x = 0
        self.component_y = 1

        box = gui.widgetBox(self.controlArea, "Variables")
        self.varlist = itemmodels.VariableListModel()
        self.varview = view = QListView(
            selectionMode=QListView.MultiSelection
        )
        view.setModel(self.varlist)
        view.selectionModel().selectionChanged.connect(self._var_changed)

        box.layout().addWidget(view)

        axes_box = gui.widgetBox(self.controlArea, "Axes")
        box = gui.widgetBox(axes_box, "Axis X", margin=0)
        box.setFlat(True)
        self.axis_x_cb = gui.comboBox(
            box, self, "component_x", callback=self._component_changed)

        box = gui.widgetBox(axes_box, "Axis Y", margin=0)
        box.setFlat(True)
        self.axis_y_cb = gui.comboBox(
            box, self, "component_y", callback=self._component_changed)

        self.infotext = gui.widgetLabel(
            gui.widgetBox(self.controlArea, "Contribution to Inertia"), "\n"
        )

        gui.rubber(self.controlArea)

        self.plot = pg.PlotWidget(background="w")
        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.warning(0)
        self.data = data

        if data is not None:
            self.varlist[:] = data.domain.variables
            self.selected_var_indices = [0, 1][:len(self.varlist)]
            self.component_x, self.component_y = 0, 1
            self.openContext(data)
            self._restore_selection()
#             self._invalidate()
        self._update_CA()

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
#         return (0, 1)
        return (self.component_x, self.component_y)

    def _var_changed(self):
        self.selected_var_indices = sorted(
            ind.row() for ind in self.varview.selectionModel().selectedRows())
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
            return
        return super().customEvent(event)

    def _update_CA(self):
        ca_vars = self.selected_vars()
        if len(ca_vars) == 0:
            return

        multi = len(ca_vars) != 2
        if multi:
            _, ctable = burt_table(self.data, ca_vars)
        else:
            ctable = contingency.get_contingency(self.data, *ca_vars[::-1])

        self.ca = correspondence(ctable, )
        axes = ["{}".format(i + 1)
                for i in range(self.ca.row_factors.shape[1])]
        self.axis_x_cb.clear()
        self.axis_x_cb.addItems(axes)
        self.axis_y_cb.clear()
        self.axis_y_cb.addItems(axes)
        self.component_x, self.component_y = self.component_x, self.component_y

        self._setup_plot()
        self._update_info()

    def _setup_plot(self):
        self.plot.clear()

        points = self.ca
        variables = self.selected_vars()
        colors = colorpalette.ColorPaletteGenerator(len(variables))

        p_axes = self._p_axes()

        if len(variables) == 2:
            row_points = self.ca.row_factors[:, p_axes]
            col_points = self.ca.col_factors[:, p_axes]
            points = [row_points, col_points]
        else:
            points = self.ca.row_factors[:, p_axes]
            counts = [len(var.values) for var in variables]
            range_indices = numpy.cumsum([0] + counts)
            ranges = zip(range_indices, range_indices[1:])
            points = [points[s:e] for s, e in ranges]

        for i, (v, points) in enumerate(zip(variables, points)):
            color_outline = colors[i]
            color_outline.setAlpha(200)
            color = QColor(color_outline)
            color.setAlpha(120)
            item = ScatterPlotItem(
                x=points[:, 0], y=points[:, 1], brush=QBrush(color),
                pen=pg.mkPen(color_outline.darker(120), width=1.5),
                size=numpy.full((points.shape[0],), 10.1),
            )
            self.plot.addItem(item)

            for name, point in zip(v.values, points):
                item = pg.TextItem(name, anchor=(0.5, 0))
                self.plot.addItem(item)
                item.setPos(point[0], point[1])

        inertia = self.ca.inertia_of_axis()
        inertia = 100 * inertia / numpy.sum(inertia)

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
            inertia = 100 * inertia / numpy.sum(inertia)

            ax1, ax2 = self._p_axes()
            self.infotext.setText(fmt.format(inertia[ax1], inertia[ax2]))


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

    table = numpy.zeros((len(values), len(values)))
    counts = [len(attr.values) for attr in variables]
    offsets = numpy.r_[0, numpy.cumsum(counts)]

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
    A = numpy.asarray(A)

    total = numpy.sum(A)
    if total > 0:
        corr_mat = A / total
    else:
        # ???
        corr_mat = A

    col_sum = numpy.sum(corr_mat, axis=0, keepdims=True)
    row_sum = numpy.sum(corr_mat, axis=1, keepdims=True)
    E = row_sum * col_sum

    D_r, D_c = row_sum.ravel() ** -1, col_sum.ravel() ** -1

    def gsvd(M, Wu, Wv):
        assert len(M.shape) == 2
        assert len(Wu.shape) == 1 and len(Wv.shape) == 1
        Wu_sqrt = numpy.sqrt(Wu)
        Wv_sqrt = numpy.sqrt(Wv)
        B = numpy.c_[Wu_sqrt] * M * numpy.r_[Wv_sqrt]
        Ub, D, Vb = numpy.linalg.svd(B, full_matrices=False)
        U = numpy.c_[Wu_sqrt ** -1] * Ub
        V = (numpy.c_[Wv_sqrt ** -1] * Vb.T).T
        return U, D, V

    U, D, V = gsvd(corr_mat - E, D_r, D_c)

    F = numpy.c_[D_r] * U * D
    G = numpy.c_[D_c] * V.T * D

    return CA(U, D, V, F, G, row_sum, col_sum)

from collections import namedtuple

CA = namedtuple("CA", ["U", "D", "V", "row_factors", "col_factors",
                       "row_sums", "column_sums"])


class CA(CA):
    def row_inertia(self):
        return self.row_sums * (self.row_factors ** 2)

    def column_inertia(self):
        return self.column_sums.T * (self.col_factors ** 2)

    def inertia_of_axis(self):
        return numpy.sum(self.row_inertia(), axis=0)


def test_main(argv=None):
    import sip
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        filename = argv[0]
    else:
        filename = "smokers_ct"

    data = Orange.data.Table(filename)
    app = QApplication([argv])
    w = OWCorrespondenceAnalysis()
    w.set_data(data)
    w.show()
    w.raise_()
    rval = app.exec_()
    w.onDeleteWidget()
    sip.delete(w)
    del w
    return rval

if __name__ == "__main__":
    sys.exit(test_main())

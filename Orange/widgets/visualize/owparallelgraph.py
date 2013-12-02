#
# OWParallelGraph.py
#
from collections import defaultdict
import os
import sys
import math

import numpy as np

from PyQt4.QtCore import QLineF, Qt, QEvent, QRect, QPoint, QPointF
from PyQt4.QtGui import QGraphicsPathItem, QPixmap, QColor, QBrush, QPen, QToolTip, QPainterPath

from Orange.canvas.utils import environ

from Orange.statistics.contingency import get_contingencies, get_contingency
from Orange.statistics.distribution import get_distribution
from Orange.widgets.settings import SettingProvider, Setting
from Orange.data import Variable
from Orange.widgets.utils.plot import OWPlot, UserAxis, AxisStart, AxisEnd, OWCurve, OWPoint, PolygonCurve, \
    xBottom, yLeft, ZOOMING
from Orange.widgets.utils.scaling import get_variable_value_indices, get_variable_values_sorted, ScaleData
VarTypes = Variable.VarTypes

NO_STATISTICS = 0
MEANS = 1
MEDIAN = 2


class OWParallelGraph(OWPlot, ScaleData, SettingProvider):
    show_distributions = Setting(False)
    show_attr_values = Setting(True)
    show_statistics = Setting(default=False)

    use_splines = Setting(False)
    alpha_value = Setting(150)
    alpha_value_2 = Setting(150)

    def __init__(self, widget, parent=None, name=None):
        widget.settingsHandler.initialize(self)
        OWPlot.__init__(self, parent, name, axes=[], widget=widget)
        ScaleData.__init__(self)

        self.update_antialiasing(False)

        self.widget = widget
        self.last_selected_curve = None
        self.enableGridXB(0)
        self.enableGridYL(0)
        self.domain_contingency = None
        self.auto_update_axes = 1
        self.old_legend_keys = []
        self.selection_conditions = {}
        self.attributes = []
        self.visualized_mid_labels = []
        self.attribute_indices = []
        self.valid_data = []

        self.selected_examples = []
        self.unselected_examples = []
        self.bottom_pixmap = QPixmap(os.path.join(environ.widget_install_dir, "icons/upgreenarrow.png"))
        self.top_pixmap = QPixmap(os.path.join(environ.widget_install_dir, "icons/downgreenarrow.png"))

    def set_data(self, data, subset_data=None, **args):
        OWPlot.setData(self, data)
        ScaleData.set_data(self, data, subset_data, **args)
        self.domain_contingency = None

    def update_data(self, attributes, mid_labels=None):
        old_selection_conditions = self.selection_conditions

        self.clear()

        if not (self.have_data or self.have_subset_data):
            return
        if len(attributes) < 2:
            return

        self.attributes = attributes
        self.attribute_indices = [self.attribute_name_index[name] for name in self.attributes]
        self.valid_data = self.get_valid_list(self.attribute_indices)

        self.visualized_mid_labels = mid_labels
        self.add_relevant_selections(old_selection_conditions)

        if self.data_has_discrete_class:
            self.discPalette.setNumberOfColors(len(self.data_domain.class_var.values))

        self.draw_curves()
        self.draw_distributions()
        self.draw_axes()
        self.draw_statistics()
        self.draw_mid_labels(mid_labels)
        self.draw_legend()

        self.replot()

    def add_relevant_selections(self, old_selection_conditions):
        """Keep only conditions related to the currently visualized attributes"""
        for name, value in old_selection_conditions.items():
            if name in self.attributes:
                self.selection_conditions[name] = value

    def draw_axes(self):
        self.remove_all_axes()
        for i in range(len(self.attributes)):
            axis_id = UserAxis + i
            a = self.add_axis(axis_id, line=QLineF(i, 0, i, 1), arrows=AxisStart | AxisEnd, zoomable=True)
            a.always_horizontal_text = True
            a.max_text_width = 100
            a.title_margin = -10
            a.text_margin = 0
            a.setZValue(5)
            self.set_axis_title(axis_id, self.data_domain[self.attributes[i]].name)
            self.set_show_axis_title(axis_id, self.show_attr_values)
            if self.show_attr_values == 1:
                attr = self.data_domain[self.attributes[i]]
                if attr.var_type == VarTypes.Continuous:
                    self.set_axis_scale(axis_id, self.attr_values[attr.name][0], self.attr_values[attr.name][1])
                elif attr.var_type == VarTypes.Discrete:
                    attribute_values = get_variable_values_sorted(self.data_domain[self.attributes[i]])
                    self.set_axis_labels(axis_id, attribute_values)

    def draw_curves(self):
        conditions = {name: self.attributes.index(name) for name in self.selection_conditions.keys()}

        def is_selected(example):
            return all(self.selection_conditions[name][0] <= example[index] <= self.selection_conditions[name][1]
                       for (name, index) in list(conditions.items()))

        selected_curves = defaultdict(list)
        background_curves = defaultdict(list)
        for row_idx, data in enumerate(self.scaled_data.T):
            if not self.valid_data[row_idx]:
                continue

            row = data[self.attribute_indices]
            color = self.select_color(row_idx)

            if is_selected(row):
                color += (self.alpha_value,)
                selected_curves[color].extend(row)
                self.selected_examples.append(row_idx)
            else:
                color += (self.alpha_value_2,)
                background_curves[color].extend(row)
                self.unselected_examples.append(row_idx)

        self._draw_curves(selected_curves)
        self._draw_curves(background_curves)

    def select_color(self, row_index):
        if self.data_has_class:
            if self.data_has_continuous_class:
                return self.contPalette.getRGB(self.no_jittering_scaled_data[self.data_class_index][row_index])
            else:
                return self.discPalette.getRGB(self.original_data[self.data_class_index][row_index])
        else:
            return 0, 0, 0

    def _draw_curves(self, selected_curves):
        n_attr = len(self.attributes)
        for color in sorted(selected_curves.keys()):
            curve = ParallelCoordinatesCurve(n_attr, selected_curves[color], color)
            curve.fitted = self.use_splines
            curve.attach(self)

    def draw_legend(self):
        if self.data_has_class:
            if self.data_domain.class_var.var_type == VarTypes.Discrete:
                self.legend().clear()
                values = get_variable_values_sorted(self.data_domain.class_var)
                for i, value in enumerate(values):
                    self.legend().add_item(self.data_domain.class_var.name, value,
                                           OWPoint(OWPoint.Rect, self.discPalette[i], self.point_width))
            else:
                values = self.attr_values[self.data_domain.class_var.name]
                decimals = self.data_domain.class_var.numberOfDecimals
                self.legend().add_color_gradient(self.data_domain.class_var.name,
                                                 ["%%.%df" % decimals % v for v in values])
        else:
            self.legend().clear()
            self.old_legend_keys = []

    def draw_mid_labels(self, mid_labels):
        if mid_labels:
            for j in range(len(mid_labels)):
                self.addMarker(mid_labels[j], j + 0.5, 1.0, alignment=Qt.AlignCenter | Qt.AlignTop)

    def draw_statistics(self):
        """Draw lines that represent standard deviation or quartiles"""
        if self.show_statistics and self.have_data:
            n_attr = len(self.attributes)
            data = []
            for i in range(n_attr):
                if self.data_domain[self.attribute_indices[i]].var_type != VarTypes.Continuous:
                    data.append([()])
                    continue  # only for continuous attributes
                array = np.compress(np.equal(self.validDataArray[self.attribute_indices[i]], 1),
                                    self.scaledData[self.attribute_indices[i]])  # remove missing values

                if not self.data_has_class or self.data_has_continuous_class:    # no class
                    if self.show_statistics == MEANS:
                        m = array.mean()
                        dev = array.std()
                        data.append([(m - dev, m, m + dev)])
                    elif self.show_statistics == MEDIAN:
                        sorted_array = np.sort(array)
                        if len(sorted_array) > 0:
                            data.append([(sorted_array[int(len(sorted_array) / 4.0)],
                                          sorted_array[int(len(sorted_array) / 2.0)],
                                          sorted_array[int(len(sorted_array) * 0.75)])])
                        else:
                            data.append([(0, 0, 0)])
                else:
                    curr = []
                    class_values = get_variable_values_sorted(self.data_domain.class_var)
                    class_value_indices = get_variable_value_indices(self.data_domain.class_var)
                    for c in range(len(class_values)):
                        scaled_val = ((class_value_indices[class_values[c]] * 2) + 1) / (2 * len(class_value_indices))
                        non_missing_values = np.compress(np.equal(self.validDataArray[self.attribute_indices[i]], 1),
                                                         self.noJitteringScaledData[self.dataClassIndex])
                        # remove missing values
                        arr_c = np.compress(np.equal(non_missing_values, scaled_val), array)
                        if len(arr_c) == 0:
                            curr.append((0, 0, 0))
                            continue
                        if self.show_statistics == MEANS:
                            m = arr_c.mean()
                            dev = arr_c.std()
                            curr.append((m - dev, m, m + dev))
                        elif self.show_statistics == MEDIAN:
                            sorted_array = np.sort(arr_c)
                            curr.append((sorted_array[int(len(arr_c) / 4.0)], sorted_array[int(len(arr_c) / 2.0)],
                                         sorted_array[int(len(arr_c) * 0.75)]))
                    data.append(curr)

            # draw vertical lines
            for i in range(len(data)):
                for c in range(len(data[i])):
                    if data[i][c] == ():
                        continue
                    x = i - 0.03 * (len(data[i]) - 1) / 2.0 + c * 0.03
                    col = QColor(self.discPalette[c])
                    col.setAlpha(self.alpha_value_2)
                    self.addCurve("", col, col, 3, OWCurve.Lines, OWPoint.NoSymbol, xData=[x, x, x],
                                  yData=[data[i][c][0], data[i][c][1], data[i][c][2]], lineWidth=4)
                    self.addCurve("", col, col, 1, OWCurve.Lines, OWPoint.NoSymbol, xData=[x - 0.03, x + 0.03],
                                  yData=[data[i][c][0], data[i][c][0]], lineWidth=4)
                    self.addCurve("", col, col, 1, OWCurve.Lines, OWPoint.NoSymbol, xData=[x - 0.03, x + 0.03],
                                  yData=[data[i][c][1], data[i][c][1]], lineWidth=4)
                    self.addCurve("", col, col, 1, OWCurve.Lines, OWPoint.NoSymbol, xData=[x - 0.03, x + 0.03],
                                  yData=[data[i][c][2], data[i][c][2]], lineWidth=4)

            # draw lines with mean/median values
            if not self.data_has_class or self.data_has_continuous_class:
                class_count = 1
            else:
                class_count = len(self.data_domain.class_var.values)
            for c in range(class_count):
                diff = - 0.03 * (class_count - 1) / 2.0 + c * 0.03
                ys = []
                xs = []
                for i in range(len(data)):
                    if data[i] != [()]:
                        ys.append(data[i][c][1])
                        xs.append(i + diff)
                    else:
                        if len(xs) > 1:
                            col = QColor(self.discPalette[c])
                            col.setAlpha(self.alpha_value_2)
                            self.addCurve("", col, col, 1, OWCurve.Lines,
                                          OWPoint.NoSymbol, xData=xs, yData=ys, lineWidth=4)
                        xs = []
                        ys = []
                col = QColor(self.discPalette[c])
                col.setAlpha(self.alpha_value_2)
                self.addCurve("", col, col, 1, OWCurve.Lines,
                              OWPoint.NoSymbol, xData=xs, yData=ys, lineWidth=4)

    def draw_distributions(self):
        """Draw distributions with discrete attributes"""
        if not (self.show_distributions and self.data_has_discrete_class and self.have_data):
            return
        class_count = len(self.data_domain.class_var.values)

        # we create a hash table of possible class values (happens only if we have a discrete class)
        sorted_class_values = get_variable_values_sorted(self.data_domain.class_var)
        if self.domain_contingency is None:
            self.domain_contingency = get_contingencies(self.raw_data)

        max_val = 1
        for attr in self.attribute_indices:
            if self.data_domain[attr].var_type != VarTypes.Discrete:
                continue
            if self.data_domain[attr] == self.data_domain.class_var:
                max_val = max(max_val, max(get_distribution(self.raw_data, attr)))
            else:
                max_val = max(max_val,
                              max([max(val or [1]) for val in list(self.domain_contingency[attr].values())] or [1]))

        for graphAttrIndex, index in enumerate(self.attribute_indices):
            attr = self.data_domain[index]
            if attr.var_type != VarTypes.Discrete:
                continue
            if self.data_domain[index] == self.data_domain.class_var:
                contingency = get_contingency(self.raw_data, self.data_domain[index], self.data_domain[index])
            else:
                contingency = self.domain_contingency[index]

            attr_len = len(attr.values)

            # we create a hash table of variable values and their indices
            sorted_variable_values = get_variable_values_sorted(self.data_domain[index])

            # create bar curve
            for j in range(attr_len):
                attribute_value = sorted_variable_values[j]
                try:
                    continuous_attribute_value = contingency[attribute_value]
                except IndexError as ex:
                    print(ex, attribute_value, contingency, file=sys.stderr)
                    continue

                for i in range(class_count):
                    class_value = sorted_class_values[i]

                    color = QColor(self.discPalette[i])
                    color.setAlpha(self.alpha_value)

                    width = float(continuous_attribute_value[class_value] * 0.5) / float(max_val)
                    y_off = float(1.0 + 2.0 * j) / float(2 * attr_len)
                    height = 0.7 / float(class_count * attr_len)

                    y_low_bottom = y_off + float(class_count * height) / 2.0 - i * height
                    curve = PolygonCurve(QPen(color),
                                         QBrush(color),
                                         xData=[graphAttrIndex, graphAttrIndex + width,
                                                graphAttrIndex + width, graphAttrIndex],
                                         yData=[y_low_bottom, y_low_bottom, y_low_bottom - height,
                                                y_low_bottom - height],
                                         tooltip=self.data_domain[index].name)
                    curve.attach(self)

    # handle tooltip events
    def event(self, ev):
        if ev.type() == QEvent.ToolTip:
            x = self.inv_transform(xBottom, ev.pos().x())
            y = self.inv_transform(yLeft, ev.pos().y())

            canvas_position = self.mapToScene(ev.pos())
            x_float = self.inv_transform(xBottom, canvas_position.x())
            contact, (index, pos) = self.testArrowContact(int(round(x_float)), canvas_position.x(), canvas_position.y())
            if contact:
                attr = self.data_domain[self.attributes[index]]
                if attr.var_type == VarTypes.Continuous:
                    condition = self.selection_conditions.get(attr.name, [0, 1])
                    val = self.attr_values[attr.name][0] + condition[pos] * (
                        self.attr_values[attr.name][1] - self.attr_values[attr.name][0])
                    str_val = attr.name + "= %%.%df" % attr.numberOfDecimals % val
                    QToolTip.showText(ev.globalPos(), str_val)
            else:
                for curve in self.items():
                    if type(curve) == PolygonCurve and \
                            curve.boundingRect().contains(x, y) and \
                            getattr(curve, "tooltip", None):
                        (name, value, total, dist) = curve.tooltip
                        count = sum([v[1] for v in dist])
                        if count == 0:
                            continue
                        tooltip_text = "Attribute: <b>%s</b><br>Value: <b>%s</b><br>" \
                                       "Total instances: <b>%i</b> (%.1f%%)<br>" \
                                       "Class distribution:<br>" % (
                                           name, value, count, 100.0 * count / float(total))
                        for (val, n) in dist:
                            tooltip_text += "&nbsp; &nbsp; <b>%s</b> : <b>%i</b> (%.1f%%)<br>" % (
                                val, n, 100.0 * float(n) / float(count))
                        QToolTip.showText(ev.globalPos(), tooltip_text[:-4])

        elif ev.type() == QEvent.MouseMove:
            QToolTip.hideText()

        return OWPlot.event(self, ev)

    def testArrowContact(self, indices, x, y):
        if type(indices) != list: indices = [indices]
        for index in indices:
            if index >= len(self.attributes) or index < 0:
                continue
            int_x = self.transform(xBottom, index)
            bottom = self.transform(yLeft,
                                    self.selection_conditions.get(self.attributes[index], [0, 1])[0])
            bottom_rect = QRect(int_x - self.bottom_pixmap.width() / 2, bottom, self.bottom_pixmap.width(),
                                self.bottom_pixmap.height())
            if bottom_rect.contains(QPoint(x, y)):
                return 1, (index, 0)
            top = self.transform(yLeft,
                                 self.selection_conditions.get(self.attributes[index], [0, 1])[1])
            top_rect = QRect(int_x - self.top_pixmap.width() / 2, top - self.top_pixmap.height(),
                            self.top_pixmap.width(),
                            self.top_pixmap.height())
            if top_rect.contains(QPoint(x, y)):
                return 1, (index, 1)
        return 0, (0, 0)

    def mousePressEvent(self, e):
        canvas_position = self.mapToScene(e.pos())
        x = self.inv_transform(xBottom, canvas_position.x())
        contact, info = self.testArrowContact(int(round(x)), canvas_position.x(), canvas_position.y())

        if contact:
            self.pressed_arrow = info
        else:
            OWPlot.mousePressEvent(self, e)

    def mouseMoveEvent(self, e):
        if hasattr(self, "pressed_arrow"):
            canvas_position = self.mapToScene(e.pos())
            y = min(1, max(0, self.inv_transform(yLeft, canvas_position.y())))
            index, pos = self.pressed_arrow
            attr = self.data_domain[self.attributes[index]]
            old_condition = self.selection_conditions.get(attr.name, [0, 1])
            old_condition[pos] = y
            self.selection_conditions[attr.name] = old_condition
            self.update_data(self.attributes, self.visualized_mid_labels)

            if attr.var_type == VarTypes.Continuous:
                val = self.attr_values[attr.name][0] + old_condition[pos] * (
                    self.attr_values[attr.name][1] - self.attr_values[attr.name][0])
                strVal = attr.name + "= %.2f" % val
                QToolTip.showText(e.globalPos(), strVal)
            if self.sendSelectionOnUpdate and self.auto_send_selection_callback:
                self.auto_send_selection_callback()

        else:
            OWPlot.mouseMoveEvent(self, e)

    def mouseReleaseEvent(self, e):
        if hasattr(self, "pressed_arrow"):
            del self.pressed_arrow
        else:
            OWPlot.mouseReleaseEvent(self, e)

    def staticMouseClick(self, e):
        if e.button() == Qt.LeftButton and self.state == ZOOMING:
            if self.tempSelectionCurve: self.tempSelectionCurve.detach()
            self.tempSelectionCurve = None
            canvasPos = self.mapToScene(e.pos())
            x = self.inv_transform(xBottom, canvasPos.x())
            y = self.inv_transform(yLeft, canvasPos.y())
            diffX = (self.axisScaleDiv(xBottom).interval().maxValue() - self.axisScaleDiv(
                xBottom).interval().minValue()) / 2.

            xmin = x - (diffX / 2.) * (x - self.axisScaleDiv(xBottom).interval().minValue()) / diffX
            xmax = x + (diffX / 2.) * (self.axisScaleDiv(xBottom).interval().maxValue() - x) / diffX
            ymin = self.axisScaleDiv(yLeft).interval().maxValue()
            ymax = self.axisScaleDiv(yLeft).interval().minValue()

            self.zoomStack.append((
                self.axisScaleDiv(xBottom).interval().minValue(), self.axisScaleDiv(xBottom).interval().maxValue(),
                self.axisScaleDiv(yLeft).interval().minValue(), self.axisScaleDiv(yLeft).interval().maxValue()))
            self.setNewZoom(xmin, xmax, ymax, ymin)
            return 1

        # if the user clicked between two lines send a list with the names of the two attributes
        elif self.widget:
            x1 = int(self.inv_transform(xBottom, e.x()))
            axis = self.axisScaleDraw(xBottom)
            self.widget.send_shown_attributes([str(axis.label(x1)), str(axis.label(x1 + 1))])
        return 0

    def removeAllSelections(self, send=1):
        self.selection_conditions = {}
        self.update_data(self.attributes, self.visualized_mid_labels)

    # draw the curves and the selection conditions
    def drawCanvas(self, painter):
        OWPlot.drawCanvas(self, painter)
        for i in range(
                int(max(0, math.floor(self.axisScaleDiv(xBottom).interval().minValue()))),
                int(min(len(self.attributes),
                        math.ceil(self.axisScaleDiv(xBottom).interval().maxValue()) + 1))):
            bottom, top = self.selection_conditions.get(self.attributes[i], (0, 1))
            painter.drawPixmap(self.transform(xBottom, i) - self.bottom_pixmap.width() / 2,
                               self.transform(yLeft, bottom), self.bottom_pixmap)
            painter.drawPixmap(self.transform(xBottom, i) - self.top_pixmap.width() / 2,
                               self.transform(yLeft, top) - self.top_pixmap.height(), self.top_pixmap)

    def auto_send_selection_callback(self):
        pass

    def clear(self):
        super().clear()

        self.attributes = []
        self.visualized_mid_labels = []
        self.selected_examples = []
        self.unselected_examples = []
        self.selection_conditions = {}


# ####################################################################
# a curve that is able to draw several series of lines
class ParallelCoordinatesCurve(OWCurve):
    def __init__(self, n_attributes, y_values, color, name=""):
        OWCurve.__init__(self, tooltip=name)
        self._item = QGraphicsPathItem(self)
        self.path = QPainterPath()
        self.fitted = False

        self.n_attributes = n_attributes
        self.n_rows = int(len(y_values) / n_attributes)

        self.set_style(OWCurve.Lines)
        if isinstance(color, tuple):
            self.set_pen(QPen(QColor(*color)))
        else:
            self.set_pen(QPen(QColor(color)))

        x_values = list(range(n_attributes)) * self.n_rows
        self.set_data(x_values, y_values)

    def update_properties(self):
        self.redraw_path()

    def redraw_path(self):
        self.path = QPainterPath()
        for segment in self.segment(self.data()):
            if self.fitted:
                self.draw_cubic_path(segment)
            else:
                self.draw_normal_path(segment)
        self._item.setPath(self.graph_transform().map(self.path))
        self._item.setPen(self.pen())

    def segment(self, data):
        for i in range(self.n_rows):
            yield data[i * self.n_attributes:(i + 1) * self.n_attributes]

    def draw_cubic_path(self, segment):
        for (x1, y1), (x2, y2) in zip(segment, segment[1:]):
            self.path.moveTo(x1, y1)
            self.path.cubicTo(QPointF(x1 + 0.5, y1),
                              QPointF(x2 - 0.5, y2), QPointF(x2, y2))

    def draw_normal_path(self, segment):
        if not segment:
            return

        x, y = segment[0]
        self.path.moveTo(x, y)
        for x, y in segment[1:]:
            self.path.lineTo(x, y)

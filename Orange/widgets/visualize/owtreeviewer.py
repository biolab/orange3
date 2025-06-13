"""Widget for visualization of tree models"""
import re
from html import escape
from typing import Optional

import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QSizePolicy, QStyle,
    QLabel, QComboBox
)
from AnyQt.QtGui import QColor, QBrush, QPen, QFontMetrics
from AnyQt.QtCore import Qt, QPointF, QSizeF, QRectF

from orangewidget.utils.combobox import ComboBoxSearch
from orangewidget.utils.itemmodels import PyListModel

from Orange.base import TreeModel, SklModel
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owtreeviewer2d import \
    GraphicsNode, GraphicsEdge, OWTreeViewer2D
from Orange.widgets.utils import to_html
from Orange.widgets.utils.localization import pl
from Orange.data import Table
from Orange.util import color_to_hex

from Orange.widgets.settings import ContextSetting, ClassValuesContextHandler, \
    Setting
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.visualize.utils.tree.skltreeadapter import SklTreeAdapter
from Orange.widgets.visualize.utils.tree.treeadapter import TreeAdapter


class PieChart(QGraphicsRectItem):
    """PieChart graphics item added at the corner of classification tree nodes
    """
    # Methods are documented in PyQt documentation
    # pylint: disable=missing-docstring
    def __init__(self, dist, r, parent):
        # pylint: disable=invalid-name
        super().__init__(parent)
        self.dist = dist
        self.r = r

    # noinspection PyPep8Naming
    def setR(self, r):
        # pylint: disable=invalid-name
        self.prepareGeometryChange()
        self.r = r

    def boundingRect(self):
        return QRectF(-self.r, -self.r, 2 * self.r, 2 * self.r)

    def paint(self, painter, option, widget=None):
        # pylint: disable=missing-docstring
        dist_sum = sum(self.dist)
        start_angle = 0
        colors = self.scene().colors
        for i in range(len(self.dist)):
            angle = self.dist[i] * 16 * 360. / dist_sum
            if angle == 0:
                continue
            painter.setBrush(QBrush(colors[i]))
            painter.setPen(QPen(colors[i]))
            painter.drawPie(-self.r, -self.r, 2 * self.r, 2 * self.r,
                            int(start_angle), int(angle))
            start_angle += angle
        painter.setPen(QPen(Qt.black))
        painter.setBrush(QBrush())
        painter.drawEllipse(-self.r, -self.r, 2 * self.r, 2 * self.r)


class TreeNode(GraphicsNode):
    """TreeNode for trees corresponding to base.Tree models"""
    # Methods are documented in PyQt documentation
    # pylint: disable=missing-docstring

    def __init__(self, tree_adapter, node_inst, parent=None):
        super().__init__(parent)
        self.tree_adapter = tree_adapter
        self.model = self.tree_adapter.model
        self.node_inst = node_inst

        fm = QFontMetrics(self.document().defaultFont())
        attr = self.tree_adapter.attribute(node_inst)
        self.attr_text_w = fm.horizontalAdvance(attr.name if attr else "")
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        self._rect = None

        if self.model.domain.class_var.is_discrete:
            self.pie = PieChart(self.tree_adapter.get_distribution(node_inst)[0], 8, self)
        else:
            self.pie = None

    def update_contents(self):
        self.prepareGeometryChange()
        self.setTextWidth(-1)
        self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        fm = QFontMetrics(self.document().defaultFont())
        attr = self.tree_adapter.attribute(self.node_inst)
        self.attr_text_w = fm.horizontalAdvance(attr.name if attr else "")
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        if self.pie is not None:
            self.pie.setPos(self.rect().right(), self.rect().center().y())

    def rect(self):
        if self._rect and self._rect.isValid():
            return self._rect
        else:
            return QRectF(QPointF(0, 0), self.document().size()).\
                adjusted(0, 0, 8, 0) | \
                (getattr(self, "_rect") or QRectF(0, 0, 1, 1))

    def set_rect(self, rect):
        self.prepareGeometryChange()
        rect = QRectF() if rect is None else rect
        self._rect = rect
        self.setTextWidth(-1)
        self.update_contents()
        self.update()

    def boundingRect(self):
        if hasattr(self, "attr"):
            attr_rect = QRectF(QPointF(0, -self.attr_text_h),
                               QSizeF(self.attr_text_w, self.attr_text_h))
        else:
            attr_rect = QRectF(0, 0, 1, 1)
        rect = self.rect().adjusted(-6, -6, 6, 6)
        return rect | attr_rect

    def paint(self, painter, option, widget=None):
        rect = self.rect()
        if self.isSelected():
            option.state ^= QStyle.State_Selected
        font = self.document().defaultFont()
        painter.setFont(font)
        if self.parent:
            draw_text = str(self.tree_adapter.short_rule(self.node_inst))
            if self.parent.x() > self.x():  # node is to the left
                fm = QFontMetrics(font)
                x = rect.width() / 2 - fm.horizontalAdvance(draw_text) - 4
            else:
                x = rect.width() / 2 + 4
            painter.drawText(QPointF(x, -self.line_descent - 1), draw_text)
        painter.save()
        painter.setBrush(self.backgroundBrush)
        if self.isSelected():
            outline = QPen(option.palette.highlight(), 3)
        else:
            outline = QPen(option.palette.dark(), 1)
        painter.setPen(outline)
        adjrect = rect.adjusted(-3, 0, 0, 0)
        if not self.tree_adapter.has_children(self.node_inst):
            painter.drawRoundedRect(adjrect, 4, 4)
        else:
            painter.drawRect(adjrect)
        painter.restore()
        painter.setClipRect(rect)
        return QGraphicsTextItem.paint(self, painter, option, widget)


class OWTreeGraph(OWTreeViewer2D):
    """Graphical visualization of tree models"""

    name = "Tree Viewer"
    icon = "icons/TreeViewer.svg"
    priority = 35
    keywords = "tree viewer"

    class Inputs:
        # Had different input names before merging from
        # Classification/Regression tree variants
        tree = Input("Tree", TreeModel, replaces=["Classification Tree", "Regression Tree"])

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True, id="selected-data")
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table, id="annotated-data")

    settingsHandler = ClassValuesContextHandler()
    target_class_index = ContextSetting(0)
    regression_colors = Setting(0)
    # None is a hint, "" means 'no hint'
    node_labels_hint: Optional[str] = ContextSetting("")
    show_intermediate = Setting(False)

    replaces = [
        "Orange.widgets.classify.owclassificationtreegraph.OWClassificationTreeGraph",
        "Orange.widgets.classify.owregressiontreegraph.OWRegressionTreeGraph"
    ]

    COL_OPTIONS = ["Default", "Number of instances", "Mean value", "Variance"]
    COL_DEFAULT, COL_INSTANCE, COL_MEAN, COL_VARIANCE = range(4)

    def __init__(self):
        super().__init__()
        self.domain = None
        self.dataset = None
        self.tree_adapter = None
        self.node_labels = None

        self.color_label = QLabel("Target class: ")
        combo = self.color_combo = ComboBoxSearch()
        combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        combo.activated[int].connect(self.color_changed)
        self.display_box.layout().addRow(self.color_label, combo)

        self.label_model = DomainModel(
            placeholder="None",
            order=(DomainModel.METAS,
                   PyListModel.Separator,
                   DomainModel.ATTRIBUTES)
        )
        combo = gui.comboBox(
            None, self, "node_labels",
            model=self.label_model,
            orientation=Qt.Horizontal,
            callback=self.label_changed,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            minimumContentsLength=8,
            tooltip="Variable that identifies the instances in nodes."
        )
        self.display_box.layout().addRow("Node labels:", combo)

        box = gui.hBox(None)
        gui.rubber(box)
        gui.checkBox(box, self, "show_intermediate",
                     "Show details in non-leaves",
                     callback=self.set_node_info)
        self.display_box.layout().addRow(box)

    def set_node_info(self):
        """Set the content of the node"""
        for node in self.scene.nodes():
            node.set_rect(QRectF())
            self.update_node_info(node)
        w = max([n.rect().width() for n in self.scene.nodes()] + [0])
        if w > self.max_node_width:
            w = self.max_node_width
        for node in self.scene.nodes():
            rect = node.rect()
            node.set_rect(QRectF(rect.x(), rect.y(), w, rect.height()))
        self.scene.fix_pos(self.root_node, 10, 10)

    def _update_node_info_attr_name(self, node, text):
        attr = self.tree_adapter.attribute(node.node_inst)
        if attr is not None:
            if text:
                text += "<hr/>"
            text += attr.name
        return text

    def activate_loaded_settings(self):
        if not self.model:
            return
        super().activate_loaded_settings()
        if self.domain.class_var.is_discrete:
            self.color_combo.setCurrentIndex(self.target_class_index)
            self.toggle_node_color_cls()
        else:
            self.color_combo.setCurrentIndex(self.regression_colors)
            self.toggle_node_color_reg()
        self.set_node_info()

    def color_changed(self, i):
        if self.domain.class_var.is_discrete:
            self.target_class_index = i
            self.toggle_node_color_cls()
            self.set_node_info()
        else:
            self.regression_colors = i
            self.toggle_node_color_reg()

    def label_changed(self):
        self.node_labels_hint = self.node_labels and self.node_labels.name
        self.set_node_info()

    def toggle_node_size(self):
        self.set_node_info()
        self.scene.update()
        self.scene_view.repaint()

    def toggle_color_cls(self):
        self.toggle_node_color_cls()
        self.set_node_info()
        self.scene.update()

    def toggle_color_reg(self):
        self.toggle_node_color_reg()
        self.set_node_info()
        self.scene.update()

    @Inputs.tree
    def ctree(self, model=None):
        """Input signal handler"""
        self.clear_scene()
        self.color_combo.clear()
        self.closeContext()
        self.model = model
        self.target_class_index = 0
        if model is None:
            self._ctree_clean()
        else:
            self._ctree_setup(model)

        self.setup_scene()
        self.Outputs.selected_data.send(None)
        self.Outputs.annotated_data.send(create_annotated_table(self.dataset, []))

    def _ctree_clean(self):
        self.infolabel.setText('No tree.')
        self.label_model.set_domain(None)
        self.root_node = None
        self.dataset = None
        self.tree_adapter = None
        self.node_labels = None

    def _ctree_setup(self, model):
        self.tree_adapter = self._get_tree_adapter(model)
        self.domain = model.domain
        self.dataset = model.instances
        class_var = self.domain.class_var
        self.scene.colors = class_var.palette
        if class_var.is_discrete:
            self.color_label.setText("Target class: ")
            self.color_combo.addItem("None")
            self.color_combo.addItems(self.domain.class_vars[0].values)
            self.color_combo.setCurrentIndex(self.target_class_index)
        else:
            self.color_label.setText("Color by: ")
            self.color_combo.addItems(self.COL_OPTIONS)
            self.color_combo.setCurrentIndex(self.regression_colors)

        self.openContext(self.domain)

        self.set_node_labels(model)
        self.root_node = self.walkcreate(self.tree_adapter.root)
        nodes = self.tree_adapter.num_nodes
        leaves = len(self.tree_adapter.leaves(self.tree_adapter.root))
        self.infolabel.setText(f'{nodes} {pl(nodes, "node")}, {leaves} {pl(leaves, "leaf|leaves")}')

    def set_node_labels(self, model):
        # Note: This function set the instance label but not the hint
        # Hints are only set by users. If the label is set heuristically
        # it will be set to the same (heuristic) value next time anyway.

        # Set node_labels to None before changing the model,
        # for the sake of hygiene
        self.node_labels = None
        self.label_model.set_domain(model.instances and model.instances.domain)

        # If we have no data or the hint say to not use labels, leave it None
        if model.instances is None or self.node_labels_hint is None:
            return

        if self.node_labels_hint in self.domain:
            # Use the hint if you can
            self.node_labels = self.domain[self.node_labels_hint]
        else:
            nunique, var = max(
                ((len(set(self.dataset.get_column(v))), v)
                 for v in self.domain.metas if v.is_string),
                default=(0, None))
            if nunique > 0.8 * len(self.dataset):
                self.node_labels = var

    def walkcreate(self, node, parent=None):
        """Create a structure of tree nodes from the given model"""
        node_obj = TreeNode(self.tree_adapter, node, parent)
        self.scene.addItem(node_obj)
        if parent:
            edge = GraphicsEdge(node1=parent, node2=node_obj)
            self.scene.addItem(edge)
            parent.graph_add_edge(edge)
        for child_inst in self.tree_adapter.children(node):
            if child_inst is not None:
                self.walkcreate(child_inst, node_obj)
        return node_obj

    def node_tooltip(self, node):
        # We use <br/> and &nbsp: styling of <li> in Qt doesn't work well
        indent = "&nbsp;&nbsp;&nbsp;"
        nbp = "<p style='white-space:pre'>"

        rule = "<br/>".join(f"{indent}– {to_html(str(rule))}"
                            for rule in self.tree_adapter.rules(node.node_inst))
        if rule:
            rule = f"<p><b>Selection</b></p><p>{rule}</p>"

        distr = self.tree_adapter.get_distribution(node.node_inst)[0]
        class_var = self.domain.class_var
        name = escape(class_var.name)
        if self.domain.class_var.is_discrete:
            total = float(sum(distr)) or 1
            show_all = len(distr) <= 2
            content = f"{nbp}<b>Distribution of</b> '{name}'</p><p>" \
                + "<table>" + "".join(
                    "<tr>"
                    f"<td><span style='color: {color_to_hex(color)}'>◼</span> "
                    f"{escape(value)}</td>"
                    f"<td>{indent}</td>"
                    f"<td align='right'>{prop:g}</td>"
                    f"<td>{indent}</td>"
                    f"<td align='right'>{prop / total * 100:.1f} %</td>"
                    "</tr>"
                    for value, color, prop
                    in zip(class_var.values, class_var.colors, distr)
                    if show_all or prop > 0) \
                + "</table>"
        else:
            mean, var = distr
            content = f"{nbp}{class_var.name} = {mean:.3g} ± {var:.3g}<br/>" + \
                f"({self.tree_adapter.num_samples(node.node_inst)} instances)</p>"

        split = self._update_node_info_attr_name(node, "")
        if split:
            split = f"{nbp}<b>Next split: </b>{split}</p>"
        return "<hr/>".join(filter(None, (rule, content, split)))

    def update_selection(self):
        if self.model is None:
            return
        nodes = [item.node_inst for item in self.scene.selectedItems()
                 if isinstance(item, TreeNode)]
        data = self.tree_adapter.get_instances_in_nodes(nodes)

        self.Outputs.selected_data.send(data)
        self.Outputs.annotated_data.send(create_annotated_table(
            self.dataset, self.tree_adapter.get_indices(nodes)))

    def send_report(self):
        if not self.model:
            return
        items = [("Tree size", self.infolabel.text()),
                 ("Edge widths",
                  ("Fixed", "Relative to root", "Relative to parent")[
                      # pylint: disable=invalid-sequence-index
                      self.line_width_method])]
        if self.domain.class_var.is_discrete:
            items.append(("Target class", self.color_combo.currentText()))
        elif self.regression_colors != self.COL_DEFAULT:
            items.append(("Color by", self.COL_OPTIONS[self.regression_colors]))
        self.report_items(items)
        self.report_plot()

    def update_node_info(self, node):
        if self.tree_adapter.has_children(node.node_inst) and not self.show_intermediate:
            text = ""
        elif self.domain.class_var.is_discrete:
            text = self.node_content_cls(node)
        else:
            text = self.node_content_reg(node)

        text = self._update_node_info_attr_name(node, text)
        if self.node_labels is not None and not self.tree_adapter.has_children(node.node_inst):
            text += "<hr/>"
            data = self.tree_adapter.get_instances_in_nodes([node.node_inst])
            var = self.node_labels
            labels = [escape(var.str_val(label))
                      for label in data.get_column(var)[:4]]
            text += ", ".join(labels)
            if len(data) > 4:
                text += ", …"

        node.setHtml(
            f'<p style="line-height: 120%; margin-bottom: 0">{text}</p>')

    def node_content_cls(self, node):
        """Update the printed contents of the node for classification trees"""
        node_inst = node.node_inst
        distr = self.tree_adapter.get_distribution(node_inst)[0]
        total = self.tree_adapter.num_samples(node_inst)
        distr = distr / np.sum(distr)
        if self.target_class_index:
            tabs = distr[self.target_class_index - 1]
            text = ""
        else:
            modus = np.argmax(distr)
            tabs = distr[modus]
            text = f"<b>{self.domain.class_vars[0].values[int(modus)]}</b><br/>"
        if tabs > 0.999:
            text += f"100%, {total}/{total}"
        else:
            text += f"{100 * tabs:2.1f}%, {int(total * tabs)}/{total}"
        return text

    def node_content_reg(self, node):
        """Update the printed contents of the node for regression trees"""
        node_inst = node.node_inst
        mean, var = self.tree_adapter.get_distribution(node_inst)[0]
        insts = self.tree_adapter.num_samples(node_inst)
        text = f"<b>{mean:.1f}</b> ± {var:.1f}<br/>"
        text += f"{insts} instances"
        return text

    def toggle_node_color_cls(self):
        """Update the node color for classification trees"""
        colors = self.scene.colors
        for node in self.scene.nodes():
            distr = node.tree_adapter.get_distribution(node.node_inst)[0]
            total = sum(distr)
            if self.target_class_index:
                p = distr[self.target_class_index - 1] / total
                color = colors[self.target_class_index - 1].lighter(
                    int(200 - 100 * p))
            else:
                modus = np.argmax(distr)
                p = distr[modus] / (total or 1)
                color = colors.value_to_qcolor(int(modus))
                color = color.lighter(int(300 - 200 * p))
            node.backgroundBrush = QBrush(color)
        self.scene.update()

    def toggle_node_color_reg(self):
        """Update the node color for regression trees"""
        def_color = QColor(192, 192, 255)
        if self.regression_colors == self.COL_DEFAULT:
            brush = QBrush(def_color.lighter(100))
            for node in self.scene.nodes():
                node.backgroundBrush = brush
        elif self.regression_colors == self.COL_INSTANCE:
            max_insts = len(self.tree_adapter.get_instances_in_nodes(
                [self.tree_adapter.root]))
            for node in self.scene.nodes():
                node_insts = len(self.tree_adapter.get_instances_in_nodes(
                    [node.node_inst]))
                node.backgroundBrush = QBrush(def_color.lighter(
                    int(120 - 20 * node_insts / max_insts)))
        elif self.regression_colors == self.COL_MEAN:
            minv = np.nanmin(self.dataset.Y)
            maxv = np.nanmax(self.dataset.Y)
            colors = self.scene.colors
            for node in self.scene.nodes():
                node_mean = self.tree_adapter.get_distribution(node.node_inst)[0][0]
                color = colors.value_to_qcolor(node_mean, minv, maxv)
                node.backgroundBrush = QBrush(color)
        else:
            nodes = list(self.scene.nodes())
            variances = [self.tree_adapter.get_distribution(node.node_inst)[0][1]
                         for node in nodes]
            max_var = max(variances)
            for node, var in zip(nodes, variances):
                node.backgroundBrush = QBrush(def_color.lighter(
                    int(120 - 20 * var / max_var)))
        self.scene.update()

    def _get_tree_adapter(self, model):
        if isinstance(model, SklModel):
            return SklTreeAdapter(model)
        return TreeAdapter(model)


if __name__ == "__main__":  # pragma: no cover
    from Orange.modelling.tree import TreeLearner
    data = Table("titanic")
    # data = Table("housing")
    clf = TreeLearner()(data)
    clf.instances = data
    WidgetPreview(OWTreeGraph).run(clf)

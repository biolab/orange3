"""Widget for visualization of tree models"""
import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsView, QGraphicsRectItem, QGraphicsTextItem, QSizePolicy, QStyle,
    QLabel, QComboBox
)
from AnyQt.QtGui import QColor, QBrush, QPen, QFontMetrics
from AnyQt.QtCore import Qt, QPointF, QSizeF, QRectF

from Orange.tree import TreeModel
from Orange.widgets.visualize.owtreeviewer2d import \
    GraphicsNode, GraphicsEdge, OWTreeViewer2D
from Orange.widgets.utils import to_html
from Orange.data import Table

from Orange.widgets.settings import ContextSetting, ClassValuesContextHandler, \
    Setting
from Orange.widgets import gui, widget
from Orange.widgets.utils.colorpalette import ContinuousPaletteGenerator
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)


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

    def __init__(self, model, node_inst, parent=None):
        super().__init__(parent)
        self.model = model
        self.node_inst = node_inst

        fm = QFontMetrics(self.document().defaultFont())
        attr = node_inst.attr
        self.attr_text_w = fm.width(attr.name if attr else "")
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        self._rect = None

        if model.domain.class_var.is_discrete:
            self.pie = PieChart(node_inst.value, 8, self)
        else:
            self.pie = None

    def update_contents(self):
        self.prepareGeometryChange()
        self.setTextWidth(-1)
        self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        fm = QFontMetrics(self.document().defaultFont())
        attr = self.node_inst.attr
        self.attr_text_w = fm.width(attr.name if attr else "")
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
        rect = self.rect().adjusted(-5, -5, 5, 5)
        return rect | attr_rect

    def paint(self, painter, option, widget=None):
        rect = self.rect()
        if self.isSelected():
            option.state ^= QStyle.State_Selected
        font = self.document().defaultFont()
        painter.setFont(font)
        if self.parent:
            draw_text = self.node_inst.description
            if self.parent.x() > self.x():  # node is to the left
                fm = QFontMetrics(font)
                x = rect.width() / 2 - fm.width(draw_text) - 4
            else:
                x = rect.width() / 2 + 4
            painter.drawText(QPointF(x, -self.line_descent - 1), draw_text)
        painter.save()
        painter.setBrush(self.backgroundBrush)
        painter.setPen(QPen(Qt.black, 3 if self.isSelected() else 0))
        adjrect = rect.adjusted(-3, 0, 0, 0)
        if not self.node_inst.children:
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
    inputs = [("Tree", TreeModel, "ctree")]
    outputs = [("Selected Data", Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Table)]

    settingsHandler = ClassValuesContextHandler()
    target_class_index = ContextSetting(0)
    regression_colors = Setting(0)

    COL_OPTIONS = ["Default", "Number of instances", "Mean value", "Variance"]
    COL_DEFAULT, COL_INSTANCE, COL_MEAN, COL_VARIANCE = range(4)

    def __init__(self):
        super().__init__()
        self.domain = None
        self.dataset = None
        self.clf_dataset = None

        self.color_label = QLabel("Target class: ")
        combo = self.color_combo = gui.OrangeComboBox()
        combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        combo.activated[int].connect(self.color_changed)
        self.display_box.layout().addRow(self.color_label, combo)

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

    @staticmethod
    def _update_node_info_attr_name(node, text):
        attr = node.node_inst.attr
        if attr is not None:
            text += "<hr/>{}".format(attr.name)
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
        else:
            self.regression_colors = i
            self.toggle_node_color_reg()

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

    def ctree(self, model=None):
        """Input signal handler"""
        self.clear_scene()
        self.color_combo.clear()
        self.closeContext()
        self.model = model
        if model is None:
            self.info.setText('No tree.')
            self.root_node = None
            self.dataset = None
        else:
            self.domain = model.domain
            self.dataset = model.instances
            if self.dataset is not None and self.dataset.domain != self.domain:
                self.clf_dataset = Table.from_table(model.domain, self.dataset)
            else:
                self.clf_dataset = self.dataset
            class_var = self.domain.class_var
            if class_var.is_discrete:
                self.scene.colors = [QColor(*col) for col in class_var.colors]
                self.color_label.setText("Target class: ")
                self.color_combo.addItem("None")
                self.color_combo.addItems(self.domain.class_vars[0].values)
                self.color_combo.setCurrentIndex(self.target_class_index)
            else:
                self.scene.colors = \
                    ContinuousPaletteGenerator(*model.domain.class_var.colors)
                self.color_label.setText("Color by: ")
                self.color_combo.addItems(self.COL_OPTIONS)
                self.color_combo.setCurrentIndex(self.regression_colors)
            self.openContext(self.domain.class_var)
            self.root_node = self.walkcreate(model.root, None)
            self.info.setText('{} nodes, {} leaves'.
                              format(model.node_count(), model.leaf_count()))
        self.setup_scene()
        self.send("Selected Data", None)
        self.send(ANNOTATED_DATA_SIGNAL_NAME,
                  create_annotated_table(self.dataset, []))

    def walkcreate(self, node_inst, parent=None):
        """Create a structure of tree nodes from the given model"""
        node = TreeNode(self.model, node_inst, parent)
        self.scene.addItem(node)
        if parent:
            edge = GraphicsEdge(node1=parent, node2=node)
            self.scene.addItem(edge)
            parent.graph_add_edge(edge)
        for child_inst in node_inst.children:
            if child_inst is not None:
                self.walkcreate(child_inst, node)
        return node

    def node_tooltip(self, node):
        return "<br>".join(to_html(rule)
                           for rule in self.model.rule(node.node_inst))

    def update_selection(self):
        if self.model is None:
            return
        nodes = [item.node_inst for item in self.scene.selectedItems()
                 if isinstance(item, TreeNode)]
        data = self.model.get_instances(nodes)
        self.send("Selected Data", data)
        self.send(ANNOTATED_DATA_SIGNAL_NAME,
                  create_annotated_table(self.dataset,
                                         self.model.get_indices(nodes)))

    def send_report(self):
        if not self.model:
            return
        items = [("Tree size", self.info.text()),
                 ("Edge widths",
                  ("Fixed", "Relative to root", "Relative to parent")[
                      # pylint: disable=invalid-sequence-index
                      self.line_width_method])]
        if self.domain.class_var.is_discrete:
            items.append(("Target class", self.color_combo.currentText()))
        elif self.regression_colors != self.COL_DEFAULT:
            items.append(("Color by", self.COL_OPTIONS[self.regression_colors]))
        self.report_items(items)
        self.report_plot(self.scene)

    def update_node_info(self, node):
        if self.domain.class_var.is_discrete:
            self.update_node_info_cls(node)
        else:
            self.update_node_info_reg(node)

    def update_node_info_cls(self, node):
        """Update the printed contents of the node for classification trees"""
        node_inst = node.node_inst
        distr = node_inst.value
        total = len(node_inst.subset)
        distr = distr / np.sum(distr)
        if self.target_class_index:
            tabs = distr[self.target_class_index - 1]
            text = ""
        else:
            modus = np.argmax(distr)
            tabs = distr[modus]
            text = self.domain.class_vars[0].values[int(modus)] + "<br/>"
        if tabs > 0.999:
            text += "100%, {}/{}".format(total, total)
        else:
            text += "{:2.1f}%, {}/{}".format(100 * tabs,
                                             int(total * tabs), total)

        text = self._update_node_info_attr_name(node, text)
        node.setHtml('<p style="line-height: 120%; margin-bottom: 0">'
                     '{}</p>'.
                     format(text))

    def update_node_info_reg(self, node):
        """Update the printed contents of the node for regression trees"""
        node_inst = node.node_inst
        mean, var = node_inst.value
        insts = len(node_inst.subset)
        text = "{:.1f} ± {:.1f}<br/>".format(mean, var)
        text += "{} instances".format(insts)
        text = self._update_node_info_attr_name(node, text)
        node.setHtml('<p style="line-height: 120%; margin-bottom: 0">{}</p>'.
                     format(text))

    def toggle_node_color_cls(self):
        """Update the node color for classification trees"""
        colors = self.scene.colors
        for node in self.scene.nodes():
            distr = node.node_inst.value
            total = sum(distr)
            if self.target_class_index:
                p = distr[self.target_class_index - 1] / total
                color = colors[self.target_class_index - 1].lighter(
                    200 - 100 * p)
            else:
                modus = np.argmax(distr)
                p = distr[modus] / (total or 1)
                color = colors[int(modus)].lighter(300 - 200 * p)
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
            max_insts = len(self.model.instances)
            for node in self.scene.nodes():
                node.backgroundBrush = QBrush(def_color.lighter(
                    120 - 20 * len(node.node_inst.subset) / max_insts))
        elif self.regression_colors == self.COL_MEAN:
            minv = np.nanmin(self.dataset.Y)
            maxv = np.nanmax(self.dataset.Y)
            fact = 1 / (maxv - minv) if minv != maxv else 1
            colors = self.scene.colors
            for node in self.scene.nodes():
                node.backgroundBrush = QBrush(
                    colors[fact * (node.node_inst.value[0] - minv)])
        else:
            nodes = list(self.scene.nodes())
            variances = [node.node_inst.value[1] for node in nodes]
            max_var = max(variances)
            for node, var in zip(nodes, variances):
                node.backgroundBrush = QBrush(def_color.lighter(
                    120 - 20 * var / max_var))
        self.scene.update()


def test():
    """Standalone test"""
    import sys
    from AnyQt.QtWidgets import QApplication
#    from Orange.classification.tree import TreeLearner
    from Orange.regression.tree import TreeLearner
    a = QApplication(sys.argv)
    ow = OWTreeGraph()
    # data = Table("iris")
    # data = Table("housing")
    # clf = TreeLearner()(data)
    from Orange.classification.tree import SklTreeLearner
    data = Table(sys.argv[1] if len(sys.argv) > 1 else "iris")
    clf = SklTreeLearner()(data)
    clf.instances = data

    ow.ctree(clf)
    ow.show()
    ow.raise_()
    a.exec_()
    ow.saveSettings()

if __name__ == "__main__":
    test()

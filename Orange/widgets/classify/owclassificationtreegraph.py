"""Widget for visualization of tree models"""

from PyQt4.QtCore import Qt, QRectF, QPointF, QSizeF
from PyQt4.QtGui import  QColor, QBrush, QPen, QFontMetrics, QStyle, \
    QSizePolicy, QGraphicsRectItem, QGraphicsTextItem

from Orange.base import Tree
from Orange.widgets.classify.owtreeviewer2d import \
    GraphicsNode, GraphicsEdge, OWTreeViewer2D
from Orange.data import Table

from Orange.widgets.settings import ContextSetting, ClassValuesContextHandler
from Orange.widgets import gui


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
        painter.setPen(QPen(Qt.white))
        painter.setBrush(QBrush())
        painter.drawEllipse(-self.r, -self.r, 2 * self.r, 2 * self.r)


class TreeNode(GraphicsNode):
    """TreeNode for trees corresponding to base.Tree models"""
    # Methods are documented in PyQt documentation
    # pylint: disable=missing-docstring

    def __init__(self, model, node_id, parent=None):
        super().__init__(parent)
        self.model = model
        self.node_id = node_id

        fm = QFontMetrics(self.document().defaultFont())
        attr = model.attribute(node_id)
        self.attr_text_w = fm.width(attr.name if attr else "")
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        self._rect = None

    def update_contents(self):
        self.prepareGeometryChange()
        self.setTextWidth(-1)
        self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        fm = QFontMetrics(self.document().defaultFont())
        attr = self.model.attribute(self.node_id)
        self.attr_text_w = fm.width(attr.name if attr else "")
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()

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
            draw_text = self.model.split_condition(
                self.node_id, self.parent.node_id)
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
        if self.model.is_leaf(self.node_id):
            painter.drawRoundedRect(adjrect, 4, 4)
        else:
            painter.drawRect(adjrect)
        painter.restore()
        painter.setClipRect(rect)
        return QGraphicsTextItem.paint(self, painter, option, widget)


class ClassificationTreeNode(TreeNode):
    def __init__(self, model, node_id, parent=None):
        super().__init__(model, node_id, parent)
        self.pie = PieChart(self.model.get_distribution(), 8, self)

    def update_contents(self):
        super().update_contents()
        self.pie.setPos(self.rect().right(), self.rect().center().y())


class OWTreeGraph(OWTreeViewer2D):
    """Graphical visualization of tree models"""

    name = "Classification Tree Viewer"
    icon = "icons/ClassificationTreeGraph.svg"
    priority = 35
    inputs = [("Classification Tree", Tree, "ctree")]
    outputs = [("Data", Table)]

    settingsHandler = ClassValuesContextHandler()
    target_class_index = ContextSetting(0)

    def __init__(self):
        super().__init__()
        self.domain = None
        self.dataset = None
        self.clf_dataset = None

        self.target_combo = gui.comboBox(
            None, self, "target_class_index", orientation=Qt.Horizontal,
            items=[], callback=self.toggle_color, contentsLength=8,
            addToLayout=False,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed))
        self.display_box.layout().addRow("Target class: ", self.target_combo)
        gui.rubber(self.controlArea)

    def set_node_info(self):
        """Set the content of the node"""
        for node in self.scene.nodes():
            node.set_rect(QRectF())
            self.update_node_info(node)
        w = max([n.rect().width() for n in self.scene.nodes()] + [0])
        if w > self.max_node_width:
            w = self.max_node_width
        for node in self.scene.nodes():
            node.set_rect(QRectF(node.rect().x(), node.rect().y(),
                                 w, node.rect().height()))
        self.scene.fix_pos(self.root_node, 10, 10)

    def _update_node_info_attr_name(self, node, text):
        attr = self.model.data_attribute(node.node_id)
        if attr is not None:
            text += "<hr/>{}".format(attr.name)
        return text

    def activate_loaded_settings(self):
        if not self.model:
            return
        super().activate_loaded_settings()
        self.set_node_info()
        self.toggle_node_color()

    def toggle_node_size(self):
        self.set_node_info()
        self.scene.update()
        self.scene_view.repaint()

    def toggle_color(self):
        self.toggle_node_color()
        self.set_node_info()
        self.scene.update()

    def ctree(self, model=None):
        """Input signal handler"""
        self.clear_scene()
        self.target_combo.clear()
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
                self.target_combo.addItem("None")
                self.target_combo.addItems(self.domain.class_vars[0].values)
                self.target_combo.setCurrentIndex(self.target_class_index)
            self.openContext(self.domain.class_var)
            self.root_node = self.walkcreate(model.root, None)
            self.scene.addItem(self.root_node)
            self.info.setText('{} nodes, {} leaves'.
                              format(model.node_count, model.leaf_count))
        self.setup_scene()
        self.send("Data", None)

    def walkcreate(self, node_id, parent=None):
        """Create a structure of tree nodes from the given model"""
        node = TreeNode(self.model, node_id, parent)
        self.scene.addItem(node)
        if parent:
            edge = GraphicsEdge(node1=parent, node2=node)
            self.scene.addItem(edge)
            parent.graph_add_edge(edge)
        for child in self.model.children(node_id):
            self.walkcreate(child, node)
        return node

    def node_tooltip(self, node):
        path_to_root = [node.node_id]
        while node.parent:
            node = node.parent
            path_to_root.append(node.node_id)
        # don't use reversed - skl trees don't like reverseiterators
        return self.model.rule(path_to_root[::-1])

    def update_selection(self):
        if self.model is None:
            return
        node_indices = [item.node_id for item in self.scene.selectedItems()
                        if isinstance(item, TreeNode)]
        data = self.model.get_instances(node_indices)
        self.send("Data", data)

    def send_report(self):
        if not self.model:
            return
        self.report_items((
            ("Tree size", self.info.text()),
            ("Edge widths",
             ("Fixed", "Relative to root", "Relative to parent")[
                 # pylint: disable=invalid-sequence-index
                 self.line_width_method]),
            ("Target class", self.target_combo.currentText())))
        self.report_plot(self.scene)

    def update_node_info(self, node):
        """Update the printed contenst of the node"""
        node_id = node.node_id
        model = self.model
        distr = model.get_distribution(node_id)
        total = model.num_instances(node_id)
        distr = distr / sum(distr)
        if self.target_class_index:
            tabs = distr[self.target_class_index - 1]
            text = ""
        else:
            modus = model.majority(node_id)
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

    def toggle_node_color(self):
        """Update the node color"""
        colors = self.scene.colors
        for node in self.scene.nodes():
            distr = self.model.get_distribution(node.node_id)
            total = sum(distr)
            if self.target_class_index:
                p = distr[self.target_class_index - 1] / total
                color = colors[self.target_class_index - 1].lighter(200 - 100 * p)
            else:
                modus = self.model.majority(node.node_id)
                p = distr[modus] / (total or 1)
                color = colors[int(modus)].lighter(400 - 300 * p)
            node.backgroundBrush = QBrush(color)
        self.scene.update()


def test():
    """Standalone test"""
    import sys
    from PyQt4.QtGui import QApplication
    from Orange.classification.tree import OrangeTreeLearner
    a = QApplication(sys.argv)
    ow = OWTreeGraph()
    data = Table("iris")
    clf = OrangeTreeLearner(max_depth=3)(data)
    clf.instances = data

    ow.ctree(clf)
    ow.show()
    ow.raise_()
    a.exec_()
    ow.saveSettings()

if __name__ == "__main__":
    test()

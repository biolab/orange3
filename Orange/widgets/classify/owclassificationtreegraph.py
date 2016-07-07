import sys
from collections import OrderedDict
import numpy

from sklearn.tree._tree import TREE_LEAF

# This is not needed yet because it is imported from owtreeviewer2d :(
from PyQt4.QtCore import Qt

from Orange.widgets.classify.owtreeviewer2d import *

from Orange.data import Table
from Orange.classification.tree import TreeClassifier
from Orange.preprocess.transformation import Indicator

from Orange.widgets.settings import ContextSetting, ClassValuesContextHandler
from Orange.widgets import gui


class OWTreeGraph(OWTreeViewer2D):
    priority = 35
    outputs = [("Data", Table)]

    def __init__(self):
        super().__init__()
        self.domain = None
        self.model = None
        self.dataset = None
        self.clf_dataset = None

        self.scene = TreeGraphicsScene(self)
        self.scene_view = TreeGraphicsView(self.scene)
        self.scene_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.mainArea.layout().addWidget(self.scene_view)
        self.toggle_zoom_slider()
        self.scene.selectionChanged.connect(self.update_selection)

    def set_node_info(self):
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
        if not node.is_leaf():
            attribute = self.domain.attributes[node.attribute()]
            if isinstance(attribute.compute_value, Indicator):
                attribute = attribute.compute_value.variable
            text += "<hr/>{}".format(attribute.name)
        return text

    def activate_loaded_settings(self):
        if not self.tree:
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
        self.clear()
        self.closeContext()
        self.model = model
        if model is None:
            self.info.setText('No tree.')
            self.tree = None
            self.root_node = None
            self.dataset = None
        else:
            self.tree = model.skl_model.tree_
            self.domain = model.domain
            self.dataset = getattr(model, "instances", None)
            if self.dataset is not None and self.dataset.domain != self.domain:
                self.clf_dataset = \
                    Table.from_table(self.model.domain, self.dataset)
            else:
                self.clf_dataset = self.dataset
            class_var = self.domain.class_var
            if class_var.is_discrete:
                self.scene.colors = [QColor(*col) for col in class_var.colors]
            self.openContext(self.domain.class_var)
            self.root_node = self.walkcreate(self.tree, 0, None)
            self.scene.addItem(self.root_node)
            self.info.setText(
                '{} nodes, {} leaves'.
                format(self.tree.node_count,
                       numpy.count_nonzero(
                            self.tree.children_left == TREE_LEAF)))

            self.scene.fix_pos(self.root_node, self._HSPACING, self._VSPACING)
            self.activate_loaded_settings()
            self.scene_view.centerOn(self.root_node.x(), self.root_node.y())
            self.update_node_tooltips()
        self.scene.update()
        self.send("Data", None)

    def walkcreate(self, tree, node_id, parent=None):
        node = self.NODE(tree, self.domain, parent, i=node_id)
        self.scene.addItem(node)
        if parent:
            edge = GraphicsEdge(node1=parent, node2=node)
            self.scene.addItem(edge)
            parent.graph_add_edge(edge)
        left_child_index = tree.children_left[node_id]
        right_child_index = tree.children_right[node_id]

        if left_child_index != TREE_LEAF:
            self.walkcreate(tree, node_id=left_child_index, parent=node)
        if right_child_index != TREE_LEAF:
            self.walkcreate(tree, node_id=right_child_index, parent=node)
        return node

    def node_tooltip(self, node):
        if node.i > 0:
            text = " AND\n".join(
                "%s %s %s" % (n, s, v) for (n, s), v in node.rule().items())
        else:
            text = "Root"
        return text

    def update_selection(self):
        if self.dataset is None or self.model is None or self.tree is None:
            return
        items = [item for item in self.scene.selectedItems()
                 if isinstance(item, self.NODE)]

        selected_leaves = [_leaf_indices(self.tree, item.node_id)
                           for item in items]

        if selected_leaves:
            selected_leaves = numpy.unique(numpy.hstack(selected_leaves))

        all_leaves = _leaf_indices(self.tree, 0)

        if len(selected_leaves) > 0:
            ind = numpy.searchsorted(all_leaves, selected_leaves, side="left")
            leaf_samples = _assign_samples(self.tree, self.clf_dataset.X)
            leaf_samples = [leaf_samples[i] for i in ind]
            indices = numpy.hstack(leaf_samples)
        else:
            indices = []

        if len(indices):
            data = self.dataset[indices]
        else:
            data = None
        self.send("Data", data)

    def send_report(self):
        if not self.tree:
            return
        self.report_items((
            ("Tree size", self.info.text()),
            ("Edge widths",
             ("Fixed", "Relative to root", "Relative to parent")[
                 self.line_width_method]),
            ("Target class", self.target_combo.currentText())))
        self.report_plot(self.scene)


class PieChart(QGraphicsRectItem):
    def __init__(self, dist, r, parent):
        super().__init__(parent)
        self.dist = dist
        self.r = r

    # noinspection PyPep8Naming
    def setR(self, r):
        self.prepareGeometryChange()
        self.r = r

    def boundingRect(self):
        return QRectF(-self.r, -self.r, 2*self.r, 2*self.r)

    def paint(self, painter, option, widget=None):
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


def _subnode_range(tree, node_id):
    right = left = node_id
    if tree.children_left[left] == TREE_LEAF:
        assert tree.children_right[node_id] == TREE_LEAF
        return node_id, node_id
    else:
        left = tree.children_left[left]
        # run down to the right most node
        while tree.children_right[right] != TREE_LEAF:
            right = tree.children_right[right]

        return left, right + 1


def _leaf_indices(tree, node_id):
    start, stop = _subnode_range(tree, node_id)
    if start == stop:
        # leaf
        return numpy.array([node_id], dtype=int)
    else:
        isleaf = tree.children_left[start: stop] == TREE_LEAF
        assert numpy.flatnonzero(isleaf).size > 0
        return start + numpy.flatnonzero(isleaf)


def _assign_samples(tree, X):
    def assign(node_id, indices):
        if tree.children_left[node_id] == TREE_LEAF:
            return [indices]
        else:
            feature_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]

            column = X[indices, feature_idx]
            leftmask = column <= thresh
            leftind = assign(tree.children_left[node_id], indices[leftmask])
            rightind = assign(tree.children_right[node_id], indices[~leftmask])
            return list.__iadd__(leftind, rightind)

    N, _ = X.shape

    items = numpy.arange(N, dtype=int)
    leaf_indices = assign(0, items)
    return leaf_indices


class TreeNode(GraphicsNode):
    def __init__(self, tree, domain, parent=None, parent_item=None,
                 i=0, distr=None):
        super().__init__(tree, parent, parent_item)
        self.distribution = distr
        self.tree = tree
        self.domain = domain
        self.i = i
        self.node_id = i
        self.parent = parent
        fm = QFontMetrics(self.document().defaultFont())
        self.attr_text_w = fm.width(str(self.attribute() if self.attribute()
                                        else ""))
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        self._rect = None

    def num_instances(self):
        """
        :return: Number of instances in a particular node.
        """
        return self.tree.n_node_samples[self.i]

    def split_condition(self):
        """
        :return: split condition to reach a particular node.
        """
        if self.i > 0:
            parent_attr = self.domain.attributes[self.parent.attribute()]
            parent_attr_cv = parent_attr.compute_value
            is_left_child = self.tree.children_left[self.parent.i] == self.i
            if isinstance(parent_attr_cv, Indicator) and \
                    hasattr(parent_attr_cv.variable, "values"):
                values = parent_attr_cv.variable.values
                return values[abs(parent_attr_cv.value - is_left_child)] \
                    if len(values) == 2 \
                    else "≠ " * is_left_child + values[parent_attr_cv.value]
            else:
                thresh = self.tree.threshold[self.parent.i]
                return "%s %s" % ([">", "≤"][is_left_child],
                                  parent_attr.str_val(thresh))
        else:
            return ""

    def rule(self):
        """
        :return:
            Rule to reach node as list of tuples (attr index, sign, threshold)
        """
        # TODO: this is easily extended to Classification Rules-compatible form
        return self.rulew(i=self.i)

    def rulew(self, i=0):
        """
        :param i:
            Index of current node.
        :return:
            Rule to reach node i, represented as list of tuples (attr name,
            sign, threshold)
        """
        if i > 0:
            parent_attr = self.domain.attributes[self.parent.attribute()]
            parent_attr_cv = parent_attr.compute_value
            is_left_child = self.tree.children_left[self.parent.i] == i
            pr = self.parent.rule()
            if isinstance(parent_attr_cv, Indicator) and \
                    hasattr(parent_attr_cv.variable, "values"):
                values = parent_attr_cv.variable.values
                attr_name = parent_attr_cv.variable.name
                sign = ["=", "≠"][is_left_child * (len(values) != 2)]
                value = values[abs(parent_attr_cv.value -
                                   is_left_child * (len(values) == 2))]
            else:
                attr_name = parent_attr.name
                sign = [">", "≤"][is_left_child]
                value = "%.3f" % self.tree.threshold[self.parent.i]
            if (attr_name, sign) in pr:
                old_val = pr[(attr_name, sign)]
                if sign == ">":
                    pr[(attr_name, sign)] = max(float(value), float(old_val))
                elif sign == "≠":
                    pr[(attr_name, sign)] = "{}, {}".format(old_val, value)
                elif sign == "≤":
                    pr[(attr_name, sign)] = min(float(value), float(old_val))
            else:
                pr[(attr_name, sign)] = value
            return pr
        else:
            return OrderedDict()

    def is_leaf(self):
        """
        :return: Node is leaf
        """
        return self.tree.children_left[self.node_id] < 0 and \
            self.tree.children_right[self.node_id] < 0

    def attribute(self):
        """
        :return: Node attribute index.
        """
        return self.tree.feature[self.node_id]

    def update_contents(self):
        self.prepareGeometryChange()
        self.setTextWidth(-1)
        self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        fm = QFontMetrics(self.document().defaultFont())
        self.attr_text_w = fm.width(str(self.attribute() if self.attribute()
                                        else ""))
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
        painter.setFont(self.document().defaultFont())
        draw_text = str(self.split_condition())
        painter.drawText(QPointF(4, -self.line_descent - 1), draw_text)
        painter.save()
        painter.setBrush(self.backgroundBrush)
        if self.isSelected():
            painter.setPen(QPen(QBrush(Qt.black), 2))
        else:
            painter.setPen(QPen(Qt.gray))
        if self.is_leaf():
            painter.drawRect(rect.adjusted(-3, 0, 0, 0))
        else:
            painter.drawRoundedRect(rect.adjusted(-3, 0, 0, 0), 4, 4)
        painter.restore()
        painter.setClipRect(rect)
        return QGraphicsTextItem.paint(self, painter, option, widget)


class ClassificationTreeNode(TreeNode):
    def __init__(self, tree, domain, parent=None, parent_item=None,
                 i=0, distr=None):
        super().__init__(tree, domain, parent, parent_item,
                         i, distr)
        self.pie = PieChart(self.get_distribution(), 8, self)

    def get_distribution(self):
        """
        :return: Distribution of class values.
        """
        if self.is_leaf():
            counts = self.tree.value[self.node_id]
        else:
            leaf_ind = _leaf_indices(self.tree, self.node_id)
            values = self.tree.value[leaf_ind]
            counts = numpy.sum(values, axis=0)

        assert counts.shape[0] == 1, "n_outputs > 1 "
        counts = counts[0]
        counts_sum = numpy.sum(counts)
        if counts_sum > 0:
            counts = counts / counts_sum
        return counts

    def majority(self):
        """
        :return:
            Majority class at node.
        """
        return numpy.argmax(self.get_distribution())

    def update_contents(self):
        super().update_contents()
        self.pie.setPos(self.rect().right(), self.rect().center().y())


class OWClassificationTreeGraph(OWTreeGraph):
    name = "Classification Tree Viewer"
    description = "A graphical visualization of a classification tree."
    icon = "icons/ClassificationTreeGraph.svg"

    settingsHandler = ClassValuesContextHandler()
    target_class_index = ContextSetting(0)

    inputs = [("Classification Tree", TreeClassifier, "ctree")]
    NODE = ClassificationTreeNode

    def __init__(self):
        super().__init__()
        self.target_combo = gui.comboBox(
            None, self, "target_class_index", orientation=Qt.Horizontal,
            items=[], callback=self.toggle_color, contentsLength=8,
            addToLayout=False,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed))
        self.display_box.layout().addRow("Target class: ", self.target_combo)
        gui.rubber(self.controlArea)

    def ctree(self, model=None):
        super().ctree(model)
        if model is not None:
            self.target_combo.clear()
            self.target_combo.addItem("None")
            self.target_combo.addItems(self.domain.class_vars[0].values)
            self.target_combo.setCurrentIndex(self.target_class_index)

    def update_node_info(self, node):
        distr = node.get_distribution()
        total = int(node.num_instances())
        if self.target_class_index:
            tabs = distr[self.target_class_index - 1]
            text = ""
        else:
            modus = node.majority()
            tabs = distr[modus]
            text = self.domain.class_vars[0].values[modus] + "<br/>"
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
        colors = self.scene.colors
        for node in self.scene.nodes():
            distr = node.get_distribution()
            total = numpy.sum(distr)
            if self.target_class_index:
                p = distr[self.target_class_index - 1] / total
                color = colors[self.target_class_index - 1].lighter(200 - 100 * p)
            else:
                modus = node.majority()
                p = distr[modus] / (total or 1)
                color = colors[int(modus)].lighter(400 - 300 * p)
            node.backgroundBrush = QBrush(color)
        self.scene.update()


if __name__ == "__main__":
    from Orange.classification.tree import TreeLearner
    a = QApplication(sys.argv)
    ow = OWClassificationTreeGraph()
    data = Table("iris")
    clf = TreeLearner(max_depth=3)(data)
    clf.instances = data

    ow.ctree(clf)
    ow.show()
    ow.raise_()
    a.exec_()
    ow.saveSettings()

import sys

import numpy

from sklearn.tree._tree import TREE_LEAF

from Orange.widgets.classify.owtreeviewer2d import *

from Orange.data import Table
from Orange.classification.tree import TreeClassifier
from Orange.widgets.utils.colorpalette import ColorPaletteDlg

from Orange.widgets.settings import \
    Setting, ContextSetting, ClassValuesContextHandler
from Orange.widgets import gui


class OWClassificationTreeGraph(OWTreeViewer2D):
    name = "Classification Tree Viewer"
    description = "Classification Tree Viewer"
    icon = "icons/ClassificationTree.svg"

    settingsHandler = ClassValuesContextHandler()
    target_class_index = ContextSetting(0)
    color_settings = Setting(None)
    selected_color_settings_index = Setting(0)

    inputs = [("Classification Tree", TreeClassifier, "ctree")]
    outputs = [("Data", Table)]

    def __init__(self):
        super().__init__()
        self.domain = None
        self.classifier = None
        self.dataset = None

        self.scene = TreeGraphicsScene(self)
        self.scene_view = TreeGraphicsView(self.scene)
        self.scene_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.mainArea.layout().addWidget(self.scene_view)
        self.toggle_zoom_slider()
        self.scene.selectionChanged.connect(self.update_selection)

        box = gui.widgetBox(self.controlArea, "Nodes", addSpace=True)
        self.target_combo = gui.comboBox(
            box, self, "target_class_index", orientation=0, items=[],
            label="Target class", callback=self.toggle_target_class)
        gui.separator(box)
        gui.button(box, self, "Set Colors", callback=self.set_colors)
        dlg = self.create_color_dialog()
        self.scene.colorPalette = dlg.getDiscretePalette("colorPalette")
        gui.rubber(self.controlArea)

    def sendReport(self):
        if self.tree:
            tclass = str(self.targetCombo.currentText())
            tsize = "%i nodes, %i leaves" % (orngTree.countNodes(self.tree),
                                             orngTree.countLeaves(self.tree))
        else:
            tclass = tsize = "N/A"
        self.reportSettings(
            "Information",
            [("Target class", tclass),
             ("Line widths",
                 ["Constant", "Proportion of all instances",
                  "Proportion of parent's instances"][self.line_width_method]),
             ("Tree size", tsize)])
        super().sendReport()

    def set_colors(self):
        dlg = self.create_color_dialog()
        if dlg.exec_():
            self.color_settings = dlg.getColorSchemas()
            self.selected_color_settings_index = dlg.selectedSchemaIndex
            self.scene.colorPalette = dlg.getDiscretePalette("colorPalette")
            self.scene.update()
            self.toggle_node_color()

    def create_color_dialog(self):
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("colorPalette", "Discrete Palette")
        c.setColorSchemas(self.color_settings,
                          self.selected_color_settings_index)
        return c

    def set_node_info(self):
        for node in self.scene.nodes():
            node.set_rect(QRectF())
            self.update_node_info(node)
        w = max([n.rect().width() for n in self.scene.nodes()] + [0])
        if w > self.max_node_width < 200:
            w = self.max_node_width
        for node in self.scene.nodes():
            node.set_rect(QRectF(node.rect().x(), node.rect().y(),
                                 w, node.rect().height()))
        self.scene.fix_pos(self.root_node, 10, 10)

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
        if not node.is_leaf():
            text += "<hr/>{}".format(
                self.domain.attributes[node.attribute()].name)
        node.setHtml('<center><p style="line-height: 120%; margin-bottom: 0">'
                     '{}</p></center>'.
                     format(text))

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

    def toggle_node_color(self):
        palette = self.scene.colorPalette
        for node in self.scene.nodes():
            distr = node.get_distribution()
            total = numpy.sum(distr)
            if self.target_class_index:
                p = distr[self.target_class_index - 1] / total
                color = palette[self.target_class_index].light(200 - 100 * p)
            else:
                modus = node.majority()
                p = distr[modus] / (total or 1)
                color = palette[int(modus)].light(400 - 300 * p)
            node.backgroundBrush = QBrush(color)
        self.scene.update()

    def toggle_target_class(self):
        self.toggle_node_color()
        self.set_node_info()
        self.scene.update()

    def ctree(self, clf=None):
        self.clear()
        self.closeContext()
        self.classifier = clf
        if clf is None:
            self.info.setText('No tree.')
            self.tree = None
            self.root_node = None
            self.dataset = None
        else:
            self.tree = clf.skl_model.tree_
            self.domain = clf.domain
            self.dataset = getattr(clf, "instances", None)
            self.target_combo.clear()
            self.target_combo.addItem("None")
            self.target_combo.addItems(self.domain.class_vars[0].values)
            self.target_class_index = 0
            self.openContext(self.domain.class_var)
            self.root_node = self.walkcreate(self.tree, 0, None)
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

    def walkcreate(self, tree, node_id, parent=None):
        node = ClassificationTreeNode(tree, self.domain, parent, None,
                                      self.scene, i=node_id)
        if parent:
            parent.graph_add_edge(
                GraphicsEdge(None, self.scene, node1=parent, node2=node))
        left_child_index = tree.children_left[node_id]
        right_child_index = tree.children_right[node_id]

        if left_child_index != TREE_LEAF:
            self.walkcreate(tree, node_id=left_child_index, parent=node)
        if right_child_index != TREE_LEAF:
            self.walkcreate(tree, node_id=right_child_index, parent=node)
        return node

    def node_tooltip(self, node):
        if node.i > 0:
            text = "<br/> AND ".join(
                "%s %s %.3f" % (self.domain.attributes[a].name, s, t)
                for a, s, t in node.rule())
        else:
            text = "Root"
        return text

    def update_selection(self):
        if self.dataset is None or self.classifier is None or self.tree is None:
            return
        data = self.dataset
        if data.domain != self.classifier.domain:
            self.dataset = data.from_table(self.classifier.domain, data)

        items = [item for item in self.scene.selectedItems()
                 if isinstance(item, ClassificationTreeNode)]

        selected_leaves = [_leaf_indices(self.tree, item.node_id)
                           for item in items]

        if selected_leaves:
            selected_leaves = numpy.unique(numpy.hstack(selected_leaves))

        all_leaves = _leaf_indices(self.tree, 0)

        if len(selected_leaves) > 0:
            ind = numpy.searchsorted(all_leaves, selected_leaves, side="left")
            leaf_samples = _assign_samples(self.tree, self.dataset.X)
            leaf_samples = [leaf_samples[i] for i in ind]
            indices = numpy.hstack(leaf_samples)
        else:
            indices = []

        if len(indices):
            data = self.dataset[indices]
        else:
            data = None
        self.send("Data", data)


class PieChart(QGraphicsRectItem):
    def __init__(self, dist, r, parent, scene):
        super().__init__(parent, scene)
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
        colors = self.scene().colorPalette
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


class ClassificationTreeNode(GraphicsNode):
    def __init__(self, tree, domain, parent=None, parent_item=None,
                 scene=None, i=0, distr=None):
        super().__init__(tree, parent, parent_item, scene)
        self.distribution = distr
        self.tree = tree
        self.domain = domain
        self.i = i
        self.node_id = i
        self.parent = parent
        self.pie = PieChart(self.get_distribution(), 8, self, scene)
        fm = QFontMetrics(self.document().defaultFont())
        self.attr_text_w = fm.width(str(self.attribute() if self.attribute()
                                        else ""))
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        self._rect = None

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
            counts /= counts_sum
        return counts

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
            sign = [">", "<="][self.tree.children_left[self.parent.i] == self.i]
            thresh = self.tree.threshold[self.parent.i]
            return "%s %s" % (
                sign, self.domain.attributes[self.attribute()].str_val(thresh))
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
            Rule to reach node i, represented as list of tuples (attr index,
            sign, threshold)
        """
        if i > 0:
            sign = "<=" if self.tree.children_left[self.parent.i] == i else ">"
            thresh = self.tree.threshold[self.parent.i]
            attr = self.parent.attribute()
            pr = self.parent.rule()
            pr.append((attr, sign, thresh))
            return pr
        else:
            return []

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

    def majority(self):
        """
        :return:
            Majority class at node.
        """
        return numpy.argmax(self.get_distribution())

    def update_contents(self):
        self.prepareGeometryChange()
        self.setTextWidth(-1)
        self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        self.pie.setPos(self.rect().right(), self.rect().center().y())
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

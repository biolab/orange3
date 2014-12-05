import sys
from Orange.widgets.classify.owtreeviewer2d import *

from Orange.data import Table
from Orange.classification.tree import ClassificationTreeLearner
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.widget import OWWidget
from numpy import argmax, zeros
from Orange.widgets.settings import \
    Setting, ContextSetting, PerfectDomainContextHandler
from Orange.widgets import gui


#BodyColor_Default = QColor(Qt.gray)
BodyColor_Default = QColor(255, 225, 10)
BodyCasesColor_Default = QColor(Qt.blue) #QColor(0, 0, 128)


class OWClassificationTreeGraph(OWTreeViewer2D):
    name = "Classification Tree Viewer"
    description = "Classification Tree Viewer"
    icon = "icons/ClassificationTree.svg"

    settingsHandler = PerfectDomainContextHandler()
    show_pies = Setting(True)
    color_settings = Setting(None)
    selected_color_settings_index = Setting(0)
    show_node_info_text = Setting(False)
    node_color_method = Setting(2)
    color_method_box = Setting(None)

    target_class_index = ContextSetting(0)

    inputs = [("ClassificationTree", ClassificationTreeClassifier, "ctree")]
    outputs = [("Examples", Table)]

    node_color_opts = [
        'Default', 'Instances in node', 'Majority class probability',
        'Target class probability', 'Target class distribution']
    node_info_buttons = [
        'Majority class', 'Majority class probability',
        'Target class probability', 'Number of instances']

    def __init__(self):
        super().__init__()

        self.scene = TreeGraphicsScene(self)
        self.scene_view = TreeGraphicsView(self.scene)
        self.scene_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.mainArea.layout().addWidget(self.scene_view)
        self.toggle_zoom_slider()

        self.scene.selectionChanged.connect(self.update_selection)

        # self.navWidget = OWWidget(self)
        # self.navWidget.lay = QVBoxLayout(self.navWidget)

        # scene = TreeGraphicsScene(self.navWidget)
        # self.treeNav = TreeNavigator(self.sceneView)
        # self.navWidget.lay.addWidget(self.treeNav)
        # self.navWidget.resize(400,400)
        # self.navWidget.setWindowTitle("Navigator")
        # self.setMouseTracking(True)

        colorbox = gui.widgetBox(self.controlArea, "Nodes", addSpace=True)

        self.color_method_box = gui.comboBox(
            colorbox, self, 'node_color_method', items=self.node_color_opts,
            callback=self.toggle_node_color)
        self.target_combo = gui.comboBox(
            colorbox, self, "target_class_index", orientation=0, items=[],
            label="Target class", callback=self.toggle_target_class)
        gui.checkBox(colorbox, self, 'show_pies',
                     'Show distribution pie charts',
                     tooltip='Show pie graph with class distribution',
                     callback=self.toggle_pies)
        gui.separator(colorbox)
        gui.button(colorbox, self, "Set Colors", callback=self.set_colors)

        node_info_box = gui.widgetBox(self.controlArea, "Show Info")
        node_info_settings = ['maj', 'majp', 'tarp', 'inst']
        self.node_info_w = []
        self.dummy = 0
        for i in range(len(self.node_info_buttons)):
            setattr(self, node_info_settings[i], i in self.node_info)
            w = gui.checkBox(
                node_info_box, self, node_info_settings[i],
                self.node_info_buttons[i], callback=self.set_node_info,
                getwidget=True)
            self.node_info_w.append(w)

        # gui.button(self.controlArea, self, "Save as", callback=self.saveGraph)
        self.node_info_sorted = list(self.node_info)
        self.node_info_sorted.sort()

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
            [("Node color", self.node_color_opts[self.node_color_method]),
             ("Target class", tclass),
             ("Data in nodes", ", ".join(
                 s for i, s in enumerate(self.node_info_buttons)
                 if self.node_info_w[i].isChecked())),
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

    def set_node_info(self, widget=None, id=None):
        flags = sum(2 ** i
                    for i, name in enumerate(['maj', 'majp', 'tarp', 'inst'])
                    if getattr(self, name))
        for n in self.scene.nodes():
            n.set_rect(QRectF())
            self.update_node_info(n, flags)
        w = max([n.rect().width() for n in self.scene.nodes()] + [0])
        if w > self.max_node_width < 200:
            w = self.max_node_width
        for n in self.scene.nodes():
            n.set_rect(QRectF(n.rect().x(), n.rect().y(), w, n.rect().height()))
        self.scene.fix_pos(self.root_node, 10, 10)

    def update_node_info(self, node, flags=31):
        lines = []
        if flags & 1:
            lines.append(self.domain.class_vars[0].values[node.majority()])
        if flags & 2:
            lines.append("%.3f" % node.get_distribution()[node.majority()])
        if flags & 4:
            lines.append("%.3f" % node.get_distribution()[self.target_class_index])
        if flags & 8:
            lines.append(str(node.num_instances()))
        text = "<br>".join(lines)
        text += "<hr>"
        if node.is_leaf():
            text += self.domain.class_vars[0].values[node.majority()]
        else:
            text += self.domain.attributes[node.attribute()].name
        node.setHtml(text)

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
        self.node_color_method = \
            self.node_color_opts.index(self.color_method_box.currentText())
        palette = self.scene.colorPalette
        for node in self.scene.nodes():
            # dist = node.tree.distribution
            if self.node_color_method == 0:
                color = BodyColor_Default
            # number of instances in node
            elif self.node_color_method == 1:
                all_cases = self.root_node.num_nodes()
                light = 200 - 100 * node.num_nodes() / (all_cases or 1)
                color = BodyCasesColor_Default.light(light)
            # majority class probability
            elif self.node_color_method == 2:
                modus = node.majority()
                p = node.get_distribution()[modus] / \
                    sum(node.get_distribution())
                light = 400 - 300 * p
                color = palette[int(modus)].light(light)
            # target class probability
            elif self.node_color_method == 3:
                p = node.get_distribution()[self.target_class_index] / \
                    sum(node.get_distribution())
                light = 200 - 100 * p
                color = palette[self.target_class_index].light(light)
            # target class distribution
            elif self.node_color_method == 4:
                all_target = int(
                    self.root_node.get_distribution()[self.target_class_index] *
                    self.root_node.num_instances())
                light = 200 - 100 * node.num_instances() * \
                    node.get_distribution()[self.target_class_index] / \
                    all_target
                color = palette[self.target_class_index].light(light)
            node.backgroundBrush = QBrush(color)
        self.scene.update()

    def toggle_target_class(self):
        if self.node_color_method in [3, 4]:
            self.toggle_node_color()
        if self.tarp:
            self.set_node_info()
        self.scene.update()

    def toggle_pies(self):
        for n in self.scene.nodes():
            n.pie.setVisible(self.show_pies and n.isVisible())
        self.scene.update()

    def ctree(self, clf=None):
        self.clear()
        if not clf:
            #self.centerRootButton.setDisabled(1)
            #self.centerNodeButton.setDisabled(0)
            self.infoa.setText('No tree.')
            self.infob.setText('')
            self.tree = None
            self.root_node = None
        else:
            self.infoa.setText('Tree found on input.')
            self.tree = clf.clf.tree_
            self.domain = clf.domain
            for name in self.domain.class_vars[0].values:
                self.target_combo.addItem(name)
            self.root_node = self.walkcreate(self.tree, None, distr=clf.distr)
            self.infoa.setText('Number of nodes: ' + str(self.root_node.num_nodes()))
            self.infob.setText('Number of leaves: ' + str(self.root_node.num_leaves()))
            self.scene.fix_pos(self.root_node, self._HSPACING,self._VSPACING)
            self.activate_loaded_settings()
            self.scene_view.centerOn(self.root_node.x(), self.root_node.y())
            self.update_node_tooltips()
            #self.centerRootButton.setDisabled(0)
            #self.centerNodeButton.setDisabled(1)
        self.scene.update()

    def walkcreate(self, tree, parent=None, level=0, i=0, distr=None):
        node = ClassificationTreeNode(tree, parent, None, self.scene,
                                      i=i, distr=distr[i])
        if parent:
            parent.graph_add_edge(
                GraphicsEdge(None, self.scene, node1=parent, node2=node))
        left_child_index = tree.children_left[i]
        right_child_index = tree.children_right[i]
        if left_child_index >= 0:
            self.walkcreate(tree, parent=node, level=level+1,
                            i=left_child_index, distr=distr)
        if right_child_index >= 0:
            self.walkcreate(tree, parent=node, level=level+1,
                            i=right_child_index, distr=distr)
        return node

    def node_tooltip(self, node):
        if node.i > 0:
            text = " AND ".join(
                "%s %s %.3f" % (self.domain.attributes[a].name, s, t)
                for a, s, t in node.rule())
        else:
            text = "Root"
        return text


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
        painter.setPen(QPen(Qt.black))
        painter.setBrush(QBrush())
        painter.drawEllipse(-self.r, -self.r, 2 * self.r, 2 * self.r)


class ClassificationTreeNode(GraphicsNode):
    def __init__(self, tree, parent=None, parent_item=None,
                 scene=None, i=0, distr=None):
        super().__init__(tree, parent, parent_item, scene)
        self.distribution = distr
        self.tree = tree
        self.i = i
        self.parent = parent
        self.pie = PieChart(self.get_distribution(), 20, self, scene)
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
        d = zeros((self.tree.value.shape[2], ), dtype="float")
        for k, v in self.distribution.items():
            d[k] = v
        return list(d / d.sum())

    def num_nodes(self):
        """
        :return: Number of nodes below particular node.
        """
        return self.num_nodesw(self.i)

    def num_nodesw(self, i=0):
        """
        :param i: index of current node.
        :return: Number of nodes below particular node.
        """
        s = 1
        if self.tree.children_left[i] > 0:
            s += self.num_nodesw(i=self.tree.children_left[i])
        if self.tree.children_right[i] > 0:
            s += self.num_nodesw(i=self.tree.children_right[i])
        return s

    def num_instances(self):
        """
        :return: Number of instances in a particular node.
        """
        return self.tree.n_node_samples[self.i]

    def num_leaves(self, i=0):
        """
        :return: Number of leaves below a particular node.
        """
        return self.num_leavesw(i = self.i)

    def num_leavesw(self, i=0):
        """
        :param i: index of current node.
        :return: Number of leaves below particular node.
        """
        s = 0
        if self.tree.children_left[i] < 0 and self.tree.children_right[i] < 0:
            return 1
        if self.tree.children_left[i] > 0:
            s += self.num_leavesw(i=self.tree.children_left[i])
        if self.tree.children_right[i] > 0:
            s += self.num_leavesw(i=self.tree.children_right[i])
        return s

    def split_condition(self):
        """
        :return: split condition to reach a particular node.
        """
        if self.i > 0:  # Node is not root
            sign = [">", "<="][self.tree.children_left[self.parent.i] == self.i]
            thresh = self.tree.threshold[self.parent.i]
            return "%s %f" % (sign, thresh)
        else:
            return ""

    def rule(self):
        """
        :return:
            Rule to reach node
            Rules are represented as list is tuples (attribute index, sign, threshold)
        """
        # TODO: this is easily extended to Classification Rules-compatible form
        return self.rulew(i=self.i)

    def rulew(self, i=0):
        """
        :param i:
            Index of current node.
        :return:
            Rule to reach node i.
            Rules are represented as list is tuples (attribute index, sign, threshold)
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
        return self.tree.children_left[self.i] < 0 and \
               self.tree.children_right[self.i] < 0

    def attribute(self):
        """
        :return: Node attribute index.
        """
        return self.attributew(i=self.i)

    def attributew(self, i=0):
        """
        :return:
            Attribute at node to split on.
        """
        return self.tree.feature[i]

    def majority(self):
        """
        :return:
            Majority class at node.
        """
        return argmax(self.get_distribution())

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
            rect = QRectF(QPointF(0,0), self.document().size())
            return rect.adjusted(
                0, 0,
                self.pie.boundingRect().width() / 2 if hasattr(self, "pie")
                else 0, 0) | \
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
        if self.isSelected():
            option.state ^= QStyle.State_Selected
        if self.isSelected():
            painter.save()
            painter.setBrush(QBrush(QColor(125, 162, 206, 192)))
            painter.drawRoundedRect(
                self.boundingRect().adjusted(-2, 1, -1, -1), 10, 10)
            painter.restore()
        painter.setFont(self.document().defaultFont())
        # painter.drawText(QPointF(0, -self.line_descent),
        #                  str(self.attribute()) if self.attribute() else "")
        draw_text = str(self.split_condition())
        painter.drawText(QPointF(0, -self.line_descent), draw_text)
        painter.save()
        painter.setBrush(self.backgroundBrush)
        rect = self.rect()
        painter.drawRoundedRect(rect.adjusted(-3, 0, 0, 0), 10, 10)
        painter.restore()
        painter.setClipRect(rect)
        return QGraphicsTextItem.paint(self, painter, option, widget)

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWClassificationTreeGraph()
    from Orange.data import Table
    ow.ctree(ClassificationTreeLearner(max_depth=3)(Table('iris')))
    ow.show()
    a.exec_()
    ow.saveSettings()

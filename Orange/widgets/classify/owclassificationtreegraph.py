"""<name>Classification Tree Graph</name>
<description>Classification tree viewer (graph view).</description>
<icon>icons/ClassificationTreeGraph.svg</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>2110</priority>
"""

from owtreeviewer2d import *
# import OWColorPalette
from Orange.widgets.utils.plot import OWPalette as OWColorPalette
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.widget import OWWidget as OWBaseWidget
from Orange.classification.tree import ClassificationTreeWrapper
import Orange

class PieChart(QGraphicsRectItem):
    def __init__(self, dist, r, parent, scene):
        QGraphicsRectItem.__init__(self, parent, scene)
        self.dist = dist
        self.r = r
        
    def setR(self, r):
        self.prepareGeometryChange()
        self.r = r
        
    def boundingRect(self):
        return QRectF(-self.r, -self.r, 2*self.r, 2*self.r)
        
    def paint(self, painter, option, widget = None):
        distSum = sum(self.dist)
        startAngle = 0
        colors = self.scene().colorPalette
        for i in range(len(self.dist)):
            angle = self.dist[i]*16 * 360./distSum
            if angle == 0: continue
            painter.setBrush(QBrush(colors[i]))
            painter.setPen(QPen(colors[i]))
            painter.drawPie(-self.r, -self.r, 2*self.r, 2*self.r, int(startAngle), int(angle))
            startAngle += angle
        painter.setPen(QPen(Qt.black))
        painter.setBrush(QBrush())
        painter.drawEllipse(-self.r, -self.r, 2*self.r, 2*self.r)

class ClassificationTreeNode(GraphicsNode):
    """
        ClassificationTreeNode graphics and statistic from Scikit learn tree.Tree object.
    """
    def __init__(self, tree, parent=None, parentItem=None, scene=None, i=0):
        GraphicsNode.__init__(self, tree, parent, parentItem, scene)
        self.tree = tree
        self.i = i
        self.parent = parent
        self.pie = PieChart(self.distribution(i=self.i), 20, self, scene)
        # self.majorityClass, self.majorityCount = max(self.tree.distribution.items(), key=lambda (key, val): val)
        fm = QFontMetrics(self.document().defaultFont())
        self.attr_text_w = fm.width(str(self.attribute() if self.attribute() else ""))
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()

    # Infer parameters and statistics from Scikit learn tree node
    def distribution(self, i=0):
        """
        :param i: index of current node.
        :return: Return prediction at node i.
        """
        return [0.7, 0.3]

    def num_nodes(self):
        """ :return: Number of nodes below particular node. """
        return self.num_nodesw(self.i)

    def num_nodesw(self, i=0):
        """
        :param i: index of current node.
        :return: Number of nodes below particular node.
        """
        s = 1
        if self.tree.children_left[i] > 0:
            s += self.num_nodesw(i = self.tree.children_left[i])
        if self.tree.children_right[i] > 0:
            s += self.num_nodesw(i = self.tree.children_right[i])
        return s


    def num_leaves(self, i=0):
        """ :return: Number of leaves below particular node. """
        return self.num_leavesw(i = self.i)

    def num_leavesw(self, i=0):
        """
        :param i: index of current node.
        :return: Number of leaves below particular node.
        """
        s = 0
        if self.tree.children_left[i] < 0 and self.tree.children_right[i] < 0:
            # Node is leaf
            return 1
        if self.tree.children_left[i] > 0:
            s += self.num_leavesw(i = self.tree.children_left[i])
        if self.tree.children_right[i] > 0:
            s += self.num_leavesw(i = self.tree.children_right[i])
        return s

    def rule(self):
        """:return: Rule to reach node"""
        # TODO: this is easily extended to Classification Rules-compatible form
        return self.rulew(i=self.i, first=True)


    def rulew(self, i=0, first=False):
        """
        :param i: index of current node.
        :return: Rule to reach node i.
        """
        if i > 0:
            sign = "&lt;=" if self.tree.children_left[self.parent.i] == i else "&gt;"
            thresh = self.tree.threshold[self.parent.i]
            attr = self.parent.attribute()
            pr = self.parent.rule()
            return (pr + " AND " if pr else "") + "%s %s %f" % (attr, sign, thresh )
        else:
            return None

    def attribute(self):
        return self.attributew(i=self.i)

    def attributew(self, i=0):
        """
            Attribute at node to split on.
        """
        return self.tree.feature[i]

    def majority(self, i=0):
        # TODO
        pass


    # Interface methods
    def updateContents(self):
        self.prepareGeometryChange()
        if getattr(self, "_rect", QRectF()).isValid() and not self.truncateText:
            self.setTextWidth(self._rect.width() - self.pie.boundingRect().width() / 2 if hasattr(self, "pie") else 0)
        else:
            self.setTextWidth(-1)
            self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        self.pie.setPos(self.rect().right(), self.rect().center().y())
        fm = QFontMetrics(self.document().defaultFont())
        self.attr_text_w = fm.width(str(self.attribute() if self.attribute() else ""))
        self.attr_text_h = fm.lineSpacing()
        self.line_descent = fm.descent()
        
    def rect(self):
        if self.truncateText and getattr(self, "_rect", QRectF()).isValid():
            return self._rect
        else:
            rect = QRectF(QPointF(0,0), self.document().size())
            return rect.adjusted(0, 0, self.pie.boundingRect().width() / 2 if hasattr(self, "pie") else 0, 0) | getattr(self, "_rect", QRectF(0,0,1,1))
    
    def setRect(self, rect):
        self.prepareGeometryChange()
        rect = QRectF() if rect is None else rect 
        self._rect = rect
        if rect.isValid() and not self.truncateText:
            self.setTextWidth(self._rect.width() - self.pie.boundingRect().width() / 2 if hasattr(self, "pie") else 0)
        else:
            self.setTextWidth(-1)
        self.updateContents()
        self.update()
        
    def boundingRect(self):
        if hasattr(self, "attr"):
            attr_rect = QRectF(QPointF(0, -self.attr_text_h), QSizeF(self.attr_text_w, self.attr_text_h))
        else:
            attr_rect = QRectF(0, 0, 1, 1)
        rect = self.rect().adjusted(-5, -5, 5, 5)
        if self.truncateText:
            return rect | attr_rect 
        else:
            return rect | GraphicsNode.boundingRect(self) | attr_rect

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            option.state = option.state.__xor__(QStyle.State_Selected)
        if self.isSelected():
            painter.save()
            painter.setBrush(QBrush(QColor(125, 162, 206, 192)))
            painter.drawRoundedRect(self.boundingRect().adjusted(-2, 1, -1, -1), 10, 10)#self.borderRadius, self.borderRadius)
            painter.restore()
        painter.setFont(self.document().defaultFont())
        painter.drawText(QPointF(0, -self.line_descent), str(self.attribute()) if self.attribute() else "")
        painter.save()
        painter.setBrush(self.backgroundBrush)
        rect = self.rect()
        painter.drawRoundedRect(rect.adjusted(-3, 0, 0, 0), 10, 10)#self.borderRadius, self.borderRadius)
        painter.restore()
        if self.truncateText:
#            self.setTextWidth(-1)`
            painter.setClipRect(rect)
        else:
            painter.setClipRect(rect | QRectF(QPointF(0, 0), self.document().size()))
        return QGraphicsTextItem.paint(self, painter, option, widget)
#        TextTreeNode.paint(self, painter, option, widget)


#BodyColor_Default = QColor(Qt.gray) 
BodyColor_Default = QColor(255, 225, 10)
BodyCasesColor_Default = QColor(Qt.blue) #QColor(0, 0, 128)

class OWClassificationTreeGraph(OWTreeViewer2D):
    settingsList = OWTreeViewer2D.settingsList+['ShowPies', "colorSettings", "selectedColorSettingsIndex"]
    # contextHandlers = {"": DomainContextHandler("", ["TargetClassIndex"], matchValues=1)}
    contextHandlers = {"": DomainContextHandler("", ["TargetClassIndex"])}
    
    nodeColorOpts = ['Default', 'Instances in node', 'Majority class probability', 'Target class probability', 'Target class distribution']
    nodeInfoButtons = ['Majority class', 'Majority class probability', 'Target class probability', 'Number of instances']
    
    def __init__(self, parent=None, signalManager = None, name='ClassificationTreeViewer2D'):
        self.ShowPies=1
        self.TargetClassIndex=0
        self.colorSettings = None
        self.selectedColorSettingsIndex = 0
        self.showNodeInfoText = False
        self.NodeColorMethod = 2
        
        OWTreeViewer2D.__init__(self, parent, signalManager, name)

        # self.inputs = [("Classification Tree", Orange.classification.tree.TreeClassifier, self.ctree)]
        self.outputs = [("Data", ExampleTable)]

        self.scene = TreeGraphicsScene(self)
        self.sceneView = TreeGraphicsView(self, self.scene)
        self.sceneView.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.mainArea.layout().addWidget(self.sceneView)
        self.toggleZoomSlider()
#        self.scene.setSceneRect(0,0,800,800)

        self.connect(self.scene, SIGNAL("selectionChanged()"), self.updateSelection)

        self.navWidget = OWBaseWidget(self)
        self.navWidget.lay=QVBoxLayout(self.navWidget)

        scene=TreeGraphicsScene(self.navWidget)
        self.treeNav = TreeNavigator(self.sceneView)
        self.navWidget.lay.addWidget(self.treeNav)
        self.navWidget.resize(400,400)
        self.navWidget.setWindowTitle("Navigator")
        self.setMouseTracking(True)
        
        colorbox = OWGUI.widgetBox(self.NodeTab, "Node Color", addSpace=True)
        
        OWGUI.comboBox(colorbox, self, 'NodeColorMethod', items=self.nodeColorOpts,
                                callback=self.toggleNodeColor)
        self.targetCombo=OWGUI.comboBox(colorbox,self, "TargetClassIndex", orientation=0, items=[],label="Target class",callback=self.toggleTargetClass)

        OWGUI.checkBox(colorbox, self, 'ShowPies', 'Show distribution pie charts', tooltip='Show pie graph with class distribution?', callback=self.togglePies)
        OWGUI.separator(colorbox)
        # OWGUI.button(colorbox, self, "Set Colors", callback=self.setColors, debuggingEnabled = 0)

        nodeInfoBox = OWGUI.widgetBox(self.NodeTab, "Show Info")
        nodeInfoSettings = ['maj', 'majp', 'tarp', 'inst']
        self.NodeInfoW = []; self.dummy = 0
        for i in range(len(self.nodeInfoButtons)):
            setattr(self, nodeInfoSettings[i], i in self.NodeInfo)
            # w = OWGUI.checkBox(nodeInfoBox, self, nodeInfoSettings[i], \
            #                               self.nodeInfoButtons[i], callback=self.setNodeInfo, getwidget=1, id=i)
            # self.NodeInfoW.append(w)

#        OWGUI.button(self.controlArea, self, "Save as", callback=self.saveGraph, debuggingEnabled = 0)
        self.NodeInfoSorted=list(self.NodeInfo)
        self.NodeInfoSorted.sort()
        
        dlg = self.createColorDialog()
        self.scene.colorPalette = dlg.getDiscretePalette("colorPalette")

        OWGUI.rubber(self.NodeTab)

    def sendReport(self):
        if self.tree:
            tclass = str(self.targetCombo.currentText())
            tsize = "%i nodes, %i leaves" % (orngTree.countNodes(self.tree), orngTree.countLeaves(self.tree))
        else:
            tclass = "N/A"
            tsize = "N/A"
            
        self.reportSettings("Information",
                            [("Node color", self.nodeColorOpts[self.NodeColorMethod]),
                             ("Target class", tclass),
                             ("Data in nodes", ", ".join(s for i, s in enumerate(self.nodeInfoButtons) if self.NodeInfoW[i].isChecked())),
                             ("Line widths", ["Constant", "Proportion of all instances", "Proportion of parent's instances"][self.LineWidthMethod]),
                             ("Tree size", tsize) ])
        OWTreeViewer2D.sendReport(self)

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedColorSettingsIndex = dlg.selectedSchemaIndex
            self.scene.colorPalette = dlg.getDiscretePalette("colorPalette")
            self.scene.update()

    def createColorDialog(self):
        c = ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("colorPalette", "Discrete Palette")
        c.setColorSchemas(self.colorSettings, self.selectedColorSettingsIndex)
        return c

    def setNodeInfo(self, widget=None, id=None):
        flags = sum(2**i for i, name in enumerate(['maj',
                        'majp', 'tarp', 'inst']) if getattr(self, name))
            
        for n in self.scene.nodes():
            n.setRect(QRectF())
            self.updateNodeInfo(n, flags)
        if True:
            w = min(max([n.rect().width() for n in self.scene.nodes()] + [0]), self.MaxNodeWidth if self.LimitNodeWidth else sys.maxint)
            for n in self.scene.nodes():
                n.setRect(QRectF(n.rect().x(), n.rect().y(), w, n.rect().height()))
        self.scene.fixPos(self.rootNode, 10, 10)
        
    def updateNodeInfo(self, node, flags=31):
        fix = lambda str: str.replace(">", "&gt;").replace("<", "&lt;")
        text = ""
        
#        text += "%s<br>" % fix(node.attr if node.attr else "")
            
        lines = []
        if flags & 1:
            start = "Majority class: " if self.showNodeInfoText else "" 
#            lines += [start + "<font color=%s>%s</font>" % (self.scene.colorPalette[node.tree.examples.domain.classVar.values.index(node.majorityClass)].name(), fix(node.majorityClass))]
            # lines += [start + fix(node.majorityClass)]
        if flags & 2:
            start = "Majority class probability: " if self.showNodeInfoText else "" 
            # lines += [start + "%.1f" % (100.0 * float(node.majorityCount) / node.tree.distribution.abs)]
        if flags & 4:
            start = "Target class probability: "  if self.showNodeInfoText else "" 
            # lines += [start + "%.1f" % (100.0 * float(node.tree.distribution[self.TargetClassIndex]) / node.tree.distribution.abs)]
        if flags & 8:
            start = "Instances: " if self.showNodeInfoText else "" 
            # lines += [start + "%i" % node.tree.distribution.cases]
        text += "<br>".join(lines)
        # if node.tree.branchSelector:
            # text += "<hr>" + "%s" % fix(node.tree.branchSelector.classVar.name)
            # pass
        # else:
            # text += "<hr>" + fix(node.majorityClass)
        #    pass


        # Debug
        text = "<p>numNodes: %d</p>" % node.num_nodes()
        text += "<p>numLeaves: %d</p>" % node.num_leaves()
        text += "<p>rule: %s</p>  " % node.rule()
        node.setHtml(text)

    def activateLoadedSettings(self):
        if not self.tree:
            return
        OWTreeViewer2D.activateLoadedSettings(self)
        self.setNodeInfo()
        self.toggleNodeColor()
        
    def toggleNodeSize(self):
        self.setNodeInfo()
        self.scene.update()
        self.sceneView.repaint()

    def toggleNodeColor(self):
        palette = self.scene.colorPalette
        for node in self.scene.nodes():
            # dist = node.tree.distribution
            if self.NodeColorMethod == 0:
                # default color
                color = BodyColor_Default
            elif self.NodeColorMethod == 1:
                # number of instances in node
                all_cases = self.tree.distribution.cases
                light = 200 - 100 * dist.cases / (all_cases or 1)
                color = BodyCasesColor_Default.light(light)
            elif self.NodeColorMethod == 2:
                # majority class probability
                # modus = dist.modus()
                # p = dist[modus] / (dist.abs or 1)
                modus = 1
                p = 0.3
                light = 400 - 300 * p
                color = palette[int(modus)].light(light)
            elif self.NodeColorMethod == 3:
                # target class probability
                # p = dist[self.TargetClassIndex] / (dist.cases or 1)
                light = 200 - 100 * p
                color = palette[self.TargetClassIndex].light(light)
            elif self.NodeColorMethod == 4:
                # target class distribution
                # all_target = self.tree.distribution[self.TargetClassIndex] or 1
                # light = 200 - 100 * dist[self.TargetClassIndex] / all_target
                color = palette[self.TargetClassIndex].light(light)
            node.backgroundBrush = QBrush(color)
        self.scene.update()

    def toggleTargetClass(self):
        if self.NodeColorMethod in [3,4]:
            self.toggleNodeColor()
        if self.tarp:
            self.setNodeInfo()
        self.scene.update()

    def togglePies(self):
        for n in self.scene.nodes():
            n.pie.setVisible(self.ShowPies and n.isVisible())
        self.scene.update()

    def ctree(self, tree=None):
        """
            Set the input tree classifier.

            :param tree:
            :return:
        """

        self.clear()
        if not tree:
            self.centerRootButton.setDisabled(1)
            self.centerNodeButton.setDisabled(0)
            self.infoa.setText('No tree.')
            self.infob.setText('')
            self.tree=None
            self.rootNode = None
        else:
            self.infoa.setText('Tree found on input.')
            self.tree = tree
            # if hasattr(self.scene, "colorPalette"):
            #   self.scene.colorPalette.setNumberOfColors(len(self.tree.distribution))
#            self.scene.setDataModel(GraphicsTree(self.tree))
            self.rootNode = self.walkcreate(self.tree, None)
            self.infoa.setText('Number of nodes: ' + str(self.rootNode.num_nodes()))
            self.infob.setText('Number of leaves: ' + str(self.rootNode.num_leaves()))
#            self.scene.addItem(self.rootNode)
            self.scene.fixPos(self.rootNode, self.HSpacing,self.VSpacing)
            self.activateLoadedSettings()
            self.sceneView.centerOn(self.rootNode.x(), self.rootNode.y())
            self.updateNodeToolTips()
            self.centerRootButton.setDisabled(0)
            self.centerNodeButton.setDisabled(1)

        self.scene.update()


    def walkcreate(self, tree, parent=None, level=0, i=0):
        """
        Recursively draw tree structure from Scikit learn Tree object.
        This method need to call ClassificationTreeNode to dispaly nodes and pies.

        :param tree:
        :param parent:
        :param level:
        :param i:
        :return:
        """
        node = ClassificationTreeNode(tree, parent, None, self.scene, i=i)
        node.borderRadius = 10
        if parent:
            parent.graph_add_edge(GraphicsEdge(None, self.scene, node1=parent, node2=node))

        # Parsing the Scikit learn structure
        left_child_index = tree.children_left[i]
        right_child_index = tree.children_right[i]

        # First draw right child, then left
        if left_child_index >= 0:
            self.walkcreate(tree, parent=node, level=level+1, i=left_child_index)
        if right_child_index >= 0:
            self.walkcreate(tree, parent=node, level=level+1, i=right_child_index)
        return node


    def nodeToolTip(self, node):
        text = node.rule()
        return text

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWClassificationTreeGraph()
    import pickle
##    a.setMainWidget(ow)
    # data = orange.ExampleTable('../../doc/datasets/voting.tab')
    # data = orange.ExampleTable(r"../../doc/datasets/zoo.tab")
    # tree = orange.TreeLearner(data, storeExamples = 1)
    tree = pickle.load(open("iris_tree.pkl", "rb"))
    ow.ctree(tree.clf.tree_)

    # here you can test setting some stuff
    ow.show()
    a.exec_()
    ow.saveSettings()

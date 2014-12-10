from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting

from PyQt4.QtCore import *
from PyQt4.QtGui import *

DefDroppletBrush = QBrush(Qt.darkGray)


class GraphNode(object):
    def __init__(self, *_, **kwargs):
        self.edges = kwargs.get("edges", set())
        
    def graph_edges(self):
        return self.edges
    
    def graph_add_edge(self, edge):
        self.edges.add(edge)
        
    def __iter__(self):
        for edge in self.edges:
            yield edge.node2
            
    def graph_nodes(self, atype=1):
        pass


class GraphEdge(object):
    def __init__(self, node1=None, node2=None, atype=1):
        self.node1 = node1
        self.node2 = node2
        self.type = atype
        node1.graph_add_edge(self)
        node2.graph_add_edge(self)


class GraphicsDroplet(QGraphicsEllipseItem):
    def __init__(self, *args):
        super().__init__(*args)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setBrush(QBrush(Qt.gray))
        self.setPen(Qt.white)
        
    def hoverEnterEvent(self, event):
        super().hoverEnterEvent(event)
        self.setBrush(QBrush(QColor(100, 100, 100)))
        self.update()
        
    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)
        self.setBrush(QBrush(QColor(200, 200, 200)))
        self.update()
        
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.parentItem().set_open(not self.parentItem().isOpen)
        if self.scene():
            self.scene().fix_pos()


# noinspection PyPep8Naming
class TextTreeNode(QGraphicsTextItem, GraphNode):
    def setBackgroundBrush(self, brush):
        if self._background_brush != brush:
            self._background_brush = QBrush(brush)
            color = brush.color()
            r, g, b, _ = color.getRgb()
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum > 30:
                self.setDefaultTextColor(Qt.black)
            else:
                self.setDefaultTextColor(Qt.white)
            self.update()

    def backgroundBrush(self):
        brush = getattr(self, "_background_brush",
                        getattr(self.scene(), "defaultItemBrush", Qt.NoBrush))
        return QBrush(brush)

    backgroundBrush = pyqtProperty(
        "QBrush", fget=backgroundBrush, fset=setBackgroundBrush,
        doc="Background brush")

    def __init__(self, tree, parent, *args, **kwargs):
        QGraphicsTextItem.__init__(self, *args)
        GraphNode.__init__(self, **kwargs)
        self._background_brush = None
        self._rect = None

        self.tree = tree
        self.parent = parent
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.droplet = GraphicsDroplet(-5, 0, 10, 10, self, self.scene())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.document().contentsChanged.connect(self.update_contents)
        self.isOpen = True
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def setHtml(self, html):
        return super().setHtml("<body>" + html + "</body>")
    
    def update_contents(self):
        self.setTextWidth(-1)
        self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        
    def set_rect(self, rect):
        self.prepareGeometryChange()
        rect = QRectF() if rect is None else rect
        self._rect = rect
        self.update_contents()
        self.update()
        
    def shape(self):
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path
    
    def rect(self):
        if getattr(self, "_rect", QRectF()).isValid():
            return self._rect
        else:
            return QRectF(QPointF(0, 0), self.document().size()) | \
                getattr(self, "_rect", QRectF(0, 0, 1, 1))
        
    def boundingRect(self):
        return self._rect if getattr(self, "_rect", QRectF()).isValid() \
            else super().boundingRect()
    
    @property  
    def branches(self):
        return [edge.node2 for edge in self.graph_edges() if edge.node1 is self]
    
    def paint(self, painter, option, widget=0):
        painter.save()
        painter.setBrush(self.backgroundBrush)
        painter.setPen(QPen(Qt.gray))
        rect = self.rect()
        painter.drawRoundedRect(rect, 4, 4)
        painter.restore()
        painter.setClipRect(rect)
        return QGraphicsTextItem.paint(self, painter, option, widget)


class GraphicsNode(TextTreeNode):
    def graph_traverse_bf(self):
        visited = set()
        queue = list(self)
        while queue:
            node = queue.pop(0)
            if node not in visited:
                yield node
                visited.add(node)
                if node.isOpen:
                    queue.extend(list(node))

    def set_open(self, do_open):
        self.isOpen = do_open
        for node in self.graph_traverse_bf():
            if node is not self:
                node.setVisible(do_open)
               
    def itemChange(self, change, value):
        if change in [QGraphicsItem.ItemPositionHasChanged,
                      QGraphicsItem.ItemVisibleHasChanged]:
            self.update_edge()
        return super().itemChange(change, value)

    # noinspection PyCallByClass,PyTypeChecker
    def update_edge(self):
        for edge in self.edges:
            if edge.node1 is self:
                QTimer.singleShot(0, edge.update_ends)
            elif edge.node2 is self:
                edge.setVisible(self.isVisible())
                
    def edge_in_point(self, edge):
        return edge.mapFromItem(
            self, QPointF(self.rect().center().x(), self.rect().y()))

    def edge_out_point(self, edge):
        return edge.mapFromItem(self.droplet, self.droplet.rect().center())
    
    def paint(self, painter, option, widget=0):
        if self.isSelected():
            option.state ^= QStyle.State_Selected
        if self.isSelected():
            rect = self.rect()
            painter.save()
            painter.setBrush(QBrush(QColor(125, 162, 206, 192)))
            painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 10, 10)
            painter.restore()
        super().paint(painter, option, widget)
        
    def boundingRect(self):
        return super().boundingRect().adjusted(-5, -5, 5, 5)
            
    def mousePressEvent(self, event):
        return super().mousePressEvent(event)


class GraphicsEdge(QGraphicsLineItem, GraphEdge):
    def __init__(self, *args, **kwargs):
        QGraphicsLineItem.__init__(self, *args)
        GraphEdge.__init__(self, **kwargs)
        self.setZValue(-30)
        
    def update_ends(self):
        try:
            self.prepareGeometryChange()
            self.setLine(QLineF(self.node1.edge_out_point(self),
                                self.node2.edge_in_point(self)))
        except RuntimeError:  # this gets called through QTimer.singleShot
                              # and might already be deleted by Qt
            pass 


class TreeGraphicsView(QGraphicsView):
    resized = pyqtSignal(QSize, name="resized")

    def __init__(self, scene, *args):
        super().__init__(scene, *args)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.HighQualityAntialiasing)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit(self.size())


class TreeGraphicsScene(QGraphicsScene):
    _HSPACING = 10
    _VSPACING = 10

    def __init__(self, master, *args):
        super().__init__(*args)
        self.master = master
        self.nodeList = []
        self.edgeList = []
        self.gx = self.gy = 10

    def fix_pos(self, node=None, x=10, y=10):
        self.gx, self.gy = x, y
        if not node:
            if self.nodes():
                node = [node for node in self.nodes() if not node.parent][0]
            else:
                return
        if not x or not y:
            x, y = self._HSPACING, self._VSPACING
        self._fix_pos(node, x, y)
        self.update()
        
    def _fix_pos(self, node, x, y):
        def brect(node):
            return node.boundingRect() | node.childrenBoundingRect()
        if node.branches and node.isOpen:
            for n in node.branches:
                x, ry = self._fix_pos(n, x,
                                      y + self._VSPACING + brect(node).height())
            x = (node.branches[0].pos().x() + node.branches[-1].pos().x()) / 2
            node.setPos(x, y)
            for e in node.edges:
                e.update_ends()
        else:
            node.setPos(self.gx, y)
            self.gx += self._HSPACING + brect(node).width()
            x += self._HSPACING + brect(node).width()
            self.gy = max(y, self.gy)
        return x, y

    def mouseMoveEvent(self, event):
        return QGraphicsScene.mouseMoveEvent(self, event)

    def mousePressEvent(self, event):
        return QGraphicsScene.mousePressEvent(self, event)
    
    def edges(self):
        return [item for item in self.items() if isinstance(item, GraphEdge)]
    
    def nodes(self):
        return [item for item in self.items() if isinstance(item, GraphNode)]


class TreeNavigator(QGraphicsView):
    def __init__(self, master_view, *_):
        super().__init__()
        self.master_view = master_view
        self.setScene(self.master_view.scene())
        self.scene().sceneRectChanged.connect(self.updateSceneRect)
        self.master_view.resized.connect(self.update_view)
        self.setRenderHint(QPainter.Antialiasing)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.master_view.centerOn(self.mapToScene(event.pos()))
            self.update_view()
        return super().mousePressEvenr(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.master_view.centerOn(self.mapToScene(event.pos()))
            self.update_view()
        return super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        QGraphicsView.resizeEvent(self, event)
        self.update_view()

    # noinspection PyPep8Naming
    def resizeView(self):
        self.update_view()

    def updateSceneRect(self, rect):
        super().updateSceneRect(rect)
        self.update_view()
        
    def update_view(self, *_):
        if self.scene():
            self.fitInView(self.scene().sceneRect())

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setBrush(QColor(100, 100, 100, 100))
        painter.setRenderHints(self.renderHints())
        painter.drawPolygon(self.viewPolygon())
        
    # noinspection PyPep8Naming
    def viewPolygon(self):
        return self.mapFromScene(
            self.master_view.mapToScene(self.master_view.viewport().rect()))


class OWTreeViewer2D(OWWidget):
    zoom = Setting(5)
    line_width_method = Setting(2)
    max_tree_depth = Setting(0)
    max_node_width = Setting(150)

    _VSPACING = 5
    _HSPACING = 5
    _TOOLTIPS_ENABLED = True
    _DEF_NODE_WIDTH = 24
    _DEF_NODE_HEIGHT = 20

    want_graph = True

    def __init__(self):
        super().__init__()
        self.root = None
        self.selected_node = None
        self.root_node = None
        self.tree = None

        box = gui.widgetBox(
            self.controlArea, 'Tree size', addSpace=20,
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.info = gui.widgetLabel(box, 'No tree.')

        layout = QGridLayout()
        layout.setVerticalSpacing(20)
        box = gui.widgetBox(self.controlArea, "Size", addSpace=True,
                            orientation=layout)
        layout.addWidget(QLabel("Zoom: "), 0, 0, Qt.AlignRight)
        layout.addWidget(gui.hSlider(
            box, self, 'zoom', minValue=1, maxValue=10, step=1,
            createLabel=False, ticks=False, addToLayout=False, addSpace=False,
            callback=self.toggle_zoom_slider), 0, 1)
        layout.addWidget(QLabel("Width: "), 1, 0, Qt.AlignRight)
        layout.addWidget(gui.hSlider(
            box, self, 'max_node_width', minValue=50, maxValue=200, step=1,
            createLabel=False, ticks=False, addToLayout=False, addSpace=False,
            callback=self.toggle_node_size), 1, 1)
        layout.addWidget(QLabel("Depth: "), 2, 0, Qt.AlignRight)
        layout.addWidget(gui.comboBox(
            box, self, 'max_tree_depth',
            items=["Unlimited"] + ["{} levels".format(x) for x in range(2, 10)],
            addToLayout=False,
            sendSelectedValue=False, callback=self.toggle_tree_depth,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed)), 2, 1)
        layout.addWidget(QLabel("Edge width: "), 3, 0, Qt.AlignRight)
        layout.addWidget(gui.comboBox(
            box, self,  'line_width_method', addToLayout=False,
            items=['Fixed', 'Relative to root', 'Relative to parent'],
            callback=self.toggle_line_width), 3, 1)
        self.resize(800, 500)

    def send_report(self):
        from PyQt4.QtSvg import QSvgGenerator
        if self.tree:
            self.reportSection("Tree")
            urlfn, filefn = self.getUniqueImageName(ext=".svg")
            svg = QSvgGenerator()
            svg.setFileName(filefn)
            ssize = self.scene.sceneRect().size()
            w, h = ssize.width(), ssize.height()
            fact = 600/w
            svg.setSize(QSize(600, h*fact))
            painter = QPainter()
            painter.begin(svg)
            self.scene.render(painter)
            painter.end()

            # from OWDlgs import OWChooseImageSizeDlg
            # self.reportImage(OWChooseImageSizeDlg(self.scene).saveImage)
            # self.report_object(self.svg_type, urlfn, width="600",
            #                    height=str(h*fact))

    def toggle_zoom_slider(self):
        k = 0.0028 * (self.zoom ** 2) + 0.2583 * self.zoom + 1.1389
        self.scene_view.setTransform(QTransform().scale(k / 2, k / 2))
        self.scene.update()

    def toggle_tree_depth(self):
        self.walkupdate(self.root_node)
        self.scene.fix_pos(self.root_node, 10, 10)
        self.scene.update()

    def toggle_line_width(self):
        root_instances = self.root_node.num_instances()
        width = 3
        for edge in self.scene.edges():
            num_inst = edge.node2.num_instances()
            if self.line_width_method == 1:
                width = 8 * num_inst / root_instances
            elif self.line_width_method == 2:
                width = 8 * num_inst / edge.node1.num_instances()
            edge.setPen(QPen(Qt.gray, width, Qt.SolidLine, Qt.RoundCap))
        self.scene.update()

    def toggle_node_size(self):
        self.set_node_info()
        self.scene.update()
        self.sceneView.repaint()

    def toggle_navigator(self):
        self.nav_widget.setHidden(not self.nav_widget.isHidden())

    def activate_loaded_settings(self):
        if not self.tree:
            return
        self.rescale_tree()
        self.scene.fix_pos(self.root_node, 10, 10)
        self.scene.update()
        self.toggle_tree_depth()
        self.toggle_line_width()

    def ctree(self, tree):
        self.clear()
        if not tree:
            self.centerRootButton.setDisabled(1)
            self.centerNodeButton.setDisabled(0)
            self.infoa.setText('No tree.')
            self.infob.setText('')
            self.tree = None
            self.root_node = None
        else:
            self.infoa.setText('Tree.')
            self.tree = tree
            self.root_node = self.walkcreate(self.tree.clf.tree_, None)
            self.scene.fix_pos(self.root_node, self._HSPACING, self._VSPACING)
            self.activate_loaded_settings()
            self.sceneView.centerOn(self.root_node.x(), self.root_node.y())
            self.update_node_tooltips()
            self.centerRootButton.setDisabled(0)
            self.centerNodeButton.setDisabled(1)
        self.scene.update()

    def walkcreate(self, tree, parent=None, level=0, i=0):
        node = GraphicsNode(tree, parent, None, self.scene)
        if parent:
            parent.graph_add_edge(GraphicsEdge(None, self.scene,
                                               node1=parent, node2=node))
        left_child_ind = tree.children_left[i]
        right_child_ind = tree.children_right[i]
        if right_child_ind >= 0:
            self.walkcreate(tree, parent=node, level=level+1, i=right_child_ind)
        if left_child_ind >= 0:
            self.walkcreate(tree, parent=node, level=level+1, i=left_child_ind)
        return node

    def walkupdate(self, node, level=0):
        if not node:
            return
        if self.max_tree_depth and self.max_tree_depth < level+1:
            node.set_open(False)
            return
        else:
            node.set_open(True)
        for n in node.branches:
            self.walkupdate(n, level + 1)

    def clear(self):
        self.tree = None
        self.scene.clear()

    def update_node_tooltips(self):
        for node in self.scene.nodes():
            node.setToolTip(self.node_tooltip(node) if self._TOOLTIPS_ENABLED
                            else "")

    def node_tooltip(self, tree):
        return "tree node"

    def rescale_tree(self):
        node_height = self._DEF_NODE_HEIGHT
        node_width = self._DEF_NODE_WIDTH
        for r in self.scene.nodeList:
            r.set_rect(r.rect().x(), r.rect().y(), node_width, node_height)
        self.scene.fix_pos()

    def update_selection(self):
        self.selected_node = (self.scene.selectedItems() + [None])[0]
        # self.centerNodeButton.setDisabled(not self.selected_node)
        # self.send("Data", self.selectedNode.tree.examples if self.selectedNode
        #           else None)

    def save_graph(self, file_name=None):
        pass
        # from OWDlgs import OWChooseImageSizeDlg
        # dlg = OWChooseImageSizeDlg(
        #     self.scene,
        #     [("Save as Dot Tree File (.dot)", self.saveDot)],
        #     parent=self)
        # dlg.exec()

    # noinspection PyTypeChecker
    def save_dot(self, filename=None):
        if filename is None:
            # noinspection PyCallByClass
            filename = QFileDialog.getSaveFileName(
                self, "Save to ...", "tree.dot", "Dot Tree File (.DOT)")
            if not filename:
                return
        # orngTree.printDot(self.tree, filename)

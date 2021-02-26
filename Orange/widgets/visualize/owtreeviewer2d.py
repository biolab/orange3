from collections import OrderedDict

from AnyQt.QtGui import (
    QBrush, QPen, QColor, QPainter, QPainterPath, QTransform
)
from AnyQt.QtWidgets import (
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QGraphicsLineItem, QGraphicsScene, QGraphicsView, QStyle, QSizePolicy,
    QFormLayout
)
from AnyQt.QtCore import (
    Qt, QRectF, QSize, QPointF, QLineF, QTimer,
    pyqtSignal, pyqtProperty
)

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting

DefDroppletBrush = QBrush(Qt.darkGray)


class GraphNode:
    def __init__(self, *_, **kwargs):
        # Implement edges as an ordered dict to get the nice speed benefits as
        # well as adding ordering, which we need to make trees deterministic
        self.__edges = kwargs.get("edges", OrderedDict())

    def graph_edges(self):
        """Get a list of the edges that stem from the node."""
        return self.__edges.keys()

    def graph_add_edge(self, edge):
        """Add an edge stemming from the node."""
        self.__edges[edge] = 0

    def __iter__(self):
        for edge in self.__edges.keys():
            yield edge.node2

    def graph_nodes(self, atype=1):
        pass


class GraphEdge:
    def __init__(self, node1=None, node2=None, atype=1):
        self.node1 = node1
        self.node2 = node2
        self.type = atype
        if node1 is not None:
            node1.graph_add_edge(self)
        if node2 is not None:
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
            if lum > 100:
                self.setDefaultTextColor(Qt.black)
            else:
                self.setDefaultTextColor(Qt.white)
            self.update()

    def backgroundBrush(self):
        brush = getattr(self, "_background_brush")
        if brush is None:
            brush = getattr(self.scene(), "defaultItemBrush", Qt.NoBrush)
        return QBrush(brush)

    backgroundBrush = pyqtProperty(
        "QBrush", fget=backgroundBrush, fset=setBackgroundBrush,
        doc="Background brush")

    def __init__(self, parent, *args, **kwargs):
        QGraphicsTextItem.__init__(self, *args)
        GraphNode.__init__(self, **kwargs)
        self._background_brush = None
        self._rect = None

        self.parent = parent
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.droplet = GraphicsDroplet(-5, 0, 10, 10, self)
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
        for edge in self.graph_edges():
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
        self.setSceneRect(QRectF(0, 0, self.gx, self.gy).adjusted(-10, -10, 100, 100))
        self.update()

    def _fix_pos(self, node, x, y):
        """Fix the position of the tree stemming from the given node."""
        def brect(node):
            """Get the bounding box of the parent rect and all its children."""
            return node.boundingRect() | node.childrenBoundingRect()

        if node.branches and node.isOpen:
            for n in node.branches:
                x, _ = self._fix_pos(n, x, y + self._VSPACING + brect(node).height())
            x = (node.branches[0].pos().x() + node.branches[-1].pos().x()) / 2
            node.setPos(x, y)
            for e in node.graph_edges():
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


class OWTreeViewer2D(OWWidget, openclass=True):
    zoom = Setting(5)
    line_width_method = Setting(2)
    max_tree_depth = Setting(0)
    max_node_width = Setting(150)

    _VSPACING = 5
    _HSPACING = 5
    _TOOLTIPS_ENABLED = True
    _DEF_NODE_WIDTH = 24
    _DEF_NODE_HEIGHT = 20

    graph_name = "scene"

    def __init__(self):
        super().__init__()
        self.selected_node = None
        self.root_node = None
        self.model = None

        box = gui.vBox(
            self.controlArea, 'Tree',
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.infolabel = gui.widgetLabel(box, 'No tree.')

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(layout.ExpandingFieldsGrow)
        box = self.display_box = gui.widgetBox(self.controlArea, "Display",
                                               orientation=layout)
        layout.addRow(
            "Zoom: ",
            gui.hSlider(box, self, 'zoom',
                        minValue=1, maxValue=10, step=1, ticks=False,
                        callback=self.toggle_zoom_slider,
                        createLabel=False, addToLayout=False))
        layout.addRow(
            "Width: ",
            gui.hSlider(box, self, 'max_node_width',
                        minValue=50, maxValue=200, step=1, ticks=False,
                        callback=self.toggle_node_size,
                        createLabel=False, addToLayout=False))
        policy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        layout.addRow(
            "Depth: ",
            gui.comboBox(box, self, 'max_tree_depth',
                         items=["Unlimited"] + [
                             "{} levels".format(x) for x in range(2, 10)],
                         addToLayout=False, sendSelectedValue=False,
                         callback=self.toggle_tree_depth, sizePolicy=policy))
        layout.addRow(
            "Edge width: ",
            gui.comboBox(box, self, 'line_width_method',
                         items=['Fixed', 'Relative to root',
                                'Relative to parent'],
                         addToLayout=False,
                         callback=self.toggle_line_width, sizePolicy=policy))
        gui.rubber(self.controlArea)

        self.scene = TreeGraphicsScene(self)
        self.scene_view = TreeGraphicsView(self.scene)
        self.scene_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.mainArea.layout().addWidget(self.scene_view)
        self.toggle_zoom_slider()
        self.scene.selectionChanged.connect(self.update_selection)

    def send_report(self):
        from AnyQt.QtSvg import QSvgGenerator

        if self.model:
            self.reportSection("Tree")
            _, filefn = self.getUniqueImageName(ext=".svg")
            svg = QSvgGenerator()
            svg.setFileName(filefn)
            ssize = self.scene.sceneRect().size()
            w, h = ssize.width(), ssize.height()
            fact = 600 / w
            svg.setSize(QSize(600, h * fact))
            painter = QPainter()
            painter.begin(svg)
            self.scene.render(painter)
            painter.end()

    def toggle_zoom_slider(self):
        k = 0.0028 * (self.zoom ** 2) + 0.2583 * self.zoom + 1.1389
        self.scene_view.setTransform(QTransform().scale(k / 2, k / 2))
        self.scene.update()

    def toggle_tree_depth(self):
        self.walkupdate(self.root_node)
        self.scene.fix_pos(self.root_node, 10, 10)
        self.scene.update()

    def toggle_line_width(self):
        if self.root_node is None:
            return

        tree_adapter = self.root_node.tree_adapter
        root_instances = tree_adapter.num_samples(self.root_node.node_inst)
        width = 3
        OFFSET = 0.20
        for edge in self.scene.edges():
            num_inst = tree_adapter.num_samples(edge.node2.node_inst)
            if self.line_width_method == 1:
                width = 8 * num_inst / root_instances + OFFSET
            elif self.line_width_method == 2:
                width = 8 * num_inst / tree_adapter.num_samples(
                    edge.node1.node_inst) + OFFSET
            edge.setPen(QPen(Qt.gray, width, Qt.SolidLine, Qt.RoundCap))
        self.scene.update()

    def toggle_node_size(self):
        self.set_node_info()
        self.scene.update()
        self.scene_view.repaint()

    def toggle_navigator(self):
        self.nav_widget.setHidden(not self.nav_widget.isHidden())

    def activate_loaded_settings(self):
        if not self.model:
            return
        self.rescale_tree()
        self.scene.fix_pos(self.root_node, 10, 10)
        self.scene.update()
        self.toggle_tree_depth()
        self.toggle_line_width()

    def clear_scene(self):
        self.scene.clear()
        self.scene.setSceneRect(QRectF())

    def setup_scene(self):
        if self.root_node is not None:
            self.scene.fix_pos(self.root_node, self._HSPACING, self._VSPACING)
            self.activate_loaded_settings()
            self.scene_view.centerOn(self.root_node.x(), self.root_node.y())
            self.update_node_tooltips()
        self.scene.update()

    def walkupdate(self, node, level=0):
        if not node:
            return
        if self.max_tree_depth and self.max_tree_depth < level + 1:
            node.set_open(False)
            return
        else:
            node.set_open(True)
        for n in node.branches:
            self.walkupdate(n, level + 1)

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
        # else None)

"""
Pythagoras tree viewer for visualizing tree structures.

The pythagoras tree viewer widget is a widget that can be plugged into any
existing widget given a tree adapter instance. It is simply a canvas that takes
and input tree adapter and takes care of all the drawing.

Types
-----
Square : namedtuple (center, length, angle)
    Since Pythagoras trees deal only with squares (they also deal with
    rectangles in the generalized form, but are completely unreadable), this
    is what all the squares are stored as.
Point : namedtuple (x, y)
    Self exaplanatory.

"""
from collections import namedtuple, defaultdict, deque
from math import pi, sqrt, cos, sin, degrees

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

# z index range, increase if needed
Z_STEP = 5000000

Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])


class PythagorasTreeViewer(QtGui.QGraphicsWidget):
    """Pythagoras tree viewer graphics widget.

    Simply pass in a tree adapter instance and a valid scene object, and the
    pythagoras tree will be added.

    Examples
    --------
    >>> from Orange.widgets.visualize.utils.tree.treeadapter import (
    >>>     TreeAdapter
    >>> )
    Pass tree through constructor.
    >>> tree_view = PythagorasTreeViewer(parent=scene, adapter=tree_adapter)

    Pass tree later through method.
    >>> tree_adapter = TreeAdapter()
    >>> scene = QtGui.QGraphicsScene()
    This is where the magic happens
    >>> tree_view = PythagorasTreeViewer(parent=scene)
    >>> tree_view.set_tree(tree_adapter)

    Both these examples set the appropriate tree and add all the squares to the
    widget instance.

    Parameters
    ----------
    parent : QGraphicsItem, optional
        The parent object that the graphics widget belongs to. Should be a
        scene.
    adapter : TreeAdapter, optional
        Any valid tree adapter instance.
    interacitive : bool, optional,
        Specify whether the widget should have an interactive display. This
        means special hover effects, selectable boxes. Default is true.

    Notes
    -----
    .. note:: The class contains two clear methods: `clear` and `clear_tree`.
        Each has  their own use.
        `clear_tree` will clear out the tree and remove any graphics items.
        `clear` will, on the other hand, clear everything, all settings
        (tooltip and color calculation functions.

        This is useful because when we want to change the size calculation of
        the Pythagora tree, we just want to clear the scene and it would be
        inconvenient to have to set color and tooltip functions again.
        On the other hand, when we want to draw a brand new tree, it is best
        to clear all settings to avoid any strange bugs - we start with a blank
        slate.

    """

    def __init__(self, parent=None, adapter=None, depth_limit=0, padding=0,
                 **kwargs):
        super().__init__(parent)

        # Instance variables
        # The tree adapter parameter will be handled at the end of init
        self.tree_adapter = None
        # The root tree node instance which is calculated inside the class
        self._tree = None
        self._padding = padding

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

        # Necessary settings that need to be set from the outside
        self._depth_limit = depth_limit
        # Provide a nice green default in case no color function is provided
        self.__calc_node_color_func = kwargs.get('node_color_func')
        self.__get_tooltip_func = kwargs.get('tooltip_func')
        self._interactive = kwargs.get('interactive', True)

        self._square_objects = {}
        self._drawn_nodes = deque()
        self._frontier = deque()

        # If a tree adapter was passed, set and draw the tree
        if adapter is not None:
            self.set_tree(adapter)

    def set_tree(self, tree_adapter):
        """Pass in a new tree adapter instance and perform updates to canvas.

        Parameters
        ----------
        tree_adapter : TreeAdapter
            The new tree adapter that is to be used.

        Returns
        -------

        """
        self.clear_tree()
        self.tree_adapter = tree_adapter

        if self.tree_adapter is not None:
            self._tree = self._calculate_tree(self.tree_adapter)
            self.set_depth_limit(tree_adapter.max_depth)
            self._draw_tree(self._tree)

    def set_depth_limit(self, depth):
        """Update the drawing depth limit.

        The drawing stops when the depth is GT the limit. This means that at
        depth 0, the root node will be drawn.

        Parameters
        ----------
        depth : int
            The maximum depth at which the nodes can still be drawn.

        Returns
        -------

        """
        self._depth_limit = depth
        self._draw_tree(self._tree)

    def set_node_color_func(self, func):
        """Set the function that will be used to calculate the node colors.

        The function must accept one parameter that represents the label of a
        given node and return the appropriate QColor object that should be used
        for the node.

        Parameters
        ----------
        func : Callable
            func :: label -> QtGui.QColor

        Returns
        -------

        """
        if func != self._calc_node_color:
            self.__calc_node_color_func = func
            self._update_node_colors()

    def _calc_node_color(self, *args):
        """Get the node color with a nice default fallback."""
        if self.__calc_node_color_func is not None:
            return self.__calc_node_color_func(*args)
        return QtGui.QColor('#297A1F')

    def set_tooltip_func(self, func):
        """Set the function that will be used the get the node tooltips.

        Parameters
        ----------
        func : Callable
            func :: label -> str

        Returns
        -------

        """
        if func != self._get_tooltip:
            self.__get_tooltip_func = func
            self._update_node_tooltips()

    def _get_tooltip(self, *args):
        """Get the node tooltip with a nice default fallback."""
        if self.__get_tooltip_func is not None:
            return self.__get_tooltip_func(*args)
        return 'Tooltip'

    def target_class_has_changed(self):
        """When the target class has changed, perform appropriate updates."""
        self._update_node_colors()
        self._update_node_tooltips()

    def tooltip_has_changed(self):
        """When the tooltip should change, perform appropriate updates."""
        self._update_node_tooltips()

    def _update_node_colors(self):
        """Update all the node colors.

        Should be called when the color method is changed and the nodes need to
        be drawn with the new colors.

        Returns
        -------

        """
        for square in self._squares():
            square.setBrush(self._calc_node_color(self.tree_adapter,
                                                  square.tree_node))

    def _update_node_tooltips(self):
        """Update all the tooltips for the squares."""
        for square in self._squares():
            square.setToolTip(self._get_tooltip(square.tree_node))

    def clear(self):
        """Clear the entire widget state."""
        self.__calc_node_color_func = None
        self.__get_tooltip_func = None
        self.clear_tree()

    def clear_tree(self):
        """Clear only the tree, keeping tooltip and color functions."""
        self.tree_adapter = None
        self._tree = None
        self._clear_scene()

    @staticmethod
    def _calculate_tree(tree_adapter):
        """Actually calculate the tree squares"""
        tree_builder = PythagorasTree()
        return tree_builder.pythagoras_tree(
            tree_adapter, tree_adapter.root, Square(Point(0, 0), 200, -pi / 2)
        )

    def _draw_tree(self, root):
        """Efficiently draw the tree with regards to the depth.

        If using a recursive approach, the tree had to be redrawn every time
        the depth was changed, which was very impractical for larger trees,
        since everything got very slow, very fast.

        In this approach, we use two queues to represent the tree frontier and
        the nodes that have already been drawn. We also store the depth. This
        way, when the max depth is increased, we do not redraw the whole tree
        but only iterate throught the frontier and draw those nodes, and update
        the frontier accordingly.
        When decreasing the max depth, we reverse the process, we clear the
        frontier, and remove nodes from the drawn nodes, and append those with
        depth max_depth + 1 to the frontier, so the frontier doesn't get
        cluttered.

        Parameters
        ----------
        root : TreeNode
            The root tree node.

        Returns
        -------

        """
        if self._tree is None:
            return
        # if this is the first time drawing the tree begin with root
        if not self._drawn_nodes:
            self._frontier.appendleft((0, root))
        # if the depth was decreased, we can clear the frontier, otherwise
        # frontier gets cluttered with non-frontier nodes
        was_decreased = self._depth_was_decreased()
        if was_decreased:
            self._frontier.clear()
        # remove nodes from drawn and add to frontier if limit is decreased
        while self._drawn_nodes:
            depth, node = self._drawn_nodes.pop()
            # check if the node is in the allowed limit
            if depth <= self._depth_limit:
                self._drawn_nodes.append((depth, node))
                break
            if depth == self._depth_limit + 1:
                self._frontier.appendleft((depth, node))

            if node.label in self._square_objects:
                self._square_objects[node.label].hide()

        # add nodes to drawn and remove from frontier if limit is increased
        while self._frontier:
            depth, node = self._frontier.popleft()
            # check if the depth of the node is outside the allowed limit
            if depth > self._depth_limit:
                self._frontier.appendleft((depth, node))
                break
            self._drawn_nodes.append((depth, node))
            self._frontier.extend((depth + 1, c) for c in node.children)

            if node.label in self._square_objects:
                self._square_objects[node.label].show()
            else:
                square_obj = InteractiveSquareGraphicsItem \
                    if self._interactive else SquareGraphicsItem
                self._square_objects[node.label] = square_obj(
                    node,
                    parent=self,
                    brush=QtGui.QBrush(
                        self._calc_node_color(self.tree_adapter, node)
                    ),
                    tooltip=self._get_tooltip(node),
                    zvalue=depth,
                )

    def _depth_was_decreased(self):
        if not self._drawn_nodes:
            return False
        # checks if the max depth was increased from the last change
        depth, node = self._drawn_nodes.pop()
        self._drawn_nodes.append((depth, node))
        # if the right most node in drawn nodes has appropriate depth, it must
        # have been increased
        return depth > self._depth_limit

    def _squares(self):
        return [node.graphics_item for _, node in self._drawn_nodes]

    def _clear_scene(self):
        for square in self._squares():
            self.scene().removeItem(square)
        self._frontier.clear()
        self._drawn_nodes.clear()
        self._square_objects.clear()

    def boundingRect(self):
        return self.childrenBoundingRect().adjusted(
            -self._padding, -self._padding, self._padding, self._padding)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        return self.boundingRect().size() + \
               QtCore.QSizeF(self._padding, self._padding)


class SquareGraphicsItem(QtGui.QGraphicsRectItem):
    """Square Graphics Item.

    Square component to draw as components for the non-interactive Pythagoras
    tree.

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    brush : QColor, optional
        The brush to be used as the backgound brush.
    pen : QPen, optional
        The pen to be used for the border.

    """

    def __init__(self, tree_node, parent=None, **kwargs):
        self.tree_node = tree_node
        self.tree_node.graphics_item = self

        center, length, angle = tree_node.square
        self._center_point = center
        self.center = QtCore.QPointF(*center)
        self.length = length
        self.angle = angle
        super().__init__(self._get_rect_attributes(), parent)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setRotation(degrees(angle))

        self.setBrush(kwargs.get('brush', QtGui.QColor('#297A1F')))
        self.setPen(kwargs.get('pen', QtGui.QPen(QtGui.QColor('#000'))))

        self.setAcceptHoverEvents(True)
        self.setZValue(kwargs.get('zvalue', 0))
        self.z_step = Z_STEP

        # calculate the correct z values based on the parent
        if self.tree_node.parent != -1:
            p = self.tree_node.parent
            # override root z step
            num_children = len(p.children)
            own_index = [1 if c.label == self.tree_node.label else 0
                         for c in p.children].index(1)

            self.z_step = int(p.graphics_item.z_step / num_children)
            base_z = p.graphics_item.zValue()

            self.setZValue(base_z + own_index * self.z_step)

    def _get_rect_attributes(self):
        """Get the rectangle attributes requrired to draw item.

        Compute the QRectF that a QGraphicsRect needs to be rendered with the
        data passed down in the constructor.

        """
        height = width = self.length
        x = self.center.x() - self.length / 2
        y = self.center.y() - self.length / 2
        return QtCore.QRectF(x, y, height, width)


class InteractiveSquareGraphicsItem(SquareGraphicsItem):
    """Interactive square graphics items.

    This is different from the base square graphics item so that it is
    selectable, and it can handle and react to hover events (highlight and
    focus own branch).

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    brush : QColor, optional
        The brush to be used as the backgound brush.
    pen : QPen, optional
        The pen to be used for the border.

    """

    timer = QtCore.QTimer()

    MAX_OPACITY = 1.
    SELECTION_OPACITY = .5
    HOVER_OPACITY = .1

    def __init__(self, tree_node, parent=None, **kwargs):
        super().__init__(tree_node, parent, **kwargs)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        self.initial_zvalue = self.zValue()
        # The max z value changes if any item is selected
        self.any_selected = False

        self.setToolTip(kwargs.get('tooltip', 'Tooltip'))

        self.timer.setSingleShot(True)

    def hoverEnterEvent(self, event):
        self.timer.stop()

        def fnc(graphics_item):
            graphics_item.setZValue(Z_STEP)
            if self.any_selected:
                if graphics_item.isSelected():
                    opacity = self.MAX_OPACITY
                else:
                    opacity = self.SELECTION_OPACITY
            else:
                opacity = self.MAX_OPACITY
            graphics_item.setOpacity(opacity)

        def other_fnc(graphics_item):
            if graphics_item.isSelected():
                opacity = self.MAX_OPACITY
            else:
                opacity = self.HOVER_OPACITY
            graphics_item.setOpacity(opacity)
            graphics_item.setZValue(self.initial_zvalue)

        self._propagate_z_values(self, fnc, other_fnc)

    def hoverLeaveEvent(self, event):

        def fnc(graphics_item):
            # No need to set opacity in this branch since it was just selected
            # and had the max value
            graphics_item.setZValue(self.initial_zvalue)

        def other_fnc(graphics_item):
            if self.any_selected:
                if graphics_item.isSelected():
                    opacity = self.MAX_OPACITY
                else:
                    opacity = self.SELECTION_OPACITY
            else:
                opacity = self.MAX_OPACITY
            graphics_item.setOpacity(opacity)

        self.timer.timeout.connect(
            lambda: self._propagate_z_values(self, fnc, other_fnc))

        self.timer.start(250)

    def _propagate_z_values(self, graphics_item, fnc, other_fnc):
        self._propagate_to_children(graphics_item, fnc)
        self._propagate_to_parents(graphics_item, fnc, other_fnc)

    def _propagate_to_children(self, graphics_item, fnc):
        # propagate function that handles graphics item to appropriate children
        fnc(graphics_item)
        for c in graphics_item.tree_node.children:
            self._propagate_to_children(c.graphics_item, fnc)

    def _propagate_to_parents(self, graphics_item, fnc, other_fnc):
        # propagate function that handles graphics item to appropriate parents
        if graphics_item.tree_node.parent != -1:
            parent = graphics_item.tree_node.parent.graphics_item
            # handle the non relevant children nodes
            for c in parent.tree_node.children:
                if c != graphics_item.tree_node:
                    self._propagate_to_children(c.graphics_item, other_fnc)
            # handle the parent node
            fnc(parent)
            # propagate up the tree
            self._propagate_to_parents(parent, fnc, other_fnc)

    def selection_changed(self):
        """Handle selection changed."""
        self.any_selected = len(self.scene().selectedItems()) > 0
        if self.any_selected:
            if self.isSelected():
                self.setOpacity(self.MAX_OPACITY)
            else:
                if self.opacity() != self.HOVER_OPACITY:
                    self.setOpacity(self.SELECTION_OPACITY)
        else:
            self.setGraphicsEffect(None)
            self.setOpacity(self.MAX_OPACITY)

    def paint(self, painter, option, widget=None):
        # Override the default selected appearance
        if self.isSelected():
            option.state ^= QtGui.QStyle.State_Selected
            rect = self.rect()
            # this must render before overlay due to order in which it's drawn
            super().paint(painter, option, widget)
            painter.save()
            pen = QtGui.QPen(QtGui.QColor(Qt.black))
            pen.setWidth(4)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(2, 2, -2, -2))
            painter.restore()
        else:
            super().paint(painter, option, widget)


class TreeNode:
    """A node in the tree structure used to represent the tree adapter

    Parameters
    ----------
    label : int
        The label of the tree node, can be looked up in the original tree.
    square : Square
        The square the represents the tree node.
    parent : TreeNode or object
        The parent of the current node. In the case of root, an object
        containing the root label of the tree adapter should be passed.
    children : tuple of TreeNode, optional, default is empty tuple
        All the children that belong to this node.

    """

    def __init__(self, label, square, parent, children=()):
        self.label = label
        self.square = square
        self.parent = parent
        self.children = children
        self.graphics_item = None

    def __str__(self):
        return '({}) -> [{}]'.format(self.parent, self.label)


class PythagorasTree:
    """Pythagoras tree.

    Contains all the logic that converts a given tree adapter to a tree
    consisting of node classes.

    """

    def __init__(self):
        # store the previous angles of each square children so that slopes can
        # be computed
        self._slopes = defaultdict(list)

    def pythagoras_tree(self, tree, node, square):
        """Get the Pythagoras tree representation in a graph like view.

        Constructs a graph using TreeNode into a tree structure. Each node in
        graph contains the information required to plot the the tree.

        Parameters
        ----------
        tree : TreeAdapter
            A tree adapter instance where the original tree is stored.
        node : int
            The node label, the root node is denoted with 0.
        square : Square
            The initial square which will represent the root of the tree.

        Returns
        -------
        TreeNode
            The root node which contains the rest of the tree.

        """
        # make sure to clear out any old slopes if we are drawing a new tree
        if node == tree.root:
            self._slopes.clear()

        children = tuple(
            self._compute_child(tree, square, child)
            for child in tree.children(node)
        )
        # make sure to pass a reference to parent to each child
        obj = TreeNode(node, square, tree.parent(node), children)
        # mutate the existing data stored in the created tree node
        for c in children:
            c.parent = obj
        return obj

    def _compute_child(self, tree, parent_square, node):
        """Compute all the properties for a single child.

        Parameters
        ----------
        tree : TreeAdapter
            A tree adapter instance where the original tree is stored.
        parent_square : Square
            The parent square of the given child.
        node : int
            The node label of the child.

        Returns
        -------
        TreeNode
            The tree node representation of the given child with the computed
            subtree.

        """
        weight = tree.weight(node)
        # the angle of the child from its parent
        alpha = weight * pi
        # the child side length
        length = parent_square.length * sin(alpha / 2)
        # the sum of the previous anlges
        prev_angles = sum(self._slopes[parent_square])

        center = self._compute_center(
            parent_square, length, alpha, prev_angles
        )
        # the angle of the square is dependent on the parent, the current
        # angle and the previous angles. Subtract PI/2 so it starts drawing at
        # 0rads.
        angle = parent_square.angle - pi / 2 + prev_angles + alpha / 2
        square = Square(center, length, angle)

        self._slopes[parent_square].append(alpha)

        return self.pythagoras_tree(tree, node, square)

    def _compute_center(self, initial_square, length, alpha, base_angle=0):
        """Compute the central point of a child square.

        Parameters
        ----------
        initial_square : Square
            The parent square representation where we will be drawing from.
        length : float
            The length of the side of the new square (the one we are computing
            the center for).
        alpha : float
            The angle that defines the size of our new square (in radians).
        base_angle : float, optional
            If the square we want to find the center for is not the first child
            i.e. its edges does not touch the base square, then we need the
            initial angle that will act as the starting point for the new
            square.

        Returns
        -------
        Point
            The central point to the new square.

        """
        parent_center, parent_length, parent_angle = initial_square
        # get the point on the square side that will be the rotation origin
        t0 = self._get_point_on_square_edge(
            parent_center, parent_length, parent_angle)
        # get the edge point that we will rotate around t0
        square_diagonal_length = sqrt(2 * parent_length ** 2)
        edge = self._get_point_on_square_edge(
            parent_center, square_diagonal_length, parent_angle - pi / 4)
        # if the new square is not the first child, we need to rotate the edge
        if base_angle != 0:
            edge = self._rotate_point(edge, t0, base_angle)

        # rotate the edge point to the correct spot
        t1 = self._rotate_point(edge, t0, alpha)

        # calculate the middle point between the rotated point and edge
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        # calculate the slope of the new square
        slope = parent_angle - pi / 2 + alpha / 2
        # using this data, we can compute the square center
        return self._get_point_on_square_edge(t2, length, slope + base_angle)

    @staticmethod
    def _rotate_point(point, around, alpha):
        """Rotate a point around another point by some angle.

        Parameters
        ----------
        point : Point
            The point to rotate.
        around : Point
            The point to perform rotation around.
        alpha : float
            The angle to rotate by (in radians).

        Returns
        -------
        Point:
            The rotated point.

        """
        temp = Point(point.x - around.x, point.y - around.y)
        temp = Point(
            temp.x * cos(alpha) - temp.y * sin(alpha),
            temp.x * sin(alpha) + temp.y * cos(alpha)
        )
        return Point(temp.x + around.x, temp.y + around.y)

    @staticmethod
    def _get_point_on_square_edge(center, length, angle):
        """Calculate the central point on the drawing edge of the given square.

        Parameters
        ----------
        center : Point
            The square center point.
        length : float
            The square side length.
        angle : float
            The angle of the square.

        Returns
        -------
        Point
            A point on the center of the drawing edge of the given square.

        """
        return Point(
            center.x + length / 2 * cos(angle),
            center.y + length / 2 * sin(angle)
        )

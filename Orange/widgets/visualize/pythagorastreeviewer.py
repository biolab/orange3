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
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict, deque
from math import pi, sqrt, cos, sin, degrees

import numpy as np
from AnyQt.QtCore import Qt, QTimer, QRectF, QSizeF
from AnyQt.QtGui import QColor, QPen
from AnyQt.QtWidgets import (
    QSizePolicy, QGraphicsItem, QGraphicsRectItem, QGraphicsWidget, QStyle
)

from Orange.widgets.utils import to_html
from Orange.widgets.visualize.utils.tree.rules import Rule
from Orange.widgets.visualize.utils.tree.treeadapter import TreeAdapter

# z index range, increase if needed
Z_STEP = 5000000

Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])


class PythagorasTreeViewer(QGraphicsWidget):
    """Pythagoras tree viewer graphics widget.

    Examples
    --------
    >>> from Orange.widgets.visualize.utils.tree.treeadapter import (
    ...     TreeAdapter
    ... )
    Pass tree through constructor.
    >>> tree_view = PythagorasTreeViewer(parent=scene, adapter=tree_adapter)

    Pass tree later through method.
    >>> tree_adapter = TreeAdapter()
    >>> scene = QGraphicsScene()
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
    interacitive : bool, optional
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
        super().__init__()
        self.parent = parent

        # In case a tree was passed, it will be handled at the end of init
        self.tree_adapter = None
        self.root = None

        self._depth_limit = depth_limit
        self._interactive = kwargs.get('interactive', True)
        self._padding = padding

        self._square_objects = {}
        self._drawn_nodes = deque()
        self._frontier = deque()

        self._target_class_index = 0

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # If a tree was passed in the constructor, set and draw the tree
        if adapter is not None:
            self.set_tree(
                adapter,
                target_class_index=kwargs.get('target_class_index'),
                weight_adjustment=kwargs.get('weight_adjustment'),
            )
            # Since `set_tree` needs to draw the entire tree to be visualized
            # properly, it overrides the `depth_limit` to max. If we specified
            # the depth limit, however, apply that afterwards-
            self.set_depth_limit(depth_limit)

    def set_tree(self, tree_adapter, weight_adjustment=lambda x: x,
                 target_class_index=0):
        """Pass in a new tree adapter instance and perform updates to canvas.

        Parameters
        ----------
        tree_adapter : TreeAdapter
            The new tree adapter that is to be used.
        weight_adjustment : callable
            A weight adjustment function that with signature `x -> x`
        target_class_index : int

        Returns
        -------

        """
        self.clear_tree()
        self.tree_adapter = tree_adapter
        self.weight_adjustment = weight_adjustment

        if self.tree_adapter is not None:
            self.root = self._calculate_tree(self.tree_adapter, self.weight_adjustment)
            self.set_depth_limit(tree_adapter.max_depth)
            self.target_class_changed(target_class_index)
            self._draw_tree(self.root)

    def set_size_calc(self, weight_adjustment):
        """Set the weight adjustment on the tree. Redraws the whole tree."""
        # Since we have to redraw the whole tree anyways, just call `set_tree`
        self.weight_adjustment = weight_adjustment
        self.set_tree(self.tree_adapter, self.weight_adjustment,
                      self._target_class_index)

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
        self._draw_tree(self.root)

    def target_class_changed(self, target_class_index=0):
        """When the target class has changed, perform appropriate updates."""
        self._target_class_index = target_class_index

        def _recurse(node):
            node.target_class_index = target_class_index
            for child in node.children:
                _recurse(child)

        _recurse(self.root)

    def tooltip_changed(self, tooltip_enabled):
        """Set the tooltip to the appropriate value on each square."""
        for square in self._squares():
            if tooltip_enabled:
                square.setToolTip(square.tree_node.tooltip)
            else:
                square.setToolTip(None)

    def clear(self):
        """Clear the entire widget state."""
        self.clear_tree()
        self._target_class_index = 0

    def clear_tree(self):
        """Clear only the tree, keeping tooltip and color functions."""
        self.tree_adapter = None
        self.root = None
        self._clear_scene()

    def _calculate_tree(self, tree_adapter, weight_adjustment):
        """Actually calculate the tree squares"""
        tree_builder = PythagorasTree(weight_adjustment=weight_adjustment)
        return tree_builder.pythagoras_tree(
            tree_adapter, tree_adapter.root, Square(Point(0, 0), 200, -pi / 2)
        )

    def _draw_tree(self, root):
        """Efficiently draw the tree with regards to the depth.

        If we used a recursive approach, the tree would have to be redrawn
        every time the depth changed, which is very impractical for larger
        trees, since drawing can take a long time.

        Using an iterative approach, we use two queues to represent the tree
        frontier and the nodes that have already been drawn. We also store the
        current depth. This way, when the max depth is increased, we do not
        redraw the entire tree but only iterate through the frontier and draw
        those nodes, and update the frontier accordingly.
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
        if self.root is None:
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

            node.target_class_index = self._target_class_index
            if node.label in self._square_objects:
                self._square_objects[node.label].show()
            else:
                square_obj = InteractiveSquareGraphicsItem \
                    if self._interactive else SquareGraphicsItem
                self._square_objects[node.label] = square_obj(
                    node, parent=self, zvalue=depth)

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
        return self.boundingRect().size() + QSizeF(self._padding, self._padding)


class SquareGraphicsItem(QGraphicsRectItem):
    """Square Graphics Item.

    Square component to draw as components for the non-interactive Pythagoras
    tree.

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    parent : QGraphicsItem

    """

    def __init__(self, tree_node, parent=None, **kwargs):
        self.tree_node = tree_node
        super().__init__(self._get_rect_attributes(), parent)
        self.tree_node.graphics_item = self

        self.setTransformOriginPoint(self.boundingRect().center())
        self.setRotation(degrees(self.tree_node.square.angle))

        self.setBrush(kwargs.get('brush', QColor('#297A1F')))
        # The border should be invariant to scaling
        pen = QPen(QColor(Qt.black))
        pen.setWidthF(0.75)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.setAcceptHoverEvents(True)
        self.setZValue(kwargs.get('zvalue', 0))
        self.z_step = Z_STEP

        # calculate the correct z values based on the parent
        if self.tree_node.parent != TreeAdapter.ROOT_PARENT:
            p = self.tree_node.parent
            # override root z step
            num_children = len(p.children)
            own_index = [1 if c.label == self.tree_node.label else 0
                         for c in p.children].index(1)

            self.z_step = int(p.graphics_item.z_step / num_children)
            base_z = p.graphics_item.zValue()

            self.setZValue(base_z + own_index * self.z_step)

    def update(self):
        self.setBrush(self.tree_node.color)
        return super().update()

    def _get_rect_attributes(self):
        """Get the rectangle attributes requrired to draw item.

        Compute the QRectF that a QGraphicsRect needs to be rendered with the
        data passed down in the constructor.

        """
        center, length, _ = self.tree_node.square
        x = center[0] - length / 2
        y = center[1] - length / 2
        return QRectF(x, y, length, length)


class InteractiveSquareGraphicsItem(SquareGraphicsItem):
    """Interactive square graphics items.

    This is different from the base square graphics item so that it is
    selectable, and it can handle and react to hover events (highlight and
    focus own branch).

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    parent : QGraphicsItem

    """

    timer = QTimer()

    MAX_OPACITY = 1.
    SELECTION_OPACITY = .5
    HOVER_OPACITY = .1

    def __init__(self, tree_node, parent=None, **kwargs):
        super().__init__(tree_node, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        self.initial_zvalue = self.zValue()
        # The max z value changes if any item is selected
        self.any_selected = False

        self.timer.setSingleShot(True)

    def update(self):
        self.setToolTip(self.tree_node.tooltip)
        return super().update()

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
        if graphics_item.tree_node.parent != TreeAdapter.ROOT_PARENT:
            parent = graphics_item.tree_node.parent.graphics_item
            # handle the non relevant children nodes
            for c in parent.tree_node.children:
                if c != graphics_item.tree_node:
                    self._propagate_to_children(c.graphics_item, other_fnc)
            # handle the parent node
            fnc(parent)
            # propagate up the tree
            self._propagate_to_parents(parent, fnc, other_fnc)

    def mouseDoubleClickEvent(self, event):
        self.tree_node.tree.reverse_children(self.tree_node.label)
        p = self.parentWidget()  # PythagorasTreeViewer
        p.set_tree(p.tree_adapter, p.weight_adjustment, self.tree_node.target_class_index)
        widget = p.parent  # OWPythagorasTree
        widget._update_main_area()

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
            option.state ^= QStyle.State_Selected
            rect = self.rect()
            # this must render before overlay due to order in which it's drawn
            super().paint(painter, option, widget)
            painter.save()
            pen = QPen(QColor(Qt.black))
            pen.setWidthF(2)
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()
        else:
            super().paint(painter, option, widget)


class TreeNode(metaclass=ABCMeta):
    """A tree node meant to be used in conjuction with graphics items.

    The tree node contains methods that are very general to any tree
    visualisation, containing methods for the node color and tooltip.

    This is an abstract class and not meant to be used by itself. There are two
    subclasses - `DiscreteTreeNode` and `ContinuousTreeNode`, which need no
    explanation. If you don't wish to deal with figuring out which node to use,
    the `from_tree` method is provided.

    Parameters
    ----------
    label : int
        The label of the tree node, can be looked up in the original tree.
    square : Square
        The square the represents the tree node.
    tree : TreeAdapter
        The tree model that the node belongs to.
    children : tuple of TreeNode, optional, default is empty tuple
        All the children that belong to this node.

    """

    def __init__(self, label, square, tree, children=()):
        self.label = label
        self.square = square
        self.tree = tree
        self.children = children
        self.parent = None
        # Properties that should update the associated graphics item
        self.__graphics_item = None
        self.__target_class_index = None

    @property
    def graphics_item(self):
        return self.__graphics_item

    @graphics_item.setter
    def graphics_item(self, graphics_item):
        self.__graphics_item = graphics_item
        self._update_graphics_item()

    @property
    def target_class_index(self):
        return self.__target_class_index

    @target_class_index.setter
    def target_class_index(self, target_class_index):
        self.__target_class_index = target_class_index
        self._update_graphics_item()

    def _update_graphics_item(self):
        if self.__graphics_item is not None:
            self.__graphics_item.update()

    @classmethod
    def from_tree(cls, label, square, tree, children=()):
        """Construct the appropriate type of node from the given tree."""
        if tree.domain.has_discrete_class:
            node = DiscreteTreeNode
        else:
            node = ContinuousTreeNode
        return node(label, square, tree, children)

    @property
    @abstractmethod
    def color(self):
        """Get the color of the node.

        Returns
        -------
        QColor

        """

    @property
    @abstractmethod
    def tooltip(self):
        """get the tooltip for the node.

        Returns
        -------
        str

        """

    @property
    def color_palette(self):
        return self.tree.domain.class_var.palette

    def _rules_str(self):
        rules = self.tree.rules(self.label)
        if rules:
            if isinstance(rules[0], Rule):
                sorted_rules = sorted(rules[:-1], key=lambda rule: rule.attr_name)
                return '<br>'.join(str(rule) for rule in sorted_rules) + \
                       '<br><b>%s</b>' % rules[-1]
            else:
                return '<br>'.join(to_html(rule) for rule in rules)
        else:
            return ''


class DiscreteTreeNode(TreeNode):
    """Discrete tree node containing methods for tree visualisations.

    Colors are defined by the data domain, and possible colorings are different
    target classes.

    """

    @property
    def color(self):
        distribution = self.tree.get_distribution(self.label)[0]
        total = np.sum(distribution)

        if self.target_class_index:
            p = distribution[self.target_class_index - 1] / total
            color = self.color_palette[self.target_class_index - 1]
            color = color.lighter(int(200 - 100 * p))
        else:
            modus = np.argmax(distribution)
            p = distribution[modus] / (total or 1)
            color = self.color_palette[int(modus)]
            color = color.lighter(int(400 - 300 * p))
        return color

    @property
    def tooltip(self):
        distribution = self.tree.get_distribution(self.label)[0]
        total = int(np.sum(distribution))
        if self.target_class_index:
            samples = distribution[self.target_class_index - 1]
            text = ''
        else:
            modus = np.argmax(distribution)
            samples = distribution[modus]
            text = self.tree.domain.class_vars[0].values[modus] + \
                '<br>'
        ratio = samples / np.sum(distribution)

        rules_str = self._rules_str()
        splitting_attr = self.tree.attribute(self.label)

        return '<p>' \
            + text \
            + '{}/{} samples ({:2.3f}%)'.format(
                int(samples), total, ratio * 100) \
            + '<hr>' \
            + ('Split by ' + splitting_attr.name
               if not self.tree.is_leaf(self.label) else '') \
            + ('<br><br>' if rules_str and not self.tree.is_leaf(self.label) else '') \
            + rules_str \
            + '</p>'


class ContinuousTreeNode(TreeNode):
    """Continuous tree node containing methods for tree visualisations.

    There are three modes of coloring:
     - None, which is a solid color
     - Mean, which colors nodes w.r.t. the mean value of all the
       instances that belong to a given node.
     - Standard deviation, which colors nodes w.r.t the standard deviation of
       all the instances that belong to a given node.

    """

    COLOR_NONE, COLOR_MEAN, COLOR_STD = range(3)
    COLOR_METHODS = {
        'None': COLOR_NONE,
        'Mean': COLOR_MEAN,
        'Standard deviation': COLOR_STD,
    }

    @property
    def color(self):
        if self.target_class_index is self.COLOR_MEAN:
            return self._color_mean()
        elif self.target_class_index is self.COLOR_STD:
            return self._color_var()
        else:
            return QColor(255, 255, 255)

    def _color_mean(self):
        """Color the nodes with respect to the mean of instances inside."""
        min_mean = np.min(self.tree.instances.Y)
        max_mean = np.max(self.tree.instances.Y)
        instances = self.tree.get_instances_in_nodes(self.label)
        mean = np.mean(instances.Y)
        return self.color_palette.value_to_qcolor(
            mean, low=min_mean, high=max_mean)

    def _color_var(self):
        """Color the nodes with respect to the variance of instances inside."""
        min_std, max_std = 0, np.std(self.tree.instances.Y)
        instances = self.tree.get_instances_in_nodes(self.label)
        std = np.std(instances.Y)
        return self.color_palette.value_to_qcolor(
            std, low=min_std, high=max_std)

    @property
    def tooltip(self):
        num_samples = self.tree.num_samples(self.label)

        instances = self.tree.get_instances_in_nodes(self.label)
        mean = np.mean(instances.Y)
        std = np.std(instances.Y)

        rules_str = self._rules_str()
        splitting_attr = self.tree.attribute(self.label)

        return '<p>Mean: {:2.3f}'.format(mean) \
            + '<br>Standard deviation: {:2.3f}'.format(std) \
            + '<br>{} samples'.format(num_samples) \
            + '<hr>' \
            + ('Split by ' + splitting_attr.name
               if not self.tree.is_leaf(self.label) else '') \
            + ('<br><br>' if rules_str and not self.tree.is_leaf(self.label) else '') \
            + rules_str \
            + '</p>'


class PythagorasTree:
    """Pythagoras tree.

    Contains all the logic that converts a given tree adapter to a tree
    consisting of node classes.

    Parameters
    ----------
    weight_adjustment : callable
        The function to be used to adjust child weights

    """

    def __init__(self, weight_adjustment=lambda x: x):
        self.adjust_weight = weight_adjustment
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

        # Calculate the adjusted child weights for the node children
        child_weights = [self.adjust_weight(tree.weight(c))
                         for c in tree.children(node)]
        total_weight = sum(child_weights)
        normalized_child_weights = [cw / total_weight for cw in child_weights]

        children = tuple(
            self._compute_child(tree, square, child, cw)
            for child, cw in zip(tree.children(node), normalized_child_weights)
        )
        # make sure to pass a reference to parent to each child
        obj = TreeNode.from_tree(node, square, tree, children)
        # mutate the existing data stored in the created tree node
        for c in children:
            c.parent = obj
        return obj

    def _compute_child(self, tree, parent_square, node, weight):
        """Compute all the properties for a single child.

        Parameters
        ----------
        tree : TreeAdapter
            A tree adapter instance where the original tree is stored.
        parent_square : Square
            The parent square of the given child.
        node : int
            The node label of the child.
        weight : float
            The weight of the node relative to its parent e.g. two children in
            relation 3:1 should have weights .75 and .25, respectively.

        Returns
        -------
        TreeNode
            The tree node representation of the given child with the computed
            subtree.

        """
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

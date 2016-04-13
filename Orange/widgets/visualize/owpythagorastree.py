# coding=utf-8
from collections import namedtuple, defaultdict, deque
from functools import lru_cache
from math import pi, sqrt, cos, sin, degrees, log

import numpy as np
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

import Orange
from Orange.classification.tree import TreeClassifier
from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.colorpalette import DefaultRGBColors
from Orange.widgets.widget import OWWidget

# Please note that all angles are in radians
Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])


class OWPythagorasTree(OWWidget):
    name = 'Pythagoras Tree'
    description = 'Generalized Pythagoras Tree for visualizing trees.'
    priority = 100

    # Enable the save as feature
    graph_name = True

    inputs = [('Classification Tree', TreeClassifier, 'set_tree')]
    outputs = [('Selected Data', Table)]

    # Settings
    zoom = settings.ContextSetting(5)
    depth_limit = settings.ContextSetting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    size_log_scale = settings.Setting(2)
    auto_commit = settings.Setting(True)

    def __init__(self):
        super().__init__()

        # Different methods to calculate the size of squares
        self.SIZE_CALCULATION = [
            ('Normal', lambda x: x),
            ('Sqrt', lambda x: sqrt(x)),
            ('Logarithmic', lambda x: log(x * self.size_log_scale)),
        ]

        # Instance variables
        self.domain = None
        self.model = None
        self.tree_adapter = None
        self.tree = None

        self.square_objects = {}
        self.drawn_nodes = deque()
        self.frontier = deque()

        self.color_palette = [QtGui.QColor(*c) for c in DefaultRGBColors]

        # CONTROL AREA
        # Tree info area
        box_info = gui.widgetBox(self.controlArea, 'Tree')
        self.info = gui.widgetLabel(box_info, label='No tree.')

        # Display controls area
        box_display = gui.widgetBox(self.controlArea, 'Display')
        gui.hSlider(
            box_display, self, 'zoom', label='Zoom',
            minValue=1, maxValue=10, step=1, ticks=False,
            callback=None)
        self.depth_slider = gui.hSlider(
            box_display, self, 'depth_limit', label='Depth',
            maxValue=0, ticks=False,
            callback=self.update_depth)
        self.target_class_combo = gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation='horizontal', items=[], contentsLength=8,
            callback=self.update_colors)
        gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation='horizontal',
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self.invalidate_tree)
        gui.hSlider(
            box_display, self, 'size_log_scale', label='Log scale',
            minValue=1, maxValue=100, step=1, ticks=False,
            callback=self.invalidate_tree)

        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

        # Bottom options
        gui.auto_commit(
            self.controlArea, self, value='auto_commit',
            label='Send selected instances', auto_label='Auto send is on')

        # GUI - MAIN AREA
        self.scene = QtGui.QGraphicsScene(self)
        self.view = QtGui.QGraphicsView(self.scene, self.mainArea)

        # Flip y axis for sane drawing
        matrix = QtGui.QMatrix()
        matrix.scale(1, -1)
        self.view.setMatrix(matrix)

        # self.view.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)

        self.mainArea.layout().addWidget(self.view)

    def set_tree(self, model=None):
        """When a different tree is given."""
        self.clear()
        self.model = model
        if model is not None:
            self.domain = model.domain
            self.invalidate_tree()
            self._update_target_class_combo()
            self._update_depth_slider()

    def invalidate_tree(self):
        """When the tree needs to be recalculated."""
        self._clear_scene()
        self._get_tree_adapter(self.model)
        self._draw_tree(self._calculate_tree())

    def update_depth(self):
        """This method should be called when the depth changes"""
        self._draw_tree(self.tree)

    def update_colors(self):
        """When the colors of the nodes need to be updated."""
        for square in self._get_scene_squares():
            square.setBrush(self._get_node_color(square.tree_node))

    def clear(self):
        """Clear all relevant data from the widget."""
        self.domain = None
        self.model = None
        self.tree_adapter = None
        self.tree = None
        self._clear_scene()

    def _update_depth_slider(self):
        # TODO figure out a way to change slider label width with larger nums
        self.depth_slider.setMaximum(self.tree_adapter.max_depth)

    def _get_tree_adapter(self, model):
        self.tree_adapter = SklTreeAdapter(
            model.skl_model.tree_,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )

    def _update_target_class_combo(self):
        self.target_class_combo.clear()
        self.target_class_combo.addItem('None')
        values = [c.title() for c in self.domain.class_vars[0].values]
        self.target_class_combo.addItems(values)
        # make sure we don't attempt to assign an index gt than the number of
        # classes
        if self.target_class_index >= len(values):
            self.target_class_index = 0
        self.target_class_combo.setCurrentIndex(self.target_class_index)

    def _calculate_tree(self):
        # Actually calculate the tree squares
        tree_builder = PythagorasTree()
        self.tree = tree_builder.pythagoras_tree(
            self.tree_adapter, 0, Square(Point(0, 0), 200, pi / 2)
        )
        return self.tree

    def _depth_was_decreased(self):
        if not self.drawn_nodes:
            return False
        # checks if the max depth was increased from the last change
        depth, node = self.drawn_nodes.pop()
        self.drawn_nodes.append((depth, node))
        # if the right most node in drawn nodes has appropriate depth, it must
        # have been increased
        return depth > self.depth_limit

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
        # if this is the first time drawing the tree begin with root
        if not self.drawn_nodes:
            self.frontier.appendleft((0, root))
        # if the depth was decreased, we can clear the frontier, otherwise
        # frontier gets cluttered with non-frontier nodes
        was_decreased = self._depth_was_decreased()
        if was_decreased:
            self.frontier.clear()
        # remove nodes from drawn and add to frontier if limit is decreased
        while self.drawn_nodes:
            depth, node = self.drawn_nodes.pop()
            # check if the node is in the allowed limit
            if depth <= self.depth_limit:
                self.drawn_nodes.append((depth, node))
                break
            if depth == self.depth_limit + 1:
                self.frontier.appendleft((depth, node))

            if node.label in self.square_objects:
                self.square_objects[node.label].hide()

        # add nodes to drawn and remove from frontier if limit is increased
        while self.frontier:
            depth, node = self.frontier.popleft()
            # check if the depth of the node is outside the allowed limit
            if depth > self.depth_limit:
                self.frontier.appendleft((depth, node))
                break
            self.drawn_nodes.append((depth, node))
            self.frontier.extend((depth + 1, c) for c in node.children)

            if node.label in self.square_objects:
                self.square_objects[node.label].show()
            else:
                self.square_objects[node.label] = SquareGraphicsItem(
                    node,
                    brush=QtGui.QBrush(self._get_node_color(node))
                )
                self.scene.addItem(self.square_objects[node.label])

    def _get_node_color(self, tree_node):
        # this is taken almost directly from the existing classification tree
        # viewer
        colors = self.color_palette
        distribution = self.tree_adapter.get_distribution(tree_node.label)[0]
        total = self.tree_adapter.num_samples(tree_node.label)

        if self.target_class_index:
            p = distribution[self.target_class_index - 1] / total
            color = colors[self.target_class_index - 1].light(200 - 100 * p)
        else:
            modus = np.argmax(distribution)
            p = distribution[modus] / (total or 1)
            color = colors[int(modus)].light(400 - 300 * p)
        return color

    def _get_scene_squares(self):
        return filter(lambda i: isinstance(i, SquareGraphicsItem),
                      self.scene.items())

    def _clear_scene(self):
        self.scene.clear()
        self.frontier.clear()
        self.drawn_nodes.clear()
        self.square_objects.clear()

    def onDeleteWidget(self):
        """When deleting the widget."""
        self.clear()
        super().onDeleteWidget()

    def commit(self):
        """Commit the selected data to output."""
        pass

    def send_report(self):
        pass


class SquareGraphicsItem(QtGui.QGraphicsRectItem):
    """Square Graphics Item.

    Square component to draw as components for the Pythagoras tree.

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

    def _get_rect_attributes(self):
        """Get the rectangle attributes requrired to draw item.

        Compute the QRectF that a QGraphicsRect needs to be rendered with the
        data passed down in the constructor.
        """
        height = width = self.length
        x = self.center.x() - self.length / 2
        y = self.center.y() - self.length / 2
        return QtCore.QRectF(x, y, height, width)


class TreeNode:
    """A node in the tree structure used to represent the tree adapter

    Parameters
    ----------
    label : int
        The label of the tree node, can be looked up in the original tree.
    square : Square
        The square the represents the tree node.
    parent : TreeNode
        The parent of the current node.
    children : tuple of TreeNode, optional
        All the children that belong to this node.

    """

    def __init__(self, label, square, parent, children=()):
        self.label = label
        self.square = square
        self.parent = parent
        self.children = children


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
        if node == 0:
            self._slopes.clear()

        children = tuple(
            self._compute_child(tree, square, child)
            for child in tree.children(node)
        )
        return TreeNode(node, square, tree.parent(node), children)

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


class SklTreeAdapter:
    """SklTreeAdapter Class.

    An abstraction on top of the scikit learn classification tree.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The raw sklearn classification tree.
    adjust_weight : function, optional
        If you want finer control over the weights of individual nodes you can
        pass in a function that takes the existsing weight and modifies it.
        The given function must have signture :: Number -> Number

    """

    def __init__(self, tree, adjust_weight=lambda x: x):
        self._tree = tree
        self._adjust_weight = adjust_weight

        # clear memoized functions
        self.weight.cache_clear()
        self._adjusted_child_weight.cache_clear()
        self.parent.cache_clear()

    @lru_cache()
    def weight(self, node):
        """Get the weight of the given node.

        The weights of the children always sum up to 1.

        Parameters
        ----------
        node : int
            The label of the node.

        Returns
        -------
        float
            The weight of the node relative to its siblings.

        """
        return self._adjust_weight(self.num_samples(node)) / \
               self._adjusted_child_weight(self.parent(node))

    @lru_cache()
    def _adjusted_child_weight(self, node):
        """Helps when dealing with adjusted weights.

        It is needed when dealing with non linear weights e.g. when calculating
        the log weight, the sum of logs of all the children will not be equal
        to the log of all the data instances.
        A simple example: log(2) + log(2) != log(4)

        Parameters
        ----------
        node : int
            The label of the node.

        Returns
        -------
        float
            The sum of all of the weights of the children of a given node.

        """
        return sum(self._adjust_weight(self.num_samples(c))
                   for c in self.children(node)) \
            if self.has_children(node) else 0

    def num_samples(self, node):
        return self._tree.n_node_samples[node]

    @lru_cache()
    def parent(self, node):
        for children in (self._tree.children_left, self._tree.children_right):
            try:
                return (children == node).nonzero()[0][0]
            except IndexError:
                continue
        return -1

    def has_children(self, node):
        return self._tree.children_left[node] != -1 \
               or self._tree.children_right[node] != -1

    def children(self, node):
        if self.has_children(node):
            return self._left_child(node), self._right_child(node)
        return ()

    def _left_child(self, node):
        return self._tree.children_left[node]

    def _right_child(self, node):
        return self._tree.children_right[node]

    def get_distribution(self, node):
        return self._tree.value[node]

    def get_impurity(self, node):
        return self._tree.impurity[node]

    @property
    def max_depth(self):
        return self._tree.max_depth

    @property
    def num_nodes(self):
        return self._tree.node_count


def main():
    import sys
    from Orange.classification.tree import TreeLearner

    argv = sys.argv
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    app = QtGui.QApplication(argv)
    ow = OWPythagorasTree()
    data = Orange.data.Table(filename)
    clf = TreeLearner(max_depth=1000)(data)
    clf.instances = data
    ow.set_tree(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()

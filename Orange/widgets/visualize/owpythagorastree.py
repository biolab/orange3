# coding=utf-8
from collections import namedtuple, defaultdict
from math import pi, sqrt, cos, sin, degrees, log
from functools import lru_cache

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

import Orange
from Orange.classification.tree import TreeClassifier
from Orange.data.table import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.colorpalette import DefaultRGBColors

# Please note that all angles are in radians
Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])


class OWPythagorasTree(OWWidget):
    name = 'Pythagoras Tree'
    description = 'Generalized Pythagoras Tree for visualizing trees.'
    priority = 100

    # Enable the save as feature
    graph_name = True

    inputs = [('Classification Tree', TreeClassifier, 'set_ctree')]
    outputs = [('Selected Data', Table)]

    # Settings
    zoom = settings.ContextSetting(5)
    depth = settings.ContextSetting(10)
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
        self.raw_tree = None
        self.tree = None

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
        gui.hSlider(
            box_display, self, 'depth', label='Depth',
            minValue=1, maxValue=10, step=1, ticks=False, callback=None)
        gui.comboBox(
            box_display, self, 'target_class_index', label='Target class',
            orientation='horizontal',
            items=[], contentsLength=8, callback=None)
        gui.comboBox(
            box_display, self, 'size_calc_idx', label='Size',
            orientation='horizontal',
            items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8,
            callback=self._invalidate)
        gui.hSlider(
            box_display, self, 'size_log_scale', label='Log scale',
            minValue=1, maxValue=100, step=1, ticks=False,
            callback=self._invalidate)

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

    def commit(self):
        pass

    def set_ctree(self, ctree=None):
        self.clear()
        self.raw_tree = ctree
        if ctree is not None:
            self._get_tree(ctree)
        self._update()

    def _invalidate(self):
        self.scene.clear()
        if self.raw_tree is not None:
            self._get_tree(self.raw_tree)
        self._update()

    def _get_tree(self, ctree):
        self.tree = SklTreeAdapter(
            ctree.skl_model.tree_,
            adjust_weight=self.SIZE_CALCULATION[self.size_calc_idx][1],
        )

    def clear(self):
        self.raw_tree = None
        self.tree = None
        self.scene.clear()

    def _update(self):
        if self.tree is not None:
            tree_builder = PythagorasTree()

            def draw_square_rec(tree_node):
                self.scene.addItem(SquareGraphicsItem(
                    *tree_node.square, brush=self.color_palette[0]
                ))
                for child in tree_node.children:
                    draw_square_rec(child)

            draw_square_rec(tree_builder.pythagoras_tree(
                self.tree, 0, Square(Point(0, 0), 200, pi / 2)
            ))
        else:
            pass

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()

    def send_report(self):
        pass


class SquareGraphicsItem(QtGui.QGraphicsRectItem):
    """Square Graphics Item.

    Square component to draw as components for the Pythagoras tree. It is
    initialized with a given center QPointF, the main side length, and the
    initial angle the sqaure is to be rendered with.

    Parameters
    ----------
    center : Point
        The central point of the square.
    length : float
        The length of the square sides.
    angle : float
        The initial angle at which the square will be rotated (in rads).
    brush : QColor, optional
        The brush to be used as the backgound brush.
    pen : QPen, optional
        The pen to be used for the border.

    """

    def __init__(self, center, length, angle, parent=None, **kwargs):
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
    square : Square
        The square the represents the tree node.
    parent : TreeNode
        The parent of the current node.
    children : tuple of TreeNode, optional
        All the children that belong to this node.

    """

    def __init__(self, square, parent, children=()):
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
        return TreeNode(square, tree.parent(node), children)

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
            return self.left_child(node), self.right_child(node)
        return ()

    def left_child(self, node):
        return self._tree.children_left[node]

    def right_child(self, node):
        return self._tree.children_right[node]


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
    ow.set_ctree(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)


if __name__ == '__main__':
    main()

import math
from collections import namedtuple, defaultdict

import Orange
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.classification.tree import TreeClassifier
from Orange.data.table import Table

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

PI = math.pi

# Please note that all angles are in radians
Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])


class OWPythagorasTree(OWWidget):
    name = "Pythagoras Tree"
    description = "Generalized Pythagoras Tree for visualizing trees."
    priority = 100

    inputs = [("Classification Tree", TreeClassifier, "set_ctree")]
    outputs = [("Selected Data", Table)]

    # Settings
    zoom = Setting(5)

    def __init__(self):
        super().__init__()

        # Instance variables
        self.raw_tree = None
        self.tree = None

        # GUI - CONTROL AREA
        layout = QtGui.QFormLayout()
        layout.setVerticalSpacing(20)
        # layout.setFieldGrowthPolicy(layout.ExpandingFieldsGrow)

        box = self.display_box = \
            gui.widgetBox(self.controlArea, "Display", addSpace=True,
                          orientation=layout)
        layout.addRow(
            "Zoom ",
            gui.hSlider(box, self, 'zoom',
                        minValue=1, maxValue=10, step=1, ticks=False,
                        callback=None,
                        createLabel=False, addToLayout=False, addSpace=False))
        # Stretch to fit the rest of the unsused area
        gui.rubber(self.controlArea)

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

    def set_ctree(self, ctree=None):
        self.clear()
        self.raw_tree = ctree
        if ctree is not None:
            self.tree = SklTreeAdapter(ctree.skl_model.tree_)

        self._update()

    def clear(self):
        self.raw_tree = None
        self.tree = None
        self._clear_scene()

    def _clear_scene(self):
        self.scene.clear()

    def _update(self):
        if self.tree is not None:
            tree_builder = PythagorasTree()

            def draw_square_rec(tree_node):
                self.scene.addItem(SquareGraphicsItem(*tree_node.square))
                for child in tree_node.children:
                    draw_square_rec(child)

            draw_square_rec(tree_builder.pythagoras_tree(
                self.tree, 0, Square(Point(0, 0), 200, PI / 2)
            ))
        else:
            pass

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()


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

    Attributes
    ----------
    center
    length
    angle

    """

    def __init__(self, center, length, angle, parent=None):
        self._center_point = center
        self.center = QtCore.QPointF(*center)
        self.length = length
        self.angle = angle
        super().__init__(self._get_rect_attributes(), parent)
        # no need to perform extra transformations where they won't be seen
        if angle % 90 != 0:
            self.setTransformOriginPoint(self.boundingRect().center())
            self.setRotation(math.degrees(angle))

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
        alpha = weight * PI
        # the child side length
        length = parent_square.length * math.sin(alpha / 2)
        # the sum of the previous anlges
        prev_angles = sum(self._slopes[parent_square])

        center = self._compute_center(
            parent_square, length, alpha, prev_angles
        )
        # the angle of the square is dependent on the parent, the current
        # angle and the previous angles. Subtract PI/2 so it starts drawing at
        # 0rads.
        angle = parent_square.angle - PI / 2 + prev_angles + alpha / 2
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
        square_diagonal_length = math.sqrt(2 * parent_length ** 2)
        edge = self._get_point_on_square_edge(
            parent_center, square_diagonal_length, parent_angle - PI / 4)
        # if the new square is not the first child, we need to rotate the edge
        if base_angle != 0:
            edge = self._rotate_point(edge, t0, base_angle)

        # rotate the edge point to the correct spot
        t1 = self._rotate_point(edge, t0, alpha)

        # calculate the middle point between the rotated point and edge
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        # calculate the slope of the new square
        slope = parent_angle - PI / 2 + alpha / 2
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
            temp.x * math.cos(alpha) - temp.y * math.sin(alpha),
            temp.x * math.sin(alpha) + temp.y * math.cos(alpha)
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
            center.x + length / 2 * math.cos(angle),
            center.y + length / 2 * math.sin(angle)
        )


class SklTreeAdapter:
    def __init__(self, tree):
        self._tree = tree

    def weight(self, node):
        return self.num_samples(node) / self.num_samples(self.parent(node))

    def num_samples(self, node):
        return self._tree.n_node_samples[node]

    def parent(self, node):
        for idx, el in enumerate(self._tree.children_left):
            if el == node:
                return idx
        for idx, el in enumerate(self._tree.children_right):
            if el == node:
                return idx
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
    clf = TreeLearner(max_depth=10)(data)
    clf.instances = data
    ow.set_ctree(clf)

    ow.show()
    ow.raise_()
    ow.handleNewSignals()
    app.exec_()

    sys.exit(0)

if __name__ == "__main__":
    main()

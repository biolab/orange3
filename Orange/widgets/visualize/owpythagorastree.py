import math
from collections import namedtuple

import Orange
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.classification.tree import TreeClassifier
from Orange.data.table import Table

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

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

            draw_square_rec(tree_builder.pythagoras_tree(self.tree))
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
        The central point of the square
    length : float
        The length of the square sides
    angle : float
        The initial angle at which the square will be rotated (in rads)

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
        """
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
    children : list of TreeNode, optional
        All the children that belong to this node.

    """
    def __init__(self, square, parent, children=[]):
        self.square = square
        self.parent = parent
        self.children = children


class PythagorasTree:
    def pythagoras_tree(self, tree, node=0,
                        square=Square(Point(0, 0), 200, 0)):
        """Get the Pythagoras tree representation in a graph like view.

        Constructs a graph using TreeNode into a tree structure. Each node in
        graph contains the information required to plot the the tree.

        Parameters
        ----------
        tree : TreeAdapter
            A tree adapter instance where the original tree is stored.
        node : int, optional
            The node label, uses 0 for the root node (the default is 0).
        square : Square, optional
            The initial square where the drawing will start from.

        Returns
        -------
        TreeNode
            The root node which contains the rest of the tree.

        """
        children = []
        if tree.has_children(node):
            left_child = tree.left_child(node)
            right_child = tree.right_child(node)

            w1, w2 = tree.weight(left_child), tree.weight(right_child)
            alpha_1 = math.pi * w2 / (w1 + w2)
            alpha_2 = math.pi * w1 / (w1 + w2)

            length_1 = square.length * math.sin(alpha_1 / 2)
            length_2 = square.length * math.sin(alpha_2 / 2)

            angle_1 = square.angle + alpha_2 / 2
            angle_2 = square.angle - alpha_1 / 2

            center_1 = self._compute_center_left(
                square, length_1, alpha_1, angle_1)
            center_2 = self._compute_center_right(
                square, length_2, alpha_2, angle_2)

            square_1 = Square(center_1, length_1, angle_1)
            square_2 = Square(center_2, length_2, angle_2)

            children.append(self.pythagoras_tree(tree, left_child, square_1))
            children.append(self.pythagoras_tree(tree, right_child, square_2))

        return TreeNode(square, tree.parent(node), children)

    def _compute_center_left(self, initial_square, length, alpha, angle):
        """Compute the central point of the left child square.

        Parameters
        ----------
        initial_square : Square
            The sqare where we are continuing from.
        length : float
            The length of the side of the new square.
        alpha : float
            The angle at which the new square is offset from its parent.
        angle : float
            The computed angle at which the entire square is offset for
            drawing.

        Returns
        -------
        Point
            The central point of the left square.

        """
        base_center, base_length, base_angle = initial_square
        # get the point on the square side and prepare it for rotation
        t0 = self._get_point_on_square_edge(
            base_center, base_length, base_angle)
        # get the edge point where we will rotate t0 around
        square_diagonal_length = math.sqrt(2 * base_length ** 2)
        edge = self._get_point_on_square_edge(
            base_center, square_diagonal_length, base_angle + math.pi / 4)

        # translate edge to origin, temp point is is copy of edge
        temp = Point(edge.x - t0.x, edge.y - t0.y)
        # rotate point around edge point and translate back
        t1 = Point(
            temp.x * math.cos(-alpha) - temp.y * math.sin(-alpha),
            temp.x * math.sin(-alpha) + temp.y * math.cos(-alpha)
        )
        # translate rotated point back to its original spot
        t1 = Point(t1.x + t0.x, t1.y + t0.y)

        # calculate the middle point between the rotated point and edge
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        return self._get_point_on_square_edge(t2, length, angle)

    def _compute_center_right(self, initial_square, length, alpha, angle):
        """Compute the central point of the right child square.

        Parameters
        ----------
        initial_square : Square
            The sqare where we are continuing from.
        length : float
            The length of the side of the new square.
        alpha : float
            The angle at which the new square is offset from its parent.
        angle : float
            The computed angle at which the entire square is offset for
            drawing.

        Returns
        -------
        Point
            The central point of the right square.

        """
        base_center, base_length, base_angle = initial_square
        # get the point on the square side and prepare it for rotation
        t0 = self._get_point_on_square_edge(
            base_center, base_length, base_angle)
        # get the edge point where we will rotate t0 around
        square_diagonal_length = math.sqrt(2 * base_length ** 2)
        edge = self._get_point_on_square_edge(
            base_center, square_diagonal_length, base_angle - math.pi / 4)

        # translate edge to origin, temp point is is copy of edge
        temp = Point(edge.x - t0.x, edge.y - t0.y)
        # rotate point around edge point and translate back
        t1 = Point(
            temp.x * math.cos(alpha) - temp.y * math.sin(alpha),
            temp.x * math.sin(alpha) + temp.y * math.cos(alpha)
        )
        # translate rotated point back to its original spot
        t1 = Point(t1.x + t0.x, t1.y + t0.y)

        # calculate the middle point between the rotated point and edge
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        return self._get_point_on_square_edge(t2, length, angle)

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
        # by default, we draw on top, so we need to offset the angle by 90degs
        rotate_90 = math.pi / 2
        return Point(
            center.x + length / 2 * math.cos(angle + rotate_90),
            center.y + length / 2 * math.sin(angle + rotate_90)
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

    def left_child(self, node):
        return self._tree.children_left[node]

    def right_child(self, node):
        return self._tree.children_right[node]


if __name__ == "__main__":
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

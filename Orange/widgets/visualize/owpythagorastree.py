import Orange
import math
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.classification.tree import TreeClassifier
from Orange.data.table import Table

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt


class OWPythagorasTree(OWWidget):
    name = "Generalized Pythagoras Tree"
    description = "Generalized Pythagoras Tree for visualizing trees."
    priority = 100

    inputs = [("Classification Tree", TreeClassifier, "set_ctree")]
    outputs = [("Selected Data", Table)]

    # Settings
    zoom = Setting(5)

    def __init__(self):
        super().__init__()

        # Instance variables
        self.cls = None
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
        self.cls = ctree
        self.tree = SklTreeAdapter(ctree.skl_model.tree_)

        self._update()

    def clear(self):
        self.cls = None
        self.tree = None
        self._clear_scene()

    def _clear_scene(self):
        self.scene.clear()

    def _update(self):
        if self.tree is not None:
            square = SquareGraphicsItem(QtCore.QPointF(0, 0), 200, 0)
            self.pythagoras_tree(0, square)
        else:
            pass

    def pythagoras_tree(self, node, square):
        tree = self.tree

        self.scene.addItem(square)
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

            center_1 = self.compute_center_left(
                square.center, square.length, square.angle, length_1, alpha_1,
                angle_1)
            center_2 = self.compute_center_right(
                square.center, square.length, square.angle, length_2, alpha_2,
                angle_2)

            square_1 = SquareGraphicsItem(center_1, length_1, angle_1)
            square_2 = SquareGraphicsItem(center_2, length_2, angle_2)

            self.pythagoras_tree(left_child, square_1)
            self.pythagoras_tree(right_child, square_2)

    def compute_center_left(self, base_center, base_length, base_angle,
                            length, alpha, angle):
        t0 = self._get_point_on_square_edge(
            base_center, base_length, base_angle)
        # get the edge point where we will rotate t0 around
        square_diagonal = math.sqrt(2 * base_length ** 2)
        edge = self._get_point_on_square_edge(
            base_center, square_diagonal, base_angle + math.pi / 4)

        # translate edge to origin
        edge -= t0
        # rotate point around edge point and translate back
        t1 = QtCore.QPointF(
            edge.x() * math.cos(-alpha) - edge.y() * math.sin(-alpha),
            edge.x() * math.sin(-alpha) + edge.y() * math.cos(-alpha)
        ) + t0

        # calculate the middle point between the rotated point and edge
        edge += t0
        t2 = (t1 + edge) / 2
        return self._get_point_on_square_edge(t2, length, angle)

    def compute_center_right(self, base_center, base_length, base_angle,
                             length, alpha, angle):
        # get the point on the square side and prepare it for rotation
        t0 = self._get_point_on_square_edge(
            base_center, base_length, base_angle)
        # get the edge point where we will rotate t0 around
        square_diagonal = math.sqrt(2 * base_length ** 2)
        edge = self._get_point_on_square_edge(
            base_center, square_diagonal, base_angle - math.pi / 4)

        # translate edge to origin
        edge -= t0
        # rotate point around edge point and translate back
        t1 = QtCore.QPointF(
            edge.x() * math.cos(alpha) - edge.y() * math.sin(alpha),
            edge.x() * math.sin(alpha) + edge.y() * math.cos(alpha)
        ) + t0

        # calculate the middle point between the rotated point and edge
        edge += t0
        t2 = (t1 + edge) / 2
        return self._get_point_on_square_edge(t2, length, angle)

    @staticmethod
    def _get_point_on_square_edge(center, length, angle):
        """
        Calculate the central point on the drawing edge of the given square
        """
        rotate_90 = math.pi / 2
        # get the point on the square edge
        t0 = QtCore.QPointF(
            center.x() + length / 2 * math.cos(angle + rotate_90),
            center.y() + length / 2 * math.sin(angle + rotate_90)
        )
        return t0

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()

    def _draw_circle(self, point):
        """
        Draw a small circle on scene for given QPoint. Meant for debugging
        """
        self.scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10)


class SquareGraphicsItem(QtGui.QGraphicsRectItem):
    """
    Square Graphics Item

    Square component to draw as components for the Pythagoras tree. It is
    initialized with a given center QPointF, the main side length, and the
    initial angle the sqaure is to be rendered with.

    Examples:
        SqaureGraphicsItem(self.view, QPointF(0, 0), 100, 15)
        Initializes a sqaure with center at (0, 0) with side of length 100
        with an initial angle of 15.
    """

    def __init__(self, center, length, angle, parent=None):
        self.center = center
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


class SklTreeAdapter(object):
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

"""Tests for the Pythagorean tree widget and associated classes."""
import math
import unittest

from Orange.classification import TreeLearner as TreeClassificationLearner
from Orange.data import Table
from Orange.regression import TreeLearner as TreeRegressionLearner
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owpythagorastree import OWPythagorasTree
from Orange.widgets.visualize.pythagorastreeviewer import (
    PythagorasTree,
    Point,
    Square,
    SquareGraphicsItem,
)


# pylint: disable=protected-access
class TestPythagorasTree(unittest.TestCase):
    """Pythagorean tree testing, make sure calculating square positions works
    properly.

    Most of the data is non trivial since the rotations and translations don't
    generally produce trivial results.
    """

    def setUp(self):
        self.builder = PythagorasTree()

    def test_get_point_on_square_edge_with_no_angle(self):
        """Get central point on square edge that is not angled."""
        point = self.builder._get_point_on_square_edge(
            center=Point(0, 0), length=2, angle=0
        )
        expected_point = Point(1, 0)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_get_point_on_square_edge_with_non_zero_angle(self):
        """Get central point on square edge that has angle"""
        point = self.builder._get_point_on_square_edge(
            center=Point(2.7, 2.77), length=1.65, angle=math.radians(20.97)
        )
        expected_point = Point(3.48, 3.07)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_compute_center_with_simple_square_angle(self):
        """Compute the center of the square in the next step given a right
        angle."""
        initial_square = Square(Point(0, 0), length=2, angle=math.pi / 2)
        point = self.builder._compute_center(
            initial_square, length=1.13, alpha=math.radians(68.57))
        expected_point = Point(1.15, 1.78)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_compute_center_with_complex_square_angle(self):
        """Compute the center of the square in the next step given a more
        complex angle."""
        initial_square = Square(
            Point(1.5, 1.5), length=2.24, angle=math.radians(63.43)
        )
        point = self.builder._compute_center(
            initial_square, length=1.65, alpha=math.radians(95.06))
        expected_point = Point(3.48, 3.07)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)

    def test_compute_center_with_complex_square_angle_with_base_angle(self):
        """Compute the center of the square in the next step when there is a
        base angle - when the square does not touch the base square on the left
        edge."""
        initial_square = Square(
            Point(1.5, 1.5), length=2.24, angle=math.radians(63.43)
        )
        point = self.builder._compute_center(
            initial_square, length=1.51, alpha=math.radians(180 - 95.06),
            base_angle=math.radians(95.06))
        expected_point = Point(1.43, 3.98)
        self.assertAlmostEqual(point.x, expected_point.x, places=1)
        self.assertAlmostEqual(point.y, expected_point.y, places=1)


class TestOWPythagorasTree(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        # Set up for output tests
        tree = TreeClassificationLearner()
        cls.model = tree(cls.data)
        cls.model.instances = cls.data

        cls.signal_name = "Tree"
        cls.signal_data = cls.model

        # Set up for widget tests
        titanic_data = Table('titanic')
        cls.titanic = TreeClassificationLearner(max_depth=1)(titanic_data)
        cls.titanic.instances = titanic_data

        housing_data = Table('housing')
        cls.housing = TreeRegressionLearner(max_depth=1)(housing_data)
        cls.housing.instances = housing_data

    def setUp(self):
        self.widget = self.create_widget(OWPythagorasTree)

    def _select_data(self):
        item = [i for i in self.widget.scene.items() if
                isinstance(i, SquareGraphicsItem)][3]
        item.setSelected(True)
        return item.tree_node.label.subset

    def get_squares(self):
        """Get all the `SquareGraphicsItems` in the widget scene."""
        return (i for i in self.widget.scene.items()
                if isinstance(i, SquareGraphicsItem))

    def get_tree_nodes(self):
        """Get all the `TreeNode` instances in the widget scene."""
        return (sq.tree_node for sq in self.get_squares())

    def set_combo_option(self, combo_box, text):
        """Set a given combo box value to some text (given that it exists)."""
        index = combo_box.findText(text)
        # This only changes the selection, need to emit signal to call callback
        combo_box.setCurrentIndex(index)
        # Apparently `currentIndexChanged` just isn't good enough...
        combo_box.activated.emit(index)

    def test_sending_classification_tree_is_drawn(self):
        self.send_signal('Tree', self.titanic)
        self.assertTrue(len(list(self.get_squares())) > 0)

    def test_sending_classification_tree_is_drawn(self):
        self.send_signal('Tree', self.housing)
        self.assertTrue(len(list(self.get_squares())) > 0)

    def test_changing_color_changes_node_coloring(self):
        """Changing the `Target class` combo box should update colors."""
        self.send_signal('Tree', self.titanic)

        # Get colors for default coloring
        def_colors_sq = self.get_squares()
        default_colors = [sq.brush().color() for sq in def_colors_sq]

        # Get colors for `Yes` class coloring
        self.set_combo_option(self.widget.target_class_combo, 'Yes')
        yes_colors_sq = self.get_squares()
        yes_colors = [sq.brush().color() for sq in yes_colors_sq]

        # Get colors for `No` class coloring
        self.set_combo_option(self.widget.target_class_combo, 'No')
        no_colors_sq = self.get_squares()
        no_colors = [sq.brush().color() for sq in no_colors_sq]

        # Make sure all the colors aren't the same in any event
        eqs = [d != y and d != n and y != n for d, y, n in
               zip(default_colors, yes_colors, no_colors)]

        self.assertTrue(all(eqs))

    def test_changing_size_adjustment_changes_sizes(self):
        self.send_signal('Tree', self.titanic)

        basic_sizing_sq = [n.square for n in self.get_tree_nodes()]

        self.set_combo_option(self.widget.size_calc_combo, 'Square root')
        sqroot_sizing_sq = [n.square for n in self.get_tree_nodes()]

        self.set_combo_option(self.widget.size_calc_combo, 'Logarithmic')
        log_sizing_sq = [n.square for n in self.get_tree_nodes()]

        eqs = [b != s and b != l and s != l for b, s, l in
               zip(basic_sizing_sq, sqroot_sizing_sq, log_sizing_sq)]

        self.assertTrue(all(eqs))

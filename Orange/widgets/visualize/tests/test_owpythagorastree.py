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
from Orange.widgets.visualize.utils.owlegend import (
    OWDiscreteLegend,
    OWContinuousLegend
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
        titanic_data = Table('titanic')[::50]
        cls.titanic = TreeClassificationLearner(max_depth=1)(titanic_data)
        cls.titanic.instances = titanic_data

        housing_data = Table('housing')[:10]
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
        return [i for i in self.widget.scene.items()
                if isinstance(i, SquareGraphicsItem)]

    def get_visible_squares(self):
        return [x for x in self.get_squares() if x.isVisible()]

    def get_tree_nodes(self):
        """Get all the `TreeNode` instances in the widget scene."""
        return (sq.tree_node for sq in self.get_squares())

    @staticmethod
    def set_combo_option(combo_box, text):
        """Set a given combo box value to some text (given that it exists)."""
        index = combo_box.findText(text)
        # This only changes the selection, need to emit signal to call callback
        combo_box.setCurrentIndex(index)
        # Apparently `currentIndexChanged` just isn't good enough...
        combo_box.activated.emit(index)

    def test_sending_classification_tree_is_drawn(self):
        self.send_signal('Tree', self.housing)
        self.assertTrue(len(self.get_squares()) > 0)

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

        # Only compare to the -1 list element since the base square is always
        # the same
        self.assertTrue(all(eqs[:-1]))

    def test_log_scale_slider(self):
        # Disabled when no tree
        self.assertFalse(self.widget.log_scale_box.isEnabled(),
                         'Should be disabled with no tree')

        self.send_signal('Tree', self.titanic)
        # No size adjustment
        self.set_combo_option(self.widget.size_calc_combo, 'Normal')
        self.assertFalse(self.widget.log_scale_box.isEnabled(),
                         'Should be disabled when no size adjustment')
        # Square root adjustment
        self.set_combo_option(self.widget.size_calc_combo, 'Square root')
        self.assertFalse(self.widget.log_scale_box.isEnabled(),
                         'Should be disabled when square root size adjustment')
        # Log adjustment
        self.set_combo_option(self.widget.size_calc_combo, 'Logarithmic')
        self.assertTrue(self.widget.log_scale_box.isEnabled(),
                        'Should be enabled when square root size adjustment')

        # Get squares for one value of log factor
        self.widget.log_scale_box.setValue(1)
        inital_sizing_sq = [n.square for n in self.get_tree_nodes()]
        # Get squares for a different value of log factor
        self.widget.log_scale_box.setValue(2)
        updated_sizing_sq = [n.square for n in self.get_tree_nodes()]

        eqs = [x != y for x, y in zip(inital_sizing_sq, updated_sizing_sq)]
        # Only compare to the -1 list element since the base square is always
        # the same
        self.assertTrue(
            all(eqs[:-1]),
            'Squares are drawn in same positions after changing log factor')

    def test_classification_tree_creates_correct_legend(self):
        self.send_signal('Tree', self.titanic)
        self.assertIsInstance(self.widget.legend, OWDiscreteLegend)

    def test_regression_tree_creates_correct_legend(self):
        self.send_signal('Tree', self.housing)
        # Put the widget into a coloring scheme that builds the legend
        # We'll put it into the the class mean coloring mode
        self.set_combo_option(self.widget.target_class_combo, 'Class mean')
        self.assertIsInstance(self.widget.legend, OWContinuousLegend)

    def test_checking_legend_checkbox_shows_and_hides_legend(self):
        self.send_signal('Tree', self.titanic)
        # Hide the legend
        self.widget.cb_show_legend.setChecked(False)
        self.assertFalse(self.widget.legend.isVisible(),
                         'Hiding legend failed')
        # Show the legend
        self.widget.cb_show_legend.setChecked(True)
        self.assertTrue(self.widget.legend.isVisible(),
                        'Showing legend failed')

    def test_checking_tooltip_shows_and_hides_tooltips(self):
        self.send_signal('Tree', self.titanic)
        square = self.get_squares()[0]
        # Hide tooltips
        self.widget.cb_show_tooltips.setChecked(False)
        self.assertEqual(square.toolTip(), '', 'Hiding tooltips failed')
        # Show tooltips
        self.widget.cb_show_tooltips.setChecked(True)
        self.assertNotEqual(square.toolTip(), '', 'Showing tooltips failed')

    def test_changing_max_depth_slider(self):
        self.send_signal('Tree', self.titanic)

        max_depth = self.widget.tree_adapter.max_depth
        num_squares_full = len(self.get_visible_squares())
        self.assertEqual(self.widget.depth_limit, max_depth,
                         'Full tree should be drawn initially')

        self.widget.depth_slider.setValue(max_depth - 1)
        num_squares_less = len(self.get_visible_squares())
        self.assertLess(num_squares_less, num_squares_full,
                        'Lowering tree depth limit did not hide squares')

        self.widget.depth_slider.setValue(max_depth + 1)
        self.assertGreater(len(self.get_visible_squares()), num_squares_less,
                           'Increasing tree depth limit did not show squares')

    def test_label_on_tree_connect_and_disconnect(self):
        regex = r'Nodes:(.+)\s*Depth:(.+)'
        # Should contain no info by default
        self.assertNotRegex(
            self.widget.info.text(), regex,
            'Initial info should not contain node or depth info')
        # Test info label for tree
        self.send_signal('Tree', self.titanic)
        self.assertRegex(
            self.widget.info.text(), regex,
            'Valid tree does not update info')
        # Remove tree from input
        self.send_signal('Tree', None)
        self.assertNotRegex(
            self.widget.info.text(), regex,
            'Initial info should not contain node or depth info')

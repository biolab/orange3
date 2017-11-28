"""Tests for the Pythagorean tree widget and associated classes."""
import math
import unittest

from os import path

from Orange.data import Table
from Orange.modelling import TreeLearner
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owpythagorastree import OWPythagorasTree
from Orange.widgets.visualize.pythagorastreeviewer import (
    PythagorasTree,
    Point,
    Square,
    SquareGraphicsItem,
)
from Orange.widgets.visualize.utils.owlegend import (
    OWDiscreteLegend,
    OWContinuousLegend,
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
        tree = TreeLearner()
        cls.model = tree(cls.data)
        cls.model.instances = cls.data

        cls.signal_name = "Tree"
        cls.signal_data = cls.model

        # Set up for widget tests
        titanic_data = Table('titanic')[::50]
        cls.titanic = TreeLearner(max_depth=1)(titanic_data)
        cls.titanic.instances = titanic_data

        housing_data = Table('housing')[:10]
        cls.housing = TreeLearner(max_depth=1)(housing_data)
        cls.housing.instances = housing_data

    def setUp(self):
        self.widget = self.create_widget(OWPythagorasTree)  # type: OWPythagorasTree

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

    def test_tree_is_drawn(self):
        self.send_signal(self.widget.Inputs.tree, self.housing)
        self.assertTrue(len(self.get_squares()) > 0)

    def _check_all_same(self, items):
        iter_items = iter(items)
        try:
            first = next(iter_items)
        except StopIteration:
            return True
        return all(first == curr for curr in iter_items)

    def test_changing_target_class_changes_node_coloring(self):
        """Changing the `Target class` combo box should update colors."""
        def _test(data_type):
            squares = []

            def _callback():
                squares.append([sq.brush().color() for sq in self.get_visible_squares()])

            simulate.combobox_run_through_all(
                self.widget.target_class_combo, callback=_callback)

            # Check that individual squares all have different colors
            squares_same = [self._check_all_same(x) for x in zip(*squares)]
            # Check that at least some of the squares have different colors
            self.assertTrue(any(x is False for x in squares_same),
                            'Colors did not change for %s data' % data_type)

        w = self.widget
        self.send_signal(w.Inputs.tree, self.titanic)
        _test('classification')
        self.send_signal(w.Inputs.tree, self.housing)
        _test('regression')

    def test_changing_size_adjustment_changes_sizes(self):
        self.send_signal(self.widget.Inputs.tree, self.titanic)
        squares = []

        def _callback():
            squares.append([sq.rect() for sq in self.get_visible_squares()])

        simulate.combobox_run_through_all(
            self.widget.size_calc_combo, callback=_callback)

        # Check that individual squares are in different position
        squares_same = [self._check_all_same(x) for x in zip(*squares)]
        # Check that at least some of the squares have different positions
        self.assertTrue(any(x is False for x in squares_same))

    def test_log_scale_slider(self):
        # Disabled when no tree
        self.assertFalse(self.widget.log_scale_box.isEnabled(),
                         'Should be disabled with no tree')

        self.send_signal(self.widget.Inputs.tree, self.titanic)
        # No size adjustment
        simulate.combobox_activate_item(self.widget.size_calc_combo, 'Normal')
        self.assertFalse(self.widget.log_scale_box.isEnabled(),
                         'Should be disabled when no size adjustment')
        # Square root adjustment
        simulate.combobox_activate_item(self.widget.size_calc_combo, 'Square root')
        self.assertFalse(self.widget.log_scale_box.isEnabled(),
                         'Should be disabled when square root size adjustment')
        # Log adjustment
        simulate.combobox_activate_item(self.widget.size_calc_combo, 'Logarithmic')
        self.assertTrue(self.widget.log_scale_box.isEnabled(),
                        'Should be enabled when square root size adjustment')

        # Get squares for one value of log factor
        self.widget.log_scale_box.setValue(1)
        inital_sizing_sq = [n.square for n in self.get_tree_nodes()]
        # Get squares for a different value of log factor
        self.widget.log_scale_box.setValue(2)
        updated_sizing_sq = [n.square for n in self.get_tree_nodes()]

        # Only compare to the -1 list element since the base square is always
        # the same
        self.assertTrue(
            any([x != y for x, y in zip(inital_sizing_sq, updated_sizing_sq)]),
            'Squares are drawn in same positions after changing log factor')

    def test_legend(self):
        """Test legend behaviour."""
        w = self.widget
        w.cb_show_legend.setChecked(True)

        self.send_signal(w.Inputs.tree, self.titanic)
        self.assertIsInstance(self.widget.legend, OWDiscreteLegend)
        self.assertTrue(self.widget.legend.isVisible())

        # The legend should be invisible when regression coloring is none
        self.send_signal(w.Inputs.tree, self.housing)
        self.assertIsInstance(w.legend, OWContinuousLegend)
        self.assertFalse(w.legend.isVisible())

        # The legend should appear when there is a coloring (2 is mean coloring)
        index = 2
        simulate.combobox_activate_index(w.target_class_combo, index)
        self.assertIsInstance(w.legend, OWContinuousLegend)
        self.assertTrue(w.legend.isVisible())

        # Check that switching back to a discrete target class works
        self.send_signal(w.Inputs.tree, self.titanic)
        self.assertIsInstance(w.legend, OWDiscreteLegend)
        self.assertTrue(w.legend.isVisible())

    def test_checking_legend_checkbox_shows_and_hides_legend(self):
        w = self.widget
        self.send_signal(w.Inputs.tree, self.titanic)
        # Hide the legend
        w.cb_show_legend.setChecked(False)
        self.assertFalse(w.legend.isVisible(), 'Hiding legend failed')
        # Show the legend
        w.cb_show_legend.setChecked(True)
        self.assertTrue(w.legend.isVisible(), 'Showing legend failed')

    def test_tooltip_changes_for_classification_target_class(self):
        """Tooltips should change when a target class is specified with a
        discrete target class."""
        w = self.widget
        w.cb_show_tooltips.setChecked(True)
        self.send_signal(w.Inputs.tree, self.titanic)
        tooltips = []

        def _callback():
            tooltips.append(self.get_visible_squares()[2].toolTip())

        simulate.combobox_run_through_all(w.target_class_combo, callback=_callback)

        self.assertFalse(self._check_all_same(tooltips))

    def test_checking_tooltip_shows_and_hides_tooltips(self):
        w = self.widget
        self.send_signal(w.Inputs.tree, self.titanic)
        square = self.get_squares()[0]
        # Hide tooltips
        w.cb_show_tooltips.setChecked(False)
        self.assertEqual(square.toolTip(), '', 'Hiding tooltips failed')
        # Show tooltips
        w.cb_show_tooltips.setChecked(True)
        self.assertNotEqual(square.toolTip(), '', 'Showing tooltips failed')

    def test_changing_max_depth_slider(self):
        w = self.widget
        self.send_signal(w.Inputs.tree, self.titanic)

        max_depth = w.tree_adapter.max_depth
        num_squares_full = len(self.get_visible_squares())
        self.assertEqual(w.depth_limit, max_depth, 'Full tree should be drawn initially')

        self.widget.depth_slider.setValue(max_depth - 1)
        num_squares_less = len(self.get_visible_squares())
        self.assertLess(num_squares_less, num_squares_full,
                        'Lowering tree depth limit did not hide squares')

        w.depth_slider.setValue(max_depth + 1)
        self.assertGreater(len(self.get_visible_squares()), num_squares_less,
                           'Increasing tree depth limit did not show squares')

    def test_label_on_tree_connect_and_disconnect(self):
        w = self.widget
        regex = r'Nodes:(.+)\s*Depth:(.+)'
        # Should contain no info by default
        self.assertNotRegex(
            self.widget.info.text(), regex,
            'Initial info should not contain node or depth info')
        # Test info label for tree
        self.send_signal(w.Inputs.tree, self.titanic)
        self.assertRegex(w.info.text(), regex, 'Valid tree does not update info')
        # Remove tree from input
        self.send_signal(w.Inputs.tree, None)
        self.assertNotRegex(
            w.info.text(), regex, 'Initial info should not contain node or depth info')

    def test_tree_determinism(self):
        """Check that the tree is drawn identically upon receiving the same
        dataset with no parameter changes."""
        n_tries = 10
        # Make sure the tree are deterministic for iris
        scene_nodes = []
        for _ in range(n_tries):
            self.send_signal(self.widget.Inputs.tree, self.signal_data)
            scene_nodes.append([n.pos() for n in self.get_visible_squares()])
        for node_row in zip(*scene_nodes):
            self.assertTrue(
                self._check_all_same(node_row),
                "The tree was not drawn identically in the %d times it was "
                "sent to widget after receiving the iris dataset." % n_tries
            )

        # Make sure trees are deterministic with data where some variables have
        # the same entropy
        data_same_entropy = Table(path.join(
            path.dirname(path.dirname(path.dirname(__file__))), "tests",
            "datasets", "same_entropy.tab"))
        data_same_entropy = TreeLearner()(data_same_entropy)
        scene_nodes = []
        for _ in range(n_tries):
            self.send_signal(self.widget.Inputs.tree, data_same_entropy)
            scene_nodes.append([n.pos() for n in self.get_visible_squares()])
        for node_row in zip(*scene_nodes):
            self.assertTrue(
                self._check_all_same(node_row),
                "The tree was not drawn identically in the %d times it was "
                "sent to widget after receiving a dataset with variables with "
                "same entropy." % n_tries
            )

    def test_keep_colors_on_sizing_change(self):
        """The color should be the same after a full recompute of the tree."""
        w = self.widget
        self.send_signal(w.Inputs.tree, self.titanic)
        colors = []

        def _callback():
            colors.append([sq.brush().color() for sq in self.get_visible_squares()])

        simulate.combobox_run_through_all(w.size_calc_combo, callback=_callback)

        # Check that individual squares all have the same color
        colors_same = [self._check_all_same(x) for x in zip(*colors)]
        self.assertTrue(all(colors_same))

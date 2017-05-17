import random
from unittest.mock import Mock

from Orange.classification.random_forest import RandomForestLearner
from Orange.data import Table
from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owpythagoreanforest import OWPythagoreanForest, \
    GridItem
from Orange.widgets.visualize.pythagorastreeviewer import PythagorasTreeViewer


class TestOWPythagoreanForest(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Set up for widget tests
        titanic_data = Table('titanic')[::50]
        cls.titanic = RandomForestLearner(n_estimators=3)(titanic_data)
        cls.titanic.instances = titanic_data

        housing_data = Table('housing')[:10]
        cls.housing = RandomForestRegressionLearner(
            n_estimators=3)(housing_data)
        cls.housing.instances = housing_data

    def setUp(self):
        self.widget = self.create_widget(OWPythagoreanForest)  # type: OWPythagoreanForest


    def get_tree_widgets(self):
        return [x for x in self.widget.scene.items()
                if isinstance(x, PythagorasTreeViewer)]

    def get_grid_items(self):
        return [x for x in self.widget.scene.items()
                if isinstance(x, GridItem)]

    def test_sending_rf_draws_trees(self):
        # No trees by default
        self.assertEqual(len(self.get_tree_widgets()), 0,
                         'No trees should be drawn when no forest on input')

        # Draw trees for classification rf
        self.send_signal('Random forest', self.titanic)
        self.assertEqual(len(self.get_tree_widgets()), 3,
                         'Incorrect number of trees when forest on input')

        # Clear trees when None
        self.send_signal('Random forest', None)
        self.assertEqual(len(self.get_tree_widgets()), 0,
                         'Trees are cleared when forest is disconnected')

        # Draw trees for regression rf
        self.send_signal('Random forest', self.housing)
        self.assertEqual(len(self.get_tree_widgets()), 3,
                         'Incorrect number of trees when forest on input')

    def test_info_label(self):
        regex = r'Trees:(.+)'
        # If no forest on input, display a message saying that
        self.assertNotRegex(self.widget.ui_info.text(), regex,
                            'Initial info should not contain info on trees')

        self.send_signal('Random forest', self.titanic)
        self.assertRegex(self.widget.ui_info.text(), regex,
                         'Valid RF does not update info')

        self.send_signal('Random forest', None)
        self.assertNotRegex(self.widget.ui_info.text(), regex,
                            'Removing RF does not clear info box')

    def test_depth_slider(self):
        self.send_signal('Random forest', self.titanic)

        trees = self.get_tree_widgets()
        for tree in trees:
            tree.set_depth_limit = Mock()

        self.widget.ui_depth_slider.setValue(0)
        for tree in trees:
            tree.set_depth_limit.assert_called_once_with(0)

    def _pick_random_tree(self):
        """Pick a random tree from all the trees on the grid.

        Returns
        -------
        PythagorasTreeViewer

        """
        return random.choice(self.get_tree_widgets())

    def _get_visible_squares(self, tree):
        return [x for _, x in tree._square_objects.items() if x.isVisible()]

    def _check_all_same(self, items):
        iter_items = iter(items)
        try:
            first = next(iter_items)
        except StopIteration:
            return True
        return all(first == curr for curr in iter_items)

    def test_changing_target_class_changes_coloring(self):
        """Changing the `Target class` combo box should update colors."""
        def _test(data_type):
            colors, tree = [], self._pick_random_tree()

            def _callback():
                colors.append([sq.brush().color() for sq in self._get_visible_squares(tree)])

            simulate.combobox_run_through_all(
                self.widget.ui_target_class_combo, callback=_callback)

            # Check that individual squares all have different colors
            squares_same = [self._check_all_same(x) for x in zip(*colors)]
            # Check that at least some of the squares have different colors
            self.assertTrue(any(x is False for x in squares_same),
                            'Colors did not change for %s data' % data_type)

        self.send_signal('Random forest', self.titanic)
        _test('classification')
        self.send_signal('Random forest', self.housing)
        _test('regression')

    def test_changing_size_adjustment_changes_sizes(self):
        self.send_signal('Random forest', self.titanic)
        squares = []
        # We have to get the same tree with an index on the grid items since
        # the tree objects are deleted and recreated with every invalidation
        tree_index = self.widget.grid_items.index(random.choice(self.get_grid_items()))

        def _callback():
            squares.append([sq.rect() for sq in self._get_visible_squares(
                self.get_tree_widgets()[tree_index])])

        simulate.combobox_run_through_all(
            self.widget.ui_size_calc_combo, callback=_callback)

        # Check that individual squares are in different position
        squares_same = [self._check_all_same(x) for x in zip(*squares)]
        # Check that at least some of the squares have different positions
        self.assertTrue(any(x is False for x in squares_same))

    def test_zoom(self):
        self.send_signal('Random forest', self.titanic)

        grid_item, zoom = self.get_grid_items()[0], self.widget.zoom

        def _destructure_rectf(r):
            return r.width(), r.height()

        iw, ih = _destructure_rectf(grid_item.boundingRect())

        # Increase the size of grid item
        self.widget.ui_zoom_slider.setValue(zoom + 1)
        lw, lh = _destructure_rectf(grid_item.boundingRect())
        self.assertTrue(iw < lw and ih < lh)

        # Decrease the size of grid item
        self.widget.ui_zoom_slider.setValue(zoom - 1)
        lw, lh = _destructure_rectf(grid_item.boundingRect())
        self.assertTrue(iw > lw and ih > lh)

    def test_keep_colors_on_sizing_change(self):
        """The color should be the same after a full recompute of the tree."""
        self.send_signal('Random forest', self.titanic)
        colors = []
        tree_index = self.widget.grid_items.index(random.choice(self.get_grid_items()))

        def _callback():
            colors.append([sq.brush().color() for sq in self._get_visible_squares(
                self.get_tree_widgets()[tree_index])])

        simulate.combobox_run_through_all(
            self.widget.ui_size_calc_combo, callback=_callback)

        # Check that individual squares all have the same color
        colors_same = [self._check_all_same(x) for x in zip(*colors)]
        self.assertTrue(all(colors_same))

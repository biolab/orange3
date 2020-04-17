# pylint: disable=missing-docstring,protected-access

from unittest.mock import Mock

from AnyQt.QtCore import Qt, QItemSelection, QItemSelectionModel

from Orange.classification.random_forest import RandomForestLearner
from Orange.data import Table
from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owpythagoreanforest import OWPythagoreanForest
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

    def test_migrate_version_1_settings(self):
        # Version 1 zoom ranged from 20-150, version 2 zoom ranges from 100-400
        # Test minimium value falls into the range
        widget_min_zoom = self.create_widget(
            OWPythagoreanForest,
            stored_settings={'zoom': 20, 'version': 2},
        )  # type: OWPythagoreanForest
        self.assertTrue(widget_min_zoom.zoom <= 400)
        self.assertTrue(widget_min_zoom.zoom >= 100)

        # Test maximum value falls into the range
        widget_max_zoom = self.create_widget(
            OWPythagoreanForest,
            stored_settings={'zoom': 150, 'version': 2},
        )  # type: OWPythagoreanForest
        self.assertTrue(widget_max_zoom.zoom <= 400)
        self.assertTrue(widget_max_zoom.zoom >= 100)

    def get_tree_widgets(self):
        model = self.widget.forest_model
        trees = []
        for idx in range(len(model)):
            scene = model.data(model.index(idx), Qt.DisplayRole)
            tree, = [item for item in scene.items()
                     if isinstance(item, PythagorasTreeViewer)]
            trees.append(tree)
        return trees

    def test_sending_rf_draws_trees(self):
        w = self.widget
        # No trees by default
        self.assertEqual(len(self.get_tree_widgets()), 0,
                         'No trees should be drawn when no forest on input')

        # Draw trees for classification rf
        self.send_signal(w.Inputs.random_forest, self.titanic)
        self.assertEqual(len(self.get_tree_widgets()), 3,
                         'Incorrect number of trees when forest on input')

        # Clear trees when None
        self.send_signal(w.Inputs.random_forest, None)
        self.assertEqual(len(self.get_tree_widgets()), 0,
                         'Trees are cleared when forest is disconnected')

        # Draw trees for regression rf
        self.send_signal(w.Inputs.random_forest, self.housing)
        self.assertEqual(len(self.get_tree_widgets()), 3,
                         'Incorrect number of trees when forest on input')

    def test_info_label(self):
        w = self.widget
        regex = r'Trees:(.+)'
        # If no forest on input, display a message saying that
        self.assertNotRegex(w.ui_info.text(), regex,
                            'Initial info should not contain info on trees')

        self.send_signal(w.Inputs.random_forest, self.titanic)
        self.assertRegex(self.widget.ui_info.text(), regex, 'Valid RF does not update info')

        self.send_signal(w.Inputs.random_forest, None)
        self.assertNotRegex(w.ui_info.text(), regex, 'Removing RF does not clear info box')

    def test_depth_slider(self):
        w = self.widget
        self.send_signal(w.Inputs.random_forest, self.titanic)

        trees = self.get_tree_widgets()
        for tree in trees:
            tree.set_depth_limit = Mock()

        w.ui_depth_slider.setValue(0)
        for tree in trees:
            tree.set_depth_limit.assert_called_once_with(0)

    def _get_first_tree(self):
        """Pick a random tree from all the trees on the grid.

        Returns
        -------
        PythagorasTreeViewer

        """
        widgets = self.get_tree_widgets()
        assert len(widgets), 'Empty list of tree widgets'
        return widgets[0]

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
        w = self.widget

        def _test(data_type):
            colors, tree = [], self._get_first_tree()

            def _callback():
                colors.append([sq.brush().color() for sq in self._get_visible_squares(tree)])

            simulate.combobox_run_through_all(w.ui_target_class_combo, callback=_callback)

            # Check that individual squares all have different colors
            squares_same = [self._check_all_same(x) for x in zip(*colors)]
            # Check that at least some of the squares have different colors
            self.assertTrue(any(x is False for x in squares_same),
                            'Colors did not change for %s data' % data_type)

        self.send_signal(w.Inputs.random_forest, self.titanic)
        _test('classification')
        self.send_signal(w.Inputs.random_forest, self.housing)
        _test('regression')

    def test_changing_size_adjustment_changes_sizes(self):
        w = self.widget
        self.send_signal(w.Inputs.random_forest, self.titanic)
        squares = []
        tree = self._get_first_tree()

        def _callback():
            squares.append([sq.rect() for sq in self._get_visible_squares(tree)])

        simulate.combobox_run_through_all(w.ui_size_calc_combo, callback=_callback)

        # Check that individual squares are in different position
        squares_same = [self._check_all_same(x) for x in zip(*squares)]
        # Check that at least some of the squares have different positions
        self.assertTrue(any(x is False for x in squares_same))

    def test_zoom(self):
        w = self.widget
        self.send_signal(w.Inputs.random_forest, self.titanic)

        min_zoom = w.ui_zoom_slider.minimum()
        max_zoom = w.ui_zoom_slider.maximum()

        # Increase the size of grid item
        w.ui_zoom_slider.setValue(max_zoom)
        item_size = w.forest_model.data(w.forest_model.index(0), Qt.SizeHintRole)
        max_w, max_h = item_size.width(), item_size.height()

        # Decrease the size of grid item
        w.ui_zoom_slider.setValue(min_zoom)
        item_size = w.forest_model.data(w.forest_model.index(0), Qt.SizeHintRole)
        min_w, min_h = item_size.width(), item_size.height()

        self.assertTrue(min_w < max_w and min_h < max_h)

    def test_keep_colors_on_sizing_change(self):
        """The color should be the same after a full recompute of the tree."""
        w = self.widget
        self.send_signal(w.Inputs.random_forest, self.titanic)
        colors = []
        tree = self._get_first_tree()

        def _callback():
            colors.append([sq.brush().color() for sq in self._get_visible_squares(tree)])

        simulate.combobox_run_through_all(w.ui_size_calc_combo, callback=_callback)

        # Check that individual squares all have the same color
        colors_same = [self._check_all_same(x) for x in zip(*colors)]
        self.assertTrue(all(colors_same))

    def select_tree(self, idx: int) -> None:
        list_view = self.widget.list_view
        index = list_view.model().index(idx)
        selection = QItemSelection(index, index)
        list_view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def test_storing_selection(self):
        # Select one of the trees
        idx = 1
        self.send_signal(self.widget.Inputs.random_forest, self.titanic)
        self.select_tree(idx)
        # Clear input
        self.send_signal(self.widget.Inputs.random_forest, None)
        # Restore previous data; context settings should be restored
        self.send_signal(self.widget.Inputs.random_forest, self.titanic)

        output = self.get_output(self.widget.Outputs.tree)
        self.assertIsNotNone(output)
        self.assertIs(output.skl_model, self.titanic.trees[idx].skl_model)

    def test_context(self):
        iris = Table("iris")
        iris_tree = RandomForestLearner()(iris)
        iris_tree.instances = iris
        self.send_signal(self.widget.Inputs.random_forest, self.titanic)
        self.widget.target_class_index = 1

        self.send_signal(self.widget.Inputs.random_forest, iris_tree)
        self.assertEqual(0, self.widget.target_class_index)

        self.widget.target_class_index = 2
        self.send_signal(self.widget.Inputs.random_forest, self.titanic)
        self.assertEqual(1, self.widget.target_class_index)

        self.send_signal(self.widget.Inputs.random_forest, iris_tree)
        self.assertEqual(2, self.widget.target_class_index)

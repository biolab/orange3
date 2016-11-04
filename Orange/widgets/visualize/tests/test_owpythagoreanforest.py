from unittest.mock import Mock

from Orange.classification.random_forest import RandomForestLearner
from Orange.data import Table
from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest
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
        self.widget = self.create_widget(OWPythagoreanForest)

    def get_tree_widgets(self):
        return [x for x in self.widget.scene.items()
                if isinstance(x, PythagorasTreeViewer)]

    def get_grid_items(self):
        return [x for x in self.widget.scene.items()
                if isinstance(x, GridItem)]

    @staticmethod
    def set_combo_option(combo_box, text):
        """Set a given combo box value to some text (given that it exists)."""
        index = combo_box.findText(text)
        # This only changes the selection, need to emit signal to call callback
        combo_box.setCurrentIndex(index)
        # Apparently `currentIndexChanged` just isn't good enough...
        combo_box.activated.emit(index)

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

    def test_target_class_for_classification_rf(self):
        self.send_signal('Random forest', self.titanic)

        trees = self.get_tree_widgets()
        for tree in trees:
            tree.target_class_has_changed = Mock()

        self.set_combo_option(self.widget.ui_target_class_combo, 'No')
        for tree in trees:
            tree.target_class_has_changed.assert_called_with()
            tree.target_class_has_changed.reset_mock()

        self.set_combo_option(self.widget.ui_target_class_combo, 'Yes')
        for tree in trees:
            tree.target_class_has_changed.assert_called_with()

    def test_target_class_for_regression_rf(self):
        self.send_signal('Random forest', self.housing)

        trees = self.get_tree_widgets()
        for tree in trees:
            tree.target_class_has_changed = Mock()

        self.set_combo_option(self.widget.ui_target_class_combo, 'Class mean')
        for tree in trees:
            tree.target_class_has_changed.assert_called_with()
            tree.target_class_has_changed.reset_mock()

        self.set_combo_option(self.widget.ui_target_class_combo,
                              'Standard deviation')
        for tree in trees:
            tree.target_class_has_changed.assert_called_with()

    def test_zoom(self):
        self.send_signal('Random forest', self.titanic)

        grid_item, zoom = self.get_grid_items()[0], self.widget.zoom

        def destructure_rectf(r):
            return r.width(), r.height()

        iw, ih = destructure_rectf(grid_item.boundingRect())

        # Increase the size of grid item
        self.widget.ui_zoom_slider.setValue(zoom + 1)
        lw, lh = destructure_rectf(grid_item.boundingRect())
        self.assertTrue(iw < lw and ih < lh)

        # Decrease the size of grid item
        self.widget.ui_zoom_slider.setValue(zoom - 1)
        lw, lh = destructure_rectf(grid_item.boundingRect())
        self.assertTrue(iw > lw and ih > lh)

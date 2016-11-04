from Orange.classification.random_forest import RandomForestLearner
from Orange.data import Table
from Orange.regression.random_forest import RandomForestRegressionLearner
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owpythagoreanforest import OWPythagoreanForest
from Orange.widgets.visualize.pythagorastreeviewer import PythagorasTreeViewer


class TestOWPythagoreanForest(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

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

    def get_tree_widgtes(self):
        return [x for x in self.widget.scene.items() \
                if isinstance(x, PythagorasTreeViewer)]

    def test_sending_rf_draws_trees(self):
        # No trees by default
        self.assertEqual(len(self.get_tree_widgtes()), 0,
                         'No trees should be drawn when no forest on input')

        # Draw trees for classification rf
        self.send_signal('Random forest', self.titanic)
        self.assertEqual(len(self.get_tree_widgtes()), 3,
                         'Incorrect number of trees when forest on input')

        # Clear trees when None
        self.send_signal('Random forest', None)
        self.assertEqual(len(self.get_tree_widgtes()), 0,
                         'Trees are cleared when forest is disconnected')

        # Draw trees for regression rf
        self.send_signal('Random forest', self.housing)
        self.assertEqual(len(self.get_tree_widgtes()), 3,
                         'Incorrect number of trees when forest on input')

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.classification import TreeLearner
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owtreeviewer import \
    OWTreeGraph


class TestOWTreeGraph(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        tree = TreeLearner()
        cls.model = tree(cls.data)
        cls.model.instances = cls.data

        cls.signal_name = "Tree"
        cls.signal_data = cls.model

    def setUp(self):
        self.widget = self.create_widget(OWTreeGraph)

    def _select_data(self):
        node = self.widget.scene.nodes()[0]
        node.setSelected(True)
        return self.model.get_indices([node.node_inst])

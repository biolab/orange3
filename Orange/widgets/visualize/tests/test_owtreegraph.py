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

    def test_target_class_changed(self):
        """Check if node content has changed after selecting target class"""
        self.send_signal(self.signal_name, self.signal_data)
        nodes = self.widget.scene.nodes()
        text = nodes[0].toPlainText()
        self.widget.color_combo.activated.emit(1)
        self.widget.color_combo.setCurrentIndex(1)
        self.assertNotEqual(nodes[0].toPlainText(), text)

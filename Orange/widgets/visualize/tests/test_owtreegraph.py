# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from os import path

from Orange.classification import TreeLearner
from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owtreeviewer import OWTreeGraph


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

        # Load a dataset that contains two variables with the same entropy
        data_same_entropy = Table(path.join(
            path.dirname(path.dirname(path.dirname(__file__))), "tests",
            "datasets", "same_entropy.tab"))
        cls.data_same_entropy = tree(data_same_entropy)
        cls.data_same_entropy.instances = data_same_entropy

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

    def test_tree_determinism(self):
        """Check that the tree is drawn identically upon receiving the same
        dataset with no parameter changes."""
        n_tries = 10

        def _check_all_same(data):
            """Check that all the elements within an iterable are identical."""
            iterator = iter(data)
            try:
                first = next(iterator)
            except StopIteration:
                return True
            return all(first == rest for rest in iterator)

        # Make sure the tree are deterministic for iris
        scene_nodes = []
        for _ in range(n_tries):
            self.send_signal(self.signal_name, self.signal_data)
            scene_nodes.append([n.pos() for n in self.widget.scene.nodes()])
        for node_row in zip(*scene_nodes):
            self.assertTrue(
                _check_all_same(node_row),
                "The tree was not drawn identically in the %d times it was "
                "sent to widget after receiving the iris dataset." % n_tries
            )

        # Make sure trees are deterministic with data where some variables have
        # the same entropy
        scene_nodes = []
        for _ in range(n_tries):
            self.send_signal(self.signal_name, self.data_same_entropy)
            scene_nodes.append([n.pos() for n in self.widget.scene.nodes()])
        for node_row in zip(*scene_nodes):
            self.assertTrue(
                _check_all_same(node_row),
                "The tree was not drawn identically in the %d times it was "
                "sent to widget after receiving a dataset with variables with "
                "same entropy." % n_tries
            )

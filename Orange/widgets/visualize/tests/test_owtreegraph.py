# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
from os import path
import unittest
from unittest.mock import Mock
import numpy as np

from Orange.classification import TreeLearner
from Orange.data import Table, ContinuousVariable, DiscreteVariable, Domain
from Orange.tree import DiscreteNode, MappedDiscreteNode, Node, NumericNode, \
    TreeModel
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

        cls.signal_name = OWTreeGraph.Inputs.tree
        cls.signal_data = cls.model

        # Load a dataset that contains two variables with the same entropy
        data_same_entropy = Table(path.join(
            path.dirname(path.dirname(path.dirname(__file__))), "tests",
            "datasets", "same_entropy.tab"))
        cls.data_same_entropy = tree(data_same_entropy)
        cls.data_same_entropy.instances = data_same_entropy

        vara = DiscreteVariable("aaa", values=("e", "f", "g"))
        root = DiscreteNode(vara, 0, np.array([42, 8]))
        root.subset = np.arange(50)

        varb = DiscreteVariable("bbb", values=tuple("ijkl"))
        child0 = MappedDiscreteNode(varb, 1, np.array([0, 1, 0, 0]), (38, 5))
        child0.subset = np.arange(16)
        child1 = Node(None, 0, (13, 3))
        child1.subset = np.arange(16, 30)
        varc = ContinuousVariable("ccc")
        child2 = NumericNode(varc, 2, 42, (78, 12))
        child2.subset = np.arange(30, 50)
        root.children = (child0, child1, child2)

        child00 = Node(None, 0, (15, 4))
        child00.subset = np.arange(10)
        child01 = Node(None, 0, (10, 5))
        child01.subset = np.arange(10, 16)
        child0.children = (child00, child01)

        child20 = Node(None, 0, (90, 4))
        child20.subset = np.arange(30, 35)
        child21 = Node(None, 0, (70, 9))
        child21.subset = np.arange(35, 50)
        child2.children = (child20, child21)

        domain = Domain([vara, varb, varc], ContinuousVariable("y"))
        t = [[i, j, k]
             for i in range(3)
             for j in range(4)
             for k in (40, 44)]
        x = np.array((t * 3)[:50])
        data = Table.from_numpy(
            domain, x, np.arange(len(x)))
        cls.tree = TreeModel(data, root)

    def setUp(self):
        self.widget = self.create_widget(OWTreeGraph)

    def _select_data(self):
        node = self.widget.scene.nodes()[0]
        node.setSelected(True)
        return self.model.get_indices([node.node_inst])

    def test_target_class_changed(self):
        """Check if node content has changed after selecting target class"""
        w = self.widget
        self.send_signal(w.Inputs.tree, self.signal_data)
        nodes = w.scene.nodes()
        text = nodes[0].toPlainText()
        w.color_combo.activated.emit(1)
        w.color_combo.setCurrentIndex(1)
        self.assertNotEqual(nodes[0].toPlainText(), text)

    def test_tree_determinism(self):
        """Check that the tree is drawn identically upon receiving the same
        dataset with no parameter changes."""
        w = self.widget
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
            self.send_signal(w.Inputs.tree, self.signal_data)
            scene_nodes.append([n.pos() for n in w.scene.nodes()])
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
            self.send_signal(w.Inputs.tree, self.data_same_entropy)
            scene_nodes.append([n.pos() for n in w.scene.nodes()])
        for node_row in zip(*scene_nodes):
            self.assertTrue(
                _check_all_same(node_row),
                "The tree was not drawn identically in the %d times it was "
                "sent to widget after receiving a dataset with variables with "
                "same entropy." % n_tries
            )

    def test_update_node_info(self):
        widget = self.widget
        self.send_signal(widget.Inputs.tree, self.signal_data)

        node = Mock()

        widget.tree_adapter = Mock()
        widget.tree_adapter.attribute = lambda *_: ContinuousVariable("foo")
        widget.node_content_cls = lambda *_: "bar<br/>ban"

        widget.tree_adapter.has_children = lambda *_: True
        widget.show_intermediate = False
        widget.update_node_info(node)
        args = node.setHtml.call_args[0][0]
        self.assertIn("foo", args)
        self.assertNotIn("bar", args)

        widget.tree_adapter.has_children = lambda *_: True
        widget.show_intermediate = True
        widget.update_node_info(node)
        args = node.setHtml.call_args[0][0]
        self.assertIn("bar<br/>ban<hr/>foo", args)

        widget.tree_adapter.has_children = lambda *_: False
        widget.show_intermediate = True
        widget.update_node_info(node)
        args = node.setHtml.call_args[0][0]
        self.assertIn("bar<br/>ban<hr/>foo", args)

        widget.tree_adapter.has_children = lambda *_: False
        widget.show_intermediate = False
        widget.update_node_info(node)
        args = node.setHtml.call_args[0][0]
        self.assertIn("bar<br/>ban<hr/>foo", args)

    def test_tree_labels(self):
        w = self.widget
        w.show_intermediate = True

        self.send_signal(w.Inputs.tree, self.tree)

        txt = w.root_node.toPlainText()
        self.assertIn("42.0 ± 8.0", txt)
        self.assertIn("50 instances", txt)
        self.assertIn("aaa", txt)

        children = [edge.node2
                    for edge in w.root_node.graph_edges()]

        txt = children[0].toPlainText()
        self.assertIn("38.0 ± 5.0", txt)
        self.assertIn("16 instances", txt)
        self.assertIn("bbb", txt)

        txt = children[1].toPlainText()
        self.assertIn("13.0 ± 3.0", txt)
        self.assertIn("14 instances", txt)

        txt = children[2].toPlainText()
        self.assertIn("78.0 ± 12.0", txt)
        self.assertIn("20 instances", txt)
        self.assertIn("ccc", txt)

        w.controls.show_intermediate.click()

        txt = w.root_node.toPlainText()
        self.assertNotIn("42.0 ± 8.0", txt)
        self.assertNotIn("50 instances", txt)
        self.assertIn("aaa", txt)

        children = [edge.node2
                    for edge in w.root_node.graph_edges()]

        txt = children[0].toPlainText()
        self.assertNotIn("38.0 ± 5.0", txt)
        self.assertNotIn("16 instances", txt)
        self.assertIn("bbb", txt)

        txt = children[1].toPlainText()
        self.assertIn("13.0 ± 3.0", txt)
        self.assertIn("14 instances", txt)

        txt = children[2].toPlainText()
        self.assertNotIn("78.0 ± 12.0", txt)
        self.assertNotIn("20 instances", txt)
        self.assertIn("ccc", txt)

    def test_select_node_labels(self):
        widget = self.widget
        combo = self.widget.controls.node_labels

        def check_labels(attr):
            column = widget.dataset.get_column(attr)
            to_str = widget.domain[attr].str_val
            for node in widget.scene.nodes():
                if widget.tree_adapter.has_children(node.node_inst):
                    continue
                subset = node.node_inst.subset
                exp = ", ".join(map(to_str, column[subset[:4]]))
                if len(subset) > 4:
                    exp += ", …"
                self.assertIn(exp, node.toHtml())

        def switch_to(attr):
            idx = combo.model().indexOf(attr and widget.domain[attr])
            combo.setCurrentIndex(idx)
            combo.activated[int].emit(idx)

        zoo = Table("zoo")
        iris = Table("iris")
        zootree = TreeLearner()(zoo)
        iristree = TreeLearner()(iris)

        self.assertIsNone(widget.node_labels)

        # Default label is a string variable (with most unique values)
        self.send_signal(widget.Inputs.tree, zootree)
        self.assertIs(widget.node_labels, zootree.domain["name"])
        check_labels("name")

        # Change to another variable
        switch_to("hair")
        check_labels("hair")

        # See that losing the data for a while keeps the label
        self.send_signal(widget.Inputs.tree, None)
        self.assertIsNone(widget.node_labels)
        self.send_signal(widget.Inputs.tree, zootree)
        check_labels("hair")

        self.send_signal(widget.Inputs.tree, iristree)
        self.assertIsNone(widget.node_labels)
        switch_to("petal length")
        check_labels("petal length")
        self.send_signal(widget.Inputs.tree, None)
        self.assertIsNone(widget.node_labels)
        self.send_signal(widget.Inputs.tree, iristree)
        self.assertEqual(widget.node_labels, iristree.domain["petal length"])
        check_labels("petal length")

        instances = zootree.instances
        zootree.instances = None
        self.send_signal(widget.Inputs.tree, zootree)
        self.assertIsNone(widget.node_labels)
        self.assertEqual(list(widget.label_model), [None])

        zootree.instances = instances
        widget.node_labels_hint = ""  # Reset to no hint
        self.send_signal(widget.Inputs.tree, zootree)
        self.assertIs(widget.node_labels, zootree.domain["name"])
        check_labels("name")

    def test_select_node_many_labels(self):
        zoo = Table("zoo")
        zootree = TreeLearner()(zoo)

        self.send_signal(self.widget.Inputs.tree, zootree)

        ta = self.widget.tree_adapter
        node = next(node for node in self.widget.scene.nodes()
                    if not ta.has_children(node.node_inst))

        var = zoo.domain["name"]
        values = zoo.get_column(var)
        ta.get_instances_in_nodes = lambda *_: zoo[[0, 50]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50]])),
                      node.toHtml())

        ta.get_instances_in_nodes = lambda *_: zoo[[0, 50, 75]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50, 75]])),
                      node.toHtml())

        ta.get_instances_in_nodes = lambda *_: zoo[[0, 50, 75, 11]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50, 75, 11]])),
                      node.toHtml())

        ta.get_instances_in_nodes = lambda *_: zoo[[0, 50, 75, 11, 3]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50, 75, 11]])) + ", …",
                      node.toHtml())

        var = zoo.domain["legs"]
        values = zoo.get_column(var)
        self.widget.node_labels = var
        ta.get_instances_in_nodes = lambda *_: zoo[[0, 50, 75, 11]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50, 75, 11]])),
                      node.toHtml())

        ta.get_instances_in_nodes = lambda *_: zoo[[0, 50, 75, 11, 3]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50, 75, 11]])) + ", …",
                      node.toHtml())

        iris = Table("iris")
        iristree = TreeLearner()(iris)
        self.send_signal(self.widget.Inputs.tree, iristree)
        var = iris.domain["petal length"]
        values = iris.get_column(var)
        self.widget.node_labels = var
        ta = self.widget.tree_adapter
        node = next(node for node in self.widget.scene.nodes()
                    if not ta.has_children(node.node_inst))

        ta.get_instances_in_nodes = lambda *_: iris[[0, 50, 75, 11, 13,
                                                     14, 15, 16]]
        self.widget.update_node_info(node)
        self.assertIn(", ".join(map(var.str_val, values[[0, 50, 75, 11]])) + ", …",
                      node.toHtml())


if __name__ == "__main__":
    unittest.main()

# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.classification.tree import \
    TreeModel, Node, DiscreteNode, MappedDiscreteNode, NumericNode
from Orange.widgets.visualize.utils.tree.treeadapter import TreeAdapter


class TestTreeAdapter(unittest.TestCase):
    def setUp(self):
        # pylint: disable=invalid-name
        v1 = self.v1 = ContinuousVariable("v1")
        v2 = self.v2 = DiscreteVariable("v2", "abc")
        v3 = self.v3 = DiscreteVariable("v3", "def")
        y = self.y = ContinuousVariable("y")
        self.domain = Domain([v1, v2, v3], y)
        self.data = Table(self.domain, np.arange(40).reshape(10, 4))
        self.root = NumericNode(v1, 0, 13, np.array([0., 42]))
        self.root.subset = np.array(np.arange(10), dtype=np.int32)
        left = self.left = DiscreteNode(v2, 1, np.array([1, 42]))
        left.subset = np.array([2, 3, 4, 5])
        left.children = [Node(None, None, np.array([x, 42])) for x in [2, 3, 4]]
        right = self.right = MappedDiscreteNode(
            v3, 2, np.array([1, 1, 0]), np.array([5, 42]))
        right.children = [Node(None, None, np.array([6, 42])),
                          None]
        right.subset = np.array([8, 9])
        self.root.children = [left, right]
        self.model = TreeModel(self.data, self.root)
        self.adapter = TreeAdapter(self.model)

    def test_adapter(self):
        adapt = self.adapter
        self.assertAlmostEqual(adapt.weight(self.left), 0.4)
        self.assertEqual(adapt.num_samples(self.left), 4)
        self.assertIs(adapt.parent(self.left), self.root)
        self.assertTrue(adapt.has_children(self.left))
        self.assertFalse(adapt.has_children(self.left.children[0]))
        self.assertFalse(adapt.is_leaf(self.left))
        self.assertTrue(adapt.is_leaf(self.left.children[0]))
        self.assertEqual(
            adapt.children(self.root),
            [self.left, self.right])
        # Test whether it skips null-children
        self.assertEqual(
            adapt.children(self.right),
            [self.right.children[0]])
        np.testing.assert_almost_equal(
            adapt.get_distribution(self.left),
            np.array([[1, 42]]))
        self.assertEqual(adapt.rules(self.right), ["v1 > 13"])
        self.assertIs(adapt.attribute(self.root), self.v1)
        self.assertEqual(adapt.leaves(self.left), self.left.children)
        self.assertEqual(adapt.leaves(self.right), [self.right.children[0]])
        self.assertEqual(
            adapt.leaves(self.root),
            self.left.children + [self.right.children[0]])
        np.testing.assert_almost_equal(
            adapt.get_instances_in_nodes(self.root).X,
            self.data.X)
        np.testing.assert_almost_equal(
            adapt.get_instances_in_nodes([self.left, self.right]).X,
            np.array([[8, 9, 10],
                      [12, 13, 14],
                      [16, 17, 18],
                      [20, 21, 22],
                      [32, 33, 34],
                      [36, 37, 38]], dtype=np.float))
        self.assertEqual(adapt.max_depth, 2)
        self.assertEqual(adapt.num_nodes, 7)
        self.assertIs(adapt.root, self.root)
        self.assertIs(adapt.domain, self.domain)

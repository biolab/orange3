# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.tree import \
    Tree, OrangeTreeModel, Node, DiscreteNode, MappedDiscreteNode, NumericNode


class TestTree:
    @classmethod
    def setUpClass(cls):
        cls.no_pruning_args = {}

    @classmethod
    def all_nodes(cls, node):
        yield node
        if node.children:
            for child in node.children:
                yield from cls.all_nodes(child)

    @classmethod
    def depth(cls, node):
        return 0 if node.children is None else \
            1 + max(cls.depth(child) for child in node.children)

    def test_get_tree(self):
        learn = self.OrangeTreeLearner()
        clf = learn(self.data)
        self.assertIsInstance(clf, OrangeTreeModel)

    def test_full_tree(self):
        table = self.data
        learn = self.OrangeTreeLearner(**self.no_pruning_args)
        clf = learn(table)
        pred = clf(table)
        self.assertTrue(np.all(table.Y.flatten() == pred))

    def test_min_samples_split(self):
        clf = self.OrangeTreeLearner(
            min_samples_split=10, min_samples_leaf=1,
            **self.no_pruning_args)(self.data)
        self.assertTrue(
            all(node.children is None or len(node.subset) >= 10
                for node in self.all_nodes(clf.root)))

    def test_min_samples_leaf(self):
        for lim in (1, 2, 30):
            args = dict(min_samples_split=1, min_samples_leaf=lim)
            args.update(self.no_pruning_args)
            clf = self.OrangeTreeLearner(**args)(self.data)
            self.assertTrue(all(len(node.subset) >= lim
                                for node in self.all_nodes(clf.root)))
            clf = self.OrangeTreeLearner(**args)(self.data_mixed)
            self.assertTrue(all(len(node.subset) >= lim
                                for node in self.all_nodes(clf.root)))
            clf = self.OrangeTreeLearner(binarize=True, **args)(self.data_mixed)
            self.assertTrue(all(len(node.subset) >= lim
                                for node in self.all_nodes(clf.root)))

    def test_max_depth(self):
        for i in (1, 2, 5):
            tree = self.OrangeTreeLearner(max_depth=i)(self.data)
            self.assertEqual(self.depth(tree.root), i)

    def test_refuse_binarize_too_many_values(self):
        clf = self.OrangeTreeLearner(binarize=True)
        lim = clf.MAX_BINARIZATION

        domain = Domain(
            [DiscreteVariable("x", ("v{}".format(i) for i in range(lim + 1)))],
            self.class_var)
        data = Table(domain, np.zeros((100, 2)))
        self.assertRaises(ValueError, clf, data)
        clf.binarize = False
        clf(data)

        domain = Domain(
            [DiscreteVariable("x", ("v{}".format(i) for i in range(lim)))],
            self.class_var)
        data = Table(domain, np.zeros((100, 2)))
        clf.binarize = True
        clf(data)
        clf.binarize = False
        clf(data)

    def test_find_mapping(self):
        clf = self.OrangeTreeLearner()

        domain = Domain([DiscreteVariable("x", values="abcdefgh"),
                         ContinuousVariable("r1"),
                         DiscreteVariable("r2", values="abcd")],
                        self.class_var)
        col_x = np.arange(80) % 8
        for mapping in (np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                        np.array([0, 0, 0, 0, 0, 1, 1, 0]),
                        np.array([0, 0, 1, 0, 0, 0, 0, 0]),
                        np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 0, 0, 0, 0, 0, 0, 1]),
                        np.array([1, 1, 1, 1, 1, 1, 1, 0]),
                        np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                        np.array([1, 1, 1, 1, 0, 1, 1, 1])):
            data = Table(domain,
                         np.vstack((col_x,
                                    np.random.random(80),
                                    np.random.randint(0, 3, 80).astype(float),
                                    mapping[col_x],)).T)
            root = clf(data).root
            self.assertIsInstance(root, MappedDiscreteNode)
            self.assertEqual(root.attr_idx, 0)
            found = root.mapping if root.mapping[0] == 0 else 1 - root.mapping
            mapping = mapping if mapping[0] == 0 else 1 - mapping
            np.testing.assert_equal(found, mapping)
            self.assertEqual(len(root.children), 2)
            self.assertIsInstance(root.children[0], Node)
            self.assertIsInstance(root.children[1], Node)

    def test_find_threshold(self):
        clf = self.OrangeTreeLearner()

        domain = Domain([ContinuousVariable("x"),
                         DiscreteVariable("r1", values="abcd"),
                         ContinuousVariable("r2")],
                        self.class_var)

        col_x = np.arange(80)
        np.random.shuffle(col_x)
        data = Table(domain,
                     np.vstack((col_x,
                                np.random.randint(0, 3, 80).astype(float),
                                np.random.random(80),
                                col_x > 30,)).T)
        root = clf(data).root
        self.assertIsInstance(root, NumericNode)
        self.assertEqual(root.attr_idx, 0)
        self.assertEqual(root.threshold, 30)
        self.assertEqual(len(root.children), 2)
        self.assertIsInstance(root.children[0], Node)
        self.assertIsInstance(root.children[1], Node)

    def test_no_data(self):
        clf = self.OrangeTreeLearner()

        domain = Domain([DiscreteVariable("r1", values="ab"),
                         DiscreteVariable("r2", values="abcd"),
                         ContinuousVariable("r3")],
                        self.class_var)

        data = Table(domain)
        tree = clf(data)
        self.assertIsInstance(tree.root, Node)
        np.testing.assert_almost_equal(tree.predict(np.array([[0., 0., 0.]])),
                                       self.blind_prediction)

    def test_all_values_missing(self):
        clf = self.OrangeTreeLearner()

        domain = Domain([DiscreteVariable("r1", values="ab"),
                         DiscreteVariable("r2", values="abcd"),
                         ContinuousVariable("r3")],
                        self.class_var)
        a = np.empty((10, 4))
        a[:, :3] = np.nan
        a[:, 3] = np.arange(10) % 2
        data = Table(domain, a)
        for clf.binarize in (False, True):
            tree = clf(data)
            self.assertIsInstance(tree.root, Node)
            np.testing.assert_almost_equal(
                tree.predict(np.array([[0., 0., 0.]])),
                self.prediction_on_0_1)

    def test_single_valued_attr(self):
        clf = self.OrangeTreeLearner()
        domain = Domain([DiscreteVariable("r1", values="a")],
                        self.class_var)
        data = Table(domain, np.array([[0, 0], [0, 1]]))
        tree = clf(data)
        self.assertIsInstance(tree.root, Node)
        np.testing.assert_almost_equal(tree.predict(np.array([[0., 0., 0.]])),
                                       self.prediction_on_0_1)


class TestClassifier(TestTree, unittest.TestCase):
    from Orange.classification import OrangeTreeLearner

    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        TestTree.setUpClass()

        cls.no_pruning_args = {'sufficient_majority': 1}

        cls.data = Table('iris')  # continuous attributes
        cls.data_mixed = Table('heart_disease')  # mixed
        cls.class_var = DiscreteVariable("y", values="nyx")
        cls.blind_prediction = np.ones((1, 3)) / 3
        cls.prediction_on_0_1 = np.array([[0.5, 0.5, 0]])


class TestRegressor(TestTree, unittest.TestCase):
    from Orange.regression import OrangeTreeLearner

    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        TestTree.setUpClass()

        cls.data = Table('housing')
        imports = Table("imports-85")
        new_domain = Domain([attr for attr in imports.domain.attributes
                             if attr.is_continuous or len(attr.values) <= 16],
                            imports.domain.class_var)
        cls.data_mixed = Table(new_domain, imports)

        cls.class_var = ContinuousVariable("y")
        cls.blind_prediction = 0
        cls.prediction_on_0_1 = 0.5


class TestTreeMixin(unittest.TestCase):
    def test_not_implemented(self):
        # pylint: disable=pointless-statement
        tree = Tree()
        with self.assertRaises(NotImplementedError):
            tree.node_count
        with self.assertRaises(NotImplementedError):
            tree.leaf_count
        with self.assertRaises(NotImplementedError):
            tree.root
        for m in (
                tree.children, tree.is_leaf, tree.num_instances, tree.attribute,
                tree.rule, tree.get_value, tree.get_instances,
                tree.data_attribute):
            self.assertRaises(NotImplementedError, m, None)
        self.assertRaises(NotImplementedError,
                          tree.split_condition, None, None)

class TestNodes(unittest.TestCase):
    def test_node(self):
        var = ContinuousVariable("y")
        node = Node(var, 42, "foo")
        self.assertEqual(node.attr, var)
        self.assertEqual(node.attr_idx, 42)
        self.assertEqual(node.value, "foo")
        self.assertIsNone(node.children)
        np.testing.assert_equal(node.subset, np.array([], dtype=np.int32))

        self.assertTrue(np.isnan(node.descend([])))

    def test_discrete_node(self):
        var = DiscreteVariable("y", values="abc")
        node = DiscreteNode(var, 2, "foo")
        self.assertEqual(node.attr, var)
        self.assertEqual(node.attr_idx, 2)
        self.assertEqual(node.value, "foo")
        self.assertIsNone(node.children)
        np.testing.assert_equal(node.subset, np.array([], dtype=np.int32))

        self.assertEqual(node.descend([3, 4, 1, 6]), 1)
        self.assertTrue(np.isnan(node.descend([3, 4, float("nan"), 6])))
        self.assertEqual(node.describe_branch(1), "b")

    def test_mapped_node(self):
        var = DiscreteVariable("y", values="abc")
        node = MappedDiscreteNode(var, 2, np.array([1, 1, 0]), "foo")
        self.assertEqual(node.attr, var)
        self.assertEqual(node.attr_idx, 2)
        self.assertEqual(node.value, "foo")
        self.assertIsNone(node.children)
        np.testing.assert_equal(node.subset, np.array([], dtype=np.int32))

        self.assertEqual(node.descend([3, 4, 0, 6]), 1)
        self.assertEqual(node.descend([3, 4, 1, 6]), 1)
        self.assertEqual(node.descend([3, 4, 2, 6]), 0)
        self.assertTrue(np.isnan(node.descend([3, 4, float("nan"), 6])))
        self.assertEqual(node.describe_branch(0), "c")
        self.assertEqual(node.describe_branch(1), "a or b")

        mapping, branches = MappedDiscreteNode.branches_from_mapping(
            np.array([2, 3, 1, 1, 0, 1, 4, 2]), int("1001", 2), 6)
        np.testing.assert_equal(
            mapping, np.array([1, 0, 0, 1, 0, 0], dtype=np.int16))
        np.testing.assert_equal(
            branches, np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int16))

    def test_numeric_node(self):
        var = ContinuousVariable("y")
        node = NumericNode(var, 2, 42, "foo")
        self.assertEqual(node.attr, var)
        self.assertEqual(node.attr_idx, 2)
        self.assertEqual(node.value, "foo")
        self.assertIsNone(node.children)
        np.testing.assert_equal(node.subset, np.array([], dtype=np.int32))

        self.assertEqual(node.descend([3, 4, 0, 6]), 0)
        self.assertEqual(node.descend([3, 4, 42, 6]), 0)
        self.assertEqual(node.descend([3, 4, 42.1, 6]), 1)
        self.assertTrue(np.isnan(node.descend([3, 4, float("nan"), 6])))
        self.assertEqual(node.describe_branch(0), "≤ 42")
        self.assertEqual(node.describe_branch(1), "> 42")


class TestTreeModel(unittest.TestCase):
    def setUp(self):
        # pylint: disable=invalid-name
        v1 = self.v1 = ContinuousVariable("v1")
        v2 = self.v2 = DiscreteVariable("v2", "abc")
        v3 = self.v3 = DiscreteVariable("v3", "def")
        y = self.y = ContinuousVariable("y")
        self.domain = Domain([v1, v2, v3], y)
        self.data = Table(self.domain, np.arange(40).reshape(10, 4))
        self.root = NumericNode(v1, 0, 13, np.array([0., 42]))
        left = DiscreteNode(v2, 1, np.array([1, 42]))
        left.children = [Node(None, None, np.array([x, 42])) for x in [2, 3, 4]]
        right = MappedDiscreteNode(v3, 2, np.array([1, 1, 0]),
                                   np.array([5, 42]))
        right.children = [Node(None, None, np.array([x, 42])) for x in [6, 7]]
        self.root.children = [left, right]

    def test_compile_and_run_cont(self):
        # I investigate, I have a warrant
        # pylint: disable=protected-access
        model = OrangeTreeModel(self.data, self.root)
        expected_values = np.vstack((np.arange(8), [42] * 8)).T
        np.testing.assert_equal(model._values, expected_values)
        self.assertEqual(model._thresholds[0], 13)
        self.assertEqual(model._thresholds.shape, (8,))

        nan = float("nan")
        x = np.array(
            [[nan, 0, 0],
             [13, nan, 0],
             [13, 0, 0],
             [13, 1, 0],
             [13, 2, 0],
             [14, 2, nan],
             [14, 2, 2],
             [14, 2, 1]], dtype=float
        )
        np.testing.assert_equal(model.get_values(x), expected_values)
        np.testing.assert_equal(model.get_values_in_python(x), expected_values)
        np.testing.assert_equal(model.get_values_by_nodes(x), expected_values)
        np.testing.assert_equal(model.predict(x), np.arange(8).astype(int))

        v1 = ContinuousVariable("d1")
        v2 = DiscreteVariable("d2", "abc")
        v3 = DiscreteVariable("d3", "def")
        y = DiscreteVariable("dy")
        domain = Domain([v1, v2, v3], y)
        data = Table(domain, np.zeros((10, 4)))
        root = NumericNode(v1, 0, 13, np.array([0., 42]))
        left = DiscreteNode(v2, 1, np.array([1, 42]))
        left.children = [Node(None, None, np.array([x, 42])) for x in [2, 3, 4]]
        right = MappedDiscreteNode(v3, 2, np.array([1, 1, 0]),
                                   np.array([5, 42]))
        right.children = [Node(None, None, np.array([x, 42])) for x in [6, 7]]
        root.children = [left, right]

        model = OrangeTreeModel(data, root)
        normalized = \
            expected_values / np.sum(expected_values, axis=1)[:, np.newaxis]
        np.testing.assert_equal(model.predict(x), normalized)

    def test_methods(self):
        model = OrangeTreeModel(self.data, self.root)
        self.assertEqual(model.node_count, 8)
        self.assertEqual(model.leaf_count, 5)
        self.assertIs(model.root, self.root)

        left, right = model.children(self.root)
        self.assertIsInstance(left, DiscreteNode)
        self.assertIsInstance(right, MappedDiscreteNode)
        self.assertEqual(model.children(model.children(left)[2]), [])

        self.assertIs(model.attribute(self.root), self.v1)
        self.assertIs(model.attribute(left), self.v2)

        self.assertEqual(model.split_condition(self.root, None), "")
        self.assertEqual(model.split_condition(left, self.root), "≤ 13")
        self.assertEqual(model.split_condition(right, self.root), "> 13")

        left.subset = np.array([2, 3])
        self.assertEqual(model.num_instances(self.root), 0)
        self.assertEqual(model.num_instances(left), 2)
        subset = model.get_instances([self.root, left])
        self.assertIsInstance(subset, Table)
        self.assertEqual(len(subset), 2)
        np.testing.assert_equal(subset.X, np.array([[8, 9, 10], [12, 13, 14]]))
        np.testing.assert_equal(subset.Y, np.array([11, 15]))

    def test_print(self):
        model = OrangeTreeModel(self.data, self.root)
        self.assertEqual(model.print_tree(), """             [ 1 42] v1 ≤ 13
             [ 2 42]     v2 a
             [ 3 42]     v2 b
             [ 4 42]     v2 c
             [ 5 42] v1 > 13
             [ 6 42]     v3 f
             [ 7 42]     v3 d or e
""")

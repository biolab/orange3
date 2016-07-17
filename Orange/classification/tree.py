"""Tree inducers: SKL and Orange's own inducer"""
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import bottleneck as bn
import sklearn.tree as skl_tree
from sklearn.tree._tree import TREE_LEAF

from Orange.classification._tree_scorers import (
    find_threshold_entropy, find_binarization_entropy)

from Orange.preprocess.transformation import Indicator
from Orange.statistics import distribution, contingency

from Orange.base import Tree, Model, Learner
from Orange.classification import SklLearner, SklModel

__all__ = ["TreeLearner", "OrangeTreeLearner"]


class Node:
    """Tree node base class; instances of this class are also used as leaves

    Attributes:
        attr (Odange.data.Variable): The attribute used for splitting
        attr_idx (int): The index of the attribute used for splitting
        class_distr (Orange.statistics.distribution.Discrete):
            class distribution on training data for this node
        children (list of Node): child branches
        subset (numpy.array): indices of data instances in this node
    """
    def __init__(self, attr, attr_idx, class_distr):
        self.attr = attr
        self.attr_idx = attr_idx
        self.class_distr = class_distr
        self.children = None
        self.subset = np.array([])

    def descend(self, inst):
        """Return the child for the given data instance"""
        return np.nan


class DiscreteNode(Node):
    """Node for discrete attributes"""
    def __init__(self, attr, attr_idx, class_distr):
        super().__init__(attr, attr_idx, class_distr)

    def descend(self, inst):
        return int(inst[self.attr_idx])

    def describe_branch(self, i):
        return self.attr.values[i]


class DiscreteNodeMapping(Node):
    """Node for discrete attributes with mapping to branches

    Attributes:
        mapping (numpy.ndarray): indices of branches for each attribute value
    """
    def __init__(self, attr, attr_idx, mapping, class_distr):
        super().__init__(attr, attr_idx, class_distr)
        self.mapping = mapping

    def descend(self, inst):
        val = inst[self.attr_idx]
        return np.nan if np.isnan(val) else self.mapping[int(val)]

    def describe_branch(self, i):
        values = [self.attr.values[j]
                  for j, v in enumerate(self.mapping) if v == i]
        if len(values) == 1:
            return values[0]
        return "{} or {}".format(", ".join(values[:-1]), values[-1])


class NumericNode(Node):
    """Node for numeric attributes

    Attributes:
        threshold (float): values lower or equal to this threshold go to the
            left branches, larger to the right
    """
    def __init__(self, attr, attr_idx, threshold, class_distr):
        super().__init__(attr, attr_idx, class_distr)
        self.threshold = threshold

    def descend(self, inst):
        val = inst[self.attr_idx]
        return np.nan if np.isnan(val) else val > self.threshold

    def describe_branch(self, i):
        return "{} {}".format(("≤", ">")[i], self.threshold)


class OrangeTreeModel(Model, Tree):
    """
    Tree classifier with proper handling of nominal attributes and binarization
    and the interface API for visualization.
    """

    def __init__(self, data, root):
        super().__init__(data.domain)
        self.instances = data
        self._root = root

    def predict(self, X):
        n = len(X)
        y = np.empty((n,))
        for i in range(n):
            x = X[i]
            node = self._root
            while True:
                next = node.descend(x)
                if np.isnan(next):
                    break
                node = node.children[next]
            y[i] = node.class_distr.modus()
        return y

    @property
    @lru_cache(10)
    def node_count(self):
        def _count(node):
            return 1 + sum(_count(c) for c in self.children(node))
        return _count(self._root)

    @property
    @lru_cache(10)
    def leaf_count(self):
        def _count(node):
            c = self.children(self)
            return not c or sum(_count(c) for c in self.children(node))
        return _count(self._root)

    @property
    def root(self):
        return self._root

    @staticmethod
    def children(node):
        return node.children or []

    @staticmethod
    def attribute(node):
        return node.attr

    @staticmethod
    def split_condition(node, parent):
        if parent is None:
            return ""
        return parent.describe_branch(parent.children.index(node))

    @staticmethod
    def rule(index_path):
        return ""

    @staticmethod
    def num_instances(node):
        return len(node.subset)

    @staticmethod
    def get_distribution(node):
        return node.class_distr

    @staticmethod
    def majority(node):
        return node.class_distr.modus()

    def get_instances(self, nodes):
        indices = np.unique(np.hstack([node.subset for node in nodes]))
        if len(indices):
            return self.instances[indices]

    def print_tree(self, node=None, level=0):
        """Simple tree printer"""
        # pylint: disable=bad-builtin
        if node is None:
            node = self.root
        if node.children is None:
            return
        for branch_no, child in enumerate(node.children):
            print("{:>20} {}{} {}".format(
                str(child.class_distr), "    " * level, node.attr.name,
                node.describe_branch(branch_no)))
            self.print_tree(child, level + 1)


class OrangeTreeLearner(Learner):
    """
    Tree inducer with proper handling of nominal attributes and binarization.
    """
    __returns__ = OrangeTreeModel

    # Binarization is exhaustive, so we set a limit on the number of values
    MAX_BINARIZATION = 16

    def __init__(
            self, *args,
            binarize=True,
            min_samples_leaf=1, min_samples_split=2, sufficient_majority=0.95,
            max_depth=None):
        super().__init__(*args)
        self.binarize = binarize
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sufficient_majority = sufficient_majority
        self.max_depth = max_depth

    def _select_attr(self, data):
        """Select the attribute for the next split.

        Returns:
            tuple with an instance of Node and a numpy array indicating
            the branch index for each data instance, or -1 if data instance
            is dropped
        """
        # Prevent false warnings by pylint
        attr = attr_no = None

        def _score_disc():
            """Scoring for discrete attributes, no binarization

            The class computes the entropy itself, not by calling other
            functions. This is to make sure that it uses the same
            definition as the below classes that compute entropy themselves
            for efficiency reasons."""
            cont = contingency.Discrete(data, attr)
            attr_distr = np.sum(cont, axis=0)
            class_distr = np.sum(cont, axis=1)  # Skip insts missing attr values
            n = sum(class_distr)
            if np.min(attr_distr) < self.min_samples_leaf:
                return None, None, None
            class_distr[class_distr <= 0] = 1
            cont[cont <= 0] = 1
            class_entr = \
                n * np.log(n) - np.sum(class_distr * np.log(class_distr))
            cont_entr = \
                np.sum(attr_distr * np.log(attr_distr)) - \
                np.sum(cont * np.log(cont))
            sc = (class_entr - cont_entr) / n / np.log(2)
            branches = data[:, attr].X.flatten()
            branches[np.isnan(branches)] = -1
            return (sc,
                    DiscreteNode(attr, attr_no, class_distr),
                    branches)

        def _score_disc_bin():
            """Scoring for discrete attributes, with binarization"""
            n_values = len(attr.values)
            if n_values == 2:
                return _score_disc()
            cont = contingency.Discrete(data, attr)
            attr_distr = np.sum(cont, axis=0)
            # Skip instances with missing value of the attribute
            class_distr_no_miss = np.sum(cont, axis=1)
            best_score, best_mapping = find_binarization_entropy(
                cont, class_distr_no_miss, attr_distr, self.min_samples_leaf)
            if best_score == 0:
                return None, None, None
            best_score *= 1 - sum(cont.unknowns) / len(data)
            best_cut = np.array(
                [int(x)
                 for x in reversed("{:>0{}b}".format(best_mapping, n_values))] +
                [-1], dtype=np.int16)
            col_x = data.X[:, attr_no].flatten()
            col_x[np.isnan(col_x)] = n_values
            return (best_score,
                    DiscreteNodeMapping(attr, attr_no, best_cut, class_distr),
                    best_cut[col_x.astype(int)])

        def _score_cont():
            """Scoring for numeric attributes"""
            col_x = data.X[:, attr_no]
            nans = np.sum(np.isnan(col_x))
            non_nans = len(col_x) - nans
            arginds = np.argsort(col_x)[:non_nans]
            best_score, best_cut = find_threshold_entropy(
                col_x, data.Y, arginds, len(class_distr), self.min_samples_leaf)
            if best_score == 0:
                return None, None, None
            best_score *= non_nans / len(col_x)
            branches = col_x > best_cut
            branches[np.isnan(col_x)] = -1
            return (best_score,
                    NumericNode(attr, attr_no, best_cut, class_distr),
                    branches)

        #######################################
        # The real _select_attr starts here
        domain = data.domain
        class_var = domain.class_var
        class_distr = distribution.Discrete(data, class_var)
        best_node = Node(None, None, class_distr)
        best_score = best_branches = None
        disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            sc, node, branches = \
                disc_scorer() if attr.is_discrete else _score_cont()
            if node is not None and (best_score is None or sc > best_score):
                best_score, best_node, best_branches = sc, node, branches
        return best_node, best_branches

    def build_tree(self, data, active_inst, level=1):
        """Induce a tree from the given data

        Returns:
            root node (Node)"""
        node_insts = data[active_inst]
        distr = distribution.Discrete(node_insts, data.domain.class_var)
        if len(data) < self.min_samples_split or \
                max(distr) >= sum(distr) * self.sufficient_majority or \
                self.max_depth is not None and level > self.max_depth:
            node, branches = Node(None, None, distr), None
        else:
            node, branches = self._select_attr(node_insts)
        node.subset = active_inst
        if branches is not None:
            node.children = [
                self.build_tree(data, active_inst[branches == branch],
                                level + 1)
                for branch in range(int(bn.nanmax(branches) + 1))]
        return node

    def fit_storage(self, data):
        if self.binarize and any(
                attr.is_discrete and len(attr.values) > self.MAX_BINARIZATION
                for attr in data.domain.attributes):
            # No fallback in the script; widgets can prevent this error
            # by providing a fallback and issue a warning about doing so
            raise ValueError("Exhaustive binarization does not handle "
                             "attributes with more than {} values".
                             format(self.MAX_BINARIZATION))

        active_inst = np.arange(len(data), dtype=int)
        model = OrangeTreeModel(data, self.build_tree(data, active_inst))
        model.root.subset = active_inst
        return model


class TreeClassifier(SklModel, Tree):
    """Wrapper for SKL's tree classifier with the interface API for
    visualizations"""
    def __init__(self, *args, **kwargs):
        SklModel.__init__(self, *args, **kwargs)
        self._cached_sample_assignments = None

    @property
    def tree(self):
        return self.skl_model.tree_

    @property
    def node_count(self):
        return self.tree.node_count

    @property
    def leaf_count(self):
        return np.count_nonzero(self.tree.children_left == TREE_LEAF)

    @property
    def root(self):
        return 0

    def attribute(self, node_index):
        return self.domain.attributes[self.tree.feature[node_index]]

    def data_attribute(self, node_index):
        attr = self.attribute(node_index)
        if attr is not None and isinstance(attr.compute_value, Indicator):
            attr = attr.compute_value.variable
        return attr

    def split_condition(self, node_index, parent_index):
        tree = self.tree
        if parent_index is None:
            return ""
        parent_attr = self.attribute(parent_index)
        parent_attr_cv = parent_attr.compute_value
        is_left_child = tree.children_left[parent_index] == node_index
        if isinstance(parent_attr_cv, Indicator) and \
                hasattr(parent_attr_cv.variable, "values"):
            values = parent_attr_cv.variable.values
            return values[abs(parent_attr_cv.value - is_left_child)] \
                if len(values) == 2 \
                else "≠ " * is_left_child + values[parent_attr_cv.value]
        else:
            thresh = tree.threshold[parent_index]
            return "%s %s" % ([">", "≤"][is_left_child],
                              parent_attr.str_val(thresh))

    def is_leaf(self, node_index):
        return self.tree.children_left[node_index] == \
               self.tree.children_right[node_index] == TREE_LEAF

    def children(self, node_index):
        tree = self.tree
        return [w[node_index] for w in (tree.children_left, tree.children_right)
                if w[node_index] != TREE_LEAF]

    def rule(self, index_path):
        tree = self.tree
        conditions = OrderedDict()
        for parent, child in zip(index_path, index_path[1:]):
            parent_attr = self.attribute(parent)
            parent_attr_cv = parent_attr.compute_value
            is_left_child = tree.children_left[parent] == child
            if isinstance(parent_attr_cv, Indicator) and \
                    hasattr(parent_attr_cv.variable, "values"):
                values = parent_attr_cv.variable.values
                attr_name = parent_attr_cv.variable.name
                sign = ["=", "≠"][is_left_child * (len(values) != 2)]
                value = values[abs(parent_attr_cv.value -
                                   is_left_child * (len(values) == 2))]
            else:
                attr_name = parent_attr.name
                sign = [">", "≤"][is_left_child]
                value = parent_attr.str_val(tree.threshold[parent])
                cond = (attr_name, sign)
            if cond in conditions:
                old_val = conditions[cond]
                if sign == ">":
                    conditions[cond] = max(float(value), float(old_val))
                elif sign == "≠":
                    conditions[cond] = "{}, {}".format(old_val, value)
                elif sign == "≤":
                    conditions[cond] = min(float(value), float(old_val))
            else:
                conditions[(attr_name, sign)] = value
        return " AND\n".join("{} {} {}".format(n, s, v)
                             for (n, s), v in conditions.items())

    def _subnode_range(self, node_id):
        right = left = node_id
        tree = self.tree
        if tree.children_left[left] == TREE_LEAF:
            return node_id, node_id
        else:
            left = tree.children_left[left]
            # run down to the right most node
            while tree.children_right[right] != TREE_LEAF:
                right = tree.children_right[right]
            return left, right + 1

    def _leaf_indices(self, node_id):
        start, stop = self._subnode_range(node_id)
        if start == stop:
            return np.array([node_id], dtype=int)
        else:
            isleaf = self.tree.children_left[start:stop] == TREE_LEAF
            assert np.flatnonzero(isleaf).size > 0
            return start + np.flatnonzero(isleaf)

    def _assign_samples(self):
        def _assign(node_id, indices):
            if tree.children_left[node_id] == TREE_LEAF:
                return [indices]
            else:
                feature_idx = tree.feature[node_id]
                thresh = tree.threshold[node_id]
                column = X[indices, feature_idx]
                leftmask = column <= thresh
                leftind = _assign(tree.children_left[node_id],
                                  indices[leftmask])
                rightind = _assign(tree.children_right[node_id],
                                   indices[~leftmask])
                return list.__iadd__(leftind, rightind)
        if self._cached_sample_assignments is None:
            tree = self.tree
            X = self.instances.X
            items = np.arange(X.shape[0], dtype=int)
            leaf_indices = _assign(0, items)
            self._cached_sample_assignments = np.array(leaf_indices)
        return self._cached_sample_assignments

    def _get_unnormalized_distribution(self, node_index):
        tree = self.tree
        if self.is_leaf(node_index):
            counts = tree.value[node_index]
        else:
            leaf_ind = self._leaf_indices(node_index)
            values = tree.value[leaf_ind]
            counts = np.sum(values, axis=0)

        assert counts.shape[0] == 1, "n_outputs > 1 "
        return counts[0]

    def get_distribution(self, node_index):
        counts = self._get_unnormalized_distribution(node_index)
        counts_sum = np.sum(counts)
        if counts_sum > 0:
            counts = counts / counts_sum
        return counts

    def majority(self, node_index):
        return np.argmax(self.get_distribution(node_index))

    def num_instances(self, node_index):
        return np.sum(self._get_unnormalized_distribution(node_index))

    def get_instances(self, node_indices):
        selected_leaves = [self._leaf_indices(node_index)
                           for node_index in node_indices]
        if selected_leaves:
            selected_leaves = np.unique(np.hstack(selected_leaves))
        all_leaves = self._leaf_indices(0)

        if len(selected_leaves) > 0:
            ind = np.searchsorted(all_leaves, selected_leaves, side="left")
            leaf_samples = self._assign_samples()
            leaf_samples = [leaf_samples[i] for i in ind]
            indices = np.hstack(leaf_samples)
        else:
            indices = []
        if len(indices):
            return self.instances[indices]


class TreeLearner(SklLearner):
    """Wrapper for SKL's tree inducer"""
    __wraps__ = skl_tree.DecisionTreeClassifier
    __returns__ = TreeClassifier
    name = 'tree'

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

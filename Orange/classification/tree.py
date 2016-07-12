"""Tree inducers: SKL and Orange's own inducer"""

import numpy as np
import bottleneck as bn
import sklearn.tree as skl_tree
from Orange.statistics import distribution, contingency

from Orange.base import Tree, Model, Learner
from Orange.classification import SklLearner, SklModel
from Orange.preprocess import (RemoveNaNClasses, Continuize,
                               RemoveNaNColumns, SklImpute, score)

__all__ = ["TreeLearner", "OrangeTreeLearner"]


class Node:
    """Tree node base class; instances of this class are also used as leaves

    Attributes:
        attr (int): The attribute used for splitting
        class_distr (Orange.statistics.distribution.Discrete):
            class distribution on training data for this node
        children (list of Node): child branches
    """
    def __init__(self, attr, class_distr):
        self.attr = attr
        self.class_distr = class_distr
        self.children = None

    def descend(self, inst):
        """Return the child for the given data instance"""
        return None

class DiscreteNode(Node):
    """Node for discrete attributes"""
    def descend(self, inst):
        return self.children[inst[self.attr]]


class DiscreteNodeMapping(Node):
    """Node for discrete attributes with mapping to branches

    Attributes:
        mapping (numpy.ndarray): indices of branches for each attribute value
    """
    def __init__(self, attr, mapping, class_distr):
        super().__init__(attr, class_distr)
        self.mapping = mapping

    def descend(self, inst):
        return self.children[self.mapping[inst[self.attr]]]


class NumericNode(Node):
    """Node for numeric attributes

    Attributes:
        threshold (float): values lower or equal to this threshold go to the
            left branches, larger to the right
    """
    def __init__(self, attr, threshold, class_distr):
        super().__init__(attr, class_distr)
        self.threshold = threshold

    def predict(self, inst):
        return self.children[inst[self.attr] > self.threshold]


class OrangeTreeLearner(Learner):
    """
    Tree inducer with proper handling of nominal attributes and binarization.
    """
    MAX_BINARIZATION = 12

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
            """Scoring for discrete attributes, no binarization"""
            attr_distr = distribution.Discrete(data, attr)
            if np.min(attr_distr) < self.min_samples_leaf:
                return None, None, None
            sc = scorer.score_data(data, attr)
            branches = data[:, attr].X.flatten()
            branches[np.isnan(branches)] = -1
            return (sc,
                    DiscreteNode(attr_no, class_distr),
                    branches)

        def _score_disc_bin():
            """Scoring for discrete attributes, with binarization"""
            n = len(attr.values)
            if n == 2:
                return _score_disc()
            if n > self.MAX_BINARIZATION:
                raise ValueError("Exhaustive binarization does not handle "
                                 "attributes with more than {} values".
                                 format(self.MAX_BINARIZATION))
            cont = contingency.Discrete(data, attr)
            nan_adjust = 1 - sum(cont.unknowns) / len(data)
            fstr = "{{:0>{}b}}".format(n)
            best_score = best_cut = None
            for i in range(1, 2 ** (n - 1)):
                left = np.array(list(fstr.format(i))).astype(bool)
                conting[:, 0] = np.sum(cont[:, left], axis=1)
                conting[:, 1] = class_distr - conting[:, 0]
                if min(np.sum(conting, axis=0)) < self.min_samples_leaf:
                    continue
                sc = scorer.from_contingency(conting, nan_adjust)
                if best_score is None or sc > best_score:
                    best_score, best_cut = sc, left
            if best_cut is None:
                return None, None, None
            best_cut = np.resize(best_cut, (len(best_cut) + 1,)).astype(int)
            best_cut[-1] = -1
            col_x = data.X[:, attr_no].flatten()
            col_x[np.isnan(col_x)] = n
            return (best_score,
                    DiscreteNodeMapping(attr_no, best_cut, class_distr),
                    best_cut[col_x.astype(int)])

        def _score_cont():
            """Scoring for numeric attributes"""
            conting[:, 0] = class_distr
            conting[:, 1] = 0
            best_score = best_cut = None
            y = data.Y
            col_x = data.X[:, attr_no]
            nans = np.sum(np.isnan(col_x))
            non_nans = len(col_x) - nans
            nan_adjust = non_nans / len(col_x)
            change = np.array([-1, 1])
            arginds = np.argsort(col_x)[:non_nans]
            for e, ne in zip(arginds, arginds[1:]):
                conting[y[e]] += change
                if col_x[ne] == col_x[e] or \
                        min(np.sum(conting, axis=0)) < self.min_samples_leaf:
                    continue
                sc = scorer.from_contingency(conting, nan_adjust)
                if best_score is None or sc > best_score:
                    best_score, best_cut = sc, col_x[e]
            if best_score is None:
                return None, None, None
            branches = col_x > best_cut
            branches[np.isnan(col_x)] = -1
            return (best_score,
                    NumericNode(attr_no, best_cut, class_distr),
                    branches)

        #######################################
        # The real _select_attr starts here
        scorer = score.InfoGain()
        domain = data.domain
        class_var = domain.class_var
        class_distr = distribution.Discrete(data, class_var)
        conting = contingency.Discrete(np.zeros((len(class_var.values), 2)))
        best_node = Node(None, class_distr)
        best_score = best_branches = None
        disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            if attr.is_discrete:
                sc, node, branches = disc_scorer()
            else:
                sc, node, branches = _score_cont()
            if node is not None and (best_score is None or sc > best_score):
                best_score, best_node, best_branches = sc, node, branches
        return best_node, best_branches

    def build_tree(self, data, level=1):
        """Induce a tree from the given data

        Returns:
            root node (Node)"""
        distr = distribution.Discrete(data, data.domain.class_var)
        if len(data) < self.min_samples_split or \
                max(distr) >= sum(distr) * self.sufficient_majority or \
                self.max_depth is not None and level > self.max_depth:
            return Node(None, distr)
        node, branches = self._select_attr(data)
        if branches is not None:
            node.children = [
                self.build_tree(data[branches == branch], level + 1)
                for branch in range(int(bn.nanmax(branches) + 1))]
        return node

    def fit_storage(self, data):
        return OrangeTreeModel(data.domain, self.build_tree(data))


class OrangeTreeModel(Model):
    """
    Tree classifier with proper handling of nominal attributes and binarization.
    """

    def __init__(self, domain, root):
        super().__init__(domain)
        self.root = root

    def print_tree(self, node=None, level=0):
        """Simple tree printer"""
        # pylint: disable=bad-builtin
        if node is None:
            node = self.root
        if node.children:
            attr = self.domain[node.attr]
            for branch_no, child in enumerate(node.children):
                print("{:>20} {}{}".format(
                    str(child.class_distr), "    " * level, attr.name), end="")
                if isinstance(node, DiscreteNode):
                    print("={}".format(attr.values[branch_no]))
                elif isinstance(node, DiscreteNodeMapping):
                    values = [attr.values[i]
                              for i, v in enumerate(node.mapping)
                              if v == branch_no]
                    if len(values) == 1:
                        print("={}".format(values[0]))
                    else:
                        print("={{{}}}".format(", ".join(values)))
                else:
                    print("{}{}".format(["<=", ">"][branch_no], node.threshold))
                self.print_tree(child, level + 1)


class TreeClassifier(SklModel, Tree):
    @property
    def tree(self):
        return self.skl_model.tree_


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

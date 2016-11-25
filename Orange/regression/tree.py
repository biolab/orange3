"""Tree inducers: SKL and Orange's own inducer"""

import numpy as np
import sklearn.tree as skl_tree

from Orange.tree import Node, DiscreteNode, MappedDiscreteNode, \
    NumericNode, TreeModel
from Orange.regression import SklLearner, SklModel, Learner
from Orange.classification import _tree_scorers

__all__ = ["SklTreeRegressionLearner", "TreeLearner"]


class TreeLearner(Learner):
    """
    Tree inducer with proper handling of nominal attributes and binarization.

    The inducer can handle missing values of attributes and target.
    For discrete attributes with more than two possible values, each value can
    get a separate branch (`binarize=False`), or values can be grouped into
    two groups (`binarize=True`, default).

    The tree growth can be limited by the required number of instances for
    internal nodes and for leafs, and by the maximal depth of the tree.

    If the tree is not binary, it can contain zero-branches.

    Args:
        binarize: if `True` the inducer will find optimal split into two
            subsets for values of discrete attributes. If `False` (default),
            each value gets its branch.
        min_samples_leaf: the minimal number of data instances in a leaf
        min_samples_split: the minimal nubmer of data instances that is split
            into subgroups
        max_depth: the maximal depth of the tree

    Returns:
        instance of OrangeTreeModel
    """
    __returns__ = TreeModel

    # Binarization is exhaustive, so we set a limit on the number of values
    MAX_BINARIZATION = 16

    def __init__(
            self, *args,
            binarize=False, min_samples_leaf=1, min_samples_split=2,
            max_depth=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.binarize = binarize
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
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
        REJECT_ATTRIBUTE = 0, None, None, 0

        def _score_disc():
            n_values = len(attr.values)
            score = _tree_scorers.compute_grouped_MSE(
                col_x, col_y, n_values, self.min_samples_leaf)
            # The score is already adjusted for missing attribute values, so
            # we don't do it here
            if score == 0:
                return REJECT_ATTRIBUTE
            branches = col_x.flatten()
            branches[np.isnan(branches)] = -1
            return score, DiscreteNode(attr, attr_no, None), branches, n_values

        def _score_disc_bin():
            n_values = len(attr.values)
            if n_values == 2:
                return _score_disc()
            score, mapping = _tree_scorers.find_binarization_MSE(
                col_x, col_y, n_values, self.min_samples_leaf)
            # The score is already adjusted for missing attribute values, so
            # we don't do it here
            if score == 0:
                return REJECT_ATTRIBUTE
            mapping, branches = MappedDiscreteNode.branches_from_mapping(
                data.X[:, attr_no], mapping, len(attr.values))
            node = MappedDiscreteNode(attr, attr_no, mapping, None)
            return score, node, branches, 2

        def _score_cont():
            """Scoring for numeric attributes"""
            nans = np.sum(np.isnan(col_x))
            non_nans = len(col_x) - nans
            arginds = np.argsort(col_x)[:non_nans]
            score, cut = _tree_scorers.find_threshold_MSE(
                col_x, col_y, arginds, self.min_samples_leaf)
            if score == 0:
                return  REJECT_ATTRIBUTE
            score *= non_nans / len(col_x)
            branches = np.full(len(col_x), -1, dtype=int)
            mask = ~np.isnan(col_x)
            branches[mask] = (col_x[mask] > cut).astype(int)
            node = NumericNode(attr, attr_no, cut, None)
            return score, node, branches, 2

        #######################################
        # The real _select_attr starts here
        domain = data.domain
        col_y = data.Y
        best_score, *best_res = REJECT_ATTRIBUTE
        best_res = [Node(None, 0, None), ] + best_res[1:]
        disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            col_x = data[:, attr_no].X.reshape((len(data),))
            sc, *res = disc_scorer() if attr.is_discrete else _score_cont()
            if res[0] is not None and sc > best_score:
                best_score, best_res = sc, res
        return best_res

    def build_tree(self, data, active_inst, level=1):
        """Induce a tree from the given data

        Returns:
            root node (Node)"""
        node_insts = data[active_inst]
        if len(node_insts) < self.min_samples_leaf:
            return None
        if len(node_insts) < self.min_samples_split or \
                self.max_depth is not None and level > self.max_depth:
            node, branches, n_children = Node(None, None, None), None, 0
        else:
            node, branches, n_children = self._select_attr(node_insts)
        mean, var = np.mean(node_insts.Y), np.var(node_insts.Y)
        node.value = np.array([mean, 1 if np.isnan(var) else var])
        node.subset = active_inst
        if branches is not None:
            node.children = [
                self.build_tree(data, active_inst[branches == br], level + 1)
                for br in range(n_children)]
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

        active_inst = np.nonzero(~np.isnan(data.Y))[0].astype(np.int32)
        root = self.build_tree(data, active_inst)
        if root is None:
            root = Node(None, 0, np.array([0., 0.]))
        root.subset = active_inst
        model = TreeModel(data, root)
        return model


class SklTreeRegressor(SklModel):
    @property
    def tree(self):
        return self.skl_model.tree_


class SklTreeRegressionLearner(SklLearner):
    __wraps__ = skl_tree.DecisionTreeRegressor
    __returns__ = SklTreeRegressor
    name = 'regression tree'

    def __init__(self, criterion="mse", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

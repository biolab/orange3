import numpy as np
import sklearn.tree as skl_tree

from Orange.tree import Node, DiscreteNode, DiscreteNodeMapping, \
    NumericNode, OrangeTreeModel
from Orange.regression import SklLearner, SklModel, Learner
from Orange.classification import _tree_scorers

__all__ = ["TreeRegressionLearner"]


class OrangeTreeLearner(Learner):
    """
    Tree inducer with proper handling of nominal attributes and binarization.
    """
    __returns__ = OrangeTreeModel

    # Binarization is exhaustive, so we set a limit on the number of values
    MAX_BINARIZATION = 16

    def __init__(
            self, *args,
            binarize=True, min_samples_leaf=1, min_samples_split=2,
            max_depth=None):
        super().__init__(*args)
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

        def _score_disc():
            score = _tree_scorers.compute_grouped_MSE(col_x, col_y)
            if score == 0:
                return None, None, None
            branches = col_x.flatten()
            branches[np.isnan(branches)] = -1
            return score, DiscreteNode(attr, attr_no, None), branches

        def _score_disc_bin():
            n_values = len(attr.values)
            if n_values == 2:
                return _score_disc
            score, mapping = _tree_scorers.find_binarization_MSE(
                col_x, col_y, self.min_samples_leaf)
            mapping, branches = DiscreteNodeMapping.branches_from_mapping(
                data.X[:, attr_no], mapping, len(attr.values))
            node = DiscreteNodeMapping(attr, attr_no, mapping, None)
            return score, node, branches

        def _score_cont():
            """Scoring for numeric attributes"""
            nans = np.sum(np.isnan(col_x))
            non_nans = len(col_x) - nans
            arginds = np.argsort(col_x)[:non_nans]
            score, cut = _tree_scorers.find_threshold_MSE(
                col_x, col_y, arginds, self.min_samples_leaf)
            if score == 0:
                return None, None, None
            score *= non_nans / len(col_x)
            branches = (col_x > cut).astype(int)
            branches[np.isnan(col_x)] = -1
            node = NumericNode(attr, attr_no, cut, None)
            return score, node, branches

        #######################################
        # The real _select_attr starts here
        domain = data.domain
        col_y = data.Y
        best_node = Node(None, None, None)
        best_score = best_branches = None
        disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            col_x = data[:, attr_no].X.reshape((len(data),))
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
        if len(node_insts) < self.min_samples_split or \
                self.max_depth is not None and level > self.max_depth:
            node, branches = Node(None, None, None), None
        else:
            node, branches = self._select_attr(node_insts)
        mean, var = np.mean(node_insts.Y), np.var(node_insts.Y)
        node.value = np.array([mean, 1 if np.isnan(var) else var])
        node.subset = active_inst
        if branches is not None:
            node.children = [
                self.build_tree(data, active_inst[branches == branch],
                                level + 1)
                for branch in range(int(np.nanmax(branches) + 1))]
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




class TreeRegressor(SklModel):
    @property
    def tree(self):
        return self.skl_model.tree_


class TreeRegressionLearner(SklLearner):
    __wraps__ = skl_tree.DecisionTreeRegressor
    __returns__ = TreeRegressor
    name = 'regression tree'

    def __init__(self, criterion="mse", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

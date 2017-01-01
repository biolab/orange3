"""Tree inducers: SKL and Orange's own inducer"""
import numpy as np
import sklearn.tree as skl_tree

from Orange.classification import SklLearner, SklModel, Learner
from Orange.classification import _tree_scorers
from Orange.statistics import distribution, contingency
from Orange.tree import Node, DiscreteNode, MappedDiscreteNode, \
    NumericNode, TreeModel

__all__ = ["SklTreeLearner", "TreeLearner"]


class TreeLearner(Learner):
    """
    Tree inducer with proper handling of nominal attributes and binarization.

    The inducer can handle missing values of attributes and target.
    For discrete attributes with more than two possible values, each value can
    get a separate branch (`binarize=False`), or values can be grouped into
    two groups (`binarize=True`, default).

    The tree growth can be limited by the required number of instances for
    internal nodes and for leafs, the sufficient proportion of majority class,
    and by the maximal depth of the tree.

    If the tree is not binary, it can contain zero-branches.

    Args:
        binarize (bool):
            if `True` the inducer will find optimal split into two
            subsets for values of discrete attributes. If `False` (default),
            each value gets its branch.

        min_samples_leaf (float):
            the minimal number of data instances in a leaf

        min_samples_split (float):
            the minimal nubmer of data instances that is
            split into subgroups

        max_depth (int): the maximal depth of the tree

        sufficient_majority (float):
            a majority at which the data is not split
            further

    Returns:
        instance of OrangeTreeModel
    """
    __returns__ = TreeModel

    # Binarization is exhaustive, so we set a limit on the number of values
    MAX_BINARIZATION = 16

    def __init__(
            self, *args, binarize=False, max_depth=None,
            min_samples_leaf=1, min_samples_split=2, sufficient_majority=0.95,
            **kwargs):
        super().__init__(*args, **kwargs)
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
        REJECT_ATTRIBUTE = 0, None, None, 0

        def _score_disc():
            """Scoring for discrete attributes, no binarization

            The class computes the entropy itself, not by calling other
            functions. This is to make sure that it uses the same
            definition as the below classes that compute entropy themselves
            for efficiency reasons."""
            n_values = len(attr.values)
            if n_values < 2:
                return REJECT_ATTRIBUTE

            x = data.X[:, attr_no].flatten()
            cont = _tree_scorers.contingency(x, len(data.domain.attributes[attr_no].values),
                                             data.Y, len(data.domain.class_var.values))
            attr_distr = np.sum(cont, axis=0)
            null_nodes = attr_distr <= self.min_samples_leaf
            # This is just for speed. If there is only a single non-null-node,
            # entropy wouldn't decrease anyway.
            if sum(null_nodes) >= n_values - 1:
                return REJECT_ATTRIBUTE
            cont[:, null_nodes] = 0
            attr_distr = np.sum(cont, axis=0)
            cls_distr = np.sum(cont, axis=1)
            n = np.sum(attr_distr)
            # Avoid log(0); <= instead of == because we need an array
            cls_distr[cls_distr <= 0] = 1
            attr_distr[attr_distr <= 0] = 1
            cont[cont <= 0] = 1
            class_entr = n * np.log(n) - np.sum(cls_distr * np.log(cls_distr))
            attr_entr = np.sum(attr_distr * np.log(attr_distr))
            cont_entr = np.sum(cont * np.log(cont))
            score = (class_entr - attr_entr + cont_entr) / n / np.log(2)
            score *= n / len(data)  # punishment for missing values
            branches = x
            branches[np.isnan(branches)] = -1
            if score == 0:
                return REJECT_ATTRIBUTE
            node = DiscreteNode(attr, attr_no, None)
            return score, node, branches, n_values

        def _score_disc_bin():
            """Scoring for discrete attributes, with binarization"""
            n_values = len(attr.values)
            if n_values <= 2:
                return _score_disc()
            cont = contingency.Discrete(data, attr)
            attr_distr = np.sum(cont, axis=0)
            # Skip instances with missing value of the attribute
            cls_distr = np.sum(cont, axis=1)
            if np.sum(attr_distr) == 0:  # all values are missing
                return REJECT_ATTRIBUTE
            best_score, best_mapping = _tree_scorers.find_binarization_entropy(
                cont, cls_distr, attr_distr, self.min_samples_leaf)
            if best_score <= 0:
                return REJECT_ATTRIBUTE
            best_score *= 1 - np.sum(cont.unknowns) / len(data)
            mapping, branches = MappedDiscreteNode.branches_from_mapping(
                data.X[:, attr_no], best_mapping, n_values)
            node = MappedDiscreteNode(attr, attr_no, mapping, None)
            return best_score, node, branches, 2

        def _score_cont():
            """Scoring for numeric attributes"""
            col_x = data.X[:, attr_no]
            nans = np.sum(np.isnan(col_x))
            non_nans = len(col_x) - nans
            arginds = np.argsort(col_x)[:non_nans]
            best_score, best_cut = _tree_scorers.find_threshold_entropy(
                col_x, data.Y, arginds,
                len(class_var.values), self.min_samples_leaf)
            if best_score == 0:
                return REJECT_ATTRIBUTE
            best_score *= non_nans / len(col_x)
            branches = np.full(len(col_x), -1, dtype=int)
            mask = ~np.isnan(col_x)
            branches[mask] = (col_x[mask] > best_cut).astype(int)
            node = NumericNode(attr, attr_no, best_cut, None)
            return best_score, node, branches, 2

        #######################################
        # The real _select_attr starts here
        domain = data.domain
        class_var = domain.class_var
        best_score, *best_res = REJECT_ATTRIBUTE
        best_res = [Node(None, None, None)] + best_res[1:]
        disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            sc, *res = disc_scorer() if attr.is_discrete else _score_cont()
            if res[0] is not None and sc > best_score:
                best_score, best_res = sc, res
        best_res[0].value = distribution.Discrete(data, class_var)
        return best_res

    def build_tree(self, data, active_inst, level=1):
        """Induce a tree from the given data

        Returns:
            root node (Node)"""
        node_insts = data[active_inst]
        distr = distribution.Discrete(node_insts, data.domain.class_var)
        if len(node_insts) < self.min_samples_leaf:
            return None
        if len(node_insts) < self.min_samples_split or \
                max(distr) >= sum(distr) * self.sufficient_majority or \
                self.max_depth is not None and level > self.max_depth:
            node, branches, n_children = Node(None, None, distr), None, 0
        else:
            node, branches, n_children = self._select_attr(node_insts)
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
            distr = distribution.Discrete(data, data.domain.class_var)
            if np.sum(distr) == 0:
                distr[:] = 1
            root = Node(None, 0, distr)
        root.subset = active_inst
        model = TreeModel(data, root)
        return model


class SklTreeClassifier(SklModel, TreeModel):
    """Wrapper for SKL's tree classifier with the interface API for
    visualizations"""
    def __init__(self, data, *args, **kwargs):
        SklModel.__init__(self, *args, **kwargs)
        self.data = data
        self._cached_sample_assignments = None
        self.root = self._build_tree(self._root())

    def rule(self, node):
        return ['foo']

    def _build_tree(self, node):
        attribute_idx = self._attribute(node)
        attribute = self.data.domain.attributes[attribute_idx]
        assert attribute.is_continuous is True
        # Since sklearn only supports numeric data, and all data passed to
        # sklearn learners is continuized first, we can saftely assume numeric
        # data
        node_obj = NumericNode(
            attribute, attr_idx=attribute_idx,
            threshold=self._threshold(attribute_idx),
            value=self._value(attribute_idx))
        node_obj.children = [
            self._build_tree(child) for child in self._children(node)]
        node_obj.subset = [1]
        return node_obj

    def _attribute(self, node):
        return self.skl_model.tree_.feature[node]

    def _threshold(self, node):
        return self.skl_model.tree_.threshold[node]

    def _value(self, node):
        # Return 0th since the distribution is store inside a 2d array
        return self.skl_model.tree_.value[node][0]

    def _root(self):
        return 0

    def _children(self, node):
        left_child = self.skl_model.tree_.children_left[node]
        right_child = self.skl_model.tree_.children_right[node]
        children = []
        if left_child > -1:
            children.append(left_child)
        if right_child > -1:
            children.append(right_child)
        return children

    def _has_children(self, node):
        return len(self._children(node)) > 0

    def _get_samples_in_leaves(self, data):
        """Get an array of instance indices that belong to each leaf.

        For a given dataset X, separate the instances out into an array, so
        they are grouped together based on what leaf they belong to.

        Examples
        --------
        Given a tree with two leaf nodes ( A <- R -> B ) and the dataset X =
        [ 10, 20, 30, 40, 50, 60 ], where 10, 20 and 40 belong to leaf A, and
        the rest to leaf B, the following structure will be returned (where
        array is the numpy array):
        [array([ 0, 1, 3 ]), array([ 2, 4, 5 ])]

        The first array represents the indices of the values that belong to the
        first leaft, so calling X[ 0, 1, 3 ] = [ 10, 20, 40 ]

        Parameters
        ----------
        data
            A matrix containing the data instances.

        Returns
        -------
        np.array
            The indices of instances belonging to a given leaf.

        """

        def assign(node_id, indices):
            if self.skl_model.tree_.children_left[node_id] == -1:
                return [indices]
            else:
                feature_idx = self.skl_model.tree_.feature[node_id]
                thresh = self.skl_model.tree_.threshold[node_id]

                column = data[indices, feature_idx]
                leftmask = column <= thresh
                leftind = assign(self.skl_model.tree_.children_left[node_id],
                                 indices[leftmask])
                rightind = assign(self.skl_model.tree_.children_right[node_id],
                                  indices[~leftmask])
                return list.__iadd__(leftind, rightind)

        n, _ = data.shape

        items = np.arange(n, dtype=int)
        leaf_indices = assign(0, items)
        return leaf_indices


class SklTreeLearner(SklLearner):
    """Wrapper for SKL's tree inducer"""
    __wraps__ = skl_tree.DecisionTreeClassifier
    __returns__ = SklTreeClassifier
    name = 'tree'

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit_storage(self, data):
        clf = self.__wraps__(**self.params)
        X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
        Y = Y.reshape(-1)
        if W is None or not self.supports_weights:
            return self.__returns__(data, clf.fit(X, Y))
        return self.__returns__(
            data, clf.fit(X, Y, sample_weight=W.reshape(-1)))

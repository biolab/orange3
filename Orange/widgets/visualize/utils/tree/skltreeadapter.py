"""Tree adapter class for sklearn trees."""
from collections import OrderedDict

import numpy as np
from Orange.widgets.visualize.utils.tree.treeadapter import BaseTreeAdapter

from Orange.misc.cache import memoize_method
from Orange.preprocess.transformation import Indicator
from Orange.widgets.visualize.utils.tree.rules import (
    DiscreteRule,
    ContinuousRule
)


class SklTreeAdapter(BaseTreeAdapter):
    """Sklear Tree Adapter.

    An abstraction on top of the scikit learn classification tree.

    Parameters
    ----------
    model : SklModel
        An sklearn tree model instance.

    """

    NO_CHILD = -1
    FEATURE_UNDEFINED = -2

    def __init__(self, model):
        super().__init__(model)
        self._tree = model.skl_model.tree_
        self._all_leaves = None

    @memoize_method(maxsize=1024)
    def weight(self, node):
        return self.num_samples(node) / self.num_samples(self.parent(node))

    def num_samples(self, node):
        return self._tree.n_node_samples[node]

    @memoize_method(maxsize=1024)
    def parent(self, node):
        for children in (self._tree.children_left, self._tree.children_right):
            try:
                return (children == node).nonzero()[0][0]
            except IndexError:
                continue
        return self.ROOT_PARENT

    def has_children(self, node):
        return (self._tree.children_left[node] != self.NO_CHILD or
                self._tree.children_right[node] != self.NO_CHILD)

    def children(self, node):
        if self.has_children(node):
            return self.__left_child(node), self.__right_child(node)
        return ()

    def __left_child(self, node):
        return self._tree.children_left[node]

    def __right_child(self, node):
        return self._tree.children_right[node]

    @memoize_method(maxsize=1024)
    def get_distribution(self, node):
        value = self._tree.value[node]
        # If regression tree, we have to compute variance by hand, we can
        # detect this because you can't have classification trees when there's
        # only one class
        if value.shape[1] == 1:
            var = np.var(self.get_instances_in_nodes(node).Y)
            variances = np.array([(var * np.ones(value.shape[0]))]).T
            value = np.hstack((value, variances))
        return value

    def get_impurity(self, node):
        return self._tree.impurity[node]

    @property
    def max_depth(self):
        return self._tree.max_depth

    @property
    def num_nodes(self):
        return self._tree.node_count

    @property
    def root(self):
        return 0

    @memoize_method(maxsize=1024)
    def rules(self, node):
        if node != self.root:
            parent = self.parent(node)
            # Convert the parent list of rules into an ordered dict
            pr = OrderedDict([(r.attr_name, r) for r in self.rules(parent)])

            parent_attr = self.attribute(parent)
            # Get the parent attribute type
            parent_attr_cv = parent_attr.compute_value

            is_left_child = self.__left_child(parent) == node

            # The parent split variable is discrete
            if isinstance(parent_attr_cv, Indicator) and \
                    hasattr(parent_attr_cv.variable, 'values'):
                values = parent_attr_cv.variable.values
                attr_name = parent_attr_cv.variable.name
                eq = not is_left_child * (len(values) != 2)
                value = values[abs(parent_attr_cv.value -
                                   is_left_child * (len(values) == 2))]
                new_rule = DiscreteRule(attr_name, eq, value)
                # Since discrete variables should appear in their own lines
                # they must not be merged, so the dict key is set with the
                # value, so the same keys can exist with different values
                # e.g. #legs ≠ 2 and #legs ≠ 4
                attr_name = attr_name + '_' + value
            # The parent split variable is continuous
            else:
                attr_name = parent_attr.name
                sign = not is_left_child
                value = self._tree.threshold[self.parent(node)]
                new_rule = ContinuousRule(attr_name, sign, value,
                                          inclusive=is_left_child)

            # Check if a rule with that attribute exists
            if attr_name in pr:
                pr[attr_name] = pr[attr_name].merge_with(new_rule)
                pr.move_to_end(attr_name)
            else:
                pr[attr_name] = new_rule

            return list(pr.values())
        else:
            return []

    def short_rule(self, node):
        return self.rules(node)[0].description

    def attribute(self, node):
        feature_idx = self.splitting_attribute(node)
        if feature_idx != self.FEATURE_UNDEFINED:
            return self.domain.attributes[self.splitting_attribute(node)]

    def splitting_attribute(self, node):
        return self._tree.feature[node]

    @memoize_method(maxsize=1024)
    def leaves(self, node):
        start, stop = self._subnode_range(node)
        if start == stop:
            # leaf
            return np.array([node], dtype=int)
        else:
            is_leaf = self._tree.children_left[start:stop] == self.NO_CHILD
            assert np.flatnonzero(is_leaf).size > 0
            return start + np.flatnonzero(is_leaf)

    def _subnode_range(self, node):
        """
        Get the range of indices where there are subnodes of the given node.

        See Also
        --------
        Orange.widgets.model.owclassificationtreegraph.OWTreeGraph
        """

        def find_largest_idx(n):
            """It is necessary to locate the node with the largest index in the
            children in order to get a good range. This is necessary with trees
            that are not right aligned, which can happen when visualising
            random forest trees."""
            if self._tree.children_left[n] == self.NO_CHILD:
                return n

            l_node = find_largest_idx(self._tree.children_left[n])
            r_node = find_largest_idx(self._tree.children_right[n])

            return max(l_node, r_node)

        right = left = node
        if self._tree.children_left[left] == self.NO_CHILD:
            assert self._tree.children_right[node] == self.NO_CHILD
            return node, node
        else:
            left = self._tree.children_left[left]
            right = find_largest_idx(right)

            return left, right + 1

    def get_samples_in_leaves(self):
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
            if self._tree.children_left[node_id] == self.NO_CHILD:
                return [indices]
            else:
                feature_idx = self._tree.feature[node_id]
                thresh = self._tree.threshold[node_id]

                column = self.instances.X[indices, feature_idx]
                leftmask = column <= thresh
                leftind = assign(self._tree.children_left[node_id],
                                 indices[leftmask])
                rightind = assign(self._tree.children_right[node_id],
                                  indices[~leftmask])
                return list.__iadd__(leftind, rightind)

        if self._all_leaves is not None:
            return self._all_leaves

        n, _ = self.instances.X.shape

        items = np.arange(n, dtype=int)
        leaf_indices = assign(0, items)
        self._all_leaves = leaf_indices
        return leaf_indices

    def get_instances_in_nodes(self, nodes):
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        node_leaves = [self.leaves(n) for n in nodes]
        if len(node_leaves) > 0:
            # get the leaves of the selected tree node
            node_leaves = np.unique(np.hstack(node_leaves))

            all_leaves = self.leaves(self.root)

            indices = np.searchsorted(all_leaves, node_leaves)
            # all the leaf samples for each leaf
            leaf_samples = self.get_samples_in_leaves()
            # filter out the leaf samples array that are not selected
            leaf_samples = [leaf_samples[i] for i in indices]
            indices = np.hstack(leaf_samples)
        else:
            indices = []

        return self.instances[indices] if len(indices) else None

    def get_indices(self, nodes):
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        node_leaves = [self.leaves(n) for n in nodes]
        if len(node_leaves) > 0:
            # get the leaves of the selected tree node
            node_leaves = np.unique(np.hstack(node_leaves))

            all_leaves = self.leaves(self.root)

            return np.searchsorted(all_leaves, node_leaves)
        return []

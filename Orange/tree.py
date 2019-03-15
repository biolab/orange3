"""Tree model used by Orange inducers, and Tree interface"""

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from Orange.base import TreeModel as TreeModelInterface


class Node:
    """Tree node base class; instances of this class are also used as leaves

    Attributes:
        attr (Odange.data.Variable): The attribute used for splitting
        attr_idx (int): The index of the attribute used for splitting
        value (object): value used for prediction (e.g. class distribution)
        children (list of Node): child branches
        subset (numpy.array): indices of data instances in this node
    """
    def __init__(self, attr, attr_idx, value):
        self.attr = attr
        self.attr_idx = attr_idx
        self.value = value
        self.children = []
        self.subset = np.array([], dtype=np.int32)
        self.description = ""
        self.condition = ()

    def descend(self, inst):
        """Return the child for the given data instance"""
        return np.nan

    def _set_child_descriptions(self, child, child_idx):
        raise NotImplementedError


class DiscreteNode(Node):
    """Node for discrete attributes"""
    def __init__(self, attr, attr_idx, value):
        super().__init__(attr, attr_idx, value)

    def descend(self, inst):
        val = inst[self.attr_idx]
        return np.nan if np.isnan(val) else int(val)

    def _set_child_descriptions(self, child, child_idx, _):
        child.condition = {child_idx}
        child.description = self.attr.values[child_idx]


class MappedDiscreteNode(Node):
    """Node for discrete attributes with mapping to branches

    Attributes:
        mapping (numpy.ndarray): indices of branches for each attribute value
    """
    def __init__(self, attr, attr_idx, mapping, value):
        super().__init__(attr, attr_idx, value)
        self.mapping = mapping

    def descend(self, inst):
        val = inst[self.attr_idx]
        return np.nan if np.isnan(val) else self.mapping[int(val)]

    @staticmethod
    def branches_from_mapping(col_x, bit_mapping, n_values):
        """
        Return mapping and branches corresponding to column x

        Args:
            col_x (np.ndarray): data in x-column
            bit_mapping (int): bitmask that specifies which attribute values
                go to the left (0) and right (1) branch
            n_values (int): the number of attribute values

        Returns:
            A tuple of two numpy array: branch indices corresponding to
            attribute values and to data instances
        """
        mapping = np.array(
            [int(x)
             for x in reversed("{:>0{}b}".format(bit_mapping, n_values))] +
            [-1], dtype=np.int16)
        col_x = col_x.flatten()  # also ensures copy
        col_x[np.isnan(col_x)] = n_values
        return mapping[:-1], mapping[col_x.astype(np.int16)]

    def _set_child_descriptions(self, child, child_idx, conditions):
        attr = self.attr
        in_brnch = {j for j, v in enumerate(self.mapping) if v == child_idx}
        if attr in conditions:
            child.condition = conditions[attr] & in_brnch
        else:
            child.condition = in_brnch
        vals = [attr.values[j] for j in sorted(child.condition)]
        if not vals:
            child.description = "(unreachable)"
        else:
            child.description = vals[0] if len(vals) == 1 else \
                "{} or {}".format(", ".join(vals[:-1]), vals[-1])


class NumericNode(Node):
    """Node for numeric attributes

    Attributes:
        threshold (float): values lower or equal to this threshold go to the
            left branch, larger to the right
    """
    def __init__(self, attr, attr_idx, threshold, value):
        super().__init__(attr, attr_idx, value)
        self.threshold = threshold

    def descend(self, inst):
        val = inst[self.attr_idx]
        return np.nan if np.isnan(val) else int(val > self.threshold)

    def _set_child_descriptions(self, child, child_idx, conditions):
        attr = self.attr
        threshold = self.threshold
        lower, upper = conditions.get(attr, (None, None))
        if child_idx == 0 and (upper is None or threshold < upper):
            upper = threshold
        elif child_idx == 1 and (lower is None or threshold > lower):
            lower = threshold
        child.condition = (lower, upper)
        child.description = \
            "{} {}".format("≤>"[child_idx], attr.str_val(threshold))


class TreeModel(TreeModelInterface):
    """
    Tree classifier with proper handling of nominal attributes and binarization
    and the interface API for visualization.
    """

    def __init__(self, data, root):
        super().__init__(data.domain)
        self.instances = data
        self.root = root

        self._values = self._thresholds = self._code = None
        self._compile()
        self._compute_descriptions()

    def _prepare_predictions(self, n):
        rootval = self.root.value
        return np.empty((n,) + rootval.shape, dtype=rootval.dtype)

    def get_values_by_nodes(self, X):
        """Prediction that does not use compiled trees; for demo only"""
        n = len(X)
        y = self._prepare_predictions(n)
        for i in range(n):
            x = X[i]
            node = self.root
            while True:
                child_idx = node.descend(x)
                if np.isnan(child_idx):
                    break
                next_node = node.children[child_idx]
                if next_node is None:
                    break
                node = next_node
            y[i] = node.value
        return y

    def get_values_in_python(self, X):
        """Prediction with compiled code, but in Python; for demo only"""
        n = len(X)
        y = self._prepare_predictions(n)
        for i in range(n):
            x = X[i]
            node_ptr = 0
            while self._code[node_ptr]:
                val = x[self._code[node_ptr + 2]]
                if np.isnan(val):
                    break
                child_ptrs = self._code[node_ptr + 3:]
                if self._code[node_ptr] == 3:
                    node_idx = self._code[node_ptr + 1]
                    next_node_ptr = child_ptrs[int(val > self._thresholds[node_idx])]
                else:
                    next_node_ptr = child_ptrs[int(val)]
                if next_node_ptr == -1:
                    break
                node_ptr = next_node_ptr
            node_idx = self._code[node_ptr + 1]
            y[i] = self._values[node_idx]
        return y

    def get_values(self, X):
        from Orange.classification import _tree_scorers
        if sp.isspmatrix_csc(X):
            func = _tree_scorers.compute_predictions_csc
        elif sp.issparse(X):
            func = _tree_scorers.compute_predictions_csr
            X = X.tocsr()
        else:
            func = _tree_scorers.compute_predictions
        return func(X, self._code, self._values, self._thresholds)

    def predict(self, X):
        predictions = self.get_values(X)
        if self.domain.class_var.is_continuous:
            return predictions[:, 0]
        else:
            sums = np.sum(predictions, axis=1)
            # This can't happen because nodes with 0 instances are prohibited
            # zeros = (sums == 0)
            # predictions[zeros] = 1
            # sums[zeros] = predictions.shape[1]
            return predictions / sums[:, np.newaxis]

    def node_count(self):
        def _count(node):
            return 1 + sum(_count(c) for c in node.children if c)
        return _count(self.root)

    def depth(self):
        def _depth(node):
            return 1 + max((_depth(child) for child in node.children if child),
                           default=0)
        return _depth(self.root) - 1

    def leaf_count(self):
        def _count(node):
            return not node.children or \
                   sum(_count(c) if c else 1 for c in node.children)
        return _count(self.root)

    def get_instances(self, nodes):
        indices = self.get_indices(nodes)
        if indices is not None:
            return self.instances[indices]

    def get_indices(self, nodes):
        subsets = [node.subset for node in nodes]
        if subsets:
            return np.unique(np.hstack(subsets))

    @staticmethod
    def climb(node):
        while node:
            yield node
            node = node.parent

    @classmethod
    def rule(cls, node):
        rules = []
        used_attrs = set()
        for node in cls.climb(node):
            if node.parent is None or node.parent.attr_idx in used_attrs:
                continue
            parent = node.parent
            attr = parent.attr
            name = attr.name
            if isinstance(parent, NumericNode):
                lower, upper = node.condition
                if upper is None:
                    rules.append("{} > {}".format(name, attr.repr_val(lower)))
                elif lower is None:
                    rules.append("{} ≤ {}".format(name, attr.repr_val(upper)))
                else:
                    rules.append("{} < {} ≤ {}".format(
                        attr.repr_val(lower), name, attr.repr_val(upper)))
            else:
                rules.append("{}: {}".format(name, node.description))
            used_attrs.add(node.parent.attr_idx)
        return rules

    def print_tree(self, node=None, level=0):
        """String representation of tree for debug purposees"""
        if node is None:
            node = self.root
        res = ""
        for child in node.children:
            res += ("{:>20} {}{} {}\n".format(
                str(child.value), "    " * level, node.attr.name,
                child.description))
            res += self.print_tree(child, level + 1)
        return res

    NODE_TYPES = [Node, DiscreteNode, MappedDiscreteNode, NumericNode]

    def _compile(self):
        def _compute_sizes(node):
            nonlocal nnodes, codesize
            nnodes += 1
            codesize += 2  # node type + node index
            if isinstance(node, MappedDiscreteNode):
                codesize += len(node.mapping)
            if node.children:
                codesize += 1 + len(node.children)  # attr index + children ptrs
                for child in node.children:
                    if child is not None:
                        _compute_sizes(child)

        def _compile_node(node):
            from Orange.classification._tree_scorers import NULL_BRANCH

            # The node is compile into the following code (np.int32)
            # [0] node type: index of type in NODE_TYPES)
            # [1] node index: serves as index into values and thresholds
            # If the node is not a leaf:
            #     [2] attribute index
            # This is followed by an array of indices of the code for children
            # nodes. The length of this array is 2 for numeric attributes or
            # **the number of attribute values** for discrete attributes
            # This is different from the number of branches if discrete values
            # are mapped to branches

            # Thresholds and class distributions are stored in separate
            # 1-d and 2-d array arrays of type np.float, indexed by node index
            # The lengths of both equal the node count; we would gain (if
            # anything) by not reserving space for unused threshold space
            if node is None:
                return NULL_BRANCH
            nonlocal code_ptr, node_idx
            code_start = code_ptr
            self._code[code_ptr] = self.NODE_TYPES.index(type(node))
            self._code[code_ptr + 1] = node_idx
            code_ptr += 2

            self._values[node_idx] = node.value
            if isinstance(node, NumericNode):
                self._thresholds[node_idx] = node.threshold
            node_idx += 1

            # pylint: disable=unidiomatic-typecheck
            if type(node) == Node:
                return code_start

            self._code[code_ptr] = node.attr_idx
            code_ptr += 1

            jump_table_size = 2 if isinstance(node, NumericNode) \
                else len(node.attr.values)
            jump_table = self._code[code_ptr:code_ptr + jump_table_size]
            code_ptr += jump_table_size
            child_indices = [_compile_node(child) for child in node.children]
            if isinstance(node, MappedDiscreteNode):
                jump_table[:] = np.array(child_indices)[node.mapping]
            else:
                jump_table[:] = child_indices

            return code_start

        nnodes = codesize = 0
        _compute_sizes(self.root)
        self._values = self._prepare_predictions(nnodes)
        self._thresholds = np.empty(nnodes)
        self._code = np.empty(codesize, np.int32)

        code_ptr = node_idx = 0
        _compile_node(self.root)

    def _compute_descriptions(self):
        def _compute_subtree(node):
            for i, child in enumerate(node.children):
                if child is None:
                    continue
                child.parent = node
                # These classes are friends
                # pylint: disable=protected-access
                node._set_child_descriptions(child, i, conditions)
                old_cond = conditions.get(node.attr)
                conditions[node.attr] = child.condition
                _compute_subtree(child)
                if old_cond is not None:
                    conditions[node.attr] = old_cond
                else:
                    del conditions[node.attr]

        conditions = OrderedDict()
        self.root.parent = None
        _compute_subtree(self.root)

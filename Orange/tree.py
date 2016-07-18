"""Tree model used by Orange inducers, and Tree interface"""

from functools import lru_cache

import numpy as np

from Orange.base import Model


class Tree:
    """Interface for tree based models.

    Defines members needed for drawing of the tree.

    The API is based on the notion of node indices. Node index can be of
    an arbitrary type; for instances ints, like in skl trees, or node instances,
    like in Orange trees.
    """

    #: Domain of data the tree was built from
    domain = None

    #: Data the tree was built from (Optional)
    instances = None

    @property
    def node_count(self):
        """Return the number of nodes"""
        raise NotImplementedError()

    @property
    def leaf_count(self):
        """Return the number of leaves"""
        raise NotImplementedError()

    @property
    def root(self):
        """Return root index"""
        raise NotImplementedError()

    def children(self, node_index):
        """Return indices of child nodes"""
        raise NotImplementedError()

    def is_leaf(self, node_index):
        """True if the node is a leaf"""
        return not self.children(node_index)

    def num_instances(self, node_index):
        """The number of training instances at the node"""
        raise NotImplementedError()

    def attribute(self, node_index):
        """Attribute whose value determines the branch"""
        raise NotImplementedError()

    def data_attribute(self, node_index):
        """The original data attribute; unless indicators are used,
        this is tha same as `attribute`"""
        return self.attribute(node_index)

    def split_condition(self, node_index, parent_index):
        """Human-readable branch description, e.g. '< 42' or 'male'"""
        raise NotImplementedError()

    def rule(self, index_path):
        """Human-readable rule with the conjunction of conditions along the
        given path

        Args:
            index_path (list): a list of node indices starting at the root"""
        raise NotImplementedError()

    def get_value(self, node_index):
        """Value stored in the node; distributions for classification,
        mean and variance for regression"""
        raise NotImplementedError()

    def get_instances(self, node_indices):
        """Get indices of training instances belonging to the node"""
        raise NotImplementedError()


class RandomForest:
    """Interface for random forest models
    """

    @property
    def trees(self):
        """Return a list of Trees in the forest

        Returns
        -------
        List[Tree]
        """


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
        self.children = None
        self.subset = np.array([])

    def descend(self, inst):
        """Return the child for the given data instance"""
        return np.nan


class DiscreteNode(Node):
    """Node for discrete attributes"""
    def __init__(self, attr, attr_idx, value):
        super().__init__(attr, attr_idx, value)

    def descend(self, inst):
        return int(inst[self.attr_idx])

    def describe_branch(self, i):
        return self.attr.values[i]


class DiscreteNodeMapping(Node):
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

    def describe_branch(self, i):
        values = [self.attr.values[j]
                  for j, v in enumerate(self.mapping) if v == i]
        if len(values) == 1:
            return values[0]
        return "{} or {}".format(", ".join(values[:-1]), values[-1])

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
        return mapping[:-1], mapping[col_x.astype(int)]


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

        self._values = self._thresholds = self._code = None
        self._compile()

    def _prepare_predictions(self, n):
        rootval = self._root.value
        return np.empty((n,) + rootval.shape, dtype=rootval.dtype)

    def predict_by_nodes(self, X):
        """Prediction that does not use compiled trees; for demo only"""
        n = len(X)
        y = self._prepare_predictions(n)
        for i in range(n):
            x = X[i]
            node = self._root
            while True:
                child_idx = node.descend(x)
                if np.isnan(child_idx):
                    break
                node = node.children[child_idx]
            y[i] = node.value
        return y

    def predict_in_python(self, X):
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
                    node_ptr = child_ptrs[int(val > self._thresholds[node_idx])]
                else:
                    node_ptr = child_ptrs[int(val)]
            node_idx = self._code[node_ptr + 1]
            y[i] = self._values[node_idx]
        return y

    def get_values(self, X):
        from Orange.classification import _tree_scorers
        return _tree_scorers.compute_predictions(
            X, self._code, self._values, self._thresholds)

    def predict(self, X):
        predictions = self.get_values(X)
        if self.domain.class_var.is_discrete:
            return predictions / np.sum(predictions, axis=1)
        else:
            return predictions[:, 0]

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
            return not node.children or sum(_count(c) for c in node.children)
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
    def get_value(node):
        return node.value

    def get_instances(self, nodes):
        subsets = [node.subset for node in nodes]
        if subsets:
            return self.instances[np.unique(np.hstack(subsets))]

    def print_tree(self, node=None, level=0):
        """Simple tree printer for debug purposes"""
        # pylint: disable=bad-builtin
        if node is None:
            node = self.root
        if node.children is None:
            return
        for branch_no, child in enumerate(node.children):
            print("{:>20} {}{} {}".format(
                str(child.value), "    " * level, node.attr.name,
                node.describe_branch(branch_no)))
            self.print_tree(child, level + 1)

    NODE_TYPES = [Node, DiscreteNode, DiscreteNodeMapping, NumericNode]

    def _compile(self):
        def _compute_sizes(node):
            nonlocal nnodes, codesize
            nnodes += 1
            codesize += 2  # node type + node index
            if isinstance(node, DiscreteNodeMapping):
                codesize += len(node.mapping)
            if node.children:
                codesize += 1 + len(node.children)  # attr index + children ptrs
                for child in node.children:
                    _compute_sizes(child)

        def _compile_node(node):
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
            # 1-d and 2-d array arrays of type np.float, indexed be node index
            # The lengths of both equal the node count; we would gain (if
            # anything) by not reserving space for unused threshold space
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
            if isinstance(node, DiscreteNodeMapping):
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

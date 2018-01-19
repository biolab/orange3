import ctypes as ct

import numpy as np
from Orange.base import Learner, Model

__all__ = ['SimpleTreeLearner']

from . import _simple_tree
_tree = ct.cdll.LoadLibrary(_simple_tree.__file__)

DiscreteNode = 0
ContinuousNode = 1
PredictorNode = 2
Classification = 0
Regression = 1
IntVar = 0
FloatVar = 1

c_int_p = ct.POINTER(ct.c_int)
c_double_p = ct.POINTER(ct.c_double)


class SIMPLE_TREE_NODE(ct.Structure):
    pass


SIMPLE_TREE_NODE._fields_ = [
    ('type', ct.c_int),
    ('children_size', ct.c_int),
    ('split_attr', ct.c_int),
    ('split', ct.c_float),
    ('children', ct.POINTER(ct.POINTER(SIMPLE_TREE_NODE))),
    ('dist', ct.POINTER(ct.c_float)),
    ('n', ct.c_float),
    ('sum', ct.c_float),
]

_tree.build_tree.restype = ct.POINTER(SIMPLE_TREE_NODE)
_tree.new_node.restype = ct.POINTER(SIMPLE_TREE_NODE)


class SimpleTreeNode:
    pass


class SimpleTreeLearner(Learner):
    """
    Classification or regression tree learner.
    Uses gain ratio for classification and mean square error for
    regression. This learner was developed to speed-up random
    forest construction, but can also be used as a standalone tree learner.

    min_instances : int, optional (default = 2)
        Minimal number of data instances in leaves. When growing the three,
        new nodes are not introduced if they would result in leaves
        with fewer instances than min_instances. Instance count is weighed.

    max_depth : int, optional (default = 1024)
        Maximal depth of tree.

    max_majority : float, optional (default = 1.0)
        Maximal proportion of majority class. When this is
        exceeded, induction stops (only used for classification).

    skip_prob : string, optional (default = 0.0)
        Data attribute will be skipped with probability ``skip_prob``.

        - if float, then skip attribute with this probability.
        - if "sqrt", then `skip_prob = 1 - sqrt(n_features) / n_features`
        - if "log2", then `skip_prob = 1 - log2(n_features) / n_features`

    bootstrap : data table, optional (default = False)
        A bootstrap dataset.

    seed : int, optional (default = 42)
        Random seed.
    """

    name = 'simple tree'

    def __init__(self, min_instances=2, max_depth=1024, max_majority=1.0,
                 skip_prob=0.0, bootstrap=False, seed=42):
        super().__init__()
        self.min_instances = min_instances
        self.max_depth = max_depth
        self.max_majority = max_majority
        self.skip_prob = skip_prob
        self.bootstrap = bootstrap
        self.seed = seed

    def fit_storage(self, data):
        return SimpleTreeModel(self, data)


class SimpleTreeModel(Model):
    def __init__(self, learner, data):
        X = np.ascontiguousarray(data.X)
        Y = np.ascontiguousarray(data.Y)
        W = np.ascontiguousarray(data.W)
        self.num_attrs = X.shape[1]
        self.dom_attr = data.domain.attributes
        self.cls_vars = list(data.domain.class_vars)
        if len(data.domain.class_vars) != 1:
            n_cls = len(data.domain.class_vars)
            raise ValueError("Number of classes should be 1: {}".format(n_cls))

        if data.domain.has_discrete_class:
            self.type = Classification
            self.cls_vals = len(data.domain.class_var.values)
        elif data.domain.has_continuous_class:
            self.type = Regression
            self.cls_vals = 0
        else:
            raise ValueError("Only Continuous and Discrete "
                             "variables are supported")

        if isinstance(learner.skip_prob, (float, int)):
            skip_prob = learner.skip_prob
        elif learner.skip_prob == 'sqrt':
            skip_prob = 1.0 - np.sqrt(X.shape[1]) / X.shape[1]
        elif learner.skip_prob == 'log2':
            skip_prob = 1.0 - np.log2(X.shape[1]) / X.shape[1]
        else:
            raise ValueError(
                "skip_prob not valid: {}".format(learner.skip_prob))

        attr_vals = []
        domain = []
        for attr in data.domain.attributes:
            if attr.is_discrete:
                attr_vals.append(len(attr.values))
                domain.append(IntVar)
            elif attr.is_continuous:
                attr_vals.append(0)
                domain.append(FloatVar)
            else:
                raise ValueError("Only Continuous and Discrete "
                                 "variables are supported")
        attr_vals = np.array(attr_vals, dtype=np.int32)
        domain = np.array(domain, dtype=np.int32)

        self.node = _tree.build_tree(
            X.ctypes.data_as(c_double_p),
            Y.ctypes.data_as(c_double_p),
            W.ctypes.data_as(c_double_p),
            X.shape[0],
            W.size,
            learner.min_instances,
            learner.max_depth,
            ct.c_float(learner.max_majority),
            ct.c_float(skip_prob),
            self.type,
            self.num_attrs,
            self.cls_vals,
            attr_vals.ctypes.data_as(c_int_p),
            domain.ctypes.data_as(c_int_p),
            learner.bootstrap,
            learner.seed)

    def predict_storage(self, data):
        X = np.ascontiguousarray(data.X)
        if self.type == Classification:
            p = np.zeros((X.shape[0], self.cls_vals))
            _tree.predict_classification(
                X.ctypes.data_as(c_double_p),
                X.shape[0],
                self.node,
                self.num_attrs,
                self.cls_vals,
                p.ctypes.data_as(c_double_p))
            return p.argmax(axis=1), p
        elif self.type == Regression:
            p = np.zeros(X.shape[0])
            _tree.predict_regression(
                X.ctypes.data_as(c_double_p),
                X.shape[0],
                self.node,
                self.num_attrs,
                p.ctypes.data_as(c_double_p))
            return p
        else:
            assert False, "Invalid prediction type"

    def __del__(self):
        if hasattr(self, "node"):
            _tree.destroy_tree(self.node, self.type)

    def __getstate__(self):
        dict = self.__dict__.copy()
        del dict['node']
        py_node = self.__to_python(self.node)
        return dict, py_node

    def __setstate__(self, state):
        dict, py_node = state
        self.__dict__.update(dict)
        self.node = self.__from_python(py_node)

    # for pickling a tree
    def __to_python(self, node):
        n = node.contents
        py_node = SimpleTreeNode()
        py_node.type = n.type
        py_node.children_size = n.children_size
        py_node.split_attr = n.split_attr
        py_node.split = n.split
        py_node.children = [
            self.__to_python(n.children[i]) for i in range(n.children_size)]
        if self.type == Classification:
            py_node.dist = [n.dist[i] for i in range(self.cls_vals)]
        else:
            py_node.n = n.n
            py_node.sum = n.sum
        return py_node

    # for unpickling a tree
    def __from_python(self, py_node):
        node = _tree.new_node(py_node.children_size, self.type, self.cls_vals)
        n = node.contents
        n.type = py_node.type
        n.children_size = py_node.children_size
        n.split_attr = py_node.split_attr
        n.split = py_node.split
        for i in range(n.children_size):
            n.children[i] = self.__from_python(py_node.children[i])
        if self.type == Classification:
            for i in range(self.cls_vals):
                n.dist[i] = py_node.dist[i]
        else:
            n.n = py_node.n
            n.sum = py_node.sum
        return node

    # for comparing two trees
    def dumps_tree(self, node):
        n = node.contents
        xs = ['{', str(n.type)]
        if n.type != PredictorNode:
            xs.append(str(n.split_attr))
            if n.type == ContinuousNode:
                xs.append('{:.5f}'.format(n.split))
        elif self.type == Classification:
            for i in range(self.cls_vals):
                xs.append('{:.2f}'.format(n.dist[i]))
        else:
            xs.append('{:.5f} {:.5f}'.format(n.n, n.sum))
        for i in range(n.children_size):
            xs.append(self.dumps_tree(n.children[i]))
        xs.append('}')
        return ' '.join(xs)

    def to_string(self, node=None, level=0):
        """Return a text-based representation of the tree.

        Parameters
        ----------
        node : LP_SIMPLE_TREE_NODE, optional (default=None)
            Tree node. Used to construct representation of the
            tree under this node.
            If not provided, node is considered root node.

        level : int, optional (defaul=0)
            Level of the node. Used for line indentation.

        Returns
        -------
        tree : str
            Text-based representation of the tree.
        """
        if node is None:
            if self.node is None:
                return '(null node)'
            else:
                node = self.node
        n = node.contents
        if self.type == Classification:
            decimals = 1
        else:
            decimals = self.domain.class_var.number_of_decimals
        if n.children_size == 0:
            if self.type == Classification:
                node_cont = [round(n.dist[i], decimals)
                             for i in range(self.cls_vals)]
                index = node_cont.index(max(node_cont))
                major_class = self.cls_vars[0].values[index]
                return ' --> %s (%s)' % (major_class, node_cont)
            else:
                node_cont = str(round(n.sum / n.n, decimals)) + ': ' + str(n.n)
                return ' --> (%s)' % node_cont
        else:
            attr = self.dom_attr[n.split_attr]
            node_desc = attr.name
            if self.type == Classification:
                node_cont = [round(n.dist[i], decimals)
                             for i in range(self.cls_vals)]
            else:
                node_cont = str(round(n.sum / n.n, decimals)) + ': ' + str(n.n)
            ret_str = '\n' + '   ' * level + '%s (%s)' % (node_desc,
                                                          node_cont)
            for i in range(n.children_size):
                if attr.is_continuous:
                    split = '<=' if i % 2 == 0 else '>'
                    split += str(round(n.split, attr.number_of_decimals))
                    ret_str += '\n' + '   ' * level + ': %s' % split
                else:
                    ret_str += '\n' + '   ' * level + ': %s' % attr.values[i]
                ret_str += self.to_string(n.children[i], level + 1)
            return ret_str

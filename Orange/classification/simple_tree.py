import os
import platform
import ctypes as ct

import numpy as np

import Orange

__all__ = ['SimpleTreeLearner']

path = os.path.dirname(os.path.abspath(__file__))
if platform.system() == 'Linux':
    _tree = ct.cdll.LoadLibrary(
        os.path.join(path, '_simple_tree.cpython-34m.so'))
elif platform.system() == 'Darwin':
    _tree = ct.cdll.LoadLibrary(os.path.join(path, '_simple_tree.so'))
elif platform.system() == 'Windows':
    _tree = ct.pydll.LoadLibrary(os.path.join(path, '_simple_tree.pyd'))
else:
    raise SystemError('System not supported: {}'.format(platform.system()))

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


class SimpleTreeLearner(Orange.classification.base.Learner):

    def __init__(self, min_instances=2, max_depth=1024, max_majority=1.0,
                 skip_prob=0.0, bootstrap=False, seed=42):
        '''
        Parameters
        ----------

        min_instance : Minimal number of instances in leaves.  Instance count
        is weighed.

        max_depth : Maximal depth of tree.

        max_majority : Maximal proportion of majority class. When this is
        exceeded, induction stops (only used for classification).

        skip_prob : an attribute will be skipped with probability
        ``skip_prob``.

          - if float, then skip attribute with this probability.
          - if "sqrt", then `skip_prob = 1 - sqrt(n_features) / n_features`
          - if "log2", then `skip_prob = 1 - log2(n_features) / n_features`

        bootstrap : Bootstrap dataset.

        seed : Random seed.
        '''

        self.min_instances = min_instances
        self.max_depth = max_depth
        self.max_majority = max_majority
        self.skip_prob = skip_prob
        self.bootstrap = bootstrap
        self.seed = seed

    def fit_storage(self, data):
        return SimpleTreeModel(self, data)


class SimpleTreeModel(Orange.classification.base.Model):

    def __init__(self, learner, data):
        self.num_attrs = data.X.shape[1]

        if len(data.domain.class_vars) != 1:
            n_cls = len(data.domain.class_vars)
            raise ValueError("Number of classes should be 1: {}".format(n_cls))

        if isinstance(data.domain.class_var, Orange.data.DiscreteVariable):
            self.type = Classification
            self.cls_vals = len(data.domain.class_var.values)
        elif isinstance(data.domain.class_var, Orange.data.ContinuousVariable):
            self.type = Regression
            self.cls_vals = 0
        else:
            assert(False)

        if isinstance(learner.skip_prob, float):
            skip_prob = learner.skip_prob
        elif learner.skip_prob == 'sqrt':
            skip_prob = 1.0 - np.sqrt(data.X.shape[1]) / data.X.shape[1]
        elif learner.skip_prob == 'log2':
            skip_prob = 1.0 - np.log2(data.X.shape[1]) / data.X.shape[1]
        else:
            raise ValueError(
                "skip_prob not valid: {}".format(learner.skip_prob))

        attr_vals = []
        domain = []
        for attr in data.domain.attributes:
            if isinstance(attr, Orange.data.DiscreteVariable):
                attr_vals.append(len(attr.values))
                domain.append(IntVar)
            elif isinstance(attr, Orange.data.ContinuousVariable):
                attr_vals.append(0)
                domain.append(FloatVar)
            else:
                assert(False)
        attr_vals = np.array(attr_vals, dtype=np.int32)
        domain = np.array(domain, dtype=np.int32)

        self.node = _tree.build_tree(
            data.X.ctypes.data_as(c_double_p),
            data.Y.ctypes.data_as(c_double_p),
            data.W.ctypes.data_as(c_double_p),
            data.X.shape[0],
            data.W.size,
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
        if self.type == Classification:
            p = np.zeros((data.X.shape[0], self.cls_vals))
            _tree.predict_classification(
                data.X.ctypes.data_as(c_double_p),
                data.X.shape[0],
                self.node,
                self.num_attrs,
                self.cls_vals,
                p.ctypes.data_as(c_double_p))
            return p.argmax(axis=1), p
        elif self.type == Regression:
            p = np.zeros(data.X.shape[0])
            _tree.predict_regression(
                data.X.ctypes.data_as(c_double_p),
                data.X.shape[0],
                self.node,
                self.num_attrs,
                p.ctypes.data_as(c_double_p))
            return p
        else:
            assert(False)

    def __del__(self):
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

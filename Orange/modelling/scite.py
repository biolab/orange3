import ctypes
import numpy as np
import re

from collections import defaultdict
from Orange.tree import TreeModel, MappedDiscreteNode, Node
from Orange.data import DiscreteVariable, Domain, Table
from Orange.classification import Learner
from Orange.statistics.distribution import Discrete

from . import _scite
lib = ctypes.pydll.LoadLibrary(_scite.__file__)

# Data constants
TERMINAL_NODE = None

# Add a class variable
CLASS_VAR = "scite_class"
CLASS_ROOT = "root"


class SciteVariable(DiscreteVariable):

    MUT_NONE = "none"
    MUT_HET = "het"
    MUT_HOM = "hom"
    MUT_UNKNOWN = "-"
    __values__ = (MUT_NONE, MUT_HET, MUT_HOM, MUT_UNKNOWN)

    """ Pre-defined values. """
    def __init__(self, name):
        super().__init__(name=name, values=self.__values__)


def generate_scite_data(n, m, val=SciteVariable.MUT_HOM):
    """
    Generate SCITE data as a lower triangular matrix. Zeroes are treated as 'none'.
    :param n: Number of cells.
    :param m: Number of genes.
    :param val: Value.
    :return: Orange data Table.
    """
    X = np.zeros((n, m))
    domain = Domain([SciteVariable(name="g-%d" % i) for i in range(m)])
    for i in range(m):
        X[i:, i] = SciteVariable.__values__.index(val)
    data = Table.from_numpy(domain, X, Y=None, W=None)
    return data


class SciteTreeModel(TreeModel):
    """ Predicts a class from a SCITE tree structure."""
    name = 'SCITE'

    def __init__(self, data, root):
        super().__init__(data, root)

    def _get_prediction_index(self, x, node=None):
        """
        Return the index of a predicted value by traversing the tree until a terminal node is reached.
        :param x: Data matrix row.
        :param node: Current node.
        :return:
        """
        if len(node.children) == 0:
            return np.argmax(node.value)
        else:
            assert isinstance(node, MappedDiscreteNode)
            child = node.children[node.mapping[int(x[node.attr_idx])]]
            return self._get_prediction_index(x, child)

    def predict(self, X):
        """
        Traverse tree for each data instance.
        :param X: Data matrix with correctly transformed domain.
        :return: Predictive distribution.
        """
        predictions = np.zeros((len(X), len(self.domain[CLASS_VAR].values)))
        for i, x in enumerate(X):
            predictions[i, self._get_prediction_index(x, self.root)] = 1
        return predictions

    def get_values(self):
        pass

    def _compile(self):
        pass


class SciteTreeLearner(Learner):
    """ SCITE wrapper for single cell inference. Returns a SciteTreeModel from data."""

    name = 'scite'
    __returns__ = SciteTreeModel

    def __init__(self, rep=2, loops=200000, fd=6.04e-5, ad1=0.21545,
                 ad2=0.21545, cc=1.299164e-05, tree_type='m', score_type='m',
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.rep = int(rep)
        self.loops = int(loops)
        self.fd = float(fd)
        self.ad1 = float(ad1)
        self.ad2 = float(ad2)
        self.cc = float(cc)
        self.tree_type = tree_type
        self.score_type = score_type
        self.tree_dot = None

    @staticmethod
    def _expand_branches(nodes, tree, domain):
        """ Expand list of nodes to a binary subtree.
            Return MappedDiscreteNode.

            Order of values is MUT_NONE, MUT_HET, MUT_HOM, MUT_UNKNOWN.

            :param nodes: A list of node indices at the current level.
            :param tree: Tree structure dictionary node_index -> [child1, child2, ..., terminal]
            :param domain: Data domain.
        """
        idx = nodes[0]

        # If length of nodes is 1, this must be a terminal node
        if len(nodes) == 1:
            terminal = Node(None, None, None)
            return terminal

        # Normal intermediate node with only one non-leaf descendant
        # The 'no' branch gets terminal node; The 'yes' branch gets the *one* child of node;
        elif len(nodes) == 2:
            mapping = np.array([0, 1, 1, 1])
            no_branch = SciteTreeLearner._expand_branches([TERMINAL_NODE], tree, domain)
            yes_branch = SciteTreeLearner._expand_branches(tree[idx], tree, domain)
            links = [no_branch, yes_branch]

        # Node with multiple descendants gets expanded to a binary tree; this node is non-terminal
        # The 'no' branch gets brother of node; The 'yes' branch gets *two or more *nodes of node;
        else:
            mapping = np.array([0, 1, 1, 0])
            no_branch = SciteTreeLearner._expand_branches(nodes[1:], tree, domain)
            yes_branch = SciteTreeLearner._expand_branches(tree[idx], tree, domain)
            links = [no_branch, yes_branch]

        node = MappedDiscreteNode(attr_idx=idx, attr=domain.attributes[idx], mapping=mapping, value=None)
        node.children = links
        return node

    @ staticmethod
    def _populate_tree(node, data, subset=None, last_parent=None):
        """
        Fill the tree with instances based on attributes.
        :param node: Tree node (starts with root).
        :param data: All data (stays fixed).
        :param subset: Subset of data indices that is passed recursively.
        :param last_parent: Last positive attribute. Passed around to determine the class value.
        :return:
            node with updated subset.
            data with updated class values.
        """
        if subset is None:
            subset = np.arange(0, len(data), dtype=np.int32)
        node.subset = subset

        if not len(node.children):
            # Reached a terminal node ; fill class values in data ; set distribution.
            if last_parent is None:
                data[subset, CLASS_VAR] = CLASS_ROOT
            else:
                data[subset, CLASS_VAR] = last_parent.attr.name
            dist = Discrete.from_data(data=data[subset], variable=data.domain[CLASS_VAR])

            if not len(subset):
                if last_parent is not None:
                    # Artificially add a pseudo count of 1
                    dist[last_parent.attr.name] = 1
                else:
                    dist[CLASS_ROOT] = 1
            node.value = dist
            return node, data

        assert isinstance(node, MappedDiscreteNode)
        assert len(node.children) == 2
        for i in range(len(node.children)):

            # Which values of the attribute are relevant for current child ; might be empty
            # To comply with all the upstream conditions, intersection with the subset is taken
            value_ids = np.where(node.mapping == i)[0]
            if len(value_ids):
                value_ids.reshape((1, len(value_ids)))
                test = np.sum(data.X[:, node.attr_idx].reshape((len(data), 1)) == value_ids, axis=1)
                child_subset = np.array(sorted(set(subset) & set(np.where(test)[0])), dtype=np.int32)
            else:
                child_subset = np.array([], dtype=np.int32)

            last_parent = node if i == 1 else last_parent
            node.children[i], data = SciteTreeLearner._populate_tree(node=node.children[i],
                                                                     data=data,
                                                                     subset=child_subset,
                                                                     last_parent=last_parent)

        # Sum the distributions of the children
        node.value = node.children[0].value + node.children[1].value
        return node, data

    @staticmethod
    def _dot_to_node(tree_dot, data):
        """ GraphViz (.dot) structure to Orange Node.
            Parse SCITE indices to correct instances and attributes.

            Nodes are counted from 1, instances from 0.

            There is an dummy root added node by SCITE, which points to the first attribute
            The attribute to which a connection points to is actually the condition in the root
            If attribute points to multiple attributes, it gets collapsed into a binary structure.
        """

        def get_node_pair(s):
            t = s.replace("s", "").split(" -> ")
            return int(t[0]) - 1, int(t[1]) - 1

        node_pat = "[0-9]+ -> [0-9]+"
        nodes = list(map(get_node_pair, re.findall(node_pat, tree_dot)))

        # Important: Terminal node marker is last in the list;
        root_idx = len(data.domain.attributes)
        tree = defaultdict(lambda: [TERMINAL_NODE, ])
        for parent, child in nodes:
            tree[parent] = [child] + tree[parent]
            tree[child]

        root = SciteTreeLearner._expand_branches(nodes=tree[root_idx],
                                                 tree=tree,
                                                 domain=data.domain)
        root, data = SciteTreeLearner._populate_tree(node=root,
                                                     data=data,
                                                     subset=None)
        return root, data

    def fit(self, X, Y, W=None):
        """ Wrap provided C++ function."""

        n_cells, n_genes = X.shape

        # Note: The output string buffer can cause problems if too small.
        n_bytes = (n_cells+1) * 100 + (n_genes + 1) * 100
        trees = ctypes.create_string_buffer(n_bytes)

        fun = lib.scite_export
        fun.restype = None
        fun.argtypes = [np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),     # X
                        ctypes.c_int,       # n_cells
                        ctypes.c_int,       # n_genes
                        ctypes.c_int,       # rep
                        ctypes.c_int,       # loops
                        ctypes.c_double,    # fd
                        ctypes.c_double,    # ad1
                        ctypes.c_double,    # ad2
                        ctypes.c_double,    # cc
                        ctypes.c_char,      # tree_type
                        ctypes.c_char,      # score_type
                        ctypes.POINTER(ctypes.c_char)   # trees (output)
                        ]
        fun(X, n_cells, n_genes, self.rep, self.loops, self.fd, self.ad1, self.ad2, self.cc,
            ord(self.tree_type), ord(self.score_type), trees)
        self.tree_dot = ctypes.cast(trees, ctypes.c_char_p).value.decode()

    def __call__(self, data):
        """
        Fit a SCITE tree model and add class to data. The class is equal to the terminal node.
        :param data: Orange data Table with cells in rows and genes in columns
        :return:
        """
        # Transfer data to C++ module and get tree structure
        X = np.ascontiguousarray(data.X, dtype=np.int32)
        self.fit(X, Y=None, W=None)

        # Add a class to the domain
        self.n_classes = data.X.shape[1] + 1
        assert CLASS_ROOT not in data.domain
        class_var = DiscreteVariable(name=CLASS_VAR,
                                     values=[CLASS_ROOT] + [att.name for att in data.domain.attributes])
        new_domain = Domain(attributes=data.domain.attributes,
                            class_vars=[class_var],
                            metas=data.domain.metas)
        new_data = data.transform(new_domain)

        # Construct a tree structure and a model
        root, data = SciteTreeLearner._dot_to_node(tree_dot=self.tree_dot, data=new_data)
        return SciteTreeModel(root=root, data=data)

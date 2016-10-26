"""Base tree adapter class with common methods needed for visualisations."""
from abc import ABCMeta, abstractmethod


class TreeAdapter(metaclass=ABCMeta):
    """Base class for tree representation.

    Any subclass should implement the methods listed in this base class. Note
    that some simple methods do not need to reimplemented e.g. is_leaf since
    it that is the opposite of has_children.

    """

    ROOT_PARENT = -1
    NO_CHILD = -1
    FEATURE_UNDEFINED = -2

    @abstractmethod
    def weight(self, node):
        """Get the weight of the given node.

        The weights of the children always sum up to 1.

        Parameters
        ----------
        node : object
            The label of the node.

        Returns
        -------
        float
            The weight of the node relative to its siblings.

        """
        pass

    @abstractmethod
    def num_samples(self, node):
        """Get the number of samples that a given node contains.

        Parameters
        ----------
        node : object
            A unique identifier of a node.

        Returns
        -------
        int

        """
        pass

    @abstractmethod
    def parent(self, node):
        """Get the parent of a given node. Return -1 if the node is the root.

        Parameters
        ----------
        node : object

        Returns
        -------
        object

        """
        pass

    @abstractmethod
    def has_children(self, node):
        """Check if the given node has any children.

        Parameters
        ----------
        node : object

        Returns
        -------
        bool

        """
        pass

    def is_leaf(self, node):
        """Check if the given node is a leaf node.

        Parameters
        ----------
        node : object

        Returns
        -------
        object

        """
        return not self.has_children(node)

    @abstractmethod
    def children(self, node):
        """Get all the children of a given node.

        Parameters
        ----------
        node : object

        Returns
        -------
        Iterable[object]
            A iterable object containing the labels of the child nodes.

        """
        pass

    @abstractmethod
    def get_distribution(self, node):
        """Get the distribution of types for a given node.

        This may be the number of nodes that belong to each different classe in
        a node.

        Parameters
        ----------
        node : object

        Returns
        -------
        Iterable[int, ...]
            The return type is an iterable with as many fields as there are
            different classes in the given node. The values of the fields are
            the number of nodes that belong to a given class inside the node.

        """
        pass

    @abstractmethod
    def get_impurity(self, node):
        """Get the impurity of a given node.

        Parameters
        ----------
        node : object

        Returns
        -------
        object

        """
        pass

    @abstractmethod
    def rules(self, node):
        """Get a list of rules that define the given node.

        Parameters
        ----------
        node : object

        Returns
        -------
        Iterable[Rule]
            A list of Rule objects, can be of any type.

        """
        pass

    @abstractmethod
    def attribute(self, node):
        """Get the attribute that splits the given tree.

        Parameters
        ----------
        node

        Returns
        -------

        """
        pass

    def is_root(self, node):
        """Check if a given node is the root node.

        Parameters
        ----------
        node

        Returns
        -------

        """
        return node == self.root

    @abstractmethod
    def leaves(self, node):
        """Get all the leavse that belong to the subtree of a given node.

        Parameters
        ----------
        node

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_instances_in_nodes(self, dataset, nodes):
        """Get all the instances belonging to a set of nodes for a given
        dataset.

        Parameters
        ----------
        dataset : Table
            A Orange Table dataset.
        nodes : iterable[TreeNode]
            A list of tree nodes for which we want the instances.

        Returns
        -------

        """
        pass

    @property
    @abstractmethod
    def max_depth(self):
        """Get the maximum depth that the tree reaches.

        Returns
        -------
        int

        """
        pass

    @property
    @abstractmethod
    def num_nodes(self):
        """Get the total number of nodes that the tree contains.

        This does not mean the number of samples inside the entire tree, just
        the number of nodes.

        Returns
        -------
        int

        """
        pass

    @property
    @abstractmethod
    def root(self):
        """Get the label of the root node.

        Returns
        -------
        object

        """
        pass

    @property
    @abstractmethod
    def domain(self):
        """Get the domain of the given tree.

        The domain contains information about the classes what the tree
        represents.

        Returns
        -------

        """
        pass

import unittest
import sys

from Orange.clustering.hierarchical import Tree


class TestTree(unittest.TestCase):
    def test_repr(self):
        tree = Tree(2)
        self.assertEqual(repr(tree), "Tree(value=2, branches=())")

        tree2 = Tree(3, (Tree(2), Tree(1)))
        self.assertEqual(repr(tree2), "Tree(value=3, branches=(Tree(value=2, branches=()), Tree(value=1, branches=())))")

        for i in range(100):
            tree = Tree(i, (tree, tree))

        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(30)
            repr(tree)  # don't care about the result, just that it doesn't crash
        finally:
            sys.setrecursionlimit(rec_limit)


if __name__ == "__main__":
    unittest.main()
import Orange
from Orange.classification.simple_tree import SimpleTreeLearner

data = Orange.data.Table("housing.tab")
tree_learner = SimpleTreeLearner(max_depth=2)
tree = tree_learner(data)
print(tree.to_string())

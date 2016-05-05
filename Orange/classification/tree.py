import sklearn.tree as skl_tree
from Orange.classification import SklLearner, SklModel
from Orange.preprocess import (RemoveNaNClasses, Continuize,
                               RemoveNaNColumns, SklImpute)
from Orange import options

__all__ = ["TreeLearner"]


class TreeClassifier(SklModel):
    pass


class TreeLearner(SklLearner):
    __wraps__ = skl_tree.DecisionTreeClassifier
    __returns__ = TreeClassifier
    name = 'tree'
    preprocessors = [RemoveNaNClasses(),
                     RemoveNaNColumns(),
                     SklImpute(),
                     Continuize()]

    CRITERIONS = (('gini', 'Gini impurity'), ('entropy', 'Entropy'))

    options = (
        options.ChoiceOption('criterion', choices=CRITERIONS),
        options.ChoiceOption(
            'max_features', choices=('auto', 'sqrt', 'log2', .25, .5, .75),
            help_text='The number of features to consider when looking for the best split.'
        ),
        options.DisableableOption(
            'max_depth', disable_value=None, disable_label='No limit',
            option=options.IntegerOption(default=1000, range=(1, 1000), step=5),
            help_text="The maximum depth of the tree",
        ),
        options.IntegerOption(
            'min_samples_split', default=2, range=(1, 1000), step=5,
            help_text='The minimum number of samples required to split an internal node'
        ),
        options.IntegerOption(
            'min_samples_leaf', default=1, range=(1, 1000), step=5,
            help_text='The minimum number of samples required to be at a leaf node'
        ),
        options.DisableableOption(
            'max_leaf_nodes', disable_value=None, disable_label='No limit',
            option=options.IntegerOption(default=1000, range=(1, 1000), step=5),
            help_text="The maximum number of leaf nodes",
        ),
        options.DisableableOption(
            'random_state', disable_value=None,
            option=options.IntegerOption('random_state')
        )
    )

    class GUI:
        main_scheme = (
            'criterion',
            options.OptionGroup('Tree', ['max_features', 'max_depth', 'min_samples_split',
                                         'min_samples_leaf', 'max_leaf_nodes']),
            'random_state',
        )

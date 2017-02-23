"""General tree learner base widget, and classification tree widget"""
from Orange.data import Table
from Orange.modelling.tree import TreeLearner
from Orange.classification.tree import TreeLearner as ClassificationTreeLearner
from Orange.widgets.model.owtree import OWTreeLearner
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Msg


class OWTreeLearner(OWTreeLearner):
    """Classification tree algorithm with forward pruning."""

    name = "Classification Tree"
    icon = "icons/ClassificationTree.svg"
    priority = 30

    LEARNER = TreeLearner

    spin_boxes = \
        OWTreeLearner.spin_boxes[:-1] + \
        (("Stop when majority reaches [%]: ",
          "limit_majority", "sufficient_majority", 51, 100),) + \
        OWTreeLearner.spin_boxes[-1:]

    replaces = ['Orange.widgets.classify.owclassificationtree.OWClassificationTree']


    class Error(OWTreeLearner.Error):
        cannot_binarize = Msg("Binarization cannot handle '{}'\n"
                              "because it has {} values. "
                              "Binarization can handle up to {}.\n"
                              "Disable 'Induce binary tree' to proceed.")

    def check_data(self):
        self.Error.cannot_binarize.clear()
        if not super().check_data():
            return False
        if not self.binary_trees:
            return True
        max_values, max_attr = max(
            ((len(attr.values), attr.name)
             for attr in self.data.domain.attributes if attr.is_discrete),
            default=(0, None))
        MAX_BINARIZATION = ClassificationTreeLearner.MAX_BINARIZATION
        if max_values > MAX_BINARIZATION:
            self.Error.cannot_binarize(
                max_attr, max_values, MAX_BINARIZATION)
            return False
        return True

    def learner_kwargs(self):
        opts = super().learner_kwargs()
        opts['sufficient_majority'] = \
            (1, self.sufficient_majority / 100)[self.limit_majority]
        return opts

    def get_learner_parameters(self):
        items = super().get_learner_parameters()
        if self.limit_majority:
            items["Pruning"] = ", " * bool(items["Pruning"]) + \
                "stop splitting when the majority class reaches {} %".format(
                    self.sufficient_majority)

    # Disable the special classification layout to be used when widgets are
    # fully merged
    add_classification_layout = OWBaseLearner.add_classification_layout


def _test():
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWTreeLearner()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

if __name__ == "__main__":
    _test()

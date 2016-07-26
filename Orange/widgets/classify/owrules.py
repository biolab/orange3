from PyQt4.QtCore import Qt

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.classification import CN2Learner, CN2SDLearner, CN2UnorderedLearner, CN2SDUnorderedLearner
from Orange.classification.rules import _RuleLearner, _RuleClassifier
from Orange.classification.rules import EntropyEvaluator, LaplaceAccuracyEvaluator, WeightedRelativeAccuracyEvaluator


class CustomRuleLearner(_RuleLearner):
    name = 'Custom rule inducer'

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.quality_evaluator = EntropyEvaluator()

    def fit(self, X, Y, W=None):
        Y = Y.astype(dtype=int)
        rule_list = self.find_rules(X, Y, W, None, self.base_rules,
                                    self.domain)
        # add the default rule, if required
        if not rule_list or rule_list and rule_list[-1].length > 0:
            rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
        return CustomRuleClassifier(domain=self.domain, rule_list=rule_list)


class CustomRuleClassifier(_RuleClassifier):
    pass


class OWRuleLearner(OWBaseLearner):
    name = "Rule Induction"
    description = ""
    icon = ""
    priority = 19

    LEARNER = CN2Learner

    learner_name = Setting("CN2 inducer")

    rule_ordering = Setting(0)
    covering_algorithm = Setting(0)
    gamma = Setting(0.7)
    beam_width = Setting(5)
    evaluation_function = Setting("Smth")
    discretize_continuous = Setting(0)
    min_covered_examples = Setting(1)
    max_rule_length = Setting(5)
    default_alpha = Setting(1.0)
    parent_alpha = Setting(1.0)
    checked_default_alpha = Setting(True)
    checked_parent_alpha = Setting(False)

    want_main_area = False
    resizing_enabled = False

    def add_main_layout(self):

        # top-level control procedure
        top_box = gui.hBox(widget=self.controlArea, box=None, addSpace=2)

        rule_ordering_box = gui.hBox(widget=top_box, box="Rule ordering")
        rule_ordering_rbs = gui.radioButtons(
            widget=rule_ordering_box, master=self, value="rule_ordering",
            callback=self.print_smth, btnLabels=("Ordered", "Unordered"))
        rule_ordering_rbs.layout().setSpacing(7)

        covering_algorithm_box = gui.hBox(
            widget=top_box, box="Covering algorithm")
        covering_algorithm_rbs = gui.radioButtons(
            widget=covering_algorithm_box, master=self,
            value="covering_algorithm", callback=self.print_smth,
            btnLabels=("Exclusive", "Weighted"))
        covering_algorithm_rbs.layout().setSpacing(7)

        insert_gamma_box = gui.vBox(widget=covering_algorithm_box, box=None)
        gui.separator(insert_gamma_box, 0, 14)
        gamma_spin = gui.doubleSpin(
            widget=insert_gamma_box, master=self, value="gamma", minv=0.0,
            maxv=1.0, step=0.01, label="γ:", orientation=Qt.Horizontal,
            callback=self.print_smth, alignment=Qt.AlignRight)

        # search bias
        middle_box = gui.vBox(widget=self.controlArea, box="Rule search")

        evaluation_function_box = gui.comboBox(
            widget=middle_box, master=self, value="evaluation_function",
            label="Evaluation measure:", orientation=Qt.Horizontal,
            items=("Smth", "Entropy", "Laplace accuracy"),
            callback=self.print_smth,
            contentsLength=4)

        beam_width_box = gui.spin(
            widget=middle_box, master=self, value="beam_width", minv=1,
            maxv=100, step=1, label="Beam width:", orientation=Qt.Horizontal,
            callback=self.print_smth, alignment=Qt.AlignRight, controlWidth=80)

        # overfitting avoidance bias
        bottom_box = gui.vBox(widget=self.controlArea, box="Rule filtering")
        min_covered_examples_box = gui.spin(
            widget=bottom_box, master=self, value="min_covered_examples", minv=1,
            maxv=100, step=1, label="Minimum rule coverage:",
            orientation=Qt.Horizontal, callback=self.print_smth,
            alignment=Qt.AlignRight, controlWidth=80)

        max_rule_length_box = gui.spin(
            widget=bottom_box, master=self, value="max_rule_length",
            minv=1, maxv=100, step=1, label="Maximum rule length:",
            orientation=Qt.Horizontal, callback=self.print_smth,
            alignment=Qt.AlignRight, controlWidth=80)

        default_alpha_spin = gui.doubleSpin(
            widget=bottom_box, master=self, value="default_alpha", minv=0.0,
            maxv=1.0, step=0.01, label="Statistical signifiance\n(default α):",
            orientation=Qt.Horizontal, callback=self.print_smth,
            alignment=Qt.AlignRight, controlWidth=80,
            checked="checked_default_alpha")

        parent_alpha_spin = gui.doubleSpin(
            widget=bottom_box, master=self, value="parent_alpha", minv=0.0,
            maxv=1.0, step=0.01, label="Relative signifiance\n(parent α):",
            orientation=Qt.Horizontal, callback=self.print_smth,
            alignment=Qt.AlignRight, controlWidth=80,
            checked="checked_parent_alpha")

        # predef_algs_box = gui.hBox(widget=self.controlArea, box="Predefined algorithms")
        # cn2_button = gui.toolButton(widget=predef_algs_box, master=self,
        #                             label="CN2")
        #
        # cn2_button = gui.toolButton(widget=predef_algs_box, master=self,
        #                             label="CN2 Unordered")
        #
        # cn2sd_button = gui.toolButton(widget=predef_algs_box, master=self,
        #                               label="CN2-SD")
        #
        # cn2_button = gui.toolButton(widget=predef_algs_box, master=self,
        #                             label="CN2-SD Unordered")

    def print_smth(self):
        pass

    def create_learner(self):
        raise NotImplementedError

    def get_learner_parameters(self):
        raise NotImplementedError

if __name__ == "__main__":
    import sys
    from PyQt4.QtGui import QApplication

    a = QApplication(sys.argv)
    ow = OWRuleLearner()
    d = Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

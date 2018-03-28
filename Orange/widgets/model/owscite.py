"""Tree learner widget"""

from AnyQt.QtCore import Qt

from Orange.modelling.scite import SciteTreeLearner, generate_scite_data
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner


class OWSciteLearner(OWBaseLearner):
    """Tree algorithm with forward pruning."""
    name = "SCITE"
    description = "A mutation tree inference algorithm based on single cell data."
    icon = "icons/Tree.svg"
    priority = 30

    LEARNER = SciteTreeLearner

    SCORE_M = "Best attachment per sample"
    SCORE_S = "Sum over all attachments"
    SCORE_TYPES = [SCORE_M, SCORE_S]
    SCORE_MAP = {0: "m", 1: "s"}

    repeats = Setting(2)
    loops = Setting(200000)
    false_discovery = Setting(6.04e-5)
    allele_dropout1 = Setting(0.21545)
    allele_dropout2 = Setting(0.21545)
    homozygous_false_discovery = Setting(1.299164e-05)
    score_type = Setting(0)

    set_repeats = Setting(True)
    set_loops = Setting(True)
    set_false_discovery = Setting(True)
    set_allele_dropout1 = Setting(True)
    set_allele_dropout2 = Setting(True)
    set_homozygous_false_discovery = Setting(True)

    spin_boxes = (
        ("MCMC Repeats: ", "set_repeats", "repeats", 1, 10, 1),
        ("MCMC Iterations: ", "set_loops", "loops", 100000, 999999, 10000),

        ("FD rate: ", "set_false_discovery", "false_discovery", 0.0, 1.0, 0.000001),
        ("Homozygous FD rate: ", "set_homozygous_false_discovery", "homozygous_false_discovery", 0.0, 1.0, 0.000001),
        ("Allellic dropout (1): ", "set_allele_dropout1", "allele_dropout1", 0.0, 1.0, 0.01),
        ("Allellic dropout (2): ", "set_allele_dropout2", "allele_dropout2", 0.0, 1.0, 0.01),
    )

    def add_main_layout(self):

        # the checkbox is put into vBox for alignemnt with other checkboxes

        # Decomposition
        box0 = gui.widgetBox(self.controlArea, 'Score type')
        gui.radioButtons(box0, self, "score_type", self.SCORE_TYPES,
                         callback=self.settings_changed)

        box1 = gui.widgetBox(self.controlArea, 'Parameters')
        for label, check, setting, fromv, tov, step in self.spin_boxes:
            gui.doubleSpin(box1, self, setting, fromv, tov, label=label,
                     checked=check, alignment=Qt.AlignRight,
                     callback=self.settings_changed, step=step,
                     checkCallback=self.settings_changed, controlWidth=80)

    def learner_kwargs(self):
        # Pylint doesn't get our Settings
        # pylint: disable=invalid-sequence-index
        return dict(rep=self.repeats, loops=self.loops, ad1=self.allele_dropout1, ad2=self.allele_dropout2,
                    cc=self.homozygous_false_discovery, fd=self.false_discovery,
                    score_type=self.SCORE_MAP[self.score_type])

    def create_learner(self):
        # pylint: disable=not-callable
        return self.LEARNER(**self.learner_kwargs())

    def set_data(self, data):
        """Set the input train data set."""
        self.Error.data_error.clear()
        self.data = data
        self.update_model()

    def update_model(self):
        self.show_fitting_failed(None)
        try:
            self.model = self.learner(self.data)
        except BaseException as exc:
            self.show_fitting_failed(exc)
        self.Outputs.model.send(self.model)


def main():
    import sys
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    ow = OWSciteLearner()
    data = generate_scite_data(10, 10)
    ow.set_data(data)
    ow.show()
    a.exec_()
    ow.saveSettings()


if __name__ == "__main__":
    main()

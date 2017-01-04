from AnyQt.QtCore import Qt

from Orange.widgets import gui
from Orange.widgets.model.owsgd import OWSGD
from Orange.widgets.settings import Setting


class OWSGD(OWSGD):
    learner_name = Setting('SGD Classification')

    def _add_algorithm_to_layout(self):
        box = gui.widgetBox(self.controlArea, 'Algorithm')
        # Classfication loss function
        self.cls_loss_function_combo = gui.comboBox(
            box, self, 'cls_loss_function_index', orientation=Qt.Horizontal,
            label='Loss function: ',
            items=list(zip(*self.cls_losses))[0],
            callback=self._on_cls_loss_change)
        param_box = gui.hBox(box)
        gui.rubber(param_box)
        self.cls_epsilon_spin = gui.spin(
            param_box, self, 'cls_epsilon', 0, 1., 1e-2, spinType=float,
            label='ε: ', controlWidth=80, alignment=Qt.AlignRight,
            callback=self.settings_changed)

        # Enable/disable appropriate controls
        self._on_cls_loss_change()


if __name__ == '__main__':
    import sys
    from AnyQt.QtWidgets import QApplication
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWSGD()
    ow.resetSettings()
    d = Table(sys.argv[1] if len(sys.argv) > 1 else 'iris')
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

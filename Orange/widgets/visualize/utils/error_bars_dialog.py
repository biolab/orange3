import sys
from typing import Optional

from AnyQt.QtCore import Signal, Qt
from AnyQt.QtWidgets import QVBoxLayout, QWidget, QComboBox, \
    QFormLayout, QLabel, QButtonGroup, QRadioButton, QLayout

from Orange.data import ContinuousVariable, Domain
from Orange.widgets.utils import disconnected
from Orange.widgets.utils.itemmodels import DomainModel


class ErrorBarsDialog(QWidget):
    changed = Signal()

    def __init__(
            self,
            parent: QWidget,
    ):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.Popup)
        self.hide()
        self.__model = DomainModel(
            separators=False,
            valid_types=(ContinuousVariable,),
            placeholder="(None)"
        )

        self.__upper_combo = upper_combo = QComboBox()
        upper_combo.setMinimumWidth(200)
        upper_combo.setModel(self.__model)
        upper_combo.currentIndexChanged.connect(self.changed)

        self.__lower_combo = lower_combo = QComboBox()
        lower_combo.setMinimumWidth(200)
        lower_combo.setModel(self.__model)
        lower_combo.currentIndexChanged.connect(self.changed)

        button_diff = QRadioButton("Difference from plotted value",
                                   checked=True)
        button_abs = QRadioButton("Absolute position on the plot")
        self.__radio_buttons = QButtonGroup()
        self.__radio_buttons.addButton(button_diff, 0)
        self.__radio_buttons.addButton(button_abs, 1)
        self.__radio_buttons.buttonClicked.connect(self.changed)

        form = QFormLayout()
        form.addRow(QLabel("Upper:"), upper_combo)
        form.addRow(QLabel("Lower:"), lower_combo)
        form.setVerticalSpacing(10)
        form.addRow(button_diff)
        form.addRow(button_abs)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(form)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

    def get_data(self) -> tuple[
        Optional[ContinuousVariable], Optional[ContinuousVariable], bool
    ]:
        upper_var, lower_var = None, None
        if self.__model:
            upper_var = self.__model[self.__upper_combo.currentIndex()]
            lower_var = self.__model[self.__lower_combo.currentIndex()]
        return upper_var, lower_var, bool(self.__radio_buttons.checkedId())

    def show_dlg(
            self,
            domain: Domain,
            x: int, y: int,
            attr_upper: Optional[ContinuousVariable] = None,
            attr_lower: Optional[ContinuousVariable] = None,
            is_abs: bool = True
    ):
        self._set_data(domain, attr_upper, attr_lower, is_abs)
        self.show()
        self.raise_()
        self.move(x, y)
        self.activateWindow()

    def _set_data(
            self,
            domain: Domain,
            upper_attr: Optional[ContinuousVariable],
            lower_attr: Optional[ContinuousVariable],
            is_abs: bool
    ):
        upper_combo, lower_combo = self.__upper_combo, self.__lower_combo
        with disconnected(upper_combo.currentIndexChanged, self.changed):
            with disconnected(lower_combo.currentIndexChanged, self.changed):
                self.__model.set_domain(domain)
                upper_combo.setCurrentIndex(self.__model.indexOf(upper_attr))
                lower_combo.setCurrentIndex(self.__model.indexOf(lower_attr))
        self.__radio_buttons.buttons()[int(is_abs)].setChecked(True)


if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from AnyQt.QtWidgets import QApplication, QPushButton

    from Orange.data import Table

    app = QApplication(sys.argv)
    w = QWidget()
    w.setFixedSize(400, 200)

    dlg = ErrorBarsDialog(w)
    dlg.changed.connect(lambda: print(dlg.get_data()))

    btn = QPushButton(w)
    btn.setText("Open")

    _domain: Domain = Table("iris").domain


    def _on_click():
        dlg.show_dlg(_domain, 500, 500, _domain.attributes[2],
                     _domain.attributes[3], is_abs=False)


    btn.clicked.connect(_on_click)

    w.show()
    sys.exit(app.exec())

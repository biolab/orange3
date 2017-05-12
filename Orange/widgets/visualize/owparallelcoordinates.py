from AnyQt.QtCore import Qt, QT_VERSION_STR

from Orange.data import Table
from Orange.widgets.widget import AttributeList
from Orange.widgets import gui, widget, settings


if QT_VERSION_STR < '5.3':
    raise RuntimeError('Parallel Coordinates requires Qt >= 5.3')


class OWParallelCoordinates(widget.OWWidget):
    name = "Parallel Coordinates"
    description = "Parallel coordinates display of multi-dimensional data."
    icon = "icons/ParallelCoordinates.svg"
    priority = 900
    inputs = [("Data", Table, 'set_data', widget.Default),
              ("Data Subset", Table, 'set_subset_data'),
              ("Features", AttributeList, 'set_shown_attributes')]
    outputs = [("Selected Data", Table, widget.Default),
               ("Other Data", Table),
               ("Features", AttributeList)]


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    ow = OWParallelCoordinates()
    ow.show()
    data = Table("iris")
    ow.set_data(data)
    ow.handleNewSignals()

    a.exec_()

    ow.saveSettings()

import copy

from PyQt4.QtGui import QFormLayout, QColor, QApplication
from PyQt4.QtCore import Qt

import numpy
import pyqtgraph as pg
import sklearn.decomposition

import Orange.data
from Orange.widgets import widget, gui, settings


class OWPCA(widget.OWWidget):
    name = "PCA"
    description = "Principal component analysis"
    icon = "icons/PCA.svg"
    priority = 3050

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Transformed data", Orange.data.Table),
               ("Components", Orange.data.Table)]
    max_components = settings.Setting(0)
    variance_covered = settings.Setting(100)
    auto_commit = settings.Setting(True)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self._invalidated = False
        self._pca = None
        self._variance_ratio = None
        self._cumulative = None
        self._line = False

        box = gui.widgetBox(self.controlArea, "Components Selection")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.components_spin = gui.spin(
            box, self, "max_components", 0, 1000,
            callback=self._update_selection,
            keyboardTracking=False
        )
        self.components_spin.setSpecialValueText("All")

        self.variance_spin = gui.spin(
            box, self, "variance_covered", 1, 100,
            callback=self._update_selection,
            keyboardTracking=False
        )
        self.variance_spin.setSuffix("%")

        form.addRow("Max components", self.components_spin)
        form.addRow("Variance covered", self.variance_spin)

        self.controlArea.layout().addStretch()

        box = gui.widgetBox(self.controlArea, "Commit")
        cb = gui.checkBox(box, self, "auto_commit", "Commit on any change")
        b = gui.button(box, self, "Commit", callback=self.commit, default=True)
        gui.setStopper(self, b, cb, "_invalidated", callback=self.commit)

        self.plot = pg.PlotWidget(background="w")

        axis = self.plot.getAxis("bottom")
        axis.setLabel("Principal Components")
        axis = self.plot.getAxis("left")
        axis.setLabel("Proportion of variance")

        self.plot.getViewBox().setMenuEnabled(False)
        self.plot.showGrid(True, True, alpha=0.5)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))

        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, data):
        self.clear()
        self.data = data

        if data is not None:
            pca = sklearn.decomposition.PCA()
            self._pca = pca.fit(self.data.X)
            self._variance_ratio = self._pca.explained_variance_ratio_
            self._cumulative = numpy.cumsum(self._variance_ratio)
            self.components_spin.setRange(0, len(self._cumulative))
            self._setup_plot()

        self.commit()

    def clear(self):
        self.data = None
        self._pca = None
        self._variance_ratio = None
        self._cumulative = None
        self._line = None
        self.plot.clear()

    def _setup_plot(self):
        explained_ratio = self._variance_ratio
        explained = self._cumulative
        (p, ) = explained.shape

        self.plot.plot(numpy.arange(p), explained_ratio,
                       pen=pg.mkPen(QColor(Qt.red), width=2),
                       antialias=True,
                       name="Variance")
        self.plot.plot(numpy.arange(p), explained,
                       pen=pg.mkPen(QColor(Qt.darkYellow), width=2),
                       antialias=True,
                       name="Cumulative Variance")

        self._line = pg.InfiniteLine(
            angle=90, pos=1, movable=True, bounds=(0, p - 1))
        self._line.setCursor(Qt.SizeHorCursor)
        self._line.setPen(pg.mkPen(QColor(Qt.darkGray), width=1.5))
        self._line.sigPositionChanged.connect(self._on_cut_changed)

        self.plot.addItem(self._line)
        self.plot.setRange(xRange=(0.0, p - 1), yRange=(0.0, 1.0))
        axis = self.plot.getAxis("bottom")
        axis.setTicks([[(i, "C{}".format(i + 1)) for i in range(p)]])

    def _on_cut_changed(self, line):
        value = line.value()
        current = self._nselected_components()
        components = int(numpy.floor(value)) + 1

        if not (self.max_components == 0 and \
                components == len(self._variance_ratio)):
            self.max_components = components

        if self._pca is not None:
            self.variance_covered = self._cumulative[components - 1] * 100

        if current != self._nselected_components():
            self._invalidate_selection()

    def _update_selection(self):
        if self._pca is None:
            return

        cut = self._nselected_components()
        if numpy.floor(self._line.value()) != cut:
            self._line.setValue(cut)

        self._invalidate_selection()

    def _nselected_components(self):
        if self._pca is None:
            return 0

        if self.max_components == 0:
            # Special "All" value
            max_comp = len(self._variance_ratio)
        else:
            max_comp = self.max_components

        var_max = self._cumulative[max_comp - 1]
        if var_max < self.variance_covered:
            cut = max_comp
        else:
            cut = numpy.searchsorted(
                self._cumulative, self.variance_covered / 100.0
            )
        return cut

    def _invalidate_selection(self):
        self._invalidated = True
        if self.auto_commit:
            self.commit()

    def commit(self):
        self._invalidated = False

        transformed = components = None
        if self._pca is not None:
            components = self._pca.components_
            ncomponents = self._nselected_components()
            pca = copy.copy(self._pca)
            pca.components_ = components[:ncomponents]
            transformed = pca.transform(self.data.X)
            transformed = Orange.data.Table(transformed)
            features = [Orange.data.ContinuousVariable("C%i" % (i + 1))
                        for i in range(components.shape[1])]
            domain = Orange.data.Domain(features)
            components = Orange.data.Table.from_numpy(domain, components)

        self.send("Transformed data", transformed)
        self.send("Components", components)


def main():
    import gc
    app = QApplication([])
    w = OWPCA()
#     data = Orange.data.Table("iris.tab")
    data = Orange.data.Table("housing.tab")
#     data = Orange.data.Table("wine.tab")
    w.set_data(data)
    w.show()
    w.raise_()
    rval = w.exec()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval

if __name__ == "__main__":
    import sys
    sys.exit(main())

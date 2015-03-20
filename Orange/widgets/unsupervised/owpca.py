
from PyQt4.QtGui import QFormLayout, QColor, QApplication
from PyQt4.QtCore import Qt

import numpy
import pyqtgraph as pg

import Orange.data
import Orange.projection
from Orange.widgets import widget, gui, settings


class OWPCA(widget.OWWidget):
    name = "PCA"
    description = "Principal component analysis"
    icon = "icons/PCA.svg"
    priority = 3050

    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Transformed data", Orange.data.Table),
               ("Components", Orange.data.Table)]
    ncomponents = settings.Setting(0)
    variance_covered = settings.Setting(100)
    auto_commit = settings.Setting(True)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self._pca = None
        self._transformed = None
        self._variance_ratio = None
        self._cumulative = None
        self._line = False

        box = gui.widgetBox(self.controlArea, "Components Selection")
        form = QFormLayout()
        box.layout().addLayout(form)

        self.components_spin = gui.spin(
            box, self, "ncomponents", 0, 1000,
            callback=self._update_selection_component_spin,
            keyboardTracking=False
        )
        self.components_spin.setSpecialValueText("All")

        self.variance_spin = gui.spin(
            box, self, "variance_covered", 1, 100,
            callback=self._update_selection_variance_spin,
            keyboardTracking=False
        )
        self.variance_spin.setSuffix("%")

        form.addRow("Components", self.components_spin)
        form.addRow("Variance covered", self.variance_spin)

        self.controlArea.layout().addStretch()

        gui.auto_commit(self.controlArea, self, "auto_commit", "Send data",
                        checkbox_label="Auto send on change")

        self.plot = pg.PlotWidget(background="w")

        axis = self.plot.getAxis("bottom")
        axis.setLabel("Principal Components")
        axis = self.plot.getAxis("left")
        axis.setLabel("Proportion of variance")

        self.plot.getViewBox().setMenuEnabled(False)
        self.plot.getViewBox().setMouseEnabled(False, False)
        self.plot.showGrid(True, True, alpha=0.5)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0))

        self.mainArea.layout().addWidget(self.plot)

    def set_data(self, data):
        self.clear()
        self.data = data

        if data is not None:
            self._transformed = None

            pca = Orange.projection.PCA()
            pca = pca(self.data)
            variance_ratio = pca.explained_variance_ratio_
            cumulative = numpy.cumsum(variance_ratio)
            self.components_spin.setRange(0, len(cumulative))

            self._pca = pca
            self._variance_ratio = variance_ratio
            self._cumulative = cumulative
            self._setup_plot()

        self.unconditional_commit()

    def clear(self):
        self.data = None
        self._pca = None
        self._transformed = None
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
            angle=90, pos=self._nselected_components() - 1, movable=True,
            bounds=(0, p - 1)
        )
        self._line.setCursor(Qt.SizeHorCursor)
        self._line.setPen(pg.mkPen(QColor(Qt.darkGray), width=1.5))
        self._line.sigPositionChanged.connect(self._on_cut_changed)

        self.plot.addItem(self._line)
        self.plot.setRange(xRange=(0.0, p - 1), yRange=(0.0, 1.0))
        axis = self.plot.getAxis("bottom")
        axis.setTicks([[(i, "C{}".format(i + 1)) for i in range(p)]])

    def _on_cut_changed(self, line):
        # cut changed by means of a cut line over the scree plot.
        value = line.value()
        current = self._nselected_components()
        components = int(numpy.floor(value)) + 1

        if not (self.ncomponents == 0 and
                components == len(self._variance_ratio)):
            self.ncomponents = components

        if self._pca is not None:
            self.variance_covered = self._cumulative[components - 1] * 100

        if current != self._nselected_components():
            self._invalidate_selection()

    def _update_selection_component_spin(self):
        # cut changed by "ncomponents" spin.
        if self._pca is None:
            return

        if self.ncomponents == 0:
            # Special "All" value
            cut = len(self._variance_ratio)
        else:
            cut = self.ncomponents
        self.variance_covered = self._cumulative[cut - 1] * 100

        if numpy.floor(self._line.value()) + 1 != cut:
            self._line.setValue(cut - 1)

        self._invalidate_selection()

    def _update_selection_variance_spin(self):
        # cut changed by "max variance" spin.
        if self._pca is None:
            return

        cut = numpy.searchsorted(self._cumulative, self.variance_covered / 100.0)
        self.ncomponents = cut

        if numpy.floor(self._line.value()) + 1 != cut:
            self._line.setValue(cut - 1)

        self._invalidate_selection()

    def _nselected_components(self):
        """Return the number of selected components."""
        if self._pca is None:
            return 0

        if self.ncomponents == 0:
            # Special "All" value
            max_comp = len(self._variance_ratio)
        else:
            max_comp = self.ncomponents

        var_max = self._cumulative[max_comp - 1]
        if var_max != numpy.floor(self.variance_covered / 100.0):
            cut = max_comp
            self.variance_covered = var_max * 100
        else:
            cut = numpy.searchsorted(
                self._cumulative, self.variance_covered / 100.0
            )
            self.ncomponents = cut
        return cut

    def _invalidate_selection(self):
        self.commit()

    def commit(self):
        transformed = components = None
        if self._pca is not None:
            components = self._pca.components_
            if self._transformed is None:
                # Compute the full transform (all components) only once.
                transformed = self._transformed = self._pca(self.data)
            else:
                transformed = self._transformed

            domain = Orange.data.Domain(
                transformed.domain.attributes[:self.ncomponents],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            transformed = Orange.data.Table.from_numpy(
                domain, transformed.X[:, :self.ncomponents], Y=transformed.Y,
                metas=transformed.metas, W=transformed.W
            )
            components = Orange.data.Table.from_numpy(None, components)

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

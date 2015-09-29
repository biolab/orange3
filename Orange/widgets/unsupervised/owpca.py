from PyQt4.QtGui import QFormLayout, QColor, QApplication, QLineEdit
from PyQt4.QtCore import Qt, QTimer

import numpy
import pyqtgraph as pg

from Orange.data import Table, Domain, StringVariable
from Orange.data.sql.table import SqlTable
import Orange.projection
from Orange.widgets import widget, gui, settings
from Orange.widgets.io import FileFormats

try:
    from orangecontrib import remote
    remotely = True
except ImportError:
    remotely = False


class OWPCA(widget.OWWidget):
    name = "PCA"
    description = "Principal component analysis with a scree-diagram."
    icon = "icons/PCA.svg"
    priority = 3050

    inputs = [("Data", Table, "set_data")]
    outputs = [("Transformed data", Table),
               ("Components", Table)]
    ncomponents = settings.Setting(2)
    variance_covered = settings.Setting(100)
    batch_size = settings.Setting(100)
    address = settings.Setting('')
    auto_update = settings.Setting(True)
    auto_commit = settings.Setting(True)

    want_graph = True

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

        self.sampling_box = gui.widgetBox(self.controlArea,
                                          "Incremental learning")

        self.addresstext = QLineEdit(box)
        self.addresstext.setPlaceholderText('Remote server')
        if self.address:
            self.addresstext.setText(self.address)
        self.sampling_box.layout().addWidget(self.addresstext)

        form = QFormLayout()
        self.sampling_box.layout().addLayout(form)
        self.batch_spin = gui.spin(
            self.sampling_box, self, "batch_size", 50, 100000, step=50,
            keyboardTracking=False)
        form.addRow("Batch size ~ ", self.batch_spin)

        self.start_button = gui.button(
            self.sampling_box, self, "Start remote computation",
            callback=self.start, autoDefault=False,
            tooltip="Start/abort computation on the server")
        self.start_button.setEnabled(False)

        gui.checkBox(self.sampling_box, self, "auto_update",
                     "Periodically fetch model", callback=self.update_model)
        self.__timer = QTimer(self, interval=2000)
        self.__timer.timeout.connect(self.get_model)

        self.sampling_box.setVisible(remotely)
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
        self.graphButton.clicked.connect(self.save_graph)

    def update_model(self):
        self.get_model()
        if self.auto_update and self.rpca and not self.rpca.ready():
            self.__timer.start(2000)
        else:
            self.__timer.stop()

    def start(self):
        if 'Abort' in self.start_button.text():
            self.rpca.abort()
            self.__timer.stop()
            self.start_button.setText("Start remote computation")
        else:
            self.address = self.addresstext.text()
            with remote.server(self.address):
                from Orange.projection.pca import RemotePCA
                maxiter = (1e5 + self.data.approx_len()) / self.batch_size * 3
                self.rpca = RemotePCA(self.data, self.batch_size, int(maxiter))
            self.update_model()
            self.start_button.setText("Abort remote computation")

    def set_data(self, data):
        self.clear()
        self.data = data

        self.start_button.setEnabled(False)
        if data is not None:
            self._transformed = None
            if remotely and isinstance(data, SqlTable):
                self.sampling_box.setVisible(True)
                self.start_button.setText("Start remote computation")
                self.start_button.setEnabled(True)
            else:
                self.sampling_box.setVisible(False)
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

    def get_model(self):
        if self.rpca is None:
            return
        if self.rpca.ready():
            self.__timer.stop()
            self.start_button.setText("Restart (finished)")
        self._pca = self.rpca.get_state()
        if self._pca is None:
            return
        self._variance_ratio = self._pca.explained_variance_ratio_
        self._cumulative = numpy.cumsum(self._variance_ratio)
        self.plot.clear()
        self._setup_plot()
        self._transformed = None
        self.commit()

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
        axis.setTicks([[(i, "PC{}".format(i + 1)) for i in range(p)]])

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

        cut = numpy.searchsorted(self._cumulative,
                                 self.variance_covered / 100.0)
        self.ncomponents = cut + 1
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
            self.ncomponents = cut = numpy.searchsorted(
                self._cumulative, self.variance_covered / 100.0) + 1
        return cut

    def _invalidate_selection(self):
        self.commit()

    def commit(self):
        transformed = components = None
        if self._pca is not None:
            if self._transformed is None:
                # Compute the full transform (all components) only once.
                self._transformed = self._pca(self.data)
            transformed = self._transformed

            domain = Domain(
                transformed.domain.attributes[:self.ncomponents],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            transformed = transformed.from_table(domain, transformed)
            dom = Domain(self._pca.orig_domain.attributes,
                         metas=[StringVariable(name='component')])
            metas = numpy.array([['PC{}'.format(i + 1)
                                  for i in range(self.ncomponents)]],
                                dtype=object).T
            components = Table(dom, self._pca.components_[:self.ncomponents],
                               metas=metas)
            components.name = 'components'

        self.send("Transformed data", transformed)
        self.send("Components", components)

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.plot.plotItem,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


def main():
    import gc
    app = QApplication([])
    w = OWPCA()
    # data = Table("iris")
    # data = Table("wine")
    data = Table("housing")
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

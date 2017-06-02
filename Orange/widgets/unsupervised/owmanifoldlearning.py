import numpy as np

from AnyQt.QtWidgets import QWidget, QVBoxLayout
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable
from Orange.projection import (MDS, Isomap, LocallyLinearEmbedding,
                               SpectralEmbedding, TSNE)
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.settings import Setting, SettingProvider
from Orange.widgets import gui


class ManifoldParametersEditor(QWidget, gui.OWComponent):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        gui.OWComponent.__init__(self, parent)
        self.parameters = {}
        self.parent_callback = parent.settings_changed

        # GUI
        self.setMinimumWidth(221)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.main_area = gui.vBox(self, spacing=0)

    def __parameter_changed(self, update_parameter, parameter_name):
        update_parameter(parameter_name)
        self.parent_callback()

    def _create_spin_parameter(self, name, minv, maxv, label):
        self.__spin_parameter_update(name)
        return gui.spin(
            self.main_area, self, name, minv, maxv, label=label,
            alignment=Qt.AlignRight, callbackOnReturn=True,
            callback=lambda f=self.__spin_parameter_update,
                            p=name: self.__parameter_changed(f, p))

    def __spin_parameter_update(self, name):
        self.parameters[name] = getattr(self, name)

    def _create_combo_parameter(self, name, label):
        self.__combo_parameter_update(name)
        items = (x[1] for x in getattr(self, name + "_values"))
        return gui.comboBox(
            self.main_area, self, name + "_index", label=label, items=items,
            orientation=Qt.Horizontal,
            callback=lambda f=self.__combo_parameter_update,
                            p=name: self.__parameter_changed(f, p))

    def __combo_parameter_update(self, name):
        index = getattr(self, name + "_index")
        values = getattr(self, name + "_values")
        self.parameters[name] = values[index][0]

    def _create_check_parameter(self, name, label):
        self.__check_parameter_update(name)
        box = gui.hBox(self.main_area)
        return gui.checkBox(
            box, self, name, label,
            callback=lambda f=self.__check_parameter_update,
                            p=name: self.__parameter_changed(f, p))

    def __check_parameter_update(self, name):
        checked = getattr(self, name)
        values = getattr(self, name + "_values")
        self.parameters[name] = values[checked]

    def _create_radio_parameter(self, name, label):
        self.__radio_parameter_update(name)
        values = (x[1] for x in getattr(self, name + "_values"))
        gui.separator(self.main_area)
        box = gui.hBox(self.main_area)
        lbl = gui.label(box, self, label + ":")
        rbt = gui.radioButtons(
            box, self, name + "_index", btnLabels=values,
            callback=lambda f=self.__radio_parameter_update,
                            p=name: self.__parameter_changed(f, p))
        rbt.layout().setAlignment(Qt.AlignTop)
        lbl.setAlignment(Qt.AlignTop)
        return rbt

    def __radio_parameter_update(self, name):
        index = getattr(self, name + "_index")
        values = getattr(self, name + "_values")
        self.parameters[name] = values[index][0]


class TSNEParametersEditor(ManifoldParametersEditor):
    _metrics = ("manhattan", "chebyshev", "jaccard", "mahalanobis", "cosine")
    metric_index = Setting(0)
    metric_values = [(x, x.capitalize()) for x in _metrics]
    # rename l2 to Euclidean
    metric_values = [("l2", "Euclidean")] + metric_values

    def __init__(self, parent):
        super().__init__(parent)
        self.metric_combo = self._create_combo_parameter(
            "metric", "Metric:")
        self.parameters["init"] = "pca"


class MDSParametersEditor(ManifoldParametersEditor):
    max_iter = Setting(300)
    init_type_index = Setting(0)
    init_type_values = (("PCA", "PCA (Torgerson)"),
                        ("random", "Random"))

    def __init__(self, parent):
        super().__init__(parent)
        self.max_iter_spin = self._create_spin_parameter(
            "max_iter", 10, 10 ** 4, "Max iterations:")
        self.random_state_radio = self._create_radio_parameter(
            "init_type", "Initialization")


class IsomapParametersEditor(ManifoldParametersEditor):
    n_neighbors = Setting(5)

    def __init__(self, parent):
        super().__init__(parent)
        self.n_neighbors_spin = self._create_spin_parameter(
            "n_neighbors", 1, 10 ** 2, "Neighbors:")


class LocallyLinearEmbeddingParametersEditor(ManifoldParametersEditor):
    n_neighbors = Setting(5)
    max_iter = Setting(100)
    method_index = Setting(0)
    method_values = (("standard", "Standard"),
                     ("modified", "Modified"),
                     ("hessian", "Hessian eigenmap"),
                     ("ltsa", "Local"))

    def __init__(self, parent):
        super().__init__(parent)
        self.method_combo = self._create_combo_parameter(
            "method", "Method:")
        self.n_neighbors_spin = self._create_spin_parameter(
            "n_neighbors", 1, 10 ** 2, "Neighbors:")
        self.max_iter_spin = self._create_spin_parameter(
            "max_iter", 10, 10 ** 4, "Max iterations:")


class SpectralEmbeddingParametersEditor(ManifoldParametersEditor):
    affinity_index = Setting(0)
    affinity_values = (("nearest_neighbors", "Nearest neighbors"),
                       ("rbf", "RBF kernel"))

    def __init__(self, parent):
        super().__init__(parent)
        self.affinity_combo = self._create_combo_parameter(
            "affinity", "Affinity:")


class OWManifoldLearning(OWWidget):
    name = "Manifold Learning"
    description = "Nonlinear dimensionality reduction."
    icon = "icons/Manifold.svg"
    priority = 2200

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        transformed_data = Output("Transformed data", Table, dynamic=False)

    MANIFOLD_METHODS = (TSNE, MDS, Isomap, LocallyLinearEmbedding,
                        SpectralEmbedding)

    tsne_editor = SettingProvider(TSNEParametersEditor)
    mds_editor = SettingProvider(MDSParametersEditor)
    isomap_editor = SettingProvider(IsomapParametersEditor)
    lle_editor = SettingProvider(LocallyLinearEmbeddingParametersEditor)
    spectral_editor = SettingProvider(SpectralEmbeddingParametersEditor)

    resizing_enabled = False
    want_main_area = False

    manifold_method_index = Setting(0)
    n_components = Setting(2)
    auto_apply = Setting(True)

    class Error(OWWidget.Error):
        n_neighbors_too_small = Msg("For chosen method and components, "
                                    "neighbors must be greater than {}")
        manifold_error = Msg("{}")
        sparse_methods = Msg('Only t-SNE method supported on sparse data')
        sparse_tsne_distance = Msg('Chebyshev, Jaccard, and Mahalanobis '
                                   'distances not supported with sparse data.')

    def __init__(self):
        self.data = None

        # GUI
        method_box = gui.vBox(self.controlArea, "Method")
        self.manifold_methods_combo = gui.comboBox(
            method_box, self, "manifold_method_index",
            items=[m.name for m in self.MANIFOLD_METHODS],
            callback=self.manifold_method_changed)

        self.params_box = gui.vBox(self.controlArea, "Parameters")

        self.tsne_editor = TSNEParametersEditor(self)
        self.mds_editor = MDSParametersEditor(self)
        self.isomap_editor = IsomapParametersEditor(self)
        self.lle_editor = LocallyLinearEmbeddingParametersEditor(self)
        self.spectral_editor = SpectralEmbeddingParametersEditor(self)
        self.parameter_editors = [
            self.tsne_editor, self.mds_editor, self.isomap_editor,
            self.lle_editor, self.spectral_editor]

        for editor in self.parameter_editors:
            self.params_box.layout().addWidget(editor)
            editor.hide()
        self.params_widget = self.parameter_editors[self.manifold_method_index]
        self.params_widget.show()

        output_box = gui.vBox(self.controlArea, "Output")
        self.n_components_spin = gui.spin(
            output_box, self, "n_components", 1, 10, label="Components:",
            alignment=Qt.AlignRight, callbackOnReturn=True,
            callback=self.settings_changed)
        self.apply_button = gui.auto_commit(
            output_box, self, "auto_apply", "&Apply",
            box=False, commit=self.apply)

    def manifold_method_changed(self):
        self.params_widget.hide()
        self.params_widget = self.parameter_editors[self.manifold_method_index]
        self.params_widget.show()
        self.apply()

    def settings_changed(self):
        self.apply()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.n_components_spin.setMaximum(len(self.data.domain.attributes)
                                          if self.data else 10)
        self.apply()

    def apply(self):
        out = None
        data = self.data
        method = self.MANIFOLD_METHODS[self.manifold_method_index]
        have_data = data is not None and len(data)
        sparse_incompat = have_data and data.is_sparse() and method != TSNE
        self.Error.clear()
        self.Error.sparse_methods(shown=sparse_incompat)
        if have_data and not sparse_incompat:
            domain = Domain([ContinuousVariable("C{}".format(i))
                             for i in range(self.n_components)],
                            data.domain.class_vars,
                            data.domain.metas)
            try:
                projector = method(**self.get_method_parameters(data, method))
                X = projector(data).embedding_
                out = Table(domain, X, data.Y, data.metas)
            except TypeError as e:
                if 'sparse' in e.args[0] and 'distance' in e.args[0]:
                    self.Error.sparse_tsne_distance()
                else:
                    raise
            except ValueError as e:
                if e.args[0] == "for method='hessian', n_neighbors " \
                                "must be greater than [n_components" \
                                " * (n_components + 3) / 2]":
                    n = self.n_components * (self.n_components + 3) / 2
                    self.Error.n_neighbors_too_small("{}".format(n))
                else:
                    self.Error.manifold_error(e.args[0])
            except np.linalg.linalg.LinAlgError as e:
                self.Error.manifold_error(str(e))
        self.Outputs.transformed_data.send(out)

    def get_method_parameters(self, data, method):
        parameters = dict(n_components=self.n_components)
        parameters.update(self.params_widget.parameters)
        if data is not None and data.is_sparse() and method == TSNE:
            parameters.update(method='exact',
                              init='random')
        return parameters

    def send_report(self):
        method = self.MANIFOLD_METHODS[self.manifold_method_index]
        self.report_items((("Method", method.name),))
        parameters = self.get_method_parameters(self.data, method)
        self.report_items("Method parameters", tuple(parameters.items()))
        if self.data:
            self.report_data("Data", self.data)


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication

    a = QApplication([])
    ow = OWManifoldLearning()
    d = Table("ionosphere")
    ow.set_data(d)
    ow.show()
    a.exec_()
    ow.saveSettings()

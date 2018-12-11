import numpy as np

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtWidgets import QFormLayout

from Orange.data import Table, Domain
from Orange.preprocess.preprocess import Preprocess, ApplyDomain
from Orange.projection import PCA, TSNE
from Orange.widgets import gui
from Orange.widgets.settings import Setting, SettingProvider
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Msg, Output


def compute_tsne(data, perplexity, iter, init):
    negative_gradient_method = 'fft' if len(data.X) > 10000 else 'bh'
    neighbor_method = 'approx' if len(data.X) > 10000 else 'exact'
    tsne = TSNE(
        perplexity=perplexity, n_iter=iter, initialization=init, theta=.8,
        early_exaggeration_iter=0, negative_gradient_method=negative_gradient_method,
        neighbors=neighbor_method, random_state=0
    )
    return tsne(data)


class OWtSNEGraph(OWScatterPlotBase):
    def update_coordinates(self):
        super().update_coordinates()
        if self.scatterplot_item is not None:
            self.view_box.setAspectLocked(True, 1)


class OWtSNE(OWDataProjectionWidget):
    name = "t-SNE"
    description = "Two-dimensional data projection with t-SNE."
    icon = "icons/TSNE.svg"
    priority = 920
    keywords = ["tsne"]

    settings_version = 3
    max_iter = Setting(300)
    perplexity = Setting(30)
    pca_components = Setting(20)

    GRAPH_CLASS = OWtSNEGraph
    graph = SettingProvider(OWtSNEGraph)
    embedding_variables_names = ("t-SNE-x", "t-SNE-y")

    #: Runtime state
    Running, Finished, Waiting = 1, 2, 3

    class Outputs(OWDataProjectionWidget.Outputs):
        preprocessor = Output("Preprocessor", Preprocess)

    class Error(OWDataProjectionWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        constant_data = Msg("Input data is constant")
        no_attributes = Msg("Data has no attributes")
        out_of_memory = Msg("Out of memory")
        optimization_error = Msg("Error during optimization\n{}")
        no_valid_data = Msg("No projection due to no valid data")

    def __init__(self):
        super().__init__()
        self.pca_data = None
        self.projection = None
        self.__update_loop = None
        # timer for scheduling updates
        self.__timer = QTimer(self, singleShot=True, interval=1,
                              timeout=self.__next_step)
        self.__state = OWtSNE.Waiting
        self.__in_next_step = False
        self.__draw_similar_pairs = False

    def _add_controls(self):
        self._add_controls_start_box()
        super()._add_controls()
        # Because sc data frequently has many genes,
        # showing all attributes in combo boxes can cause problems
        # QUICKFIX: Remove a separator and attributes from order
        # (leaving just the class and metas)
        self.models = self.gui.points_models
        for model in self.models:
            model.order = model.order[:-2]

    def _add_controls_start_box(self):
        box = gui.vBox(self.controlArea, True)
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )

        form.addRow(
            "Max iterations:",
            gui.spin(box, self, "max_iter", 1, 2000, step=50))

        form.addRow(
            "Perplexity:",
            gui.spin(box, self, "perplexity", 1, 100, step=1))

        box.layout().addLayout(form)

        gui.separator(box, 10)
        self.runbutton = gui.button(box, self, "Run", callback=self._toggle_run)

        gui.separator(box, 10)
        gui.hSlider(box, self, "pca_components", label="PCA components:",
                    minValue=2, maxValue=50, step=1)

    def check_data(self):
        def error(err):
            err()
            self.data = None

        super().check_data()
        if self.data is not None:
            if len(self.data) < 2:
                error(self.Error.not_enough_rows)
            elif not self.data.domain.attributes:
                error(self.Error.no_attributes)
            elif not self.data.is_sparse() and \
                    np.allclose(self.data.X - self.data.X[0], 0):
                error(self.Error.constant_data)
            elif not self.data.is_sparse() and \
                    np.all(~np.isfinite(self.data.X)):
                error(self.Error.no_valid_data)

    def get_embedding(self):
        if self.data is None:
            self.valid_data = None
            return None
        elif self.projection is None:
            embedding = np.random.normal(size=(len(self.data), 2))
        else:
            embedding = self.projection.embedding.X
        self.valid_data = np.ones(len(embedding), dtype=bool)
        return embedding

    def _toggle_run(self):
        if self.__state == OWtSNE.Running:
            self.stop()
            self.commit()
        else:
            self.start()

    def start(self):
        if not self.data or self.__state == OWtSNE.Running:
            self.graph.update_coordinates()
        elif self.__state in (OWtSNE.Finished, OWtSNE.Waiting):
            self.__start()

    def stop(self):
        if self.__state == OWtSNE.Running:
            self.__set_update_loop(None)

    def pca_preprocessing(self):
        if self.pca_data is not None and \
                self.pca_data.X.shape[1] == self.pca_components:
            return
        pca = PCA(n_components=self.pca_components, random_state=0)
        model = pca(self.data)
        self.pca_data = model(self.data)

    def __start(self):
        self.pca_preprocessing()
        initial = 'random' if self.projection is None \
            else self.projection.embedding.X
        step_size = 50

        def update_loop(data, max_iter, step, embedding):
            # NOTE: this code MUST NOT call into QApplication.processEvents
            done = False
            iterations_done = 0

            while not done:
                step_iter = min(max_iter - iterations_done, step)
                projection = compute_tsne(
                    data, self.perplexity, step_iter, embedding)
                embedding = projection.embedding.X
                iterations_done += step_iter
                if iterations_done >= max_iter:
                    done = True

                yield projection, iterations_done / max_iter

        self.__set_update_loop(update_loop(
            self.pca_data, self.max_iter, step_size, initial))
        self.progressBarInit(processEvents=None)

    def __set_update_loop(self, loop):
        if self.__update_loop is not None:
            self.__update_loop.close()
            self.__update_loop = None
            self.progressBarFinished(processEvents=None)

        self.__update_loop = loop

        if loop is not None:
            self.setBlocking(True)
            self.progressBarInit(processEvents=None)
            self.setStatusMessage("Running")
            self.runbutton.setText("Stop")
            self.__state = OWtSNE.Running
            self.__timer.start()
        else:
            self.setBlocking(False)
            self.setStatusMessage("")
            self.runbutton.setText("Start")
            self.__state = OWtSNE.Finished
            self.__timer.stop()

    def __next_step(self):
        if self.__update_loop is None:
            return

        assert not self.__in_next_step
        self.__in_next_step = True

        loop = self.__update_loop
        self.Error.out_of_memory.clear()
        self.Error.optimization_error.clear()
        try:
            projection, progress = next(self.__update_loop)
            assert self.__update_loop is loop
        except StopIteration:
            self.__set_update_loop(None)
            self.unconditional_commit()
        except MemoryError:
            self.Error.out_of_memory()
            self.__set_update_loop(None)
        except Exception as exc:
            self.Error.optimization_error(str(exc))
            self.__set_update_loop(None)
        else:
            self.progressBarSet(100.0 * progress, processEvents=None)
            self.projection = projection
            self.graph.update_coordinates()
            self.graph.update_density()
            # schedule next update
            self.__timer.start()

        self.__in_next_step = False

    def setup_plot(self):
        super().setup_plot()
        self.start()

    def commit(self):
        super().commit()
        self.send_preprocessor()

    def _get_projection_data(self):
        if self.data is None:
            return None
        if self.projection is None:
            variables = self._get_projection_variables()
        else:
            variables = self.projection.domain.attributes
        data = self.data.transform(
            Domain(self.data.domain.attributes,
                   self.data.domain.class_vars,
                   self.data.domain.metas + variables))
        data.metas[:, -2:] = self.get_embedding()
        return data

    def send_preprocessor(self):
        prep = None
        if self.data is not None and self.projection is not None:
            prep = ApplyDomain(self.projection.domain, self.projection.name)
        self.Outputs.preprocessor.send(prep)

    def clear(self):
        super().clear()
        self.__set_update_loop(None)
        self.__state = OWtSNE.Waiting
        self.pca_data = None
        self.projection = None

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 3:
            if "selection_indices" in settings:
                settings["selection"] = settings["selection_indices"]

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


if __name__ == "__main__":
    data = Table("iris")
    WidgetPreview(OWtSNE).run(
        set_data=data,
        set_subset_data=data[np.random.choice(len(data), 10)])

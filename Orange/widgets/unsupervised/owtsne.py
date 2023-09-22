import numpy as np
import warnings
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout
from functools import partial
from types import SimpleNamespace as namespace
from typing import Optional  # pylint: disable=unused-import

from Orange.data import Table, Domain
from Orange.data.util import array_equal
from Orange.misc import DistMatrix
from Orange.preprocess import preprocess
from Orange.projection import PCA
from Orange.projection import manifold
from Orange.widgets import gui
from Orange.widgets.settings import SettingProvider, ContextSetting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Msg
from orangewidget.utils.signals import Input

_STEP_SIZE = 25
_MAX_PCA_COMPONENTS = 50
_DEFAULT_PCA_COMPONENTS = 20

INITIALIZATIONS = [("PCA", "pca"), ("Spectral", "spectral")]
DISTANCE_METRICS = [("Euclidean", "l2"), ("Manhattan", "l1"), ("Cosine", "cosine")]


class Task(namespace):
    """Completely determines the t-SNE task spec and intermediate results."""
    data = None                     # type: Optional[Table]
    distance_matrix = None          # type: Optional[DistMatrix]

    preprocessed_data = None        # type: Optional[Table]

    normalize = None                # type: Optional[bool]
    normalized_data = None          # type: Optional[Table]

    use_pca_preprocessing = None    # type: Optional[bool]
    pca_components = None           # type: Optional[int]
    pca_projection = None           # type: Optional[Table]

    distance_metric = None          # type: Optional[str]
    perplexity = None               # type: Optional[float]
    multiscale = None               # type: Optional[bool]
    exaggeration = None             # type: Optional[float]
    initialization_method = None    # type: Optional[str]
    initialization = None           # type: Optional[np.ndarray]
    affinities = None               # type: Optional[openTSNE.affinity.Affinities]
    tsne_embedding = None           # type: Optional[manifold.TSNEModel]
    iterations_done = 0             # type: int

    # These attributes need not be set by the widget
    tsne = None                     # type: Optional[manifold.TSNE]
    # `effective_data` stores the current working matrix which should be used
    # for any steps depending on the data matrix. For instance, normalization
    # should use the effective data (presumably the original data), and set it
    # to the normalized version upon completion. This can then later be used for
    # PCA preprocessing
    effective_data = None           # type: Optional[Table]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set `effective_data` to `data` if provided and effective data is not
        # provided
        if self.effective_data is None and self.data is not None:
            self.effective_data = self.data

    class ValidationError(ValueError):
        pass

    def validate(self) -> "Task":
        def error(msg):
            raise Task.ValidationError(msg)

        if self.data is None and self.distance_matrix is None:
            error("Both `distance_matrix` and `data` cannot be `None`")

        if self.distance_matrix is not None:
            if self.distance_metric != "precomputed":
                error(
                    "`distance_metric` must be set to `precomputed` when using "
                    "a distance matrix"
                )
            if self.initialization_method != "spectral":
                error(
                    "`initialization_method` must be set to `spectral` when "
                    "using a distance matrix"
                )

        if self.distance_matrix is None:
            if self.distance_metric == "precomputed":
                error(
                    "`distance_metric` cannot be set to `precomputed` when no "
                    "distance matrix is provided"
                )

        if self.data is not None and self.data.is_sparse():
            if self.normalize:
                error("Data normalization is not supported for sparse data")

        return self


def apply_tsne_preprocessing(tsne, data):
    return tsne.preprocess(data)


def data_normalization(data):
    normalization = preprocess.Normalize()
    return normalization(data)


def pca_preprocessing(data, n_components):
    projector = PCA(n_components=n_components, random_state=0)
    model = projector(data)
    return model(data)


def prepare_tsne_obj(n_samples: int, initialization_method: str,
                     distance_metric: str, perplexity: float,
                     multiscale: bool, exaggeration: float):
    """Automatically determine the best parameters for the given data set."""
    # Compute perplexity settings for multiscale
    if multiscale:
        perplexity = min((n_samples - 1) / 3, 50), min((n_samples - 1) / 3, 500)
    else:
        perplexity = perplexity

    return manifold.TSNE(
        n_components=2,
        initialization=initialization_method,
        metric=distance_metric,
        perplexity=perplexity,
        multiscale=multiscale,
        exaggeration=exaggeration,
        random_state=0,
    )


class TSNERunner:
    @staticmethod
    def compute_tsne_preprocessing(task: Task, state: TaskState, **_) -> None:
        state.set_status("Preprocessing data...")
        task.preprocessed_data = apply_tsne_preprocessing(task.tsne, task.effective_data)
        task.effective_data = task.preprocessed_data
        state.set_partial_result(("preprocessed_data", task))

    @staticmethod
    def compute_normalization(task: Task, state: TaskState, **_) -> None:
        state.set_status("Normalizing data...")
        task.normalized_data = data_normalization(task.effective_data)
        task.effective_data = task.normalized_data
        state.set_partial_result(("normalized_data", task))

    @staticmethod
    def compute_pca(task: Task, state: TaskState, **_) -> None:
        # Perform PCA preprocessing
        state.set_status("Computing PCA...")
        pca_projection = pca_preprocessing(task.effective_data, task.pca_components)
        # Apply t-SNE's preprocessors to the data
        task.pca_projection = task.tsne.preprocess(pca_projection)
        task.effective_data = task.pca_projection
        state.set_partial_result(("pca_projection", task))

    @staticmethod
    def compute_initialization(task: Task, state: TaskState, **_) -> None:
        # Prepare initial positions for t-SNE
        state.set_status("Preparing initialization...")
        if task.initialization_method == "pca":
            x = task.effective_data.X
        elif task.initialization_method == "spectral":
            assert task.affinities is not None
            x = task.affinities.P
        else:
            raise RuntimeError(
                f"Unrecognized initialization scheme `{task.initialization_method}`!"
            )
        task.initialization = task.tsne.compute_initialization(x)
        state.set_partial_result(("initialization", task))

    @staticmethod
    def compute_affinities(task: Task, state: TaskState, **_) -> None:
        state.set_status("Finding nearest neighbors...")

        if task.distance_metric == "precomputed":
            assert task.distance_matrix is not None
            x = task.distance_matrix
        else:
            assert task.data is not None
            assert task.effective_data is not None
            x = task.effective_data.X

        task.affinities = task.tsne.compute_affinities(x)
        state.set_partial_result(("affinities", task))

    @staticmethod
    def compute_tsne(task: Task, state: TaskState, progress_callback=None) -> None:
        tsne = task.tsne

        state.set_status("Running optimization...")

        # If this the first time we're computing t-SNE (otherwise we may just
        # be resuming optimization), we have to assemble the tsne object
        if task.tsne_embedding is None:
            # Assemble a t-SNE embedding object and convert it to a TSNEModel
            task.tsne_embedding = tsne.prepare_embedding(
                task.affinities, task.initialization
            )

            if task.distance_metric == "precomputed":
                x = task.distance_matrix
            else:
                assert task.effective_data is not None
                x = task.effective_data

            task.tsne_embedding = tsne.convert_embedding_to_model(
                x, task.tsne_embedding
            )
            state.set_partial_result(("tsne_embedding", task))

            if state.is_interruption_requested():
                return

        total_iterations_needed = tsne.early_exaggeration_iter + tsne.n_iter

        def run_optimization(tsne_params: dict, iterations_needed: int) -> bool:
            """Run t-SNE optimization phase. Return value indicates whether the
            optimization was interrupted."""
            while task.iterations_done < iterations_needed:
                # Step size can't be larger than the remaining number of iterations
                step_size = min(_STEP_SIZE, iterations_needed - task.iterations_done)
                task.tsne_embedding = task.tsne_embedding.optimize(
                    step_size, **tsne_params
                )
                task.iterations_done += step_size
                state.set_partial_result(("tsne_embedding", task))
                if progress_callback is not None:
                    # The current iterations must be divided by the total
                    # number of iterations, not the number of iterations in the
                    # current phase (iterations_needed)
                    progress_callback(task.iterations_done / total_iterations_needed)

                if state.is_interruption_requested():
                    return True

        # Run early exaggeration phase
        was_interrupted = run_optimization(
            dict(exaggeration=tsne.early_exaggeration, momentum=0.8, inplace=False),
            iterations_needed=tsne.early_exaggeration_iter,
        )
        if was_interrupted:
            return
        # Run regular optimization phase
        run_optimization(
            dict(exaggeration=tsne.exaggeration, momentum=0.8, inplace=False),
            iterations_needed=total_iterations_needed,
        )

    @classmethod
    def run(cls, task: Task, state: TaskState) -> Task:
        task.validate()

        # Assign weights to each job indicating how much time will be spent on each
        weights = {"preprocessing": 1, "normalization": 1, "pca": 1, "init": 1, "aff": 25, "tsne": 50}
        total_weight = sum(weights.values())

        # Prepare the tsne object and add it to the spec
        if task.distance_matrix is not None:
            n_samples = task.distance_matrix.shape[0]
        else:
            assert task.data is not None
            n_samples = task.data.X.shape[0]

        task.tsne = prepare_tsne_obj(
            n_samples,
            task.initialization_method,
            task.distance_metric,
            task.perplexity,
            task.multiscale,
            task.exaggeration,
        )

        job_queue = []
        # Add the tasks that still need to be run to the job queue
        if task.distance_metric != "precomputed":
            task.effective_data = task.data
            if task.preprocessed_data is None:
                job_queue.append((cls.compute_tsne_preprocessing, weights["preprocessing"]))

            if task.normalize and task.normalized_data is None:
                job_queue.append((cls.compute_normalization, weights["normalization"]))

            if task.use_pca_preprocessing and task.pca_projection is None:
                job_queue.append((cls.compute_pca, weights["pca"]))

        if task.affinities is None:
            job_queue.append((cls.compute_affinities, weights["aff"]))

        if task.initialization is None:
            job_queue.append((cls.compute_initialization, weights["init"]))

        total_iterations = task.tsne.early_exaggeration_iter + task.tsne.n_iter
        if task.tsne_embedding is None or task.iterations_done < total_iterations:
            job_queue.append((cls.compute_tsne, weights["tsne"]))

        job_queue = [(partial(f, task, state), w) for f, w in job_queue]

        # Ensure the effective data is set to the appropriate, potentially
        # precomputed matrix
        task.effective_data = task.data
        if task.preprocessed_data is not None:
            task.effective_data = task.preprocessed_data
        if task.normalize and task.normalized_data is not None:
            task.effective_data = task.normalized_data
        if task.use_pca_preprocessing and task.pca_projection is not None:
            task.effective_data = task.pca_projection

        # Figure out the total weight of the jobs
        job_weight = sum(j[1] for j in job_queue)
        progress_done = total_weight - job_weight
        for job, job_weight in job_queue:

            def _progress_callback(val):
                state.set_progress_value(
                    (progress_done + val * job_weight) / total_weight * 100
                )

            if state.is_interruption_requested():
                return task

            # Execute the job
            job(progress_callback=_progress_callback)
            # Update the progress bar according to the weights assigned to
            # each job
            progress_done += job_weight
            state.set_progress_value(progress_done / total_weight * 100)

        return task


class OWtSNEGraph(OWScatterPlotBase):
    def update_coordinates(self):
        super().update_coordinates()
        if self.scatterplot_item is not None:
            self.view_box.setAspectLocked(True, 1)


class invalidated:
    # pylint: disable=invalid-name
    preprocessed_data = normalized_data = pca_projection = initialization = \
        affinities = tsne_embedding = False

    def __set__(self, instance, value):
        # `self._invalidate = True` should invalidate everything
        self.preprocessed_data = value
        self.normalized_data = value
        self.pca_projection = value
        self.initialization = value
        self.affinities = value
        self.tsne_embedding = value

    def __bool__(self):
        # If any of the values are invalidated, this should return true
        return (
            self.preprocessed_data or self.normalized_data or self.pca_projection or
            self.initialization or self.affinities or self.tsne_embedding
        )

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(
            "=".join([k, str(getattr(self, k))])
            for k in ["preprocessed_data", "normalized_data", "pca_projection",
                      "initialization", "affinities", "tsne_embedding"]
        ))


class OWtSNE(OWDataProjectionWidget, ConcurrentWidgetMixin):
    name = "t-SNE"
    description = "Two-dimensional data projection with t-SNE."
    icon = "icons/TSNE.svg"
    priority = 920
    keywords = "t-sne, tsne"

    settings_version = 4
    perplexity = ContextSetting(30)
    multiscale = ContextSetting(False)
    exaggeration = ContextSetting(1)
    initialization_method_idx = ContextSetting(0)
    distance_metric_idx = ContextSetting(0)

    normalize = ContextSetting(True)
    use_pca_preprocessing = ContextSetting(True)
    pca_components = ContextSetting(_DEFAULT_PCA_COMPONENTS)

    GRAPH_CLASS = OWtSNEGraph
    graph = SettingProvider(OWtSNEGraph)
    embedding_variables_names = ("t-SNE-x", "t-SNE-y")

    # Use `invalidated` descriptor, so we don't break the usage of
    # `_invalidated` in `OWDataProjectionWidget`, but still allow finer control
    # over which parts of the embedding to invalidate
    _invalidated = invalidated()

    class Inputs(OWDataProjectionWidget.Inputs):
        distances = Input("Distances", DistMatrix)

    class Information(OWDataProjectionWidget.Information):
        modified = Msg("The parameter settings have been changed. Press "
                       "\"Start\" to rerun with the new settings.")

    class Warning(OWDataProjectionWidget.Warning):
        consider_using_pca_preprocessing = Msg(
            "The input data contains a large number of features, which may slow"
            " down t-SNE computation. Consider enabling PCA preprocessing."
        )

    class Error(OWDataProjectionWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        not_enough_cols = Msg("Input data needs at least 2 attributes")
        constant_data = Msg("Input data is constant")
        no_valid_data = Msg("No projection due to no valid data")

        distance_matrix_not_symmetric = Msg("Distance matrix is not symmetric")
        distance_matrix_too_small = Msg("Input matrix must be at least 2x2")

        dimension_mismatch = Msg("Data and distance dimensions do not match")

    def __init__(self):
        OWDataProjectionWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        # Distance matrix from `Distances` signal
        self.distance_matrix = None       # type: Optional[DistMatrix]
        # Data table from the `self.matrix.row_items` (if present)
        self.distance_matrix_data = None  # type: Optional[Table]
        # Data table from `Data` signal
        self.signal_data = None           # type: Optional[Table]

        # Intermediate results
        self.preprocessed_data = None     # type: Optional[Table]
        self.normalized_data = None       # type: Optional[Table]
        self.pca_projection = None        # type: Optional[Table]
        self.initialization = None        # type: Optional[np.ndarray]
        self.affinities = None            # type: Optional[openTSNE.affinity.Affinities]
        self.tsne_embedding = None        # type: Optional[manifold.TSNEModel]
        self.iterations_done = 0          # type: int

    @property
    def normalize_(self):
        should_normalize = self.normalize
        if self.distance_matrix is not None:
            should_normalize = False
        if self.data is not None:
            if self.data.is_sparse():
                should_normalize = False
        return should_normalize

    @property
    def use_pca_preprocessing_(self):
        should_use_pca_preprocessing = self.use_pca_preprocessing
        if self.distance_matrix is not None:
            should_use_pca_preprocessing = False
        return should_use_pca_preprocessing

    @property
    def effective_data(self):
        return self.data.transform(Domain(self.effective_variables))

    def _add_controls(self):
        self._add_controls_start_box()
        super()._add_controls()

    def _add_controls_start_box(self):
        self.preprocessing_box = gui.vBox(self.controlArea, box="Preprocessing")
        self.normalize_cbx = gui.checkBox(
            self.preprocessing_box, self, "normalize", "Normalize data",
            callback=self._normalize_data_changed, stateWhenDisabled=False,
        )
        self.pca_preprocessing_cbx = gui.checkBox(
            self.preprocessing_box, self, "use_pca_preprocessing", "Apply PCA preprocessing",
            callback=self._pca_preprocessing_changed, stateWhenDisabled=False,
        )
        self.pca_component_slider = gui.hSlider(
            self.preprocessing_box, self, "pca_components", label="PCA Components:",
            minValue=2, maxValue=_MAX_PCA_COMPONENTS, step=1,
            callback=self._pca_slider_changed,
        )

        self.parameter_box = gui.vBox(self.controlArea, box="Parameters")
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )

        self.initialization_combo = gui.comboBox(
            self.controlArea, self, "initialization_method_idx",
            items=[m[0] for m in INITIALIZATIONS],
            callback=self._invalidate_initialization,
        )
        form.addRow("Initialization:", self.initialization_combo)

        self.distance_metric_combo = gui.comboBox(
            self.controlArea, self, "distance_metric_idx",
            items=[m[0] for m in DISTANCE_METRICS],
            callback=self._invalidate_affinities,
        )
        form.addRow("Distance metric:", self.distance_metric_combo)

        self.perplexity_spin = gui.spin(
            self.controlArea, self, "perplexity", 1, 500, step=1,
            alignment=Qt.AlignRight, addToLayout=False,
            callback=self._invalidate_affinities,
        )
        form.addRow("Perplexity:", self.perplexity_spin)

        form.addRow(gui.checkBox(
            self.controlArea, self, "multiscale", label="Preserve global structure",
            callback=self._multiscale_changed, addToLayout=False
        ))

        sbe = gui.hBox(self.controlArea, False, addToLayout=False)
        gui.hSlider(
            sbe, self, "exaggeration", minValue=1, maxValue=4, step=0.25,
            intOnly=False, labelFormat="%.2f",
            callback=self._invalidate_tsne_embedding,
        )
        form.addRow("Exaggeration:", sbe)

        self.parameter_box.layout().addLayout(form)

        self.run_button = gui.button(
            self.parameter_box, self, "Start", callback=self._toggle_run
        )

    # GUI control callbacks
    def _normalize_data_changed(self):
        # We only care about the normalization checkbox if there is no distance
        # matrix provided and if the data are not sparse. This is not user-
        # settable anyway, but is triggered when we programmatically
        # enable/disable the checkbox in`enable_controls`
        if self.distance_matrix is None and not self.data.is_sparse():
            self._invalidate_normalized_data()

    def _pca_preprocessing_changed(self):
        # We only care about the PCA checkbox if there is no distance
        # matrix provided. This is not user-settable anyway, but is triggered
        # when we programmatically enable/disable the checkbox in
        # `enable_controls`
        if self.distance_matrix is None:
            self.controls.pca_components.box.setEnabled(self.use_pca_preprocessing)

            self._invalidate_pca_projection()

            should_warn_pca = False
            if self.data is not None and not self.use_pca_preprocessing:
                if len(self.data.domain.attributes) >= _MAX_PCA_COMPONENTS:
                    should_warn_pca = True
            self.Warning.consider_using_pca_preprocessing(shown=should_warn_pca)

    def _pca_slider_changed(self):
        # We only care about the PCA slider if there is no distance
        # matrix provided. This is not user-settable anyway, but is triggered
        # when we programmatically enable/disable the checkbox in
        # `enable_controls`
        if self.distance_matrix is None:
            self._invalidate_pca_projection()

    def _multiscale_changed(self):
        form = self.parameter_box.layout().itemAt(0)
        assert isinstance(form, QFormLayout)
        form.labelForField(self.perplexity_spin).setDisabled(self.multiscale)
        self.controls.perplexity.setDisabled(self.multiscale)

        self._invalidate_affinities()

    # Invalidation cascade
    def _invalidate_preprocessed_data(self):
        self._invalidated.preprocessed_data = True
        self._invalidate_normalized_data()

    def _invalidate_normalized_data(self):
        self._invalidated.normalized_data = True
        self._invalidate_pca_projection()

    def _invalidate_pca_projection(self):
        self._invalidated.pca_projection = True
        self._invalidate_affinities()

    def _invalidate_affinities(self):
        self._invalidated.affinities = True
        self._invalidate_tsne_embedding()

    def _invalidate_initialization(self):
        self._invalidated.initialization = True
        self._invalidate_tsne_embedding()

    def _invalidate_tsne_embedding(self):
        self._invalidated.tsne_embedding = True
        self._stop_running_task()
        self._set_modified(True)

    def _stop_running_task(self):
        self.cancel()
        self.run_button.setText("Start")

    def _set_modified(self, state):
        """Mark the widget (GUI) as containing modified state."""
        if self.data is None:
            # Does not apply when we have no data
            state = False
        self.Information.modified(shown=state)

    def check_data(self):
        self.Error.dimension_mismatch.clear()
        self.Error.not_enough_rows.clear()
        self.Error.not_enough_cols.clear()
        self.Error.no_valid_data.clear()
        self.Error.constant_data.clear()

        if self.data is None:
            return

        def error(err):
            err()
            self.data = None

        if (
            self.data is not None and self.distance_matrix is not None and
            len(self.data) != len(self.distance_matrix)
        ):
            error(self.Error.dimension_mismatch)

        # The errors below are relevant only if the distance matrix is not
        # provided
        if self.distance_matrix is not None:
            return

        if len(self.data) < 2:
            error(self.Error.not_enough_rows)

        elif len(self.data.domain.attributes) < 2:
            error(self.Error.not_enough_cols)

        elif not self.data.is_sparse():
            if np.all(~np.isfinite(self.data.X)):
                error(self.Error.no_valid_data)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Degrees of freedom .*", RuntimeWarning)
                    if np.nan_to_num(np.nanstd(self.data.X, axis=0)).sum() == 0:
                        error(self.Error.constant_data)

    def check_distance_matrix(self):
        self.Error.distance_matrix_not_symmetric.clear()
        self.Error.distance_matrix_too_small.clear()

        if self.distance_matrix is None:
            return

        def error(err):
            err()
            self.distance_matrix = self.distance_matrix_data = None

        # Check for matrix validity
        if self.distance_matrix is not None:
            if not self.distance_matrix.is_symmetric():
                error(self.Error.distance_matrix_not_symmetric)
            elif len(self.distance_matrix) < 2:
                error(self.Error.distance_matrix_too_small)

    def get_embedding(self):
        if self.tsne_embedding is None:
            self.valid_data = None
            return None

        embedding = self.tsne_embedding.embedding.X
        self.valid_data = np.ones(len(embedding), dtype=bool)
        return embedding

    def _toggle_run(self):
        # If no data, there's nothing to do
        if self.data is None:
            return

        # Pause task
        if self.task is not None:
            self.cancel()
            self.run_button.setText("Resume")
            self.commit.deferred()
        # Resume task
        else:
            self.run()

    @Inputs.data
    def set_data(self, data):
        self.signal_data = data
        # Data checking will be performed in `handleNewSignals` since the data
        # can also be set from the `distance_matrix.row_items` if no additional
        # data is provided

    @Inputs.distances
    def set_distances(self, matrix: DistMatrix):
        had_distance_matrix = self.distance_matrix is not None
        prev_distance_matrix = self.distance_matrix

        self.distance_matrix = matrix
        self.distance_matrix_data = matrix.row_items if matrix is not None else None
        self.check_distance_matrix()

        # If there was no distance matrix before, but there is data now, invalidate
        if self.distance_matrix is not None and not had_distance_matrix:
            self._invalidated = True

        # If the new distance matrix is invalid or None, invalidate
        elif self.distance_matrix is None and had_distance_matrix:
            self._invalidated = True

        # If the distance matrix has changed, invalidate
        elif (
            had_distance_matrix and self.distance_matrix is not None and
            not array_equal(prev_distance_matrix, self.distance_matrix)
        ):
            self._invalidated = True

    def handleNewSignals(self):
        had_data = self.data is not None
        prev_data = self.effective_data if had_data else None

        self.cancel()  # clear any running jobs
        self.data = None
        self.closeContext()

        if self.signal_data is not None:
            self.data = self.signal_data
        elif self.distance_matrix_data is not None:
            self.data = self.distance_matrix_data

        self.check_data()

        # If we have any errors, there's something wrong with the inputs or
        # their combination, so we clear the graph and the outputs
        if len(self.Error.active) > 0:
            self.clear()
            self._invalidated = True
            # Set data to None so that the output signal will be cleared
            self.data = None
            self.init_attr_values()
            self.commit.now()
            return

        # We only invalidate based on data if there is no distance matrix, as
        # otherwise, the embedding will remain in-tact
        if self.distance_matrix is None:
            # If there was no data before, but there is data now, invalidate
            if self.data is not None and not had_data:
                self._invalidated = True

            # If the new data is invalid or None, invalidate
            elif self.data is None and had_data:
                self._invalidated = True

            # If the data table has changed, invalidate
            elif (
                had_data and self.data is not None and
                not array_equal(prev_data.X, self.effective_data.X)
            ):
                self._invalidated = True

        self.init_attr_values()
        self.openContext(self.data)
        self.enable_controls()

        if self._invalidated:
            self.clear()
            self.input_changed.emit(self.data)

        # We don't bother with the granular invalidation flags because
        # `super().handleNewSignals` will just set all of them to False or will
        # do nothing. However, it's important we remember its state because we
        # won't call `run` if needed. `run` also relies on the state of
        # `_invalidated` to properly set the intermediate values to None
        prev_invalidated = bool(self._invalidated)
        super().handleNewSignals()
        self._invalidated = prev_invalidated

        if self._invalidated:
            self.run()

    def init_attr_values(self):
        super().init_attr_values()

        if self.data is not None:
            n_attrs = len(self.data.domain.attributes)
            max_components = min(_MAX_PCA_COMPONENTS, n_attrs)
            should_use_pca = len(self.data.domain.attributes) > 10
        else:
            max_components = _MAX_PCA_COMPONENTS
            should_use_pca = False

        # We set this to the default number of components here, so it resets
        # properly, any previous settings will be restored from context
        # settings a little later
        self.controls.pca_components.setMaximum(max_components)
        self.controls.pca_components.setValue(_DEFAULT_PCA_COMPONENTS)

        self.exaggeration = 1
        self.normalize = True
        self.use_pca_preprocessing = should_use_pca
        self.distance_metric_idx = 0
        self.initialization_method_idx = 0

    def enable_controls(self):
        super().enable_controls()

        has_distance_matrix = self.distance_matrix is not None
        has_data = self.data is not None

        # When we disable controls in the form layout, we also want to ensure
        # the labels are disabled, to be consistent with the preprocessing box
        form = self.parameter_box.layout().itemAt(0)
        assert isinstance(form, QFormLayout)

        # Reset all tooltips and controls
        self.normalize_cbx.setDisabled(False)
        self.normalize_cbx.setToolTip("")

        self.pca_preprocessing_cbx.setDisabled(False)
        self.pca_preprocessing_cbx.setToolTip("")

        self.initialization_combo.setDisabled(False)
        self.initialization_combo.setToolTip("")
        form.labelForField(self.initialization_combo).setDisabled(False)

        self.distance_metric_combo.setDisabled(False)
        self.distance_metric_combo.setToolTip("")
        form.labelForField(self.distance_metric_combo).setDisabled(False)

        if has_distance_matrix:
            self.normalize_cbx.setDisabled(True)
            self.normalize_cbx.setToolTip(
                "Precomputed distances provided. Preprocessing is unnecessary!"
            )

            self.pca_preprocessing_cbx.setDisabled(True)
            self.pca_preprocessing_cbx.setToolTip(
                "Precomputed distances provided. Preprocessing is unnecessary!"
            )

            # Only spectral init is valid with a precomputed distance matrix
            spectral_init_idx = self.initialization_combo.findText("Spectral")
            self.initialization_combo.setCurrentIndex(spectral_init_idx)
            self.initialization_combo.setDisabled(True)
            self.initialization_combo.setToolTip(
                "Only spectral intialization is supported with precomputed "
                "distance matrices."
            )
            form.labelForField(self.initialization_combo).setDisabled(True)

            self.distance_metric_combo.setDisabled(True)
            self.distance_metric_combo.setCurrentIndex(-1)
            self.distance_metric_combo.setToolTip(
                "Precomputed distances provided."
            )
            form.labelForField(self.distance_metric_combo).setDisabled(True)

        # Normalization isn't supported on sparse data, as this would
        # require centering and normalizing the matrix
        if not has_distance_matrix and has_data and self.data.is_sparse():
            self.normalize_cbx.setDisabled(True)
            self.normalize_cbx.setToolTip(
                "Data normalization is not supported on sparse matrices."
            )

        # Disable slider parent, because we want to disable the labels too
        self.pca_component_slider.parent().setEnabled(self.use_pca_preprocessing_)

        # Disable the perplexity spin box if multiscale is turned on
        self.perplexity_spin.setDisabled(self.multiscale)
        form.labelForField(self.perplexity_spin).setDisabled(self.multiscale)

    def run(self):
        # Reset invalidated values as indicated by the flags
        if self._invalidated.preprocessed_data:
            self.preprocessed_data = None
        if self._invalidated.normalized_data:
            self.normalized_data = None
        if self._invalidated.pca_projection:
            self.pca_projection = None
        if self._invalidated.affinities:
            self.affinities = None
        if self._invalidated.initialization:
            self.initialization = None
        if self._invalidated.tsne_embedding:
            self.iterations_done = 0
            self.tsne_embedding = None

        self._set_modified(False)
        self._invalidated = False

        # When the data is invalid, it is set to `None` and an error is set,
        # therefore it would be erroneous to clear the error here
        if self.data is not None:
            self.run_button.setText("Stop")

        # Cancel current running task
        self.cancel()

        if self.data is None and self.distance_matrix is None:
            return

        initialization_method = INITIALIZATIONS[self.initialization_method_idx][1]
        distance_metric = DISTANCE_METRICS[self.distance_metric_idx][1]
        if self.distance_matrix is not None:
            distance_metric = "precomputed"
            initialization_method = "spectral"

        task = Task(
            data=self.data,
            distance_matrix=self.distance_matrix,
            # Preprocessed data
            preprocessed_data=self.preprocessed_data,
            # Normalization
            normalize=self.normalize_,
            normalized_data=self.normalized_data,
            # PCA preprocessing
            use_pca_preprocessing=self.use_pca_preprocessing_,
            pca_components=self.pca_components,
            pca_projection=self.pca_projection,
            # t-SNE parameters
            initialization_method=initialization_method,
            initialization=self.initialization,
            distance_metric=distance_metric,
            perplexity=self.perplexity,
            multiscale=self.multiscale,
            exaggeration=self.exaggeration,
            affinities=self.affinities,
            # Misc
            tsne_embedding=self.tsne_embedding,
            iterations_done=self.iterations_done,
        )
        return self.start(TSNERunner.run, task)

    def __ensure_task_same_for_preprocessing(self, task: Task):
        if task.distance_metric != "precomputed":
            assert task.data is self.data
            assert isinstance(task.preprocessed_data, Table) and \
                len(task.preprocessed_data) == len(self.data)

    def __ensure_task_same_for_normalization(self, task: Task):
        assert task.normalize == self.normalize_
        if task.normalize and task.distance_metric != "precomputed":
            assert task.data is self.data
            assert isinstance(task.normalized_data, Table) and \
                len(task.normalized_data) == len(self.data)

    def __ensure_task_same_for_pca(self, task: Task):
        assert task.use_pca_preprocessing == self.use_pca_preprocessing_
        if task.use_pca_preprocessing and task.distance_metric != "precomputed":
            assert task.data is self.data
            assert task.pca_components == self.pca_components
            assert isinstance(task.pca_projection, Table) and \
                len(task.pca_projection) == len(self.data)

    def __ensure_task_same_for_initialization(self, task: Task):
        if self.distance_matrix is not None:
            n_samples = self.distance_matrix.shape[0]
        else:
            initialization_method = INITIALIZATIONS[self.initialization_method_idx][1]
            # If distance matrix is provided, the control value will be set to
            # whatever it was from the context, but we will use `spectral`
            assert task.initialization_method == initialization_method
            assert self.data is not None
            n_samples = self.data.X.shape[0]
        assert isinstance(task.initialization, np.ndarray) and \
            len(task.initialization) == n_samples

    def __ensure_task_same_for_affinities(self, task: Task):
        assert task.perplexity == self.perplexity
        assert task.multiscale == self.multiscale
        distance_metric = DISTANCE_METRICS[self.distance_metric_idx][1]
        # Precomputed distances will never match the combo box value
        if task.distance_metric != "precomputed":
            assert task.distance_metric == distance_metric

    def __ensure_task_same_for_embedding(self, task: Task):
        assert task.exaggeration == self.exaggeration
        if self.distance_matrix is not None:
            n_samples = self.distance_matrix.shape[0]
        else:
            assert self.data is not None
            n_samples = self.data.X.shape[0]
        assert isinstance(task.tsne_embedding, manifold.TSNEModel) and \
            len(task.tsne_embedding.embedding) == n_samples

    def on_partial_result(self, value):
        # type: (Tuple[str, Task]) -> None
        which, task = value

        if which == "preprocessed_data":
            self.__ensure_task_same_for_preprocessing(task)
            self.preprocessed_data = task.preprocessed_data
        elif which == "normalized_data":
            self.__ensure_task_same_for_preprocessing(task)
            self.__ensure_task_same_for_normalization(task)
            self.normalized_data = task.normalized_data
        elif which == "pca_projection":
            self.__ensure_task_same_for_preprocessing(task)
            self.__ensure_task_same_for_normalization(task)
            self.__ensure_task_same_for_pca(task)
            self.pca_projection = task.pca_projection
        elif which == "initialization":
            self.__ensure_task_same_for_preprocessing(task)
            self.__ensure_task_same_for_normalization(task)
            self.__ensure_task_same_for_pca(task)
            self.__ensure_task_same_for_initialization(task)
            self.initialization = task.initialization
        elif which == "affinities":
            self.__ensure_task_same_for_preprocessing(task)
            self.__ensure_task_same_for_normalization(task)
            self.__ensure_task_same_for_pca(task)
            self.__ensure_task_same_for_affinities(task)
            self.affinities = task.affinities
        elif which == "tsne_embedding":
            self.__ensure_task_same_for_preprocessing(task)
            self.__ensure_task_same_for_normalization(task)
            self.__ensure_task_same_for_pca(task)
            self.__ensure_task_same_for_initialization(task)
            self.__ensure_task_same_for_affinities(task)
            self.__ensure_task_same_for_embedding(task)

            prev_embedding, self.tsne_embedding = self.tsne_embedding, task.tsne_embedding
            self.iterations_done = task.iterations_done
            # If this is the first partial result we've gotten, we've got to
            # set up the plot
            if prev_embedding is None:
                self.setup_plot()
            # Otherwise, just update the point positions
            else:
                self.graph.update_coordinates()
                self.graph.update_density()
        else:
            raise RuntimeError(
                "Unrecognized partial result called with `%s`" % which
            )

    def on_done(self, task):
        # type: (Task) -> None
        self.run_button.setText("Start")
        # NOTE: All of these have already been set by on_partial_result,
        # we double-check that they are aliases
        if task.preprocessed_data is not None:
            self.__ensure_task_same_for_preprocessing(task)
            assert task.preprocessed_data is self.preprocessed_data
        if task.normalized_data is not None:
            self.__ensure_task_same_for_normalization(task)
            assert task.normalized_data is self.normalized_data
        if task.pca_projection is not None:
            self.__ensure_task_same_for_pca(task)
            assert task.pca_projection is self.pca_projection
        if task.initialization is not None:
            self.__ensure_task_same_for_initialization(task)
            assert task.initialization is self.initialization
        if task.affinities is not None:
            assert task.affinities is self.affinities
        if task.tsne_embedding is not None:
            self.__ensure_task_same_for_embedding(task)
            assert task.tsne_embedding is self.tsne_embedding

        self.commit.deferred()

    def cancel(self):
        self.run_button.setText("Start")
        super().cancel()

    def clear(self):
        """Clear widget state. Note that this doesn't clear the data."""
        super().clear()
        self.cancel()
        self.preprocessed_data = None
        self.normalized_data = None
        self.pca_projection = None
        self.initialization = None
        self.affinities = None
        self.tsne_embedding = None
        self.iterations_done = 0

    def onDeleteWidget(self):
        self.clear()
        self.data = None
        self.shutdown()
        super().onDeleteWidget()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 3:
            if "selection_indices" in settings:
                settings["selection"] = settings["selection_indices"]
        if version < 4:
            settings.pop("max_iter", None)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


if __name__ == "__main__":
    import sys
    data = Table(sys.argv[1] if len(sys.argv) > 1 else "iris")
    from Orange.distance import Euclidean
    dist_matrix = Euclidean(data, normalize=True)
    WidgetPreview(OWtSNE).run(
        set_distances=dist_matrix,
        set_subset_data=data[np.random.choice(len(data), 10)],
    )

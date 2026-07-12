from AnyQt.QtCore import Qt
from sklearn.preprocessing import normalize

from Orange.data import Table
from Orange.modelling import KNNLearner
from Orange.preprocess import PreprocessorList
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview

class NormalizeInstancesL2:
    """Normalize each instance to unit L2 norm."""
    _reprable_module = True

    def __call__(self, data):
        if data is None or data.X is None or data.X.shape[0] == 0:
            return data

        return Table.from_numpy(
            data.domain,
            normalize(data.X, norm="l2", axis=1),
            Y=data.Y,
            metas=data.metas,
            W=data.W if data.has_weights() else None,
        )

    def __repr__(self):
        return "Normalize instances (L2)"

class OWKNNLearner(OWBaseLearner):
    name = "kNN"
    description = "Predict according to the nearest training instances."
    icon = "icons/KNN-symbolic.svg"
    replaces = [
        "Orange.widgets.classify.owknn.OWKNNLearner",
        "Orange.widgets.regression.owknnregression.OWKNNRegression",
    ]
    priority = 20
    keywords = "knn, k nearest, knearest, neighbor, neighbour"

    LEARNER = KNNLearner

    weights = ["uniform", "distance"]
    metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]

    weights_options = ["Uniform", "By Distances"]
    metrics_options = ["Euclidean", "Manhattan", "Chebyshev", "Mahalanobis"]

    n_neighbors = Setting(5)
    metric_index = Setting(0)
    weight_index = Setting(0)
    normalize_instances_l2 = Setting(False)

    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.vBox(self.controlArea, "Neighbors")
        self.n_neighbors_spin = gui.spin(
            box, self, "n_neighbors", 1, 100, label="Number of neighbors:",
            alignment=Qt.AlignRight, callback=self.settings_changed,
            controlWidth=80)
        self.metrics_combo = gui.comboBox(
            box, self, "metric_index", orientation=Qt.Horizontal,
            label="Metric:", items=self.metrics_options,
            callback=self._on_metric_changed)
        self.normalize_l2_checkbox = gui.checkBox(
            box, self, "normalize_instances_l2",
            label="Normalize instances (L2)",
            callback=self.settings_changed)
        self.normalize_l2_checkbox.setToolTip(
            "Normalize each instance to unit L2 norm before learning. "
            "Recommended for embeddings when using Euclidean distance.")
        self.weights_combo = gui.comboBox(
            box, self, "weight_index", orientation=Qt.Horizontal,
            label="Weight:", items=self.weights_options,
            callback=self.settings_changed)
        self._update_l2_checkbox()

    def _on_metric_changed(self):
        self._update_l2_checkbox()
        self.settings_changed()

    def _update_l2_checkbox(self):
        # false positive, pylint: disable=invalid-sequence-index
        enabled = self.metrics[self.metric_index] == "euclidean"
        self.normalize_l2_checkbox.setEnabled(enabled)
        if not enabled and self.normalize_instances_l2:
            self.normalize_instances_l2 = False
            self.normalize_l2_checkbox.setChecked(False)

    def create_learner(self):
        preprocessors = self.preprocessors

        if self.normalize_instances_l2:
            l2_normalizer = NormalizeInstancesL2()

            if preprocessors is None:
                preprocessors = l2_normalizer
            elif isinstance(preprocessors, PreprocessorList):
                preprocessors = PreprocessorList(
                    preprocessors.preprocessors + [l2_normalizer]
                )
            else:
                preprocessors = PreprocessorList(
                    [preprocessors, l2_normalizer]
                )

        return self.LEARNER(
            n_neighbors=self.n_neighbors,
            # false positive, pylint: disable=invalid-sequence-index
            metric=self.metrics[self.metric_index],
            weights=self.weights[self.weight_index],
            preprocessors=preprocessors,
        )

    def get_learner_parameters(self):
        return (("Number of neighbours", self.n_neighbors),
                # false positive, pylint: disable=invalid-sequence-index
                ("Metric", self.metrics_options[self.metric_index]),
                ("Normalize instances", 
                    "L2" if self.normalize_instances_l2 else "No",
                ),
                # false positive, pylint: disable=invalid-sequence-index
                ("Weight", self.weights_options[self.weight_index]),)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWKNNLearner).run(Table("iris"))

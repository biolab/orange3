from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.modelling import RandomForestLearner
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg


class OWRandomForest(OWBaseLearner):
    name = "Random Forest"
    description = "Predict using an ensemble of decision trees."
    icon = "icons/RandomForest.svg"
    replaces = [
        "Orange.widgets.classify.owrandomforest.OWRandomForest",
        "Orange.widgets.regression.owrandomforestregression.OWRandomForestRegression",
    ]
    priority = 40
    keywords = []

    LEARNER = RandomForestLearner

    n_estimators = settings.Setting(10)
    max_features = settings.Setting(5)
    use_max_features = settings.Setting(False)
    use_random_state = settings.Setting(False)
    max_depth = settings.Setting(3)
    use_max_depth = settings.Setting(False)
    min_samples_split = settings.Setting(5)
    use_min_samples_split = settings.Setting(True)
    index_output = settings.Setting(0)
    class_weight = settings.Setting(False)

    class Error(OWBaseLearner.Error):
        not_enough_features = Msg("Insufficient number of attributes ({})")

    class Warning(OWBaseLearner.Warning):
        class_weights_used = Msg("Weighting by class may decrease performance.")

    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.vBox(self.controlArea, 'Basic Properties')
        self.n_estimators_spin = gui.spin(
            box, self, "n_estimators", minv=1, maxv=10000, controlWidth=80,
            alignment=Qt.AlignRight, label="Number of trees: ",
            callback=self.settings_changed)
        self.max_features_spin = gui.spin(
            box, self, "max_features", 2, 50, controlWidth=80,
            label="Number of attributes considered at each split: ",
            callback=self.settings_changed, checked="use_max_features",
            checkCallback=self.settings_changed, alignment=Qt.AlignRight,)
        self.random_state = gui.checkBox(
            box, self, "use_random_state", label="Replicable training",
            callback=self.settings_changed,
            attribute=Qt.WA_LayoutUsesWidgetRect)
        self.weights = gui.checkBox(
            box, self,
            "class_weight", label="Balance class distribution",
            callback=self.settings_changed,
            tooltip="Weigh classes inversely proportional to their frequencies.",
            attribute=Qt.WA_LayoutUsesWidgetRect
        )

        box = gui.vBox(self.controlArea, "Growth Control")
        self.max_depth_spin = gui.spin(
            box, self, "max_depth", 1, 50, controlWidth=80,
            label="Limit depth of individual trees: ", alignment=Qt.AlignRight,
            callback=self.settings_changed, checked="use_max_depth",
            checkCallback=self.settings_changed)
        self.min_samples_split_spin = gui.spin(
            box, self, "min_samples_split", 2, 1000, controlWidth=80,
            label="Do not split subsets smaller than: ",
            callback=self.settings_changed, checked="use_min_samples_split",
            checkCallback=self.settings_changed, alignment=Qt.AlignRight)

    def create_learner(self):
        self.Warning.class_weights_used.clear()
        common_args = {"n_estimators": self.n_estimators}
        if self.use_max_features:
            common_args["max_features"] = self.max_features
        if self.use_random_state:
            common_args["random_state"] = 0
        if self.use_max_depth:
            common_args["max_depth"] = self.max_depth
        if self.use_min_samples_split:
            common_args["min_samples_split"] = self.min_samples_split
        if self.class_weight:
            common_args["class_weight"] = "balanced"
            self.Warning.class_weights_used()

        return self.LEARNER(preprocessors=self.preprocessors, **common_args)

    def check_data(self):
        self.Error.not_enough_features.clear()
        if super().check_data():
            n_features = len(self.data.domain.attributes)
            if self.use_max_features and self.max_features > n_features:
                self.Error.not_enough_features(n_features)
                self.valid_data = False
        return self.valid_data

    def get_learner_parameters(self):
        """Called by send report to list the parameters of the learner."""
        return (
            ("Number of trees", self.n_estimators),
            ("Maximal number of considered features",
             self.max_features if self.use_max_features else "unlimited"),
            ("Replicable training", ["No", "Yes"][self.use_random_state]),
            ("Maximal tree depth",
             self.max_depth if self.use_max_depth else "unlimited"),
            ("Stop splitting nodes with maximum instances",
             self.min_samples_split if self.use_min_samples_split else
             "unlimited"),
            ("Class weights", self.class_weight)
        )


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWRandomForest).run(Table("iris"))

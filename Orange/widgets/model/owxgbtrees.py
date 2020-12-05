from typing import Union, Tuple

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.modelling import XGBLearner, XGBRFLearner

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWXGBTrees(OWBaseLearner):
    name = "Gradient Boosted Trees"
    description = "Predict using gradient boosted trees."
    icon = "icons/XGB.svg"
    priority = 45
    keywords = ["xgboost", "gradient", "boost", "tree", "forest", "xgb", "gb"]

    LEARNER = XGBLearner

    n_estimators = Setting(100)
    boost_random_forest = Setting(False)

    max_depth = Setting(6)
    learning_rate = Setting(0.3)
    gamma = Setting(0)

    subsample = Setting(1)
    colsample_bytree = Setting(1)
    colsample_bylevel = Setting(1)
    colsample_bynode = Setting(1)

    reg_alpha = Setting(0)
    reg_lambda = Setting(1)

    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.vBox(self.controlArea, "Parameters")
        gui.spin(
            box, self, "n_estimators", 1, 10000,
            label="Number of trees:", alignment=Qt.AlignRight,
            callback=self.settings_changed, controlWidth=80
        )
        gui.checkBox(
            box, self, "boost_random_forest", label="Boost random forest",
            callback=self.settings_changed
        )

        box = gui.vBox(self.controlArea, "Booster")
        gui.spin(
            box, self, "max_depth", 1, 50, controlWidth=80,
            label="Maximum tree depth: ", alignment=Qt.AlignRight,
            callback=self.settings_changed
        )
        gui.doubleSpin(
            box, self, "learning_rate", 0, 1, 0.001, controlWidth=80,
            label="Learning rate: ", alignment=Qt.AlignRight,
            callback=self.settings_changed
        )
        gui.doubleSpin(
            box, self, "gamma", 0, 10000, 0.001, controlWidth=80,
            label="Minimum split loss: ", alignment=Qt.AlignRight,
            callback=self.settings_changed
        )

        box = gui.vBox(self.controlArea, "Subsampling")
        gui.doubleSpin(
            box, self, "subsample", 0.01, 1, 0.01, controlWidth=80,
            label="Proportion of training instances: ",
            alignment=Qt.AlignRight, callback=self.settings_changed
        )
        gui.doubleSpin(
            box, self, "colsample_bytree", 0.01, 1, 0.01, controlWidth=80,
            label="Proportion of features for each tree: ",
            alignment=Qt.AlignRight, callback=self.settings_changed
        )
        gui.doubleSpin(
            box, self, "colsample_bylevel", 0.01, 1, 0.01, controlWidth=80,
            label="Proportion of features for each level: ",
            alignment=Qt.AlignRight, callback=self.settings_changed
        )
        gui.doubleSpin(
            box, self, "colsample_bynode", 0.01, 1, 0.01, controlWidth=80,
            label="Proportion of features for each split: ",
            alignment=Qt.AlignRight, callback=self.settings_changed
        )

        box = gui.vBox(self.controlArea, "Regularization")
        gui.doubleSpin(
            box, self, "reg_alpha", 0, 1000, 1e-5, controlWidth=80,
            label="L1 regularization: ", alignment=Qt.AlignRight,
            callback=self.settings_changed
        )
        gui.doubleSpin(
            box, self, "reg_lambda", 0, 1000, 1e-5, controlWidth=80,
            label="L2 regularization: ", alignment=Qt.AlignRight,
            callback=self.settings_changed
        )

    def create_learner(self) -> Union[XGBLearner, XGBRFLearner]:
        params = ("n_estimators", "max_depth", "learning_rate", "gamma",
                  "subsample", "colsample_bytree", "colsample_bylevel",
                  "colsample_bynode", "reg_alpha", "reg_lambda")
        kwargs = {param: getattr(self, param) for param in params}
        learner = XGBRFLearner if self.boost_random_forest else XGBLearner
        return learner(preprocessors=self.preprocessors, **kwargs)

    def get_learner_parameters(self) -> Tuple:
        return (
            ("Number of trees", self.n_estimators),
            ("Boost random forest",
             ["No", "Yes"][int(self.boost_random_forest)]),
            ("Maximum tree depth", self.max_depth),
            ("Learning rate", self.learning_rate),
            ("Minimum split loss", self.gamma),
            ("Proportion of training instances", self.subsample),
            ("Proportion of features for each tree", self.colsample_bytree),
            ("Proportion of features for each level", self.colsample_bylevel),
            ("Proportion of features for each split", self.colsample_bynode),
            ("L1 regularization", self.reg_alpha),
            ("L2 regularization", self.reg_lambda),
        )


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWXGBTrees).run(Table("iris"))

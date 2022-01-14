from itertools import chain
from random import randint
from typing import Tuple, Dict, Callable, Type

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QStandardItem, QStandardItemModel
from AnyQt.QtWidgets import QWidget, QVBoxLayout

from Orange.base import Learner
from Orange.data import Table
from Orange.modelling import GBLearner

try:
    from Orange.modelling import CatGBLearner
except ImportError:
    CatGBLearner = None
try:
    from Orange.modelling import XGBLearner, XGBRFLearner
except ImportError:
    XGBLearner = XGBRFLearner = None

from Orange.widgets import gui
from Orange.widgets.settings import Setting, SettingProvider
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview


class LearnerItemModel(QStandardItemModel):
    LEARNERS = [
        (GBLearner, "", ""),
        (XGBLearner, "Extreme Gradient Boosting (xgboost)", "xgboost"),
        (XGBRFLearner, "Extreme Gradient Boosting Random Forest (xgboost)",
         "xgboost"),
        (CatGBLearner, "Gradient Boosting (catboost)", "catboost"),
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self._add_data()

    def _add_data(self):
        for cls, opt_name, lib in self.LEARNERS:
            item = QStandardItem()
            imported = bool(cls)
            name = cls.name if imported else opt_name
            item.setData(f"{name}", Qt.DisplayRole)
            item.setEnabled(imported)
            if not imported:
                item.setToolTip(f"{lib} is not installed")
            self.appendRow(item)


class BaseEditor(QWidget, gui.OWComponent):
    learner_class: Type[Learner] = NotImplemented
    n_estimators: int = NotImplemented
    learning_rate: float = NotImplemented
    random_state: bool = NotImplemented
    max_depth: int = NotImplemented

    def __init__(self, parent: OWBaseLearner):
        QWidget.__init__(self, parent)
        gui.OWComponent.__init__(self, parent)
        self.settings_changed: Callable = parent.settings_changed

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._layout: QWidget = gui.vBox(self, spacing=6, margin=0)

        self._add_main_layout()

    def _add_main_layout(self):
        common_args = {"callback": self.settings_changed,
                       "alignment": Qt.AlignRight, "controlWidth": 80}
        self.basic_box = gui.vBox(self._layout, "Basic Properties")
        gui.spin(
            self.basic_box, self, "n_estimators", 1, 10000,
            label="Number of trees:", **common_args
        )
        gui.doubleSpin(
            self.basic_box, self, "learning_rate", 0, 1, 0.001,
            label="Learning rate: ", **common_args
        )
        gui.checkBox(
            self.basic_box, self, "random_state", label="Replicable training",
            callback=self.settings_changed, attribute=Qt.WA_LayoutUsesWidgetRect
        )

        self.growth_box = gui.vBox(self._layout, "Growth Control")
        gui.spin(
            self.growth_box, self, "max_depth", 1, 50,
            label="Limit depth of individual trees: ", **common_args
        )

        self.sub_box = gui.vBox(self._layout, "Subsampling")

    def get_arguments(self) -> Dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": 0 if self.random_state else randint(1, 1000000),
            "max_depth": self.max_depth,
        }

    def get_learner_parameters(self) -> Tuple:
        return (
            ("Method", self.learner_class.name),
            ("Number of trees", self.n_estimators),
            ("Learning rate", self.learning_rate),
            ("Replicable training", "Yes" if self.random_state else "No"),
            ("Maximum tree depth", self.max_depth),
        )


class RegEditor(BaseEditor):
    LAMBDAS = list(chain([x / 10000 for x in range(1, 10)],
                         [x / 1000 for x in range(1, 20)],
                         [x / 100 for x in range(2, 20)],
                         [x / 10 for x in range(2, 9)],
                         range(1, 20),
                         range(20, 100, 5),
                         range(100, 1001, 100)))

    lambda_index: int = NotImplemented

    @property
    def lambda_(self):
        return self.LAMBDAS[int(self.lambda_index)]

    def _add_main_layout(self):
        super()._add_main_layout()
        # Basic properties
        box = self.basic_box
        gui.separator(box, height=1)
        gui.widgetLabel(box, "Regularization:")
        gui.hSlider(
            box, self, "lambda_index", minValue=0, createLabel=False,
            maxValue=len(self.LAMBDAS) - 1, callback=self._set_lambda_label,
            callback_finished=self.settings_changed
        )
        box2 = gui.hBox(box)
        box2.layout().setAlignment(Qt.AlignCenter)
        self.lambda_label = gui.widgetLabel(box2, "")
        self._set_lambda_label()

    def _set_lambda_label(self):
        self.lambda_label.setText("Lambda: {}".format(self.lambda_))

    def get_arguments(self) -> Dict:
        params = super().get_arguments()
        params["reg_lambda"] = self.lambda_
        return params

    def get_learner_parameters(self) -> Tuple:
        return super().get_learner_parameters() + (
            ("Regularization strength", self.lambda_),
        )


class GBLearnerEditor(BaseEditor):
    learner_class = GBLearner
    n_estimators = Setting(100)
    learning_rate = Setting(0.1)
    random_state = Setting(True)
    subsample = Setting(1)
    max_depth = Setting(3)
    min_samples_split = Setting(2)

    def _add_main_layout(self):
        super()._add_main_layout()
        # Subsampling
        gui.doubleSpin(
            self.sub_box, self, "subsample", 0.05, 1, 0.05,
            controlWidth=80, alignment=Qt.AlignRight,
            label="Fraction of training instances: ",
            callback=self.settings_changed
        )
        # Growth control
        gui.spin(
            self.growth_box, self, "min_samples_split", 2, 1000,
            controlWidth=80, label="Do not split subsets smaller than: ",
            alignment=Qt.AlignRight, callback=self.settings_changed
        )

    def get_arguments(self) -> Dict:
        params = super().get_arguments()
        params["subsample"] = self.subsample
        params["min_samples_split"] = self.min_samples_split
        return params

    def get_learner_parameters(self) -> Tuple:
        return super().get_learner_parameters() + (
            ("Fraction of training instances", self.subsample),
            ("Stop splitting nodes with maximum instances",
             self.min_samples_split),
        )


class CatGBLearnerEditor(RegEditor):
    learner_class = CatGBLearner
    n_estimators = Setting(100)
    learning_rate = Setting(0.3)
    random_state = Setting(True)
    max_depth = Setting(6)
    lambda_index = Setting(55)  # 3
    colsample_bylevel = Setting(1)

    def _add_main_layout(self):
        super()._add_main_layout()
        # Subsampling
        gui.doubleSpin(
            self.sub_box, self, "colsample_bylevel", 0.05, 1, 0.05,
            controlWidth=80, alignment=Qt.AlignRight,
            label="Fraction of features for each tree: ",
            callback=self.settings_changed
        )

    def get_arguments(self) -> Dict:
        params = super().get_arguments()
        params["colsample_bylevel"] = self.colsample_bylevel
        return params

    def get_learner_parameters(self) -> Tuple:
        return super().get_learner_parameters() + (
            ("Fraction of features for each tree", self.colsample_bylevel),
        )


class XGBBaseEditor(RegEditor):
    learner_class = XGBLearner
    n_estimators = Setting(100)
    learning_rate = Setting(0.3)
    random_state = Setting(True)
    max_depth = Setting(6)
    lambda_index = Setting(53)  # 1
    subsample = Setting(1)
    colsample_bytree = Setting(1)
    colsample_bylevel = Setting(1)
    colsample_bynode = Setting(1)

    def _add_main_layout(self):
        super()._add_main_layout()
        # Subsampling
        common_args = {"callback": self.settings_changed,
                       "alignment": Qt.AlignRight, "controlWidth": 80}
        gui.doubleSpin(
            self.sub_box, self, "subsample", 0.05, 1, 0.05,
            label="Fraction of training instances: ", **common_args
        )
        gui.doubleSpin(
            self.sub_box, self, "colsample_bytree", 0.05, 1, 0.05,
            label="Fraction of features for each tree: ", **common_args
        )
        gui.doubleSpin(
            self.sub_box, self, "colsample_bylevel", 0.05, 1, 0.05,
            label="Fraction of features for each level: ", **common_args
        )
        gui.doubleSpin(
            self.sub_box, self, "colsample_bynode", 0.05, 1, 0.05,
            label="Fraction of features for each split: ", **common_args
        )

    def get_arguments(self) -> Dict:
        params = super().get_arguments()
        params["subsample"] = self.subsample
        params["colsample_bytree"] = self.colsample_bytree
        params["colsample_bylevel"] = self.colsample_bylevel
        params["colsample_bynode"] = self.colsample_bynode
        return params

    def get_learner_parameters(self) -> Tuple:
        return super().get_learner_parameters() + (
            ("Fraction of training instances", self.subsample),
            ("Fraction of features for each tree", self.colsample_bytree),
            ("Fraction of features for each level", self.colsample_bylevel),
            ("Fraction of features for each split", self.colsample_bynode),
        )


class XGBLearnerEditor(XGBBaseEditor):
    learner_class = XGBLearner


class XGBRFLearnerEditor(XGBBaseEditor):
    learner_class = XGBRFLearner


class OWGradientBoosting(OWBaseLearner):
    name = "Gradient Boosting"
    description = "Predict using gradient boosting on decision trees."
    icon = "icons/GradientBoosting.svg"
    priority = 45
    keywords = ["catboost", "gradient", "boost", "tree", "forest",
                "xgb", "gb", "extreme"]

    LEARNER: Learner = GBLearner
    editor: BaseEditor = None

    gb_editor = SettingProvider(GBLearnerEditor)
    xgb_editor = SettingProvider(XGBLearnerEditor)
    xgbrf_editor = SettingProvider(XGBRFLearnerEditor)
    catgb_editor = SettingProvider(CatGBLearnerEditor)
    method_index = Setting(0)

    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.vBox(self.controlArea, "Method")
        gui.comboBox(
            box, self, "method_index", model=LearnerItemModel(self),
            callback=self.__method_changed
        )
        self.gb_editor = GBLearnerEditor(self)
        self.xgb_editor = XGBLearnerEditor(self)
        self.xgbrf_editor = XGBRFLearnerEditor(self)
        self.catgb_editor = CatGBLearnerEditor(self)
        self.editors = [self.gb_editor, self.xgb_editor,
                        self.xgbrf_editor, self.catgb_editor]

        editor_box = gui.widgetBox(self.controlArea)
        for editor in self.editors:
            editor_box.layout().addWidget(editor)
            editor.hide()

        if self.editors[int(self.method_index)].learner_class is None:
            self.method_index = 0
        self.editor = self.editors[int(self.method_index)]
        self.editor.show()

    def __method_changed(self):
        self.editor.hide()
        self.editor = self.editors[int(self.method_index)]
        self.editor.show()
        self.settings_changed()

    def create_learner(self) -> Learner:
        learner = self.editor.learner_class
        kwargs = self.editor.get_arguments()
        return learner(preprocessors=self.preprocessors, **kwargs)

    def get_learner_parameters(self) -> Tuple:
        return self.editor.get_learner_parameters()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWGradientBoosting).run(Table("iris"))

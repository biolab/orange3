import json
import unittest
from typing import Type
from unittest.mock import Mock

from Orange.classification import GBClassifier

from Orange.classification import XGBClassifier, XGBRFClassifier
from Orange.classification import CatGBClassifier
from Orange.data import Table
from Orange.preprocess.score import Scorer
from Orange.regression import GBRegressor

from Orange.regression import XGBRegressor, XGBRFRegressor
from Orange.regression import CatGBRegressor
from Orange.widgets.model.owgradientboosting import OWGradientBoosting, \
    LearnerItemModel, GBLearnerEditor, XGBLearnerEditor, XGBRFLearnerEditor, \
    CatGBLearnerEditor, BaseEditor
from Orange.widgets.settings import SettingProvider
from Orange.widgets.tests.base import WidgetTest, ParameterMapping, \
    WidgetLearnerTestMixin, datasets, simulate, GuiTest
from Orange.widgets.widget import OWWidget


def get_tree_train_params(model):
    ln = json.loads(model.skl_model.get_booster().save_config())["learner"]
    try:
        return ln["gradient_booster"]["tree_train_param"]
    except KeyError:
        return ln["gradient_booster"]["updater"]["grow_colmaker"]["train_param"]


def create_parent(editor_class):
    class DummyWidget(OWWidget):
        name = "Mock"
        settings_changed = Mock()
        editor = SettingProvider(editor_class)

    return DummyWidget()


class TestLearnerItemModel(GuiTest):
    def test_model(self):
        classifiers = [GBClassifier, XGBClassifier,
                       XGBRFClassifier, CatGBClassifier]
        widget = create_parent(CatGBLearnerEditor)
        model = LearnerItemModel(widget)
        n_items = 4
        self.assertEqual(model.rowCount(), n_items)
        for i in range(n_items):
            self.assertEqual(model.item(i).isEnabled(),
                             classifiers[i] is not None)


class BaseEditorTest(GuiTest):
    EditorClass: Type[BaseEditor] = None

    def setUp(self):
        super().setUp()
        editor_class = self.EditorClass
        self.widget = create_parent(editor_class)
        self.editor = editor_class(self.widget)  # pylint: disable=not-callable

    def tearDown(self) -> None:
        self.widget.deleteLater()
        super().tearDown()


class TestGBLearnerEditor(BaseEditorTest):
    EditorClass = GBLearnerEditor

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                "random_state": 0, "subsample": 1, "min_samples_split": 2}
        self.assertDictEqual(self.editor.get_arguments(), args)

    def test_learner_parameters(self):
        params = (("Method", "Gradient Boosting (scikit-learn)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.1),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 3),
                  ("Fraction of training instances", 1),
                  ("Stop splitting nodes with maximum instances", 2))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = GBClassifier()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(params["learning_rate"], self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["min_samples_split"],
                         self.editor.min_samples_split)
        self.assertTrue(self.editor.random_state)  # different than default
        self.assertIsNone(params["random_state"])

    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = GBRegressor()
        model = booster(data)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(params["learning_rate"], self.editor.learning_rate)
        self.assertEqual(params["max_depth"], self.editor.max_depth)
        self.assertEqual(params["subsample"], self.editor.subsample)
        self.assertEqual(params["min_samples_split"],
                         self.editor.min_samples_split)
        self.assertTrue(self.editor.random_state)  # different than default
        self.assertIsNone(params["random_state"])


class TestXGBLearnerEditor(BaseEditorTest):
    EditorClass = XGBLearnerEditor

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "reg_lambda": 1, "colsample_bytree": 1, "colsample_bylevel": 1,
                "colsample_bynode": 1, "subsample": 1, "random_state": 0}
        self.assertDictEqual(self.editor.get_arguments(), args)

    def test_learner_parameters(self):
        params = (("Method", "Extreme Gradient Boosting (xgboost)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.3),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 6),
                  ("Regularization strength", 1),
                  ("Fraction of training instances", 1),
                  ("Fraction of features for each tree", 1),
                  ("Fraction of features for each level", 1),
                  ("Fraction of features for each split", 1))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = XGBClassifier()
        model = booster(data)
        params = model.skl_model.get_params()
        tp = get_tree_train_params(model)
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(
            round(float(tp["learning_rate"]), 1), self.editor.learning_rate
        )
        self.assertEqual(int(tp["max_depth"]), self.editor.max_depth)
        self.assertEqual(float(tp["reg_lambda"]), self.editor.lambda_)
        self.assertEqual(int(tp["subsample"]), self.editor.subsample)
        self.assertEqual(int(tp["colsample_bytree"]), self.editor.colsample_bytree)
        self.assertEqual(int(tp["colsample_bylevel"]), self.editor.colsample_bylevel)
        self.assertEqual(int(tp["colsample_bynode"]), self.editor.colsample_bynode)

    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = XGBRegressor()
        model = booster(data)
        params = model.skl_model.get_params()
        tp = get_tree_train_params(model)
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(
            round(float(tp["learning_rate"]), 1), self.editor.learning_rate
        )
        self.assertEqual(int(tp["max_depth"]), self.editor.max_depth)
        self.assertEqual(float(tp["reg_lambda"]), self.editor.lambda_)
        self.assertEqual(int(tp["subsample"]), self.editor.subsample)
        self.assertEqual(int(tp["colsample_bytree"]), self.editor.colsample_bytree)
        self.assertEqual(int(tp["colsample_bylevel"]), self.editor.colsample_bylevel)
        self.assertEqual(int(tp["colsample_bynode"]), self.editor.colsample_bynode)


class TestXGBRFLearnerEditor(BaseEditorTest):
    EditorClass = XGBRFLearnerEditor

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "reg_lambda": 1, "colsample_bytree": 1, "colsample_bylevel": 1,
                "colsample_bynode": 1, "subsample": 1, "random_state": 0}
        self.assertDictEqual(self.editor.get_arguments(), args)

    def test_learner_parameters(self):
        params = (("Method",
                   "Extreme Gradient Boosting Random Forest (xgboost)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.3),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 6),
                  ("Regularization strength", 1),
                  ("Fraction of training instances", 1),
                  ("Fraction of features for each tree", 1),
                  ("Fraction of features for each level", 1),
                  ("Fraction of features for each split", 1))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = XGBRFClassifier()
        model = booster(data)
        params = model.skl_model.get_params()
        tp = get_tree_train_params(model)
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(
            round(float(tp["learning_rate"]), 1), self.editor.learning_rate
        )
        self.assertEqual(int(tp["max_depth"]), self.editor.max_depth)
        self.assertEqual(float(tp["reg_lambda"]), self.editor.lambda_)
        self.assertEqual(int(tp["subsample"]), self.editor.subsample)
        self.assertEqual(int(tp["colsample_bytree"]), self.editor.colsample_bytree)
        self.assertEqual(int(tp["colsample_bylevel"]), self.editor.colsample_bylevel)
        self.assertEqual(int(tp["colsample_bynode"]), self.editor.colsample_bynode)

    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = XGBRFRegressor()
        model = booster(data)
        params = model.skl_model.get_params()
        tp = get_tree_train_params(model)
        self.assertEqual(params["n_estimators"], self.editor.n_estimators)
        self.assertEqual(
            round(float(tp["learning_rate"]), 1), self.editor.learning_rate
        )
        self.assertEqual(int(tp["max_depth"]), self.editor.max_depth)
        self.assertEqual(float(tp["reg_lambda"]), self.editor.lambda_)
        self.assertEqual(int(tp["subsample"]), self.editor.subsample)
        self.assertEqual(int(tp["colsample_bytree"]), self.editor.colsample_bytree)
        self.assertEqual(int(tp["colsample_bylevel"]), self.editor.colsample_bylevel)
        self.assertEqual(int(tp["colsample_bynode"]), self.editor.colsample_bynode)


class TestCatGBLearnerEditor(BaseEditorTest):
    EditorClass = CatGBLearnerEditor

    def test_arguments(self):
        args = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
                "reg_lambda": 3, "colsample_bylevel": 1, "random_state": 0}
        self.assertDictEqual(self.editor.get_arguments(), args)

    def test_learner_parameters(self):
        params = (("Method", "Gradient Boosting (catboost)"),
                  ("Number of trees", 100),
                  ("Learning rate", 0.3),
                  ("Replicable training", "Yes"),
                  ("Maximum tree depth", 6),
                  ("Regularization strength", 3),
                  ("Fraction of features for each tree", 1))
        self.assertTupleEqual(self.editor.get_learner_parameters(), params)

    def test_default_parameters_cls(self):
        data = Table("heart_disease")
        booster = CatGBClassifier()
        model = booster(data)
        params = model.cat_model.get_all_params()
        self.assertEqual(self.editor.n_estimators, 100)
        self.assertEqual(params["iterations"], 1000)
        self.assertEqual(params["depth"], self.editor.max_depth)
        self.assertEqual(params["l2_leaf_reg"], self.editor.lambda_)
        self.assertEqual(params["rsm"], self.editor.colsample_bylevel)
        self.assertEqual(self.editor.learning_rate, 0.3)
        # params["learning_rate"] is automatically defined so don't test it

    def test_default_parameters_reg(self):
        data = Table("housing")
        booster = CatGBRegressor()
        model = booster(data)
        params = model.cat_model.get_all_params()
        self.assertEqual(self.editor.n_estimators, 100)
        self.assertEqual(params["iterations"], 1000)
        self.assertEqual(params["depth"], self.editor.max_depth)
        self.assertEqual(params["l2_leaf_reg"], self.editor.lambda_)
        self.assertEqual(params["rsm"], self.editor.colsample_bylevel)
        self.assertEqual(self.editor.learning_rate, 0.3)
        # params["learning_rate"] is automatically defined so don't test it


class TestOWGradientBoosting(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWGradientBoosting,
                                         stored_settings={"auto_apply": False})
        self.init()

        controls = self.widget.editor.controls
        self.parameters = [
            ParameterMapping("n_estimators", controls.n_estimators, [500, 10]),
            ParameterMapping("learning_rate", controls.learning_rate),
            ParameterMapping("max_depth", controls.max_depth),
            ParameterMapping("min_samples_split", controls.min_samples_split),
        ]

    def test_scorer(self):
        self.assertIsInstance(
            self.get_output(self.widget.Outputs.learner), Scorer
        )

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)

    def test_xgb_params(self):
        simulate.combobox_activate_index(self.widget.controls.method_index, 1)
        editor = self.widget.editor
        controls = editor.controls
        reg_slider = controls.lambda_index

        self.parameters = [
            ParameterMapping("n_estimators", controls.n_estimators, [500, 10]),
            ParameterMapping("learning_rate", controls.learning_rate),
            ParameterMapping("max_depth", controls.max_depth),
            ParameterMapping("reg_lambda", reg_slider,
                             values=[editor.LAMBDAS[0], editor.LAMBDAS[-1]],
                             getter=lambda: editor.LAMBDAS[reg_slider.value()],
                             setter=lambda val: reg_slider.setValue(
                                 editor.LAMBDAS.index(val))),
        ]
        self.test_parameters()

    def test_methods(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        method_cb = self.widget.controls.method_index
        for i, cls in enumerate(LearnerItemModel.LEARNERS):
            simulate.combobox_activate_index(method_cb, i)
            self.click_apply()
            self.assertIsInstance(self.widget.learner, cls)


if __name__ == "__main__":
    unittest.main()

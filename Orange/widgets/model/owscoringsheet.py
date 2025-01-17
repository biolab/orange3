from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.base import Model
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.widget import Msg
from Orange.widgets import gui
from Orange.widgets.settings import Setting

from Orange.classification.scoringsheet import ScoringSheetLearner


class ScoringSheetRunner:
    @staticmethod
    def run(learner: ScoringSheetLearner, data: Table, state: TaskState) -> Model:
        if data is None:
            return None
        state.set_status("Learning...")
        model = learner(data)
        return model


class OWScoringSheet(OWBaseLearner, ConcurrentWidgetMixin):
    name = "Scoring Sheet"
    description = "A fast and explainable classifier."
    icon = "icons/ScoringSheet.svg"
    replaces = ["orangecontrib.prototypes.widgets.owscoringsheet.OWScoringSheet"]
    priority = 75
    keywords = "scoring sheet"

    LEARNER = ScoringSheetLearner

    class Inputs(OWBaseLearner.Inputs):
        pass

    class Outputs(OWBaseLearner.Outputs):
        pass

    # Preprocessing
    num_attr_after_selection = Setting(20)

    # Scoring Sheet Settings
    num_decision_params = Setting(5)
    max_points_per_param = Setting(5)
    custom_features_checkbox = Setting(False)
    num_input_features = Setting(1)

    # Warning messages
    class Information(OWBaseLearner.Information):
        custom_num_of_input_features = Msg(
            "If the number of input features used is too low for the number of decision \n"
            "parameters, the number of decision parameters will be adjusted to fit the model."
        )

    def __init__(self):
        ConcurrentWidgetMixin.__init__(self)
        OWBaseLearner.__init__(self)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Preprocessing")

        self.num_attr_after_selection_spin = gui.spin(
            box,
            self,
            "num_attr_after_selection",
            minv=1,
            maxv=100,
            step=1,
            label="Number of Attributes After Feature Selection:",
            orientation=Qt.Horizontal,
            alignment=Qt.AlignRight,
            callback=self.settings_changed,
            controlWidth=45,
        )

        box = gui.vBox(self.controlArea, "Model Parameters")

        gui.spin(
            box,
            self,
            "num_decision_params",
            minv=1,
            maxv=50,
            step=1,
            label="Maximum Number of Decision Parameters:",
            orientation=Qt.Horizontal,
            alignment=Qt.AlignRight,
            callback=self.settings_changed,
            controlWidth=45,
        )

        gui.spin(
            box,
            self,
            "max_points_per_param",
            minv=1,
            maxv=100,
            step=1,
            label="Maximum Points per Decision Parameter:",
            orientation=Qt.Horizontal,
            alignment=Qt.AlignRight,
            callback=self.settings_changed,
            controlWidth=45,
        )

        gui.checkBox(
            box,
            self,
            "custom_features_checkbox",
            label="Custom number of input features",
            callback=[self.settings_changed, self.custom_input_features],
        )

        self.custom_features = gui.spin(
            box,
            self,
            "num_input_features",
            minv=1,
            maxv=50,
            step=1,
            label="Number of Input Features Used:",
            orientation=Qt.Horizontal,
            alignment=Qt.AlignRight,
            callback=self.settings_changed,
            controlWidth=45,
        )

        self.custom_input_features()

    def custom_input_features(self):
        self.custom_features.setEnabled(self.custom_features_checkbox)
        if self.custom_features_checkbox:
            self.Information.custom_num_of_input_features()
        else:
            self.Information.custom_num_of_input_features.clear()
        self.apply()

    @Inputs.data
    def set_data(self, data):
        self.cancel()
        super().set_data(data)

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        self.cancel()
        super().set_preprocessor(preprocessor)

        # Enable or disable the spin box based on whether a preprocessor is set
        self.num_attr_after_selection_spin.setEnabled(preprocessor is None)
        if preprocessor:
            self.Information.ignored_preprocessors()
        else:
            self.Information.ignored_preprocessors.clear()

    def create_learner(self):
        return self.LEARNER(
            num_attr_after_selection=self.num_attr_after_selection,
            num_decision_params=self.num_decision_params,
            max_points_per_param=self.max_points_per_param,
            num_input_features=(
                self.num_input_features if self.custom_features_checkbox else None
            ),
            preprocessors=self.preprocessors,
        )

    def update_model(self):
        self.cancel()
        self.show_fitting_failed(None)
        self.model = None
        if self.data is not None:
            self.start(ScoringSheetRunner.run, self.learner, self.data)
        else:
            self.Outputs.model.send(None)

    def get_learner_parameters(self):
        return (
            self.num_decision_params,
            self.max_points_per_param,
            self.num_input_features,
        )

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Model):
        assert isinstance(result, Model) or result is None
        self.model = result
        self.Outputs.model.send(result)

    def on_exception(self, ex):
        self.cancel()
        self.Outputs.model.send(None)
        if isinstance(ex, BaseException):
            self.show_fitting_failed(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWScoringSheet).run()

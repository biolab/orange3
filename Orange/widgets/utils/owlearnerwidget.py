from PyQt4 import QtGui

import numpy as np
from PyQt4.QtCore import QTimer, Qt

from Orange.classification.base_classification import LearnerClassification
from Orange.data import Table
from Orange.wrappers import BaseWrapper, WrappersMix
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, WidgetMetaClass


class DefaultWidgetChannelsMetaClass(WidgetMetaClass):
    """Metaclass that adds default inputs and outputs objects.
    """

    REQUIRED_ATTRIBUTES = []

    def __new__(mcls, name, bases, attrib):
        # check whether it is abstract class
        if attrib.get('name', False):
            # Ensure all needed attributes are present
            if not all(attr in attrib for attr in mcls.REQUIRED_ATTRIBUTES):
                raise AttributeError("'{name}' must have '{attrs}' attributes"
                                     .format(name=name, attrs="', '".join(mcls.REQUIRED_ATTRIBUTES)))

            attrib['outputs'] = mcls.update_channel(
                mcls.default_outputs(attrib),
                attrib.get('outputs', [])
            )

            attrib['inputs'] = mcls.update_channel(
                mcls.default_inputs(attrib),
                attrib.get('inputs', [])
            )

            mcls.add_extra_attributes(name, attrib)

        return super().__new__(mcls, name, bases, attrib)

    @classmethod
    def default_inputs(cls, attrib):
        return []

    @classmethod
    def default_outputs(cls, attrib):
        return []

    @classmethod
    def update_channel(cls, channel, items):
        item_names = set(item[0] for item in channel)

        for item in items:
            if not item[0] in item_names:
                channel.append(item)

        return channel

    @classmethod
    def add_extra_attributes(cls, name, attrib):
        return attrib


class OWBaseLearnerMeta(DefaultWidgetChannelsMetaClass):
    """Metaclass that adds default inputs (table, preprocess) and
    outputs (learner, model) for learner widgets.
    """

    REQUIRED_ATTRIBUTES = ['LEARNER']

    @classmethod
    def default_inputs(cls, attrib):
        return [("Data", Table, "set_data"), ("Preprocessor", Preprocess, "set_preprocessor")]

    @classmethod
    def default_outputs(cls, attrib):
        learner_class = attrib['LEARNER']
        if issubclass(learner_class, LearnerClassification):
            model_name = 'Classifier'
        else:
            model_name = 'Predictor'

        attrib['OUTPUT_MODEL_NAME'] = model_name
        return [("Learner", learner_class),
                (model_name, attrib['LEARNER'].__returns__)]

    @classmethod
    def add_extra_attributes(cls, name, attrib):
        if 'learner_name' not in attrib:
            attrib['learner_name'] = Setting(attrib['name'])
        attrib['learner_settings'] = Setting({})
        return attrib


class OWBaseLearner(OWWidget, metaclass=OWBaseLearnerMeta):
    """Abstract widget for classification/regression learners.

    Attributes:
        LEARNER: Learner class
        learner (LEARNER): an instance of LEARNER
        learner_name (str): name of learner


    Notes:
        All learner widgets should define learner class LEARNER.
        LEARNER should have __returns__ attribute.

        Overwrite `create_learner`, `add_main_layout` and
        `get_learner_parameters` in case LEARNER has extra parameters.

    """
    LEARNER = None

    want_main_area = False
    resizing_enabled = False

    DATA_ERROR_ID = 1
    OUTDATED_LEARNER_WARNING_ID = 2

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = False

        if self.is_wrapper() and not hasattr(self, 'learner'):
            self.learner = self.LEARNER(**self.learner_settings)
            self.learner.callback = self.settings_changed
        else:
            self.learner = None

        self.model = None
        self.preprocessors = None
        self.outdated_settings = False
        self.setup_layout()
        QTimer.singleShot(0, self.apply)

    def create_learner(self):
        """Creates a learner with current configuration.

        Returns:
            Leaner: an instance of Orange.base.learner subclass.
        """
        if self.is_wrapper():
            return self.learner
        return self.LEARNER(preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        """Creates an `OrderedDict` or a sequence of pairs with current model
        configuration.

        Returns:
            OrderedDict or List: (option, value) pairs or dict
        """
        if self.is_wrapper():
            return ((v.option.verbose_name, v.value)
                    for v in self.learner._values.values())
        return []

    def set_preprocessor(self, preprocessor):
        """Add user-set preprocessors before the default, mandatory ones"""
        self.preprocessors = ((preprocessor,) if preprocessor else ()) + tuple(self.LEARNER.preprocessors)
        self.apply()

    @check_sql_input
    def set_data(self, data):
        """Set the input train data set."""
        self.error(self.DATA_ERROR_ID)
        self.data = data
        if data is not None and data.domain.class_var is None:
            self.error(self.DATA_ERROR_ID, "Data has no target variable")
            self.data = None

        self.update_model()

    def apply(self):
        """Applies changes and sends a new model."""
        self.update_learner()
        self.update_model()

    def update_learner(self):
        if self.is_wrapper():
            self.create_learner()
        else:
            self.learner = self.create_learner()
        self.learner.name = self.learner_name
        self.send("Learner", self.learner)
        self.outdated_settings = False
        self.warning(self.OUTDATED_LEARNER_WARNING_ID)

    def update_model(self):
        if self.check_data():
            self.model = self.learner(self.data)
            self.model.name = self.learner_name
            self.model.instances = self.data
            self.valid_data = True

        self.send(self.OUTPUT_MODEL_NAME, self.model)

    def check_data(self):
        self.valid_data = False
        if self.data is not None and self.learner is not None:
            self.error(self.DATA_ERROR_ID)
            if not self.learner.check_learner_adequacy(self.data.domain):
                self.error(self.DATA_ERROR_ID, self.learner.learner_adequacy_err_msg)
            elif len(np.unique(self.data.Y)) < 2:
                self.error(self.DATA_ERROR_ID,
                           "Data contains single target value. "
                           "There is nothing to learn.")
            else:
                self.valid_data = True
        return self.valid_data

    def settings_changed(self, *args, **kwargs):
        self.outdated_settings = True
        if self.is_wrapper():
            self.learner_settings = self.learner.state

        self.warning(self.OUTDATED_LEARNER_WARNING_ID,
                     "Press Apply to submit changes.")

    def send_report(self):
        self.report_items((('Learner', self.learner.name),))

        if self.learner.name != self.learner_name:
            self.report_items((("Model name", self.learner_name),))

        model_parameters = self.get_learner_parameters()
        if model_parameters:
            self.report_items("Model parameters", model_parameters)

        if self.data:
            self.report_data("Data", self.data)

    # GUI
    def setup_layout(self):
        self.add_learner_name_widget()
        self.add_main_layout()
        self.add_bottom_buttons()

    def add_main_layout(self):
        """Creates layout with the learner configuration widgets.

        Override this method for laying out any learner-specific parameter controls.
        See setup_layout() method for execution order.
        """
        if self.is_wrapper():
            self.learner.callback = self.settings_changed
            self.main_layout = self.learner.options_layout()
            self.main_layout.setMargin(8)
            self.controlArea.layout().addLayout(self.main_layout)

    def add_learner_name_widget(self):
        gui.lineEdit(self.controlArea, self, 'learner_name', box='Name',
                     tooltip='The name will identify this model in other widgets')

    def add_bottom_buttons(self):
        box = gui.hBox(self.controlArea, True)
        box.layout().addWidget(self.report_button)
        gui.separator(box, 15)
        self.apply_button = gui.button(box, self, "&Apply", callback=self.apply,
                                       disabled=0, default=True)

    def is_wrapper(self):
        return issubclass(self.LEARNER, BaseWrapper)


class OWBaseMultipleLearner(OWBaseLearner):
    ORIENTATION = Qt.Vertical

    @property
    def learner(self):
        return self.learners.checked

    @learner.setter
    def learner(self, learner):
        pass

    def __init__(self):
        self.learners = WrappersMix(self.Learners)
        self.learners.callback = self.settings_changed
        super().__init__()

    def add_main_layout(self):
        box = QtGui.QWidget()
        self.learners.options_layout(parent=box, orientation=self.ORIENTATION)
        self.controlArea.layout().addWidget(box)

from Orange.data import Table
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, WidgetMetaClass


class _OWAcceptsPreprocessor:
    """
    Accepts Preprocessor input.

    Requires `LEARNER` attribute with default `LEARNER.preprocessors` be set on it.

    Sets `self.preprocessors` tuple.

    Calls `apply()` method after setting preprocessors.
    """
    inputs = [("Preprocessor", Preprocess, "set_preprocessor")]

    def set_preprocessor(self, preproc):
        """Add user-set preprocessors before the default, mandatory ones"""
        self.preprocessors = ((preproc,) if preproc else ()) + tuple(self.LEARNER.preprocessors)
        self.apply()


class OWProvidesLearner(_OWAcceptsPreprocessor):
    """
    Base class for all classification / regression learner-providing widgets
    that extend it.
    """

    LEARNER = None
    OUTPUT_MODEL_NAME = None
    OUTPUT_MODEL_CLASS = None

    # because of backward compatibility we can't overwrite this attributes
    # inputs = [("Data", Table, "set_data")] + _OWAcceptsPreprocessor.inputs
    # outputs = [
    #     ("Learner", LEARNER),
    #     (OUTPUT_MODEL_NAME, OUTPUT_MODEL_CLASS)
    # ]
    # model_name = Setting(OUTPUT_MODEL_NAME)

    def __init__(self):
        super().__init__()

        self.data = None
        self.learner = None
        self.model = None
        # here can be gui configuration (e.g. gui.lineEdit for model_name)
        # we can add method get_setting_box()
        #
        # gui.lineEdit
        # box = get_setting_box()
        # apply and report buttons

    @check_sql_input
    def set_data(self, data):
        """Set the input train data set."""
        self.error(0)
        self.data = data
        if data is not None and data.domain.class_var is None:
            self.error(0, "Data has no target variable")
            self.data = None
        self.update_model()

    def apply(self):
        self.update_learner()
        self.update_model()

    def update_learner(self):
        # self.learner.name = self.model_name
        # self.send("Learner", self.learner)
        raise NotImplementedError()

    def update_model(self):
        self.model = None
        if self.data is not None:
            self.error(1)
            if not self.learner.check_learner_adequacy(self.data.domain):
                self.error(1, self.learner.learner_adequacy_err_msg)
            else:
                self.model = self.learner(self.data)
                self.model.name = self.model_name
                self.model.instances = self.data

        self.send(self.OUTPUT_MODEL_NAME, self.model)

    def send_report(self):
        raise NotImplementedError()


class ProviderMetaClass(WidgetMetaClass):
    """ Metaclass that add inputs, outputs and model_name setting.
    """
    def __new__(mcls, name, bases, attrib):

        # check whether it is abstract class
        if attrib.get('name', False):

            for attr in ['LEARNER', 'OUTPUT_MODEL_NAME', 'OUTPUT_MODEL_CLASS']:
                if attr not in attrib:
                    raise AttributeError("'{name}' must have  ")

            # allows have outputs = []
            if attrib.get('outputs', None) is None:
                attrib['outputs'] = [
                    ("Learner", attrib['LEARNER']),
                    (attrib['OUTPUT_MODEL_NAME'], attrib['OUTPUT_MODEL_CLASS'])
                ]
            # adds extra outputs
            if attrib.get('extra_outputs', None):
                attrib['outputs'] += attrib.pop('extra_outputs')

            if attrib.get('inputs', None) is None:
                attrib['inputs'] = [("Data", Table, "set_data")] + \
                                   _OWAcceptsPreprocessor.inputs

            if attrib.get('model_name', None) is None:
                attrib['model_name'] = Setting(attrib['OUTPUT_MODEL_CLASS'])

        return super(ProviderMetaClass, mcls).__new__(mcls, name, bases, attrib)


class OWProvidesLearnerI(OWProvidesLearner, OWWidget, metaclass=ProviderMetaClass):
    pass

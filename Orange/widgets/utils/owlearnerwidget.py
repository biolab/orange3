from Orange.preprocess.preprocess import Preprocess


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

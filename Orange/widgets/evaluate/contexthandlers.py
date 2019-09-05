from Orange.data import Variable
from Orange.widgets import settings


class EvaluationResultsContextHandler(settings.ContextHandler):
    """Context handler for evaluation results"""

    def open_context(self, widget, classes, classifier_names):
        if isinstance(classes, Variable):
            if classes.is_discrete:
                classes = classes.values
            else:
                classes = None
        super().open_context(widget, classes, classifier_names)

    def new_context(self, classes, classifier_names):
        context = super().new_context()
        context.classes = classes
        context.classifier_names = classifier_names
        return context

    def match(self, context, classes, classifier_names):
        if classifier_names != context.classifier_names:
            return self.NO_MATCH
        elif isinstance(classes, Variable) and classes.is_continuous:
            return (self.PERFECT_MATCH if context.classes is None
                    else self.NO_MATCH)
        else:
            return (self.PERFECT_MATCH if context.classes == classes
                    else self.NO_MATCH)

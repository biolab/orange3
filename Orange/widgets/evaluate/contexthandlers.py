from Orange.widgets import settings
from Orange.widgets.utils import getdeepattr


class EvaluationResultsContextHandler(settings.ContextHandler):
    def __init__(self, targetAttr, selectedAttr):
        super().__init__()
        self.targetAttr, self.selectedAttr = targetAttr, selectedAttr

    #noinspection PyMethodOverriding
    def match(self, context, cnames, cvalues):
        return (cnames, cvalues) == (
            context.classifierNames, context.classValues) and 2

    def fast_save(self, widget, name, value):
        context = widget.current_context
        if name == self.targetAttr:
            context.targetClass = value
        elif name == self.selectedAttr:
            context.selectedClassifiers = list(value)

    def settings_from_widget(self, widget):
        super().settings_from_widget(widget)
        context = widget.current_context
        context.targetClass = getdeepattr(widget, self.targetAttr)
        context.selectedClassifiers = list(getdeepattr(self.selectedAttr))

    def settings_to_widget(self, widget):
        super().settings_to_widget(widget)
        context = widget.current_context
        if context.targetClass is not None:
            setattr(widget, self.targetAttr, context.targetClass)
        if context.selectedClassifiers is not None:
            setattr(widget, self.selectedAttr, context.selectedClassifiers)

    #noinspection PyMethodOverriding
    def find_or_create_context(self, widget, results):
        cnames = [c.name for c in results.classifiers]
        cvalues = results.classValues
        context, isNew = super().find_or_create_context(
            widget, results.classifierNames, results.classValues)
        if isNew:
            context.classifierNames = results.classifierNames
            context.classValues = results.classValues
            context.selectedClassifiers = None
            context.targetClass = None
        return context, isNew

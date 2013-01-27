from Orange.widgets import settings

class EvaluationResultsContextHandler(settings.ContextHandler):
    def __init__(self, targetAttr, selectedAttr):
        super().__init__()
        self.targetAttr, self.selectedAttr = targetAttr, selectedAttr

    #noinspection PyMethodOverriding
    def match(self, context, cnames, cvalues):
        return (cnames, cvalues) == (
            context.classifierNames, context.classValues) and 2

    def fastSave(self, widget, name, value):
        context = widget.currentContext
        if name == self.targetAttr:
            context.targetClass = value
        elif name == self.selectedAttr:
            context.selectedClassifiers = list(value)

    def settingsFromWidget(self, widget):
        super().settingsFromWidget(widget)
        context = widget.currentContext
        context.targetClass = widget.getdeepattr(self.targetAttr)
        context.selectedClassifiers = list(widget.getdeepattr(self.selectedAttr))

    def settingsToWidget(self, widget):
        super().settingsToWidget(widget)
        context = widget.currentContext
        if context.targetClass is not None:
            setattr(widget, self.targetAttr, context.targetClass)
        if context.selectedClassifiers is not None:
            setattr(widget, self.selectedAttr, context.selectedClassifiers)

    #noinspection PyMethodOverriding
    def findOrCreateContext(self, widget, results):
        cnames = [c.name for c in results.classifiers]
        cvalues = results.classValues
        context, isNew = super().findOrCreateContext(
            widget, results.classifierNames, results.classValues)
        if isNew:
            context.classifierNames = results.classifierNames
            context.classValues = results.classValues
            context.selectedClassifiers = None
            context.targetClass = None
        return context, isNew

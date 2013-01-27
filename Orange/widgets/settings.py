import time
import copy
import itertools

from Orange import data

class Context:
    def __init__(self, **argkw):
        self.time = time.time()
        self.__dict__.update(argkw)

    def __getstate__(self):
        s = dict(self.__dict__)
        for nc in getattr(self, "noCopy", []):
            if nc in s:
                del s[nc]
        return s


class Setting:
    """A description of a setting: the default value and flags.
       The default can be either an (immutable!) object or a callable that is
       used to set the default value.

       When the default is callable that should not be called (which should
       be uncommon), the call can be prevented by setting the flag
       NOT_CALLABLE.
    """
    def __init__(self, default, flags=0, **data):
        self.default = default
        self.flags = flags
        self.__dict__.update(data)


class SettingsHandler:
    """Holds the decription of widget's settings, stored as a dict
       whose keys are attribute names and values are instances of Setting
    """

    NOT_CALLABLE = 1
    """Flag telling that the initialization of the widget should not call
    the object with the default value, although it is callable"""

    def __init__(self):
        self.settings = {}

    def initialize(self, widget):
        """Sets the widget's attributes whose defaults are given as callables
           typically a list"""
        for name, setting in self.settings.items():
            if callable(setting.default) and not (
                    setting.flags & SettingsHandler.NOT_CALLABLE):
                setattr(widget, name, setting.default())


class ContextHandler(SettingsHandler):
    """Base class for setting handlers that can handle contexts."""

    CONTEXT = 2
    """A flag that marks an attribute as context-dependent"""

    maxSavedContexts = 50

    def __init__(self):
        super().__init__()
        self.globalContexts = []

    def initialize(self, widget):
        """Initialize the widget: call the inherited initialization and
        add an attribute 'contextSettings' to the widget. This method
        does not open a context."""
        super().initialize(widget)
        widget.contextSettings = self.globalContexts

    def newContext(self):
        """Create a new context."""
        return Context()

    def openContext(self, widget, *arg, **argkw):
        """Open a context by finding one and setting the widget data or
        creating one and fill with the data from the widget."""
        widget.currentContext, isNew = \
            self.findOrCreateContext(widget, *arg, **argkw)
        if isNew:
            self.settingsFromWidget(widget)
        else:
            self.settingsToWidget(widget)

    def match(self, c, *arg, **argkw):
        raise SystemError(self.__class__.__name__ + " does not overload match")

    def findOrCreateContext(self, widget, *arg, **argkw):
        bestContext, bestScore = None, 0
        for i, c in enumerate(widget.contextSettings):
            score = self.match(c, *arg, **argkw)
            if score == 2:
                self.moveContextUp(widget, i)
                return bestContext, False
            if score > bestScore: # 0 is not OK!
                bestContext, bestScore = c, score
        if bestContext:
            # if cloneIfImperfect should be disabled, change this and the
            # addContext below
            context = self.cloneContext(bestContext)
        else:
            context = self.newContext()
        self.addContext(widget, context)
        return context, bestContext is None

    def moveContextUp(self, widget, index):
        setting = widget.contextSettings.pop(index)
        setting.time = time.time()
        widget.contextSettings.insert(0, setting)

    def addContext(self, widget, setting):
        s = widget.contextSettings
        s.insert(0, setting)
        del s[len(s):]

    def cloneContext(self, context):
        return copy.deepcopy(context)

    def closeContext(self, widget):
        self.settingsFromWidget(widget)

    def fastSave(self, widget, name, value):
        pass

    def settingsToWidget(self, widget):
        widget.retrieveSpecificSettings()

    def settingsFromWidget(self, widget):
        widget.storeSpecificSettings()

    def mergeBack(self, widget):
        # this should happen if the schema is loaded from file?
        globs = self.globalContexts
        if widget.contextSettings is not globs:
            ids = {id(c) for c in globs}
            globs += (c for c in widget.contextSettings if id(c) not in ids)
            globs.sort(key=lambda c: -c.time)
            del globs[self.maxSavedContexts:]


class DomainContextHandler(ContextHandler):
    # Flags for Settings
    REQUIRED = 0
    OPTIONAL = 4
    REQUIRED_IF_SELECTED = 8
    NOT_ATTRIBUTE = 16
    LIST = 32
    EXCLUDE_ATTRIBUTES = 64
    INCLUDE_METAS = 128

    REQUIREMENT_MASK = 12

    # Flags for the handler
    MATCH_VALUES_NONE, MATCH_VALUES_CLASS, MATCH_VALUES_ALL = range(3)

    def __init__(self, maxAttributesToPickle=100, matchValues=0,
                 reservoir=None, attributes_in_res=True, metas_in_res=False):
        super().__init__()
        self.maxAttributesToPickle = maxAttributesToPickle
        self.matchValues = matchValues
        self.reservoir = reservoir
        self.attributes_in_res = attributes_in_res
        self.metas_in_res = metas_in_res

        self.hasOrdinaryAttributes = attributes_in_res
        self.hasMetaAttributes = metas_in_res
        for s in self.settings:
            if s.flags & self.CONTEXT and not s.flags & self.NOT_ATTRIBUTE:
                if not s.flags & self.EXCLUDE_ATTRIBUTES:
                    self.hasOrdinaryAttributes = True
                if s.flags & self.INCLUDE_METAS:
                    self.hasMetaAttributes = True

    def encodeDomain(self, domain):
        def encode(lst, values):
            if values:
                return {v.name:
                            v.values if isinstance(v, data.DiscreteVariable)
                            else v.var_type
                        for v in lst}
            else:
                return {v.name: v.var_type for v in lst}

        match = self.matchValues
        if self.hasOrdinaryAttributes:
            if match == self.MATCH_VALUES_CLASS:
                attributes = encode(domain.attributes, False)
                attributes.update(encode(domain.class_vars, True))
            else:
                attributes = encode(domain, match == self.MATCH_VALUES_ALL)
        else:
            attributes = None

        if self.hasMetaAttributes:
            metas = encode(domain.metas, match == self.MATCH_VALUES_ALL)
        else:
            metas = None

        return attributes, metas


    #noinspection PyMethodOverriding,PyTupleAssignmentBalance
    def findOrCreateContext(self, widget, domain):
        if not domain:
            return None, False

        if not isinstance(domain, data.Domain):
            domain = domain.domain

        encodedDomain = self.encodeDomain(domain)
        context, isNew = super().findOrCreateContext(self, widget,
                                                     domain, *encodedDomain)
        if len(encodedDomain) == 2:
            context.attributes, context.metas = encodedDomain
        else:
            context.attributes, context.classVar, context.metas = encodedDomain

        if self.hasOrdinaryAttributes:
            context.orderedDomain = [(v.name, v.var_type) for v in domain]
        else:
            context.orderedDomain = []
        if self.hasMetaAttributes:
            context.orderedDomain += [(v.name, v.var_type)
                                      for v in domain.metas]
        if isNew:
            context.values = {}
            context.noCopy = ["orderedDomain"]
        return context, isNew


    def settingsToWidget(self, widget):
        def attrSet(attrs):
            if isinstance(attrs, dict):
                try:
                    return set(attrs.items())
                except TypeError:
                    return list(attrs.items())
            elif isinstance(attrs, bool):
                return {}
            else:
                return set()

        super().settingsToWidget(widget)

        context = widget.currentContext
        attrItemsSet = attrSet(context.attributes)
        metaItemsSet = attrSet(context.metas)
        excluded = set()

        for name, setting in self.settings.items():
            flags = setting.flags
            if name not in context.values:
                continue
            value = context.values[name]

            if not flags & self.LIST:
                # TODO: is setattr supposed to check that we do not assign
                # values that are optional and do not exist? is context
                # cloning's filter enough to get rid of such attributes?
                setattr(widget, name, value[0])
                if not flags & self.NOT_ATTRIBUTE:
                    excluded.add(value)
            else:
                newLabels, newSelected = [], []
                has_selection = hasattr(setting, "selected")
                if has_selection:
                    oldSelected = context.values.get(setting.selected, [])
                    for i, saved in enumerate(value):
                        if (not flags & self.EXCLUDE_ATTRIBUTES and (
                                saved in context.attributes or
                                saved in attrItemsSet
                            ) or
                            flags & self.INCLUDE_METAS and (
                                saved in context.metas or
                                saved in metaItemsSet
                            )):
                            if i in oldSelected:
                                newSelected.append(len(newLabels))
                            newLabels.append(saved)
                context.values[name] = newLabels
                setattr(widget, name, value)
                excluded |= set(value)
                if has_selection:
                    context.values[setting.selected] = newSelected
                    # first 'name', then 'selected' - this gets signalled to Qt
                    setattr(widget, setting.selected, newSelected)

        if self.reservoir is not None:
            ll = [a for a in context.orderedDomain if a not in excluded and (
                  self.attributes_in_res and
                      context.attributes.get(a[0], None) == a[1] or
                  self.metas_in_res and context.metas.get(a[0], None) == a[1])]
            setattr(widget, self.reservoir, ll)


    def settingsFromWidget(self, widget):
        super().settingsFromWidget(widget)
        context = widget.currentContext
        context.values = {}
        for name, setting in self.settings.items():
            value = widget.getdeepattr(name)
            if not setting.flags & self.LIST:
                self.saveLow(widget, name, value, setting.flags)
            else:
                context.values[name] = copy.copy(value) # shallow copy
                if hasattr(setting, "selected"):
                    context.values[setting.selected] = list(
                        widget.getdeepattr(setting.selected))

    def fastSave(self, widget, name, value):
        context = widget.currentContext
        if context:
            for sname, setting in self.settings.items():
                if name == sname:
                    if setting.flags & self.LIST:
                        context.values[name] = copy.copy(value) # shallow copy
                    else:
                        self.saveLow(widget, name, value, setting.flags)
                    return
                if name == getattr(setting, "selected", ""):
                    context.values[setting.selected] = list(value)
                    return

    def saveLow(self, widget, name, value, flags):
        context = widget.currentContext
        value = copy.copy(value)
        if isinstance(value, str):
            valtype = (not flags & self.EXCLUDE_ATTRIBUTES and
                       context.attributes.get(value, -1))
            if valtype == -1:
                valtype = (flags & self.INCLUDE_METAS and
                           context.attributes.get(value, -1))
            context.values[name] = value, valtype # -1: not an attribute
        else:
            context.values[name] = value, -2

    def __varExists(self, value, flags, attributes, metas):
        return (not flags & self.EXCLUDE_ATTRIBUTES
                and attributes.get(value[0], -1) == value[1]
                or
                flags & self.INCLUDE_METAS
                and metas.get(value[0], -1) == value[1])


    #noinspection PyMethodOverriding
    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return 2
        filled = potentiallyFilled = 0
        for name, setting in self.settings.items():
            flags = setting.flags
            if flags & self.NOT_ATTRIBUTE:
                continue
            value = context.values.get(name, None)
            if not value:
                continue
            if flags & self.LIST:
                if flags & self.REQUIREMENT_MASK == self.REQUIRED:
                    potentiallyFilled += len(value)
                    filled += len(value)
                    for item in value:
                        if not self.__varExists(item, flags, attrs, metas):
                            return 0
                else:
                    selectedRequired = (setting.flags & self.REQUIREMENT_MASK
                                        == self.REQUIRED_IF_SELECTED)
                    selected = context.values.get(setting.selected, [])
                    potentiallyFilled += len(selected)
                    for i in selected:
                        if self.__varExists(value[i], flags, attrs, metas):
                            filled += 1
                        else:
                            if selectedRequired:
                                return 0
            else:
                potentiallyFilled += 1
                if value[1] >= 0:
                    if self.__varExists(value, flags, attrs, metas):
                        filled += 1
                    else:
                        if flags & self.REQUIRED:
                            return 0
        if not potentiallyFilled:
            return 0.1
        else:
            return filled / potentiallyFilled


    #noinspection PyMethodOverriding
    def cloneContext(self, context, domain, attrs, metas):
        context = copy.deepcopy(context)
        for name, setting in self.settings.items():
            flags = setting.flags
            value = context.values.get(name, None)
            if value is None:
                continue
            if flags & self.LIST:
                sel_name = getattr(setting, "selected", None)
                if sel_name is not None:
                    selected = context.values.get(sel_name, [])
                    selected.sort()
                    nextSel = selected and selected[0] or -1
                else:
                    selected = None
                    nextSel = -1
                i = j = realI = 0
                while i < len(value):
                    if self.__varExists(value[i], flags, attrs, metas):
                        if nextSel == realI:
                            selected[j] -= realI - i
                            j += 1
                            nextSel = j < len(selected) and selected[j] or -1
                        i += 1
                    else:
                        del value[i]
                        if nextSel == realI:
                            del selected[j]
                            nextSel = j < len(selected) and selected[j] or -1
                    realI += 1
                if sel_name is not None:
                    context.values[sel_name] = selected[:j]
            else:
                if (value[1] >= 0 and
                    not self.__varExists(value, flags, attrs, metas)):
                        del context.values[name]
        context.attributes, context.metas = attrs, metas
        context.orderedDomain = [(attr.name, attr.var_type) for attr in
                                 itertools.chain(domain, domain.metas)]
        return context

    def mergeBack(self, widget):
        globs = self.globalContexts
        mp = self.maxAttributesToPickle
        if widget.contextSettings is not globs:
            ids = {id(c) for c in globs}
            globs += (c for c in widget.contextSettings if id(c) not in ids and (
                (c.attributes and len(c.attributes) or 0) +
                (c.class_vars and len(c.class_vars) or 0) +
                (c.metas and len(c.metas) or 0)) <= mp)
            globs.sort(key=lambda c: -c.time)
            del globs[self.maxSavedContexts:]
        else:
            for i in range(len(globs)-1, -1, -1):
                c = globs[i]
                if ((c.attributes and len(c.attributes) or 0) +
                    (c.class_vars and len(c.class_vars) or 0) +
                    (c.metas and len(c.metas) or 0) >= mp):
                        del globs[i]



class ClassValuesContextHandler(ContextHandler):
    #noinspection PyMethodOverriding
    def findOrCreateContext(self, widget, classes):
        if isinstance(classes, data.Variable):
            if isinstance(classes, data.DiscreteVariable):
                classes = classes.values
            else:
                classes = None
        context, isNew = super().findOrCreateContext(widget, classes)
        context.classes = classes
        if isNew:
            context.values = {}
        return context, isNew

    #noinspection PyMethodOverriding
    def match(self, context, classes):
        if isinstance(classes, data.ContinuousVariable):
            return context.classes is None and 2
        else:
            return context.classes == classes and 2

    def settingsToWidget(self, widget):
        super().settingsToWidget(widget)
        context = widget.currentContext
        for name, setting in self.settings.items():
            setattr(widget, name, context.values[name])

    def settingsFromWidget(self, widget):
        super().settingsFromWidget(widget)
        context = widget.currentContext
        values = context.values = {}
        for name, setting in self.settings.items():
            value = widget.getdeepattr(name)
            values[name] = copy.copy(value)

    def fastSave(self, widget, name, value):
        if name in self.settings:
            widget.currentContext.values[name] = copy.copy(value)




### Requires the same the same attributes in the same order
### The class overloads domain encoding and matching.
### Due to different encoding, it also needs to overload saveLow and
### cloneContext (which is the same as the ContextHandler's)
### We could simplify some other methods, but prefer not to replicate the code
class PerfectDomainContextHandler(DomainContextHandler):
    def encodeDomain(self, domain):
        if self.matchValues == 2:
            def encode(vars):
                return tuple(
                    (v.name, v.values if isinstance(v, data.DiscreteVariable)
                             else v.var_type)
                    for v in vars)
        else:
            def encode(vars):
                return tuple((v.name, v.var_type) for v in vars)
        return (encode(domain.attributes),
                encode(domain.class_vars),
                encode(domain.metas))


    #noinspection PyMethodOverriding
    def match(self, context, domain, attributes, class_vars, metas):
        return (attributes, class_vars, metas) == (
                context.attributes, context.class_vars, context.metas) and 2

    def saveLow(self, widget, name, value, flags):
        context = widget.currentContext
        if isinstance(value, str):
            atype = -1
            if not flags & self.EXCLUDE_ATTRIBUTES:
                for aname, atype in itertools.chain(context.attributes,
                                                     context.class_vars):
                    if aname == value:
                        break
            if atype == -1 and flags & self.INCLUDE_METAS:
                for aname, values in itertools.chain(context.attributes,
                                                     context.class_vars):
                    if aname == value:
                        break
            context.values[name] = value, copy.copy(atype)
        else:
            context.values[name] = value, -2


    def cloneContext(self, context, _, *__):
        import copy
        return copy.deepcopy(context)


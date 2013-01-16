import time, copy, orange
from string import *

contextStructureVersion = 100

class Context:
    def __init__(self, **argkw):
        self.time = time.time()
        self.__dict__.update(argkw)

    def __getstate__(self):
        s = dict(self.__dict__)
        for nc in getattr(self, "noCopy", []):
            if s.has_key(nc):
                del s[nc]
        return s

    
class ContextHandler:
    maxSavedContexts = 50
    
    def __init__(self, contextName = "", cloneIfImperfect = True, findImperfect = True, syncWithGlobal = True, contextDataVersion = 0, **args):
        self.contextName = contextName
        self.localContextName = "localContexts"+contextName
        self.cloneIfImperfect, self.findImperfect = cloneIfImperfect, findImperfect
        self.contextDataVersion = contextDataVersion
        self.syncWithGlobal = syncWithGlobal
        self.globalContexts = []
        self.__dict__.update(args)

    def newContext(self):
        return Context()

    def openContext(self, widget, *arg, **argkw):
        context, isNew = self.findOrCreateContext(widget, *arg, **argkw)
        if context:
            if isNew:
                self.settingsFromWidget(widget, context)
            else:
                self.settingsToWidget(widget, context)
        return context

    def initLocalContext(self, widget, parentContext=None):
        if not hasattr(widget, self.localContextName):
            if parentContext is None:
                globalContexts = self.globalContexts
            else:
                # parentContext is a Schema level context repo 
                globalContexts = parentContext.globalContexts
                
            if self.syncWithGlobal:
                setattr(widget, self.localContextName, globalContexts)
            else:
                setattr(widget, self.localContextName, copy.deepcopy(globalContexts))
        
    def findOrCreateContext(self, widget, *arg, **argkw):        
        index, context, score = self.findMatch(widget, self.findImperfect, *arg, **argkw)
        if context:
            if index < 0:
                self.addContext(widget, context)
            else:
                self.moveContextUp(widget, index)
            return context, False
        else:
            context = self.newContext()
            self.addContext(widget, context)
            return context, True

    def closeContext(self, widget, context):
        self.settingsFromWidget(widget, context)

    def fastSave(self, context, widget, name, value):
        pass

    def settingsToWidget(self, widget, context):
        cb = getattr(widget, "settingsToWidgetCallback" + self.contextName, None)
        return cb and cb(self, context)

    def settingsFromWidget(self, widget, context):
        cb = getattr(widget, "settingsFromWidgetCallback" + self.contextName, None)
        return cb and cb(self, context)

    def findMatch(self, widget, imperfect = True, *arg, **argkw):
        bestI, bestContext, bestScore = -1, None, -1
        for i, c in enumerate(getattr(widget, self.localContextName)):
            score = self.match(c, imperfect, *arg, **argkw)
            if score == 2:
                return i, c, score
            if score and score > bestScore:
                bestI, bestContext, bestScore = i, c, score

        if bestContext and self.cloneIfImperfect:
            if hasattr(self, "cloneContext"):
                bestContext = self.cloneContext(bestContext, *arg, **argkw)
            else:
                bestContext = copy.deepcopy(bestContext)
            bestI = -1
                
        return bestI, bestContext, bestScore
            
    def moveContextUp(self, widget, index):
        localContexts = getattr(widget, self.localContextName)
        l = getattr(widget, self.localContextName)
        context = l.pop(index)
        context.time = time.time()
        l.insert(0, context)

    def addContext(self, widget, context):
        l = getattr(widget, self.localContextName)
        l.insert(0, context)
        while len(l) > self.maxSavedContexts:
            del l[-1]

    def mergeBack(self, widget):
        if not self.syncWithGlobal or getattr(widget, self.localContextName) is not  self.globalContexts:
            self.globalContexts.extend([c for c in getattr(widget, self.localContextName) if c not in self.globalContexts])
            self.globalContexts.sort(lambda c1,c2: -cmp(c1.time, c2.time))
            self.globalContexts[:] = self.globalContexts[:self.maxSavedContexts]


class ContextField:
    def __init__(self, name, flags = 0, **argkw):
        self.name = name
        self.flags = flags
        self.__dict__.update(argkw)
    

class DomainContextHandler(ContextHandler):
    Optional, SelectedRequired, Required = range(3)
    RequirementMask = 3
    NotAttribute = 4
    List = 8
    RequiredList = Required + List
    SelectedRequiredList = SelectedRequired + List
    ExcludeOrdinaryAttributes, IncludeMetaAttributes = 16, 32

    MatchValuesNo, MatchValuesClass, MatchValuesAttributes = range(3)
    
    def __init__(self, contextName, fields = [],
                 cloneIfImperfect = True, findImperfect = True, syncWithGlobal = True,
                 maxAttributesToPickle = 100, matchValues = 0, 
                 forceOrdinaryAttributes = False, forceMetaAttributes = False, contextDataVersion = 0, **args):
        ContextHandler.__init__(self, contextName, cloneIfImperfect, findImperfect, syncWithGlobal, contextDataVersion = contextDataVersion, **args)
        self.maxAttributesToPickle = maxAttributesToPickle
        self.matchValues = matchValues
        self.fields = []
        hasMetaAttributes = hasOrdinaryAttributes = False

        for field in fields:
            if isinstance(field, ContextField):
                self.fields.append(field)
                if not field.flags & self.NotAttribute:
                    hasOrdinaryAttributes = hasOrdinaryAttributes or not field.flags & self.ExcludeOrdinaryAttributes
                    hasMetaAttributes = hasMetaAttributes or field.flags & self.IncludeMetaAttributes
            elif type(field)==str:
                self.fields.append(ContextField(field, self.Required))
                hasOrdinaryAttributes = True
            # else it's a tuple
            else:
                flags = field[1]
                if isinstance(field[0], list):
                    self.fields.extend([ContextField(x, flags) for x in field[0]])
                else:
                    self.fields.append(ContextField(field[0], flags))
                if not flags & self.NotAttribute:
                    hasOrdinaryAttributes = hasOrdinaryAttributes or not flags & self.ExcludeOrdinaryAttributes
                    hasMetaAttributes = hasMetaAttributes or flags & self.IncludeMetaAttributes
                    
        self.hasOrdinaryAttributes, self.hasMetaAttributes = hasOrdinaryAttributes, hasMetaAttributes
        
    def encodeDomain(self, domain):
        if self.matchValues == 2:
            attributes = self.hasOrdinaryAttributes and \
                         dict([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                                for attr in domain])
            metas = self.hasMetaAttributes and \
                         dict([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                                for attr in domain.getmetas().values()])
        else:
            if self.hasOrdinaryAttributes:
                attributes = dict([(attr.name, attr.varType) for attr in domain.attributes])
                classVar = domain.classVar
                if classVar:
                    if self.matchValues and classVar.varType == orange.VarTypes.Discrete:
                        attributes[classVar.name] = classVar.values
                    else:
                        attributes[classVar.name] = classVar.varType
            else:
                attributes = False

            metas = self.hasMetaAttributes and dict([(attr.name, attr.varType) for attr in domain.getmetas().values()]) or {}

        return attributes, metas
    
    def findOrCreateContext(self, widget, domain):
        if not domain:
            return None, False
        
        if not isinstance(domain, orange.Domain):
            domain = domain.domain
            
        encodedDomain = self.encodeDomain(domain)
        context, isNew = ContextHandler.findOrCreateContext(self, widget, domain, *encodedDomain)
        if not context:
            return None, False
        
        if len(encodedDomain) == 2:
            context.attributes, context.metas = encodedDomain
        else:
            context.attributes, context.classVar, context.metas = encodedDomain

        metaIds = domain.getmetas().keys()
        metaIds.sort()
        context.orderedDomain = []
        if self.hasOrdinaryAttributes:
            context.orderedDomain.extend([(attr.name, attr.varType) for attr in domain])
        if self.hasMetaAttributes:
            context.orderedDomain.extend([(domain[i].name, domain[i].varType) for i in metaIds])

        if isNew:
            context.values = {}
            context.noCopy = ["orderedDomain"]
        return context, isNew

#    def exists(self, field, value, context):
#        for check, what in ((not field.flags & self.ExcludeOrdinaryAttributes, context.attributes),
#                            (field.flags & self.IncludeMetaAttributes, context.metas)):
#            if check:
#                inDomainType = context.attributes.get(value[0])
#                if isinstance(savedType, list):
#                    if what.get(value[0], False) == savedType 
#                                f
#                    if saved
             
    def settingsToWidget(self, widget, context):
        ContextHandler.settingsToWidget(self, widget, context)
        excluded = {}
        addOrdinaryTo = []
        addMetaTo = []

        def set_if_all_hashable(iter):
            try:
                return set(iter)
            except TypeError:
                return list(iter)
            
        def attrSet(attrs):
            if isinstance(attrs, dict):
                return set_if_all_hashable(attrs.items())
            elif isinstance(attrs, bool):
                return {}
            else:
                return set([])
            
        attrItemsSet = attrSet(context.attributes)
        metaItemsSet = attrSet(context.metas)
        for field in self.fields:
            name, flags = field.name, field.flags

            excludes = getattr(field, "reservoir", [])
            if excludes:
                if not isinstance(excludes, list):
                    excludes = [excludes]
                for exclude in excludes:
                    excluded.setdefault(exclude, [])
                    if not (flags & self.NotAttribute + self.ExcludeOrdinaryAttributes):
                        addOrdinaryTo.append(exclude)
                    if flags & self.IncludeMetaAttributes:
                        addMetaTo.append(exclude)                    

            if not context.values.has_key(name):
                continue
            
            value = context.values[name]

            if not flags & self.List:
# TODO: is setattr supposed to check that we do not assign values that are optional and do not exist?
# is context cloning's filter enough to get rid of such attributes?
                setattr(widget, name, value[0])
                for exclude in excludes:
                    excluded[exclude].append(value)

            else:
                newLabels, newSelected = [], []
                oldSelected = hasattr(field, "selected") and context.values.get(field.selected, []) or []
                for i, saved in enumerate(value):
                    if not flags & self.ExcludeOrdinaryAttributes and (saved in context.attributes or saved in attrItemsSet) \
                       or flags & self.IncludeMetaAttributes and (saved in context.metas or saved in metaItemsSet):
                        if i in oldSelected:
                            newSelected.append(len(newLabels))
                        newLabels.append(saved)

                context.values[name] = newLabels
                setattr(widget, name, value)

                if hasattr(field, "selected"):
                    context.values[field.selected] = newSelected
                    setattr(widget, field.selected, context.values[field.selected])

                for exclude in excludes:
                    excluded[exclude].extend(value)

        for name, values in excluded.items():
            addOrd, addMeta = name in addOrdinaryTo, name in addMetaTo
            ll = [a for a in context.orderedDomain if a not in values and ((addOrd and context.attributes.get(a[0], None) == a[1]) or (addMeta and context.metas.get(a[0], None) == a[1]))]
            setattr(widget, name, ll)
            

    def settingsFromWidget(self, widget, context):
        ContextHandler.settingsFromWidget(self, widget, context)
        context.values = {}
        for field in self.fields:
            if not field.flags & self.List:
                self.saveLow(context, widget, field.name, widget.getdeepattr(field.name), field.flags)
            else:
                value = widget.getdeepattr(field.name)
                # shallow copy of the list
                context.values[field.name] = copy.copy(value) #type(value)(value)
                if hasattr(field, "selected"):
                    context.values[field.selected] = list(widget.getdeepattr(field.selected))

    def fastSave(self, context, widget, name, value):
        if context:
            for field in self.fields:
                if name == field.name:
                    if field.flags & self.List:
                        # shallow copy of the list
                        context.values[field.name] = copy.copy(value) #type(value)(value)
                    else:
                        self.saveLow(context, widget, name, value, field.flags)
                    return
                if name == getattr(field, "selected", None):
                    context.values[field.selected] = list(value)
                    return

    def saveLow(self, context, widget, field, value, flags):
        # The code below uses type(value)(value) to make at least a shallow copy of the
        # attributes. This will mostly make copies of integers, floats and strings, but
        # it is crucial to make copies of lists
        value = copy.copy(value) #type(value)(value)
        if isinstance(value, str):
            valtype = not flags & self.ExcludeOrdinaryAttributes and context.attributes.get(value, -1)
            if valtype == -1:
                valtype = flags & self.IncludeMetaAttributes and context.attributes.get(value, -1)
            context.values[field] = value, valtype # -1 means it's not an attribute
        else:
            context.values[field] = value, -2

    def attributeExists(self, value, flags, attributes, metas):
        return not flags & self.ExcludeOrdinaryAttributes and attributes.get(value[0], -1) == value[1] \
                or flags & self.IncludeMetaAttributes and metas.get(value[0], -1) == value[1]
    
    def match(self, context, imperfect, domain, attributes, metas):
        if (attributes, metas) == (context.attributes, context.metas):
            return 2
        if not imperfect:
            return 0

        filled = potentiallyFilled = 0
        for field in self.fields:
            flags = field.flags
            value = context.values.get(field.name, None)
            if value:
                if flags & self.List:
                    if flags & self.RequirementMask == self.Required:
                        potentiallyFilled += len(value)
                        filled += len(value)
                        for item in value:
                            if not self.attributeExists(item, flags, attributes, metas): 
                                return 0
                    else:
                        selectedRequired = field.flags & self.RequirementMask == self.SelectedRequired
                        selected = context.values.get(field.selected, [])
                        potentiallyFilled += len(selected) 
                        for i in selected:
                            # TODO: shouldn't we check the attribute type here, too? or should we change self.saveLow for these field types then?
                            if (not flags & self.ExcludeOrdinaryAttributes and value[i] in attributes
                                 or flags & self.IncludeMetaAttributes and value[i] in metas): 
                                filled += 1
                            else:
                                if selectedRequired:
                                    return 0
                else:
                    potentiallyFilled += 1
                    if value[1] >= 0:
                        if (not flags & self.ExcludeOrdinaryAttributes and attributes.get(value[0], None) == value[1]
                             or flags & self.IncludeMetaAttributes and metas.get(value[0], None) == value[1]): 
                            filled += 1
                        else:
                            if flags & self.Required:
                                return 0

            if not potentiallyFilled:
                return 1.0
            else:
                return filled / float(potentiallyFilled)

    def cloneContext(self, context, domain, attributes, metas):
        import copy
        context = copy.deepcopy(context)
        
        for field in self.fields:
            value = context.values.get(field.name, None)
            if value:
                if field.flags & self.List:
                    i = j = realI = 0
                    selected = context.values.get(field.selected, [])
                    selected.sort()
                    nextSel = selected and selected[0] or None
                    while i < len(value):
                        if not self.attributeExists(value[i], field.flags, attributes, metas):
                            del value[i]
                            if nextSel == realI:
                                del selected[j]
                                nextSel = j < len(selected) and selected[j] or None
                        else:
                            if nextSel == realI:
                                selected[j] -= realI - i
                                j += 1
                                nextSel = j < len(selected) and selected[j] or None
                            i += 1
                        realI += 1
                    if hasattr(field, "selected"):
                        context.values[field.selected] = selected[:j]
                else:
                    if value[1] >= 0 and not self.attributeExists(value, field.flags, attributes, metas):
                        del context.values[field.name]
                        
        context.attributes, context.metas = attributes, metas
        context.orderedDomain = [(attr.name, attr.varType) for attr in domain]
        return context

    # this is overloaded to get rid of the huge domains
    def mergeBack(self, widget):
        if not self.syncWithGlobal or getattr(widget, self.localContextName) is not self.globalContexts:
            self.globalContexts.extend([c for c in getattr(widget, self.localContextName) if c not in self.globalContexts])
            mp = self.maxAttributesToPickle
            self.globalContexts[:] = filter(lambda c: (c.attributes and len(c.attributes) or 0) + (c.metas and len(c.metas) or 0) < mp, self.globalContexts)
            self.globalContexts.sort(lambda c1,c2: -cmp(c1.time, c2.time))
            self.globalContexts[:] = self.globalContexts[:self.maxSavedContexts]

    
class ClassValuesContextHandler(ContextHandler):
    def __init__(self, contextName, fields = [], syncWithGlobal = True, contextDataVersion = 0, **args):
        ContextHandler.__init__(self, contextName, False, False, syncWithGlobal, contextDataVersion = contextDataVersion, **args)
        if isinstance(fields, list):
            self.fields = fields
        else:
            self.fields = [fields]
        
    def findOrCreateContext(self, widget, classes):
        if isinstance(classes, orange.Variable):
            classes = classes.varType == orange.VarTypes.Discrete and classes.values
        if not classes:
            return None, False
        context, isNew = ContextHandler.findOrCreateContext(self, widget, classes)
        if not context:
            return None, False
        context.classes = classes
        if isNew:
            context.values = {}
        return context, isNew

    def settingsToWidget(self, widget, context):
        ContextHandler.settingsToWidget(self, widget, context)
        for field in self.fields:
            setattr(widget, field, context.values[field])
            
    def settingsFromWidget(self, widget, context):
        ContextHandler.settingsFromWidget(self, widget, context)
        # shallow copy!
        values = context.values = {}
        for field in self.fields:
            value = widget.getdeepattr(field)
            values[field] = copy.copy(value) #type(value)(value)

    def fastSave(self, context, widget, name, value):
        if context and name in self.fields:
            # shallow copy!
            context.values[name] = copy.copy(value) #type(value)(value)

    def match(self, context, imperfect, classes):
        return context.classes == classes and 2

    def cloneContext(self, context, domain, encodedDomain):
        import copy
        return copy.deepcopy(context)
        


### Requires the same the same attributes in the same order
### The class overloads domain encoding and matching.
### Due to different encoding, it also needs to overload saveLow and cloneContext
### (the latter gets really simple now).
### We could simplify some other methods, but prefer not to replicate the code
###
### Note that forceOrdinaryAttributes is here True by default!
class PerfectDomainContextHandler(DomainContextHandler):
    def __init__(self, contextName = "", fields = [],
                 syncWithGlobal = True, **args):
            DomainContextHandler.__init__(self, contextName, fields, False, False, syncWithGlobal, **args)

        
    def encodeDomain(self, domain):
        if self.matchValues == 2:
            attributes = tuple([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                         for attr in domain])
            classVar = domain.classVar
            if classVar:
                classVar = classVar.name, classVar.varType != orange.VarTypes.Discrete and classVar.varType or classVar.values
            metas = dict([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                         for attr in domain.getmetas().values()])
        else:
            attributes = tuple([(attr.name, attr.varType) for attr in domain.attributes])
            classVar = domain.classVar
            if classVar:
                classVar = classVar.name, classVar.varType
            metas = dict([(attr.name, attr.varType) for attr in domain.getmetas().values()])
        return attributes, classVar, metas
    


    def match(self, context, imperfect, domain, attributes, classVar, metas):
        return (attributes, classVar, metas) == (context.attributes, context.classVar, context.metas) and 2


    def saveLow(self, context, widget, field, value, flags):
        if isinstance(value, str):
            if not flags & self.ExcludeOrdinaryAttributes:
                attr = [x[1] for x in context.attributes if x[0] == value]
            if not attr and context.classVar and context.classVar[0] == value:
                attr = [context.classVar[1]]
            if not attr and flags & self.IncludeMetaAttributes:
                attr = [x[1] for x in context.metas if x[0] == value]

            value = copy.copy(value) #type(value)(value)
            if attr:
                context.values[field] = value, attr[0]
            else:
                context.values[field] = value, -1
        else:
            context.values[field] = value, -2


    def cloneContext(self, context, domain, encodedDomain):
        import copy
        context = copy.deepcopy(context)
        

class EvaluationResultsContextHandler(ContextHandler):
    def __init__(self, contextName, targetAttr, selectedAttr,
                 syncWithGlobal = True, **args):
            self.targetAttr, self.selectedAttr = targetAttr, selectedAttr
            ContextHandler.__init__(self, contextName, False, False, syncWithGlobal, **args)
        
    def match(self, context, imperfect, cnames, cvalues):
        return (cnames, cvalues) == (context.classifierNames, context.classValues) and 2

    def fastSave(self, context, widget, name, value):
        if context:
            if name == self.targetAttr:
                context.targetClass = value
            elif name == self.selectedAttr:
                context.selectedClassifiers = list(value)

    def settingsFromWidget(self, widget, context):
        context.targetClass = widget.getdeepattr(self.targetAttr)
        context.selectedClassifiers = list(widget.getdeepattr(self.selectedAttr))

    def settingsToWidget(self, widget, context):
        if context.targetClass is not None:
            setattr(widget, self.targetAttr, context.targetClass)
        if context.selectedClassifiers is not None:
            setattr(widget, self.selectedAttr, context.selectedClassifiers)
            
    def cloneContext(self, context, domain):
        import copy
        context = copy.deepcopy(context)
        
    def findOrCreateContext(self, widget, results):
        if not results:
            return None, False
        cnames = [c.name for c in results.classifiers]
        cvalues = results.classValues
        context, isNew = ContextHandler.findOrCreateContext(self, widget, results.classifierNames, results.classValues)
        if not context:
            return None, False
        if isNew:
            context.classifierNames = results.classifierNames
            context.classValues = results.classValues
            context.selectedClassifiers = None
            context.targetClass = None
        return context, isNew

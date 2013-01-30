# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    manager, that handles correct processing of widget signals
#

import sys, os, time
import copy
import logging
import logging.handlers
import builtins

import Orange

from .utils import debugging

Single = 2
Multiple = 4

Default = 8
NonDefault = 16

Explicit = 32  # Explicit - only connected if specifically requested or the only possibility

Dynamic = 64  # Dynamic output signal


class InputSignal(object):
    def __init__(self, name, signalType, handler,
                 parameters=Single + NonDefault,
                 oldParam=0):
        self.name = name
        self.type = signalType
        self.handler = handler

        if type(parameters) == str:
            parameters = eval(parameters)   # parameters are stored as strings
        # if we have the old definition of parameters then transform them
        if parameters in [0, 1]:
            self.single = parameters
            self.default = not oldParam
            self.explicit = 0
            return

        if not (parameters & Single or parameters & Multiple):
            parameters += Single
        if not (parameters & Default or parameters & NonDefault):
            parameters += NonDefault

        self.single = parameters & Single
        self.default = parameters & Default
        self.explicit = parameters & Explicit


class OutputSignal(object):
    def __init__(self, name, signalType, parameters=Single + NonDefault):
        self.name = name
        self.type = signalType

        if type(parameters) == str:
            parameters = eval(parameters)

        if parameters in [0, 1]:  # old definition of parameters
            self.default = not parameters
            self.single = 1
            self.explicit = 0
            self.dynamic = 0
            return

        if not (parameters & Default or parameters & NonDefault):
            parameters += NonDefault

        self.single = parameters & Single
        self.default = parameters & Default
        self.explicit = parameters & Explicit

        self.dynamic = parameters & Dynamic
        if self.dynamic and self.single:
            print("Output signal can not be Multiple and Dynamic")
            self.dynamic = 0


def canConnect(output, input, dynamic=False):
    ret = issubclass(output.type, input.type)
    if output.dynamic and dynamic:
        ret = ret or issubclass(input.type,output.type)
    return ret


def rgetattr(obj, name):
    while "." in name:
        first, name = name.split(".", 1)
        obj = getattr(obj, first)
    return getattr(obj, name)


def resolveSignal(signal, globals={}):
    """If `signal.type` is a string (used by orngRegistry) return
    the signal copy with the resolved `type`, else return the signal
    unchanged.
    """
    if isinstance(signal.type, str):
        type_name = signal.type
        if "." in type_name:
            module, name = type_name.split(".", 1)
            if module in globals:
                #  module and type are imported in the globals
                module = globals[module]
                sig_type = rgetattr(module, name)
            else:
                module, name = type_name.rsplit(".", 1)
                module = __import__(module, fromlist=[name], globals=globals)
                sig_type = getattr(module, name)
        else:
            if type_name in globals:
                sig_type = globals[type_name]
            elif hasattr(builtins, type_name):
                sig_type = getattr(builtins, type_name)
            else:
                raise NameError(type_name)

        signal = copy.copy(signal)
        signal.type = sig_type
    return signal

class SignalLink(object):
    def __init__(self, widgetFrom, outputSignal, widgetTo, inputSignal, enabled=True):
        self.widgetFrom = widgetFrom
        self.widgetTo = widgetTo

        self.outputSignal = outputSignal
        self.inputSignal = inputSignal

        #TODO inputSignal is sometimes (after updating widget and rereading
        #it from module instead from a registry?) 'str' instead of 'type'
        if issubclass(outputSignal.type, inputSignal.type):
            self.dynamic = False
        else:
            self.dynamic = outputSignal.dynamic

        self.enabled = enabled

        self.signalNameFrom = self.outputSignal.name
        self.signalNameTo = self.inputSignal.name


    def canEnableDynamic(self, obj):
        """ Can dynamic signal link be enabled for `obj`?
        """
        return isinstance(obj, self.inputSignal.type)


# class that allows to process only one signal at a time
class SignalWrapper(object):
    def __init__(self, widget, method):
        self.widget = widget
        self.method = method

    def __call__(self, *k):
        manager = self.widget.signalManager
        if not manager:
            manager = signalManager

        manager.signalProcessingInProgress += 1
        try:
            self.method(*k)
        finally:
            manager.signalProcessingInProgress -= 1
            if not manager.signalProcessingInProgress:
                manager.processNewSignals(self.widget)


class SignalManager(object):
    widgets = []    # topologically sorted list of widgets
    links = {}      # dicionary. keys: widgetFrom, values: [SignalLink, ...]
    freezing = 0            # do we want to process new signal immediately
    signalProcessingInProgress = 0 # this is set to 1 when manager is propagating new signal values

    def __init__(self, *args):
        self.debugFile = None
        self.verbosity = debugging.orngVerbosity
        self.stderr = sys.stderr
        self._seenExceptions = {}
        self.widgetQueue = []
        self.asyncProcessingEnabled = False

        from .utils import environ
        if not hasattr(self, "log"):
            SignalManager.log = logging.getLogger("SignalManager")
            self.logFileName = os.path.join(environ.canvas_settings_dir,
                "signalManager.log")
            try:
                self.log.addHandler(logging.handlers.RotatingFileHandler(self.logFileName, maxBytes=2**20, backupCount=2))
            except:
                pass
            self.log.setLevel(logging.INFO)

        self.log.info("Signal Manager started")

        self.stdout = sys.stdout

        class err(object):
            def write(myself, str):
                self.log.error(str[:-1] if str.endswith("\n") else str)
            def flush(myself):
                pass
        self.myerr = err()

        if debugging.orngDebuggingEnabled:
            self.debugHandler = logging.FileHandler(debugging.orngDebuggingFileName, mode="wb")
            self.log.addHandler(self.debugHandler)
            self.log.setLevel(logging.DEBUG if debugging.orngVerbosity > 0
            else logging.INFO)
            sys.excepthook = self.exceptionHandler
            sys.stderr = self.myerr

    def setDebugMode(self, debugMode = 0, debugFileName = "signalManagerOutput.txt", verbosity = 1):
        self.verbosity = verbosity

        if debugMode:
            handler = logging.FileHandler(debugFileName, "wb")
            self.log.addHandler(handler)

            sys.excepthook = self.exceptionHandler

            sys.stderr = self.myerr

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # DEBUGGING FUNCTION

    def closeDebugFile(self):
        sys.stderr = self.stderr

    def addEvent(self, strValue, object = None, eventVerbosity = 1):
        info = str(strValue)
        if isinstance(object, Orange.data.Table):
            name = " " + getattr(object, "name", "")
            info += ". Token type = ExampleTable" + name + ". len = " + str(len(object))
        elif type(object) == list:
            info += ". Token type = %s. Value = %s" % (str(type(object)), str(object[:10]))
        elif object != None:
            info += ". Token type = %s. Value = %s" % (str(type(object)), str(object)[:100])
        if eventVerbosity > 0:
            self.log.debug(info)
        else:
            self.log.info(info)


    def exceptionSeen(self, type, value, tracebackInfo):
        import traceback, os
        shortEStr = "".join(traceback.format_exception(type, value, tracebackInfo))[-2:]
        return shortEStr in self._seenExceptions

    def exceptionHandler(self, type, value, tracebackInfo):
        import traceback, os, io

        # every exception show only once
        shortEStr = "".join(traceback.format_exception(type, value, tracebackInfo))[-2:]
        if shortEStr in self._seenExceptions:
            return
        self._seenExceptions[shortEStr] = 1

        list = traceback.extract_tb(tracebackInfo, 10)
        space = "\t"
        totalSpace = space
        message = io.StringIO()
        message.write("Unhandled exception of type %s\n" % ( str(type)))
        message.write("Traceback:\n")

        for i, (file, line, funct, code) in enumerate(list):
            if not code:
                continue
            message.write(totalSpace + "File: " + os.path.split(file)[1] + " in line %4d\n" %(line))
            message.write(totalSpace + "Function name: %s\n" % (funct))
            message.write(totalSpace + "Code: " + code + "\n")
            totalSpace += space

        message.write(totalSpace[:-1] + "Exception type: " + str(type) + "\n")
        message.write(totalSpace[:-1] + "Exception value: " + str(value)+ "\n")
        self.log.error(message.getvalue())
#        message.flush()

    # ----------------------------------------------------------
    # ----------------------------------------------------------

    # freeze/unfreeze signal processing. If freeze=1 no signal will be processed until freeze is set back to 0
    def setFreeze(self, freeze, startWidget = None):
        """ Freeze/unfreeze signal processing. If freeze=1 no signal will be
        processed until freeze is set back to 0

        """
        self.freezing = max(freeze, 0)
        if freeze > 0:
            self.addEvent("Freezing signal processing (%s)" % str(freeze), startWidget)
        elif freeze == 0:
            self.addEvent("Unfreezing signal processing", startWidget)
        else:
            self.addEvent("Invalid freeze value! (by %s)", startWidget, eventVerbosity=0)

        if self.freezing == 0 and self.widgets != []:
            self.processNewSignals(self.widgets[0]) # always start processing from the first
#            if startWidget:
#                self.processNewSignals(startWidget)
#            else:
#                self.processNewSignals(self.widgets[0])

    def addWidget(self, widget):
        """ Add `widget` to the `widgets` list
        """

        self.addEvent("Added widget " + widget.captionTitle, eventVerbosity = 2)

        if widget not in self.widgets:
            self.widgets.append(widget)
#            widget.connect(widget, SIGNAL("blockingStateChanged(bool)"), self.onStateChanged)

    def removeWidget(self, widget):
        """ Remove widget from the `widgets` list
        """
#        if self.verbosity >= 2:
        self.addEvent("Remove widget " + widget.captionTitle, eventVerbosity = 2)
        self.widgets.remove(widget)
        if widget in self.links:
            del self.links[widget]

    def getLinks(self, widgetFrom=None, widgetTo=None, signalNameFrom=None, signalNameTo=None):
        """ Return a list of matching SignalLinks
        """
        links = []
        if widgetFrom is None:
            widgets = self.widgets # search all widgets
        else:
            widgets = [widgetFrom]
        for w in widgets:
            for link in self.links.get(w, []):
                if (widgetFrom is None or widgetFrom is link.widgetFrom) and \
                   (widgetTo is None or widgetTo is link.widgetTo) and \
                   (signalNameFrom is None or signalNameFrom == link.signalNameFrom) and \
                   (signalNameTo is None or signalNameTo == link.signalNameTo):
                        links.append(link)

        return links

    def getLinkWidgetsIn(self, widget, signalName):
        """ Return a list of widgets that connect to `widget`'s input `signalName`
        """
        links = self.getLinks(None, widget, None, signalName)
        return [link.widgetFrom for link in links]


    def getLinkWidgetsOut(self, widget, signalName):
        """ Return a list of widgets that connect to `widget`'s output `signalName`
        """
        links = self.getLinks(widget, None, signalName, None)
        return [link.widgetTo for link in links]



    def canConnect(self, widgetFrom, widgetTo, dynamic=True):
        return any(canConnect(out, in_, dynamic)
                   for out in widgetFrom.outputs for in_ in widgetTo.inputs)


    def proposePossibleLinks(self, widgetFrom, widgetTo, dynamic=True):
        """ Return a ordered list of (OutputSignal, InputSignal, weight) tuples that
        can connect both widgets
        """
        # Get signals that are Single links and already connected to input widget
        links = self.getLinks(None, widgetTo)
        alreadyConnected = [link.signalNameTo for link in links if link.inputSignal.single]

        def weight(outS, inS):
            if outS.explicit or inS.explicit:
                # Zero weight for explicit signals
                weight = 0
            else:
                check = [not outS.dynamic, inS.name not in alreadyConnected, bool(inS.default), bool(outS.default)] #Dynamic signals are lasts
                weights = [2**i for i in range(len(check), 0, -1)]
                weight = sum([w for w, c in zip(weights, check) if c])
            return weight

        possibleLinks = []
        for outS in widgetFrom.outputs:
            for inS in widgetTo.inputs:
                if canConnect(outS, inS, dynamic):
                    possibleLinks.append((outS, inS, weight(outS, inS)))

        return sorted(possibleLinks, key=lambda link: link[-1], reverse=True)


    def inputSignal(self, widget, name):
        for tt in widget.inputs:
            if tt.name == name:
                return tt

    def outputSignal(self, widget, name):
        for tt in widget.outputs:
            if tt.name == name:
                return tt


    def addLink(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo, enabled):
        self.addEvent("Add link from " + widgetFrom.captionTitle + " to " + widgetTo.captionTitle, eventVerbosity = 2)

        ## would this link create a cycle
        if self.existsPath(widgetTo, widgetFrom):
            return 0
        # check if signal names still exist
        found = 0
        output_names = [t.name for t in widgetFrom.outputs]
        found = signalNameFrom in output_names

        if not found:
            if signalNameFrom in _CHANNEL_NAME_MAP and \
                    _CHANNEL_NAME_MAP[signalNameFrom] in output_names:
                self.addEvent("Widget changed its output signal  %r name. Changed to %r." % (signalNameFrom, _CHANNEL_NAME_MAP[signalNameFrom]),
                              eventVerbosity=1)
                signalNameFrom = _CHANNEL_NAME_MAP[signalNameFrom]
                found = 1

        if not found:
            print("Error. Widget %s changed its output signals. It does not have signal %s anymore." % (str(getattr(widgetFrom, "captionTitle", "")), signalNameFrom))
            return 0

        found = 0
        input_names = [t.name for t in widgetTo.inputs]
        found = signalNameTo in input_names

        if not found:
            if signalNameTo in _CHANNEL_NAME_MAP and \
                    _CHANNEL_NAME_MAP[signalNameTo] in input_names:
                self.addEvent("Widget changed its input signal  %r name. Changed to %r." % (signalNameFrom, _CHANNEL_NAME_MAP[signalNameTo]),
                              eventVerbosity=1)
                signalNameTo = _CHANNEL_NAME_MAP[signalNameTo]
                found = 1

        if not found:
            print("Error. Widget %s changed its input signals. It does not have signal %s anymore." % (str(getattr(widgetTo, "captionTitle", "")), signalNameTo))
            return 0

        if widgetFrom in self.links:
            if self.getLinks(widgetFrom, widgetTo, signalNameFrom, signalNameTo):
                print("connection ", widgetFrom, " to ", widgetTo, " alread exists. Error!!")
                return

        link = SignalLink(widgetFrom, self.outputSignal(widgetFrom, signalNameFrom),
                          widgetTo, self.inputSignal(widgetTo, signalNameTo), enabled=enabled)
        self.links[widgetFrom] = self.links.get(widgetFrom, []) + [link]

        widgetTo.addInputConnection(widgetFrom, signalNameTo)

        # if there is no key for the signalNameFrom, create it and set its id=None and data = None
        if signalNameFrom not in widgetFrom.linksOut:
            widgetFrom.linksOut[signalNameFrom] = {None:None}

        # if channel is enabled, send data through it
        if enabled:
            self.pushAllOnLink(link)

        # reorder widgets if necessary
        if self.widgets.index(widgetFrom) > self.widgets.index(widgetTo):
            self.fixTopologicalOrdering()
#            self.widgets.remove(widgetTo)
#            self.widgets.append(widgetTo)   # appent the widget at the end of the list
#            self.fixPositionOfDescendants(widgetTo)

        return 1

    # fix position of descendants of widget so that the order of widgets in self.widgets is consistent with the schema
    def fixPositionOfDescendants(self, widget):
        for link in self.links.get(widget, []):
            widgetTo = link.widgetTo
            self.widgets.remove(widgetTo)
            self.widgets.append(widgetTo)
            self.fixPositionOfDescendants(widgetTo)

    def fixTopologicalOrdering(self):
        """ fix the widgets topological ordering
        """
        order = []
        visited = set()
        # TODO This does not work in Python 3.0
        # E.g. TypeError: unorderable types: OWFile() < OWFile()
        queue = sorted([w for w in self.widgets if not self.getLinks(None, w)])
        while queue:
            w = queue.pop(0)
            order.append(w)
            visited.add(w)
            linked = set([link.widgetTo for link in self.getLinks(w)])
            queue.extend(sorted(linked.difference(queue)))
        self.widgets[:] = order


    def findSignals(self, widgetFrom, widgetTo):
        """ Return a list of (outputName, inputName) for links between widgets
        """
        links = self.getLinks(widgetFrom, widgetTo)
        return [(link.signalNameFrom, link.signalNameTo) for link in links]


    def isSignalEnabled(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo):
        """ Is signal enabled
        """
        links = self.getLinks(widgetFrom, widgetTo, signalNameFrom, signalNameTo)
        if links:
            return links[0].enabled
        else:
            return False


    def removeLink(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo):
        """ Remove link
        """
        self.addEvent("Remove link from " + widgetFrom.captionTitle + " to " + widgetTo.captionTitle, eventVerbosity = 2)

        # no need to update topology, just remove the link
        if widgetFrom in self.links:
            links = self.getLinks(widgetFrom, widgetTo, signalNameFrom, signalNameTo)
            if len(links) != 1:
                print("Error removing a link with none or more then one entries")
                return

            link = links[0]
            self.purgeLink(link)

            self.links[widgetFrom].remove(link)
            if not self.freezing and not self.signalProcessingInProgress:
                self.processNewSignals(widgetFrom)
        widgetTo.removeInputConnection(widgetFrom, signalNameTo)


    # ############################################
    # ENABLE OR DISABLE LINK CONNECTION

    def setLinkEnabled(self, widgetFrom, widgetTo, enabled, justSend = False):
        """ Set `enabled` state for links between widgets.
        """
        for link in self.getLinks(widgetFrom, widgetTo):
            if not justSend:
                link.enabled = enabled
            if enabled:
                self.pushAllOnLink(link)

        if enabled:
            self.processNewSignals(widgetTo)


    def getLinkEnabled(self, widgetFrom, widgetTo):
        """ Is any link between widgets enabled
        """
        return any(link.enabled for link in self.getLinks(widgetFrom, widgetTo))


    # widget widgetFrom sends signal with name signalName and value value
    def send(self, widgetFrom, signalNameFrom, value, id):
        """ Send signal `signalNameFrom` from `widgetFrom` with `value` and `id`
        """
        # add all target widgets new value and mark them as dirty
        # if not freezed -> process dirty widgets
        self.addEvent("Send data from " + widgetFrom.captionTitle + ". Signal = " + signalNameFrom, value, eventVerbosity = 2)

        if widgetFrom not in self.links:
            return

        for link in self.getLinks(widgetFrom, None, signalNameFrom, None):
            self.pushToLink(link, value, id)

        if not self.freezing and not self.signalProcessingInProgress:
            self.processNewSignals(widgetFrom)

    # when a new link is created, we have to
    def sendOnNewLink(self, widgetFrom, widgetTo, signals):
        for (signalNameFrom, signalNameTo) in signals:
            for link in self.getLinks(widgetFrom, widgetTo, signalNameFrom, signalNameTo):
                self.pushAllOnLink(link)


    def pushAllOnLink(self, link):
        """ Send all data on link
        """
        for key in link.widgetFrom.linksOut[link.signalNameFrom].keys():
            self.pushToLink(link, link.widgetFrom.linksOut[link.signalNameFrom][key], key)


    def purgeLink(self, link):
        """ Clear all data on link (i.e. send None for all keys)
        """
        for key in link.widgetFrom.linksOut[link.signalNameFrom].keys():
            self.pushToLink(link, None, key)

    def pushToLink(self, link, value, id):
        """ Send value with id on link
        """
        if link.enabled:
            if link.dynamic:
                dyn_enable = link.canEnableDynamic(value)
                self.setDynamicLinkEnabled(link, dyn_enable)
                if not dyn_enable:
                    value = None
            link.widgetTo.updateNewSignalData(link.widgetFrom, link.signalNameTo,
                                          value, id, link.signalNameFrom)

    def processNewSignals(self, firstWidget=None):
        """ Process new signals starting from `firstWidget`
        """

        if len(self.widgets) == 0 or self.signalProcessingInProgress or self.freezing:
            return

        if firstWidget not in self.widgets or self.widgetQueue:
            firstWidget = self.widgets[0]   # if some window that is not a widget started some processing we have to process new signals from the first widget

        self.addEvent("Process new signals starting from " + firstWidget.captionTitle, eventVerbosity = 2)

        skipWidgets = self.getBlockedWidgets() # Widgets that are blocking


        # start propagating
        self.signalProcessingInProgress = 1

        index = self.widgets.index(firstWidget)
        for i in range(index, len(self.widgets)):
            if self.widgets[i] in skipWidgets:
                continue

            while self.widgets[i] in self.widgetQueue:
                self.widgetQueue.remove(self.widgets[i])
            if self.widgets[i].needProcessing:
                self.addEvent("Processing " + self.widgets[i].captionTitle)
                try:
                    self.widgets[i].processSignals()
                except Exception:
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that it doesn't crash canvas

                if self.widgets[i].isBlocking():
                    if not self.asyncProcessingEnabled:
                        self.addEvent("Widget %s blocked during signal processing. Aborting." % self.widgets[i].captionTitle)
                        break
                    else:
                        self.addEvent("Widget %s blocked during signal processing." % self.widgets[i].captionTitle)

                    # If during signal processing the widget changed state to
                    # blocking we skip all of its descendants
                    skipWidgets.update(self.widgetDescendants(self.widgets[i]))
            if self.freezing:
                self.addEvent("Signals frozen during processing of " + self.widgets[i].captionTitle + ". Aborting.")
                break

        # we finished propagating
        self.signalProcessingInProgress = 0

        if self.widgetQueue:
            # if there are still some widgets on queue
            self.processNewSignals(None)

    def scheduleSignalProcessing(self, widget=None):
        self.widgetQueue.append(widget)
        self.processNewSignals(widget)


    def existsPath(self, widgetFrom, widgetTo):
        """ Is there a path between `widgetFrom` and `widgetTo`
        """
        # is there a direct link
        if widgetFrom not in self.links:
            return 0

        for link in self.links[widgetFrom]:
            if link.widgetTo == widgetTo:
                return 1

        # is there a nondirect link
        for link in self.links[widgetFrom]:
            if self.existsPath(link.widgetTo, widgetTo):
                return 1

        # there is no link...
        return 0


    def widgetDescendants(self, widget):
        """ Return all widget descendants of `widget`
        """
        queue = [widget]
        queue_set = set(queue)

        index = self.widgets.index(widget)
        for i in range(index, len(self.widgets)):
            widget = self.widgets[i]
            if widget not in queue:
                continue
            linked = [link.widgetTo for link in self.links.get(widget, []) if link.enabled]
            for w in linked:
                if w not in queue_set:
                    queue.append(widget)
                    queue_set.add(widget)
        return queue


    def isWidgetBlocked(self, widget):
        """ Is this widget or any of its up-stream connected widgets blocked.
        """
        if widget.isBlocking():
            return True
        else:
            widgets = [link.widgetFrom for link in self.getLinks(None, widget, None, None)]
            if widgets:
                return any(self.isWidgetBlocked(w) for w in widgets)
            else:
                return False


    def getBlockedWidgets(self):
        """ Return a set of all widgets that are blocked.
        """
        blocked = set()
        for w in self.widgets:
            if w not in blocked and w.isBlocking():
                blocked.update(self.widgetDescendants(w))
        return blocked


    def freeze(self, widget=None):
        """ Return a context manager that freezes the signal processing
        """
        signalManager = self
        class freezer(object):
            def __enter__(self):
                self.push()
                return self

            def __exit__(self, *args):
                self.pop()

            def push(self):
                signalManager.setFreeze(signalManager.freezing + 1)

            def pop(self):
                signalManager.setFreeze(signalManager.freezing - 1, widget)

        return freezer()

    def setDynamicLinkEnabled(self, link, enabled):
        import PyQt4.QtCore as QtCore
        link.widgetFrom.emit(QtCore.SIGNAL("dynamicLinkEnabledChanged(PyQt_PyObject, bool)"), link, enabled)


# Channel renames.

_CHANNEL_NAME_MAP = \
    {'Additional Tables': 'Additional Data',
     'Attribute Definitions': 'Feature Definitions',
     'Attribute List': 'Features',
     'Attribute Pair': 'Interacting Features',
     'Attribute Selection List': 'Features',
     'Attribute Statistics': 'Feature Statistics',
     'Attribute selection': 'Features',
     'Attributes': 'Features',
     'Choosen Tree': 'Selected Tree',
     'Covered Examples': 'Covered Data',
     'Data Instances': 'Data',
     'Data Table': 'Data',
     'Distance Matrix': 'Distances',
     'Distance matrix': 'Distances',
     'Distances btw. Instances': 'Distances',
     'Example Subset': 'Data Subset',
     'Example Table': 'Data',
     'Examples': 'Data',
     'Examples A': 'Data A',
     'Examples B': 'Data B',
     'Examples with Z-scores': 'Data with z-score',
     'Graph with ExampleTable': 'Graph with Data',
     'Input Data': 'Data',
     'Input Table': 'Data',
     'Instances': 'Data',
     'Items Distance Matrix': 'Distances',
     'Items Subset': 'Item Subset',
     'Items to Mark': 'Marked Items',
     'KNN Classifier': 'kNN Classifier',
     'Marked Examples': 'Marked Data',
     'Matching Examples': 'Matching Data',
     'Merged Examples A+B': 'Merged Data A+B',
     'Merged Examples B+A': 'Merged Data B+A',
     'Mismatching Examples': 'Mismatched Data',
     'Non-Matching Examples': 'Unmatched Data',
     'Output Data': 'Data',
     'Output Table': 'Data',
     'Preprocessed Example Table': 'Preprocessed Data',
     'Primary Table': 'Primary Data',
     'Reduced Example Table': 'Reduced Data',
     'Remaining Examples': 'Remaining Data',
     'SOMMap': 'SOM',
     'Sample': 'Data Sample',
     'Selected Attributes List': 'Selected Features',
     'Selected Examples': 'Selected Data',
     'Selected Instances': 'Selected Data',
     'Selected Items Distance Matrix': 'Distance Matrix',
     'Shuffled Data Table': 'Shuffled Data',
     'Train Data': 'Training Data',
     'Training data': 'Data',
     'Unselected Examples': 'Other Data',
     'Unselected Items': 'Other Items',
     }

# create a global instance of signal manager
globalSignalManager = SignalManager()


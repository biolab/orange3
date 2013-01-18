#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#
import sys, time, os, pickle
from functools import reduce

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Orange.canvas.orngSignalManager import *
from Orange.canvas.utils import environ

from Orange.widgets import settings, gui

from Orange import data as orange_data

ERROR = 0
WARNING = 1

TRUE=1
FALSE=0

def unisetattr(self, name, value, grandparent):
    if "." in name:
        names = name.split(".")
        lastname = names.pop()
        obj = reduce(lambda o, n: getattr(o, n, None),  names, self)
    else:
        lastname, obj = name, self

    if not obj:
        print("unable to set setting ", name, " to value ", value)
    else:
        if hasattr(grandparent, "__setattr__") and isinstance(obj, grandparent):
            grandparent.__setattr__(obj, lastname,  value)
        else:
            setattr(obj, lastname, value)
#            obj.__dict__[lastname] = value

    controlledAttributes = hasattr(self, "controlledAttributes") and getattr(self, "controlledAttributes", None)
    controlCallback = controlledAttributes and controlledAttributes.get(name, None)
    if controlCallback:
        for callback in controlCallback:
            callback(value)
#        controlCallback(value)

    # controlled things (checkboxes...) never have __attributeControllers
    else:
        if hasattr(self, "__attributeControllers"):
            for controller, myself in self.__attributeControllers.keys():
                if getattr(controller, myself, None) != self:
                    del self.__attributeControllers[(controller, myself)]
                    continue

                controlledAttributes = hasattr(controller, "controlledAttributes") and getattr(controller, "controlledAttributes", None)
                if controlledAttributes:
                    fullName = myself + "." + name

                    controlCallback = controlledAttributes.get(fullName, None)
                    if controlCallback:
                        for callback in controlCallback:
                            callback(value)

                    else:
                        lname = fullName + "."
                        dlen = len(lname)
                        for controlled in controlledAttributes.keys():
                            if controlled[:dlen] == lname:
                                self.setControllers(value, controlled[dlen:], controller, fullName)
                                # no break -- can have a.b.c.d and a.e.f.g; needs to set controller for all!


    # if there are any context handlers, call the fastsave to write the value into the context
    if hasattr(self, "contextHandlers") and hasattr(self, "currentContexts"):
        for contextName, contextHandler in self.contextHandlers.items():
            contextHandler.fastSave(self.currentContexts.get(contextName), self, name, value)



class ControlledAttributesDict(dict):
    def __init__(self, master):
        self.master = master

    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, [value])
        else:
            dict.__getitem__(self, key).append(value)
        self.master.setControllers(self.master, key, self.master, "")



##################
# this definitions are needed only to define ExampleTable as subclass of ExampleTableWithClass
class AttributeList(list):
    pass

class ExampleList(list):
    pass

widgetId = 0

class BaseWidgetClass(type(QDialog)):
    def __new__(mcs, name, bases, dict):
        cls = type.__new__(mcs, name, bases, dict)
        if not hasattr(cls, "settings"):
            cls.settings = settings.SettingsHandler()
        for name, value in cls.__dict__.items():
            if isinstance(value, settings.Setting):
                cls.settings.settings[name] = value
                setattr(cls, name, value.default)
        return cls


class OWBaseWidget(QDialog, metaclass=BaseWidgetClass):
    def __init__(self, parent=None, signalManager=None, title="Orange BaseWidget", modal=FALSE, savePosition = False, resizingEnabled = 1, **args):
        super().__init__()

        self.currentContexts = {}   # the "currentContexts" MUST be the first thing assigned to a widget
        self._useContexts = 1       # do you want to use contexts
        self._owInfo = 1            # currently disabled !!!
        self._owWarning = 1         # do we want to see warnings
        self._owError = 1           # do we want to see errors
        self._owShowStatus = 0      # do we want to see warnings and errors in status bar area of the widget
        self._guiElements = []      # used for automatic widget debugging
        for key in args:
            if key in ["_owInfo", "_owWarning", "_owError", "_owShowStatus", "_useContexts", "_category", "_settingsFromSchema"]:
                self.__dict__[key] = args[key]        # we cannot use __dict__.update(args) since we can have many other

        if resizingEnabled:
            QDialog.__init__(self, parent, Qt.Window)
        else:
            QDialog.__init__(self, parent, Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)# | Qt.WindowMinimizeButtonHint)

        # do we want to save widget position and restore it on next load
        self.savePosition = savePosition
        if savePosition:
            self.settingsList = getattr(self, "settingsList", []) + ["widgetShown", "savedWidgetGeometry"]

        # directories are better defined this way, otherwise .ini files get written in many places
        self.__dict__.update(environ.directories)
        try:
            self.__dict__["thisWidgetDir"] = os.path.dirname(sys.modules[self.__class__.__module__].__file__)
        except:
            pass

        self.setCaption(title.replace("&","")) # used for widget caption
        self.setFocusPolicy(Qt.StrongFocus)

        # number of control signals, that are currently being processed
        # needed by signalWrapper to know when everything was sent
        self.parent = parent
        self.needProcessing = 0     # used by signalManager
        if not signalManager: self.signalManager = globalSignalManager        # use the global instance of signalManager  - not advised
        else:                 self.signalManager = signalManager              # use given instance of signal manager

        self.inputs = []     # signalName:(dataType, handler, onlySingleConnection)
        self.outputs = []    # signalName: dataType
        self.wrappers = []    # stored wrappers for widget events
        self.linksIn = {}      # signalName : (dirty, widgetFrom, handler, signalData)
        self.linksOut = {}       # signalName: (signalData, id)
        self.connections = {}   # dictionary where keys are (control, signal) and values are wrapper instances. Used in connect/disconnect
        self.controlledAttributes = ControlledAttributesDict(self)
        self.progressBarHandler = None  # handler for progress bar events
        self.processingHandler = None   # handler for processing events
        self.eventHandler = None
        self.callbackDeposit = []
        self.startTime = time.time()    # used in progressbar

        self.widgetStateHandler = None
        self.widgetState = {"Info":{}, "Warning":{}, "Error":{}}

        self.settings.initialize(self)

        if hasattr(self, "contextHandlers"):
            for contextHandler in self.contextHandlers.values():
                contextHandler.initLocalContext(self)
        self.loadSettings()

        global widgetId
        widgetId += 1
        self.widgetId = widgetId

        self._private_thread_pools = {}
        self.asyncCalls = []
        self.asyncBlock = False

        self.connect(self, SIGNAL("blockingStateChanged(bool)"), lambda bool :self.signalManager.log.info("Blocking state changed %s %s" % (str(self), str(bool))))


    # uncomment this when you need to see which events occured
    """
    def event(self, e):
        #eventDict = dict([(0, 'None'), (1, 'Timer'), (2, 'MouseButtonPress'), (3, 'MouseButtonRelease'), (4, 'MouseButtonDblClick'), (5, 'MouseMove'), (6, 'KeyPress'), (7, 'KeyRelease'), (8, 'FocusIn'), (9, 'FocusOut'), (10, 'Enter'), (11, 'Leave'), (12, 'Paint'), (13, 'Move'), (14, 'Resize'), (15, 'Create'), (16, 'Destroy'), (17, 'Show'), (18, 'Hide'), (19, 'Close'), (20, 'Quit'), (21, 'Reparent'), (22, 'ShowMinimized'), (23, 'ShowNormal'), (24, 'WindowActivate'), (25, 'WindowDeactivate'), (26, 'ShowToParent'), (27, 'HideToParent'), (28, 'ShowMaximized'), (30, 'Accel'), (31, 'Wheel'), (32, 'AccelAvailable'), (33, 'CaptionChange'), (34, 'IconChange'), (35, 'ParentFontChange'), (36, 'ApplicationFontChange'), (37, 'ParentPaletteChange'), (38, 'ApplicationPaletteChange'), (40, 'Clipboard'), (42, 'Speech'), (50, 'SockAct'), (51, 'AccelOverride'), (60, 'DragEnter'), (61, 'DragMove'), (62, 'DragLeave'), (63, 'Drop'), (64, 'DragResponse'), (70, 'ChildInserted'), (71, 'ChildRemoved'), (72, 'LayoutHint'), (73, 'ShowWindowRequest'), (80, 'ActivateControl'), (81, 'DeactivateControl'), (1000, 'User')])
        eventDict = dict([(0, "None"), (130, "AccessibilityDescription"), (119, "AccessibilityHelp"), (86, "AccessibilityPrepare"), (114, "ActionAdded"), (113, "ActionChanged"), (115, "ActionRemoved"), (99, "ActivationChange"), (121, "ApplicationActivated"), (122, "ApplicationDeactivated"), (36, "ApplicationFontChange"), (37, "ApplicationLayoutDirectionChange"), (38, "ApplicationPaletteChange"), (35, "ApplicationWindowIconChange"), (68, "ChildAdded"), (69, "ChildPolished"), (71, "ChildRemoved"), (40, "Clipboard"), (19, "Close"), (82, "ContextMenu"), (52, "DeferredDelete"), (60, "DragEnter"), (62, "DragLeave"), (61, "DragMove"), (63, "Drop"), (98, "EnabledChange"), (10, "Enter"), (150, "EnterEditFocus"), (124, "EnterWhatsThisMode"), (116, "FileOpen"), (8, "FocusIn"), (9, "FocusOut"), (97, "FontChange"), (159, "GraphicsSceneContextMenu"), (164, "GraphicsSceneDragEnter"), (166, "GraphicsSceneDragLeave"), (165, "GraphicsSceneDragMove"), (167, "GraphicsSceneDrop"), (163, "GraphicsSceneHelp"), (160, "GraphicsSceneHoverEnter"), (162, "GraphicsSceneHoverLeave"), (161, "GraphicsSceneHoverMove"), (158, "GraphicsSceneMouseDoubleClick"), (155, "GraphicsSceneMouseMove"), (156, "GraphicsSceneMousePress"), (157, "GraphicsSceneMouseRelease"), (168, "GraphicsSceneWheel"), (18, "Hide"), (27, "HideToParent"), (127, "HoverEnter"), (128, "HoverLeave"), (129, "HoverMove"), (96, "IconDrag"), (101, "IconTextChange"), (83, "InputMethod"), (6, "KeyPress"), (7, "KeyRelease"), (89, "LanguageChange"), (90, "LayoutDirectionChange"), (76, "LayoutRequest"), (11, "Leave"), (151, "LeaveEditFocus"), (125, "LeaveWhatsThisMode"), (88, "LocaleChange"), (153, "MenubarUpdated"), (43, "MetaCall"), (102, "ModifiedChange"), (4, "MouseButtonDblClick"), (2, "MouseButtonPress"), (3, "MouseButtonRelease"), (5, "MouseMove"), (109, "MouseTrackingChange"), (13, "Move"), (12, "Paint"), (39, "PaletteChange"), (131, "ParentAboutToChange"), (21, "ParentChange"), (75, "Polish"), (74, "PolishRequest"), (123, "QueryWhatsThis"), (14, "Resize"), (117, "Shortcut"), (51, "ShortcutOverride"), (17, "Show"), (26, "ShowToParent"), (50, "SockAct"), (112, "StatusTip"), (100, "StyleChange"), (87, "TabletMove"), (92, "TabletPress"), (93, "TabletRelease"), (171, "TabletEnterProximity"), (172, "TabletLeaveProximity"), (1, "Timer"), (120, "ToolBarChange"), (110, "ToolTip"), (78, "UpdateLater"), (77, "UpdateRequest"), (111, "WhatsThis"), (118, "WhatsThisClicked"), (31, "Wheel"), (132, "WinEventAct"), (24, "WindowActivate"), (103, "WindowBlocked"), (25, "WindowDeactivate"), (34, "WindowIconChange"), (105, "WindowStateChange"), (33, "WindowTitleChange"), (104, "WindowUnblocked"), (126, "ZOrderChange"), (169, "KeyboardLayoutChange"), (170, "DynamicPropertyChange")])
        if eventDict.has_key(e.type()):
            print str(self.windowTitle()), eventDict[e.type()]
        return QDialog.event(self, e)
    """

    def getIconNames(self, iconName):
        if type(iconName) == list:      # if canvas sent us a prepared list of valid names, just return those
            return iconName

        names = []
        name, ext = os.path.splitext(iconName)
        for num in [16, 32, 42, 60]:
            names.append("%s_%d%s" % (name, num, ext))
        fullPaths = []
        for paths in [(self.widgetDir, name), (self.widgetDir, "icons", name), (os.path.dirname(sys.modules[self.__module__].__file__), "icons", name)]:
            for name in names + [iconName]:
                fname = os.path.join(*paths)
                if os.path.exists(fname):
                    fullPaths.append(fname)
            if fullPaths != []:
                break

        if len(fullPaths) > 1 and fullPaths[-1].endswith(iconName):
            fullPaths.pop()     # if we have the new icons we can remove the default icon
        return fullPaths


    def setWidgetIcon(self, iconName):
        iconNames = self.getIconNames(iconName)

        icon = QIcon()
        for name in iconNames:
            pix = QPixmap(name)
            icon.addPixmap(pix)
#            frame = QPixmap(os.path.join(self.widgetDir, "icons/frame.png"))
#            icon = QPixmap(iconName)
#            result = QPixmap(icon.size())
#            painter = QPainter()
#            painter.begin(result)
#            painter.drawPixmap(0,0, frame)
#            painter.drawPixmap(0,0, icon)
#            painter.end()

        self.setWindowIcon(icon)


    # ##############################################
    def createAttributeIconDict(self):
        return gui.getAttributeIcons()

    def isDataWithClass(self, data, wantedVarType=None, checkMissing=False):
        self.error([1234, 1235, 1236])
        if not data:
            return 0
        if not data.domain.classVar:
            self.error(1234, "A data set with a class attribute is required.")
            return 0
        if wantedVarType and data.domain.classVar.varType != wantedVarType:
            self.error(1235, "Unable to handle %s class." % (data.domain.classVar.varType == orange_data.Variable.VarTypes.Discrete and "discrete" or "continuous"))
            return 0
        if checkMissing and not orange.Preprocessor_dropMissingClasses(data):
            self.error(1236, "Unable to handle data set with no known classes")
            return 0
        return 1

    # call processEvents(), but first remember position and size of widget in case one of the events would be move or resize
    # call this function if needed in __init__ of the widget
    def safeProcessEvents(self):
        keys = ["widgetShown"]
        vals = [(key, getattr(self, key, None)) for key in keys]
        qApp.processEvents()
        for (key, val) in vals:
            if val != None:
                setattr(self, key, val)


    # this function is called at the end of the widget's __init__ when the widgets is saving its position and size parameters
    def restoreWidgetPosition(self):
        if self.savePosition:
            geometry = getattr(self, "savedWidgetGeometry", None)
            restored = False
            if geometry is not None:
                restored = self.restoreGeometry(QByteArray(geometry))

            if restored:
                space = qApp.desktop().availableGeometry(self)
                frame, geometry = self.frameGeometry(), self.geometry()

                #Fix the widget size to fit inside the available space
                width = min(space.width() - (frame.width() - geometry.width()), geometry.width())
                height = min(space.height() - (frame.height() - geometry.height()), geometry.height())
                self.resize(width, height)

                #Move the widget to the center of available space if it is currently outside it
                if not space.contains(self.frameGeometry()):
                    x = max(0, space.width() / 2 - width / 2)
                    y = max(0, space.height() / 2 - height / 2)

                    self.move(x, y)

#            geometry.move(frameOffset) #Make sure the title bar is shown
#            self.setGeometry(geometry.intersected(space.adjusted(-frameOffset.x(), -frameOffset.y(), 0, 0)))


#            if self.isWindow():
#                frame = self.frameGeometry()
#                if space.topLeft() != QPoint(0, 0):
#                    self.move(self.geometry().topLeft() - frame.topLeft())
#            if getattr(self, "widgetXPosition", None) != None and getattr(self, "widgetYPosition", None) != None:
##                print self.captionTitle, "restoring position", self.widgetXPosition, self.widgetYPosition, "to", max(self.widgetXPosition, 0), max(self.widgetYPosition, 0)
#                self.move(max(self.widgetXPosition, space.x()), max(self.widgetYPosition, space.y()))
#            if getattr(self,"widgetWidth", None) != None and getattr(self,"widgetHeight", None) != None:
#                self.resize(min(self.widgetWidth, space.width()), min(self.widgetHeight, space.height()))
#            frame = self.frameGeometry()
#            area = lambda rect: rect.width() * rect.height()
#            if area(frame.intersected(space)) < area(frame):
#                self.move(max(min(space.right() - frame.width(), frame.x()), space.x()),
#                          max(min(space.height() - frame.height(), frame.y()), space.y()))

    # this is called in canvas when loading a schema. it opens the widgets that were shown when saving the schema
    def restoreWidgetStatus(self):
        if self.savePosition and getattr(self, "widgetShown", None):
            self.show()

    # when widget is resized, save new width and height into widgetWidth and widgetHeight. some widgets can put this two
    # variables into settings and last widget shape is restored after restart
    def resizeEvent(self, ev):
        QDialog.resizeEvent(self, ev)
        # Don't store geometry if the widget is not visible
        # (the widget receives the resizeEvent before showEvent and we must not
        # overwrite the the savedGeometry before then)
        if self.savePosition and self.isVisible():
            self.savedWidgetGeometry = str(self.saveGeometry())


    # when widget is moved, save new x and y position into widgetXPosition and widgetYPosition. some widgets can put this two
    # variables into settings and last widget position is restored after restart
    # Commented out because of Ubuntu (on call to restoreGeometry calls move event saving pos (0, 0)
#    def moveEvent(self, ev):
#        QDialog.moveEvent(self, ev)
#        if self.savePosition:
#            self.widgetXPosition = self.frameGeometry().x()
#            self.widgetYPosition = self.frameGeometry().y()
#            self.savedWidgetGeometry = str(self.saveGeometry())

    # set widget state to hidden
    def hideEvent(self, ev):
        if self.savePosition:
            self.widgetShown = 0
            self.savedWidgetGeometry = str(self.saveGeometry())
        QDialog.hideEvent(self, ev)

    # override the default show function.
    # after show() we must call processEvents because show puts some LayoutRequests in queue
    # and we must process them immediately otherwise the width(), height(), ... of elements in the widget will be wrong
#    def show(self):
#        QDialog.show(self)
#        qApp.processEvents()

    # set widget state to shown
    def showEvent(self, ev):
        QDialog.showEvent(self, ev)
        if self.savePosition:
            self.widgetShown = 1

        self.restoreWidgetPosition()

    def closeEvent(self, ev):
        if self.savePosition:
            self.savedWidgetGeometry = str(self.saveGeometry())
        QDialog.closeEvent(self, ev)

    def wheelEvent(self, event):
        """ Silently accept the wheel event. This is to ensure combo boxes
        and other controls that have focus don't receive this event unless
        the cursor is over them.

        """
        event.accept()

    def setCaption(self, caption):
        if self.parent != None and isinstance(self.parent, QTabWidget):
            self.parent.setTabText(self.parent.indexOf(self), caption)
        else:
            # we have to save caption title in case progressbar will change it
            self.captionTitle = str(caption)
            self.setWindowTitle(caption)

    # put this widget on top of all windows
    def reshow(self):
        self.show()
        self.raise_()
        self.activateWindow()


    def send(self, signalName, value, id = None):
        if not self.hasOutputName(signalName):
            print("Warning! Signal '%s' is not a valid signal name for the '%s' widget. Please fix the signal name." % (signalName, self.captionTitle))

        if signalName in self.linksOut:
            self.linksOut[signalName][id] = value
        else:
            self.linksOut[signalName] = {id:value}

        self.signalManager.send(self, signalName, value, id)


    def getdeepattr(self, attr, **argkw):
        try:
            return reduce(lambda o, n: getattr(o, n, None),  attr.split("."), self)
        except:
            if "default" in argkw:
                return argkw[default]
            else:
                raise AttributeError("'%s' has no attribute '%s'" % (self, attr))


    # Set all settings
    # settings - the map with the settings
    def setSettings(self,settings):
        for key in settings:
            self.__setattr__(key, settings[key])
        #self.__dict__.update(settings)

    # Get all settings
    # returns map with all settings
    def getSettings(self, alsoContexts = True, globalContexts=False):
        settings = {}
        if hasattr(self, "settingsList"):
            for name in self.settingsList:
                try:
                    settings[name] =  self.getdeepattr(name)
                except:
                    #print "Attribute %s not found in %s widget. Remove it from the settings list." % (name, self.captionTitle)
                    pass

        if alsoContexts:
            self.synchronizeContexts()
            contextHandlers = getattr(self, "contextHandlers", {})
            for contextHandler in list(contextHandlers.values()):
                contextHandler.mergeBack(self)
#                settings[contextHandler.localContextName] = contextHandler.globalContexts
# Instead of the above line, I found this; as far as I recall this was a fix
# for some bugs related to, say, Select Attributes not handling the context
# attributes properly, but I dare not add it without understanding what it does.
# Here it is, if these contexts give us any further trouble.
                if (contextHandler.syncWithGlobal and contextHandler.globalContexts is getattr(self, contextHandler.localContextName)) or globalContexts:
                    settings[contextHandler.localContextName] = contextHandler.globalContexts
                else:
                    local_contexts = getattr(self, contextHandler.localContextName, None)
                    if local_contexts:
                        settings[contextHandler.localContextName] = local_contexts
                    ###
                settings[contextHandler.localContextName+"Version"] = (settings.contextStructureVersion, contextHandler.contextDataVersion)

        return settings


    def getSettingsFile(self, file):
        if file is None:
            file = os.path.join(self.widget_settings_dir, self.captionTitle + ".ini")
            if not os.path.exists(file):
                try:
                    f = open(file, "wb")
                    pickle.dump({}, f)
                    f.close()
                except IOError:
                    return
        if isinstance(file, str):
            if os.path.exists(file):
                return open(file, "rb")
        else:
            return file


    # Loads settings from the widget's .ini file
    def loadSettings(self, file = None):
        file = self.getSettingsFile(file)
        if file:
            try:
                settings = pickle.load(file)
            except Exception as ex:
                print("Failed to load settings!", repr(ex), file=sys.stderr)
                settings = None

            if hasattr(self, "_settingsFromSchema"):
                if settings: settings.update(self._settingsFromSchema)
                else:        settings = self._settingsFromSchema

            # can't close everything into one big try-except since this would mask all errors in the below code
            if settings:
                if hasattr(self, "settingsList"):
                    self.setSettings(settings)

                contextHandlers = getattr(self, "contextHandlers", {})
                for contextHandler in contextHandlers.values():
                    localName = contextHandler.localContextName

                    structureVersion, dataVersion = settings.get(localName+"Version", (0, 0))
                    if (structureVersion < settings.contextStructureVersion or dataVersion < contextHandler.contextDataVersion)\
                            and localName in settings:
                        del settings[localName]
                        delattr(self, localName)
                        contextHandler.initLocalContext(self)

                    if not hasattr(self, "_settingsFromSchema"): #When running stand alone widgets
                        if contextHandler.syncWithGlobal:
                            local_contexts = settings.get(localName, None)
                            if local_contexts is not None:
                                contextHandler.globalContexts = local_contexts
                        else:
                            setattr(self, localName, contextHandler.globalContexts)


    def saveSettings(self, file = None):
        settings = self.getSettings(globalContexts=True)
        if settings:
            if file is None:
                file = os.path.join(self.widget_settings_dir, self.captionTitle + ".ini")
            if isinstance(file, str):
                file = open(file, "wb")
            pickle.dump(settings, file)

    # Loads settings from string str which is compatible with pickle
    def loadSettingsStr(self, str):
        if str == None or str == "":
            return

        settings = pickle.loads(str)
        self.setSettings(settings)

        contextHandlers = getattr(self, "contextHandlers", {})
        for contextHandler in contextHandlers.values():
            localName = contextHandler.localContextName
            if localName in settings:
                structureVersion, dataVersion = settings.get(localName+"Version", (0, 0))
                if structureVersion < settings.contextStructureVersion or dataVersion < contextHandler.contextDataVersion:
                    del settings[localName]
                    delattr(self, localName)
                    contextHandler.initLocalContext(self)
                else:
                    setattr(self, localName, settings[localName])

    # return settings in string format compatible with pickle
    def saveSettingsStr(self):
        settings = self.getSettings()
        return pickle.dumps(settings)

    def onDeleteWidget(self):
        pass

    # this function is only intended for derived classes to send appropriate signals when all settings are loaded
    def activateLoadedSettings(self):
        pass

    # reimplemented in other widgets
    def setOptions(self):
        pass

    # does widget have a signal with name in inputs
    def hasInputName(self, name):
        for input in self.inputs:
            if name == input[0]: return 1
        return 0

    # does widget have a signal with name in outputs
    def hasOutputName(self, name):
        for output in self.outputs:
            if name == output[0]: return 1
        return 0

    def getInputType(self, signalName):
        for input in self.inputs:
            if input[0] == signalName: return input[1]
        return None

    def getOutputType(self, signalName):
        for output in self.outputs:
            if output[0] == signalName: return output[1]
        return None

    # ########################################################################
    def connect(self, control, signal, method, type=Qt.AutoConnection):
        wrapper = SignalWrapper(self, method)
        self.connections[(control, signal)] = wrapper   # save for possible disconnect
        self.wrappers.append(wrapper)
        QDialog.connect(control, signal, wrapper, type)
        #QWidget.connect(control, signal, method)        # ordinary connection useful for dialogs and windows that don't send signals to other widgets


    def disconnect(self, control, signal, method=None):
        wrapper = self.connections[(control, signal)]
        QDialog.disconnect(control, signal, wrapper)


    def getConnectionMethod(self, control, signal):
        if (control, signal) in self.connections:
            wrapper = self.connections[(control, signal)]
            return wrapper.method
        else:
            return None


    def signalIsOnlySingleConnection(self, signalName):
        for i in self.inputs:
            input = InputSignal(*i)
            if input.name == signalName: return input.single

    def addInputConnection(self, widgetFrom, signalName):
        for i in range(len(self.inputs)):
            if self.inputs[i][0] == signalName:
                handler = self.inputs[i][2]
                break

        existing = []
        if signalName in self.linksIn:
            existing = self.linksIn[signalName]
            for (dirty, widget, handler, data) in existing:
                if widget == widgetFrom: return             # no need to add new tuple, since one from the same widget already exists
        self.linksIn[signalName] = existing + [(0, widgetFrom, handler, [])]    # (dirty, handler, signalData)
        #if not self.linksIn.has_key(signalName): self.linksIn[signalName] = [(0, widgetFrom, handler, [])]    # (dirty, handler, signalData)

    # delete a link from widgetFrom and this widget with name signalName
    def removeInputConnection(self, widgetFrom, signalName):
        if signalName in self.linksIn:
            links = self.linksIn[signalName]
            for i in range(len(self.linksIn[signalName])):
                if widgetFrom == self.linksIn[signalName][i][1]:
                    self.linksIn[signalName].remove(self.linksIn[signalName][i])
                    if self.linksIn[signalName] == []:  # if key is empty, delete key value
                        del self.linksIn[signalName]
                    return

    # return widget, that is already connected to this singlelink signal. If this widget exists, the connection will be deleted (since this is only single connection link)
    def removeExistingSingleLink(self, signal):
        for i in self.inputs:
            input = InputSignal(*i)
            if input.name == signal and not input.single: return None

        for signalName in self.linksIn.keys():
            if signalName == signal:
                widget = self.linksIn[signalName][0][1]
                del self.linksIn[signalName]
                return widget

        return None


    def handleNewSignals(self):
        # this is called after all new signals have been handled
        # implement this in your widget if you want to process something only after you received multiple signals
        pass

    # signal manager calls this function when all input signals have updated the data
    def processSignals(self):
        if self.processingHandler:
            self.processingHandler(self, 1)    # focus on active widget
        newSignal = 0        # did we get any new signals

        # we define only a way to handle signals that have defined a handler function
        for signal in self.inputs:        # we go from the first to the last defined input
            key = signal[0]
            if key in self.linksIn:
                for i in range(len(self.linksIn[key])):
                    (dirty, widgetFrom, handler, signalData) = self.linksIn[key][i]
                    if not (handler and dirty): continue
                    newSignal = 1

                    qApp.setOverrideCursor(Qt.WaitCursor)
                    try:
                        for (value, id, nameFrom) in signalData:
                            if self.signalIsOnlySingleConnection(key):
                                self.printEvent("ProcessSignals: Calling %s with %s" % (handler, value), eventVerbosity = 2)
                                handler(value)
                            else:
                                self.printEvent("ProcessSignals: Calling %s with %s (%s, %s)" % (handler, value, nameFrom, id), eventVerbosity = 2)
                                handler(value, (widgetFrom, nameFrom, id))
                    except:
                        type, val, traceback = sys.exc_info()
                        sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that we don't crash other widgets
                    qApp.restoreOverrideCursor()

                    self.linksIn[key][i] = (0, widgetFrom, handler, []) # clear the dirty flag

        if newSignal == 1:
            self.handleNewSignals()

        while self.isBlocking():
            self.thread().msleep(50)
            qApp.processEvents()

        if self.processingHandler:
            self.processingHandler(self, 0)    # remove focus from this widget
        self.needProcessing = 0

    # set new data from widget widgetFrom for a signal with name signalName
    def updateNewSignalData(self, widgetFrom, signalName, value, id, signalNameFrom):
        if signalName not in self.linksIn: return
        for i in range(len(self.linksIn[signalName])):
            (dirty, widget, handler, signalData) = self.linksIn[signalName][i]
            if widget == widgetFrom:
                if self.linksIn[signalName][i][3] == []:
                    self.linksIn[signalName][i] = (1, widget, handler, [(value, id, signalNameFrom)])
                else:
                    found = 0
                    for j in range(len(self.linksIn[signalName][i][3])):
                        (val, ID, nameFrom) = self.linksIn[signalName][i][3][j]
                        if ID == id and nameFrom == signalNameFrom:
                            self.linksIn[signalName][i][3][j] = (value, id, signalNameFrom)
                            found = 1
                    if not found:
                        self.linksIn[signalName][i] = (1, widget, handler, self.linksIn[signalName][i][3] + [(value, id, signalNameFrom)])
        self.needProcessing = 1

    # ############################################
    # PROGRESS BAR FUNCTIONS

    progressBarValueChanged = pyqtSignal(float)
    """Progress bar value has changed"""

    processingStateChanged = pyqtSignal(int)
    """Processing state has changed"""

    def progressBarInit(self):
        self.progressBarValue = 0
        self.startTime = time.time()
        self.setWindowTitle(self.captionTitle + " (0% complete)")
        if self.progressBarHandler:
            self.progressBarHandler(self, 0)
        self.processingStateChanged.emit(1)

    def progressBarSet(self, value):
        if value > 0:
            self.__progressBarValue = value
            usedTime = max(1, time.time() - self.startTime)
            totalTime = (100.0 * usedTime) / float(value)
            remainingTime = max(0, totalTime - usedTime)
            h = int(remainingTime / 3600)
            min = int((remainingTime - h * 3600) / 60)
            sec = int(remainingTime - h * 3600 - min * 60)
            if h > 0:
                text = "%(h)d:%(min)02d:%(sec)02d" % vars()
            else:
                text = "%(min)d:%(sec)02d" % vars()
            self.setWindowTitle(self.captionTitle + " (%(value).2f%% complete, remaining time: %(text)s)" % vars())
        else:
            self.setWindowTitle(self.captionTitle + " (0% complete)")
        if self.progressBarHandler:
            self.progressBarHandler(self, value)

        self.progressBarValueChanged.emit(value)

        qApp.processEvents()

    def progressBarValue(self):
        return self.__progressBarValue

    progressBarValue = pyqtProperty(float, fset=progressBarSet,
                                    fget=progressBarValue)

    def progressBarAdvance(self, value):
        self.progressBarSet(self.progressBarValue + value)

    def progressBarFinished(self):
        self.setWindowTitle(self.captionTitle)
        if self.progressBarHandler:
            self.progressBarHandler(self, 101)
        self.processingStateChanged.emit(0)

    # handler must be a function, that receives 2 arguments. First is the widget instance, the second is the value between -1 and 101
    def setProgressBarHandler(self, handler):
        self.progressBarHandler = handler

    def setProcessingHandler(self, handler):
        self.processingHandler = handler

    def setEventHandler(self, handler):
        self.eventHandler = handler

    def setWidgetStateHandler(self, handler):
        self.widgetStateHandler = handler


    # if we are in debug mode print the event into the file
    def printEvent(self, text, eventVerbosity = 1):
        self.signalManager.addEvent(self.captionTitle + ": " + text, eventVerbosity = eventVerbosity)
        if self.eventHandler:
            self.eventHandler(self.captionTitle + ": " + text, eventVerbosity)

    def openWidgetHelp(self):
        if "widgetInfo" in self.__dict__:  # This widget is on a canvas.
            qApp.canvasDlg.helpWindow.showHelpFor(self.widgetInfo, True)

    def focusInEvent(self, *ev):
        #print "focus in"
        #if qApp.canvasDlg.settings["synchronizeHelp"]:  on ubuntu: pops up help window on first widget focus for every widget
        #    qApp.canvasDlg.helpWindow.showHelpFor(self, True)
        QDialog.focusInEvent(self, *ev)


    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Help, Qt.Key_F1):
            self.openWidgetHelp()
#            e.ignore()
        elif (int(e.modifiers()), e.key()) in OWBaseWidget.defaultKeyActions:
            OWBaseWidget.defaultKeyActions[int(e.modifiers()), e.key()](self)
        else:
            QDialog.keyPressEvent(self, e)

    def information(self, id = 0, text = None):
        self.setState("Info", id, text)
        #self.setState("Warning", id, text)

    def warning(self, id = 0, text = ""):
        self.setState("Warning", id, text)
        #self.setState("Info", id, text)        # if we want warning just set information

    def error(self, id = 0, text = ""):
        self.setState("Error", id, text)

    def setState(self, stateType, id, text):
        changed = 0
        if type(id) == list:
            for val in id:
                if val in self.widgetState[stateType]:
                    self.widgetState[stateType].pop(val)
                    changed = 1
        else:
            if type(id) == str:
                text = id; id = 0       # if we call information(), warning(), or error() function with only one parameter - a string - then set id = 0
            if not text:
                if id in self.widgetState[stateType]:
                    self.widgetState[stateType].pop(id)
                    changed = 1
            else:
                self.widgetState[stateType][id] = text
                changed = 1

        if changed:
            if self.widgetStateHandler:
                self.widgetStateHandler()
            elif text: # and stateType != "Info":
                self.printEvent(stateType + " - " + text)

            if type(id) == list:
                for i in id:
                    self.emit(SIGNAL("widgetStateChanged(QString, int, QString)"),
                              stateType, i, "")
            else:
                self.emit(SIGNAL("widgetStateChanged(QString, int, QString)"),
                          stateType, id, text or "")
                #qApp.processEvents()
        return changed

    widgetStateChanged = pyqtSignal(str, int, str)
    """Widget state has changed first arg is the state type
    ('Info', 'Warning' or 'Error') the second is the message id
    and finally the message string."""

    def widgetStateToHtml(self, info=True, warning=True, error=True):
        pixmaps = self.getWidgetStateIcons()
        items = []
        iconPath = {"Info": "canvasIcons:information.png",
                    "Warning": "canvasIcons:warning.png",
                    "Error": "canvasIcons:error.png"}
        for show, what in [(info, "Info"), (warning, "Warning"),(error, "Error")]:
            if show and self.widgetState[what]:
                items.append('<img src="%s" style="float: left;"> %s' % (iconPath[what], "\n".join(self.widgetState[what].values())))
        return "<br>".join(items)

    @classmethod
    def getWidgetStateIcons(cls):
        if not hasattr(cls, "_cached__widget_state_icons"):
            iconsDir = os.path.join(environ.canvas_install_dir, "icons")
            QDir.addSearchPath("canvasIcons",os.path.join(environ.canvas_install_dir,
                "icons/"))
            info = QPixmap("canvasIcons:information.png")
            warning = QPixmap("canvasIcons:warning.png")
            error = QPixmap("canvasIcons:error.png")
            cls._cached__widget_state_icons = \
                    {"Info": info, "Warning": warning, "Error": error}
        return cls._cached__widget_state_icons

    def synchronizeContexts(self):
        if hasattr(self, "contextHandlers"):
            for contextName, handler in self.contextHandlers.items():
                context = self.currentContexts.get(contextName, None)
                if context:
                    handler.settingsFromWidget(self, context)

    def openContext(self, contextName="", *arg):
        if not self._useContexts:
            return
        handler = self.contextHandlers[contextName]
        context = handler.openContext(self, *arg)
        if context:
            self.currentContexts[contextName] = context


    def closeContext(self, contextName=""):
        if not self._useContexts:
            return
        curcontext = self.currentContexts.get(contextName)
        if curcontext:
            self.contextHandlers[contextName].closeContext(self, curcontext)
            del self.currentContexts[contextName]

    def settingsToWidgetCallback(self, handler, context):
        pass

    def settingsFromWidgetCallback(self, handler, context):
        pass

    def setControllers(self, obj, controlledName, controller, prefix):
        while obj:
            if prefix:
#                print "SET CONTROLLERS: %s %s + %s" % (obj.__class__.__name__, prefix, controlledName)
                if "attributeController" in obj.__dict__:
                    obj.__dict__["__attributeControllers"][(controller, prefix)] = True
                else:
                    obj.__dict__["__attributeControllers"] = {(controller, prefix): True}

            parts = controlledName.split(".", 1)
            if len(parts) < 2:
                break
            obj = getattr(obj, parts[0], None)
            prefix += parts[0]
            controlledName = parts[1]

    def __setattr__(self, name, value):
        return unisetattr(self, name, value, QDialog)

    defaultKeyActions = {}

    if sys.platform == "darwin":
        defaultKeyActions = {
            (Qt.ControlModifier, Qt.Key_M): lambda self: self.showMaximized if self.isMinimized() else self.showMinimized(),
            (Qt.ControlModifier, Qt.Key_W): lambda self: self.setVisible(not self.isVisible())}


    def scheduleSignalProcessing(self):
        self.signalManager.scheduleSignalProcessing(self)

    def setBlocking(self, state=True):
        """ Set blocking flag for this widget. While this flag is set this
        widget and all its descendants will not receive any new signals from
        the signal manager
        """
        self.asyncBlock = state
        self.emit(SIGNAL("blockingStateChanged(bool)"), self.asyncBlock)
        if not self.isBlocking():
            self.scheduleSignalProcessing()


    def isBlocking(self):
        """ Is this widget blocking signal processing. Widget is blocking if
        asyncBlock value is True or any AsyncCall objects in asyncCalls list
        has blocking flag set
        """
        return self.asyncBlock or any(a.blocking for a in self.asyncCalls)

    def asyncExceptionHandler(self, exception):
        (etype, value, tb) = exception
        import traceback
        sys.excepthook(etype, value, tb)

    def asyncFinished(self, async, string):
        """ Remove async from asyncCalls, update blocking state
        """

        index = self.asyncCalls.index(async)
        async = self.asyncCalls.pop(index)

        if async.blocking and not self.isBlocking():
            # if we are responsible for unblocking
            self.emit(SIGNAL("blockingStateChanged(bool)"), False)
            self.scheduleSignalProcessing()

        async.disconnect(async, SIGNAL("finished(PyQt_PyObject, QString)"), self.asyncFinished)
        self.emit(SIGNAL("asyncCallsStateChange()"))



    def asyncCall(self, func, args=(), kwargs={}, name=None, onResult=None, onStarted=None, onFinished=None, onError=None, blocking=True, thread=None, threadPool=None):
        """ Return an OWConcurent.AsyncCall object func, args and kwargs
        set and signals connected.
        """
        from functools import partial
        from OWConcurrent import AsyncCall

        asList = lambda slot: slot if isinstance(slot, list) else ([slot] if slot else [])

        onResult = asList(onResult)
        onStarted = asList(onStarted) #+ [partial(self.setBlocking, True)]
        onFinished = asList(onFinished) #+ [partial(self.blockSignals, False)]
        onError = asList(onError) or [self.asyncExceptionHandler]

        async = AsyncCall(func, args, kwargs, thread=thread, threadPool=threadPool)
        async.name = name if name is not None else ""

        for slot in  onResult:
            async.connect(async, SIGNAL("resultReady(PyQt_PyObject)"), slot, Qt.QueuedConnection)
        for slot in onStarted:
            async.connect(async, SIGNAL("starting()"), slot, Qt.QueuedConnection)
        for slot in onFinished:
            async.connect(async, SIGNAL("finished(QString)"), slot, Qt.QueuedConnection)
        for slot in onError:
            async.connect(async, SIGNAL("unhandledException(PyQt_PyObject)"), slot, Qt.QueuedConnection)

        self.addAsyncCall(async, blocking)

        return async

    def addAsyncCall(self, async, blocking=True):
        """ Add AsyncCall object to asyncCalls list (will be removed
        once it finishes processing).

        """
        ## TODO: make this thread safe

        async.connect(async, SIGNAL("finished(PyQt_PyObject, QString)"), self.asyncFinished)

        async.blocking = blocking

        if blocking:
            # if we are responsible for blocking
            state = any(a.blocking for a in self.asyncCalls)
            self.asyncCalls.append(async)
            if not state:
                self.emit(SIGNAL("blockingStateChanged(bool)"), True)
        else:
            self.asyncCalls.append(async)

        self.emit(SIGNAL("asyncCallsStateChange()"))


def blocking(method):
    """ Return method that sets blocking flag while executing
    """
    from functools import wraps
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        old = self._blocking
        self.setBlocking(True)
        try:
            return method(self, *args, **kwargs)
        finally:
            self.setBlocking(old)


if __name__ == "__main__":
    a=QApplication(sys.argv)
    oww=OWBaseWidget(adfaf=1)
    oww.show()
    a.exec_()
    oww.saveSettings()

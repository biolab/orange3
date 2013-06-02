import math, os
from functools import reduce

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import Orange

YesNo = NoYes = ("No", "Yes")
groupBoxMargin = 7

def id_generator(id):
    while True:
        id += 1
        yield id

OrangeUserRole = id_generator(Qt.UserRole)

enter_icon = None

def getdeepattr(obj, attr, **argkw):
    if type(obj) is dict:
        return obj.get(attr)
    try:
        return reduce(lambda o, n: getattr(o, n),  attr.split("."), obj)
    except:
            raise AttributeError("'%s' has no attribute '%s'" % (obj, attr))


def getEnterIcon():
    global enter_icon
    if not enter_icon:
        enter_icon = QIcon(os.path.dirname(__file__) + "/icons/Dlg_enter.png")
    return enter_icon


# constructs a box (frame) if not none, and returns the right master widget
def widgetBox(widget, box=None, orientation='vertical', addSpace=False, sizePolicy = None, margin = -1, spacing = -1, flat = 0, addToLayout = 1, stretch=0):
    if box:
        b = QGroupBox(widget)
        if isinstance(box, str): # if you pass 1 for box, there will be a box, but no text
            b.setTitle(" "+box.strip()+" ")
        if margin == -1: margin = groupBoxMargin
        b.setFlat(flat)
    else:
        b = QWidget(widget)
        if margin == -1: margin = 0
    if addToLayout and widget.layout() is not None:
        widget.layout().addWidget(b, stretch)

    if isinstance(orientation, QLayout):
        b.setLayout(orientation)
    elif orientation == 'horizontal' or not orientation:
        b.setLayout(QHBoxLayout())
##        b.setSizePolicy(sizePolicy or QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum))
    else:
        b.setLayout(QVBoxLayout())
##        b.setSizePolicy(sizePolicy or QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
    if sizePolicy:
        b.setSizePolicy(sizePolicy)

    if spacing == -1: spacing = 4
    b.layout().setSpacing(spacing)
    if margin != -1:
        b.layout().setMargin(margin)

    if addSpace and isinstance(addSpace, bool):
        separator(widget)
    elif addSpace and isinstance(addSpace, int):
        separator(widget, addSpace, addSpace)
    elif addSpace:
        separator(widget)

    return b

def indentedBox(widget, sep=20, orientation = True, addSpace=False):
    r = widgetBox(widget, orientation = "horizontal", spacing=0)
    separator(r, sep, 0)

    if addSpace and isinstance(addSpace, bool):
        separator(widget)
    elif addSpace and isinstance(addSpace, int):
        separator(widget, 0, addSpace)
    elif addSpace:
        separator(widget)

    return widgetBox(r, orientation = orientation)

def widgetLabel(widget, label=None, labelWidth=None, addToLayout = 1):
    if label is not None:
        lbl = QLabel(label, widget)
        if labelWidth:
            lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
        if widget.layout() is not None and addToLayout: widget.layout().addWidget(lbl)
    else:
        lbl = None

    return lbl


import re
__re_frmt = re.compile(r"(^|[^%])%\((?P<value>[a-zA-Z]\w*)\)")

def label(widget, master, label, labelWidth = None):
    lbl = QLabel("", widget)
    if widget.layout() is not None: widget.layout().addWidget(lbl)

    reprint = CallFrontLabel(lbl, label, master)
    for mo in __re_frmt.finditer(label):
        master.controlledAttributes[mo.group("value")] = reprint
    reprint()

    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())

    return lbl


class SpinBoxWFocusOut(QSpinBox):
    def __init__(self, min, max, step, bi):
        QSpinBox.__init__(self, bi)
        self.setRange(min, max)
        self.setSingleStep(step)
        self.inSetValue = False
        self.enterButton = None

    def onChange(self, value):
        if not self.inSetValue:
            self.placeHolder.hide()
            self.enterButton.show()

    def onEnter(self):
        if self.enterButton.isVisible():
            self.enterButton.hide()
            self.placeHolder.show()
            if self.cback:
                self.cback(int(str(self.text())))
            if self.cfunc:
                self.cfunc()

    # doesn't work: it's probably LineEdit's focusOut that we should (and can't) catch
    def focusOutEvent(self, *e):
        QSpinBox.focusOutEvent(self, *e)
        if self.enterButton and self.enterButton.isVisible():
            self.onEnter()

    def setValue(self, value):
        self.inSetValue = True
        QSpinBox.setValue(self, value)
        self.inSetValue = False


def checkWithSpin(widget, master, label, min, max, checked, value, posttext = None, step = 1, tooltip=None,
                  checkCallback=None, spinCallback=None, getwidget=None,
                  labelWidth=None, debuggingEnabled = 1, controlWidth=55,
                  callbackOnReturn = False):
    return spin(widget, master, value, min, max, step, None, label, labelWidth, 0, tooltip,
                spinCallback, debuggingEnabled, controlWidth, callbackOnReturn, checked, checkCallback, posttext)



def spin(widget, master, value, min, max, step=1,
         box=None, label=None, labelWidth=None, orientation=None, tooltip=None,
         callback=None, debuggingEnabled = 1, controlWidth = None, callbackOnReturn = False,
         checked = "", checkCallback = None, posttext = None, addToLayout=True,
         alignment = Qt.AlignLeft, keyboardTracking=True):
    if box or label and not checked:
        b = widgetBox(widget, box, orientation)
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False

    if not hasHBox and (checked or callback and callbackOnReturn or posttext):
        bi = widgetBox(b, "", 0)
    else:
        bi = b

    if checked:
        wb = checkBox(bi, master, checked, label, labelWidth = labelWidth, callback=checkCallback, debuggingEnabled = debuggingEnabled)
    elif label:
        b.label = widgetLabel(b, label, labelWidth)


    wa = bi.control = SpinBoxWFocusOut(min, max, step, bi)
    wa.setAlignment(alignment)
    wa.setKeyboardTracking(keyboardTracking) # If false it wont emit valueChanged signals while editing the text
    if addToLayout and bi.layout() is not None:
        bi.layout().addWidget(wa)
    # must be defined because of the setText below
    if controlWidth:
        wa.setFixedWidth(controlWidth)
    if tooltip:
        wa.setToolTip(tooltip)
    if value:
        wa.setValue(getdeepattr(master, value))

    cfront, wa.cback, wa.cfunc = connectControl(wa, master, value, callback, not (callback and callbackOnReturn) and "valueChanged(int)", CallFrontSpin(wa))

    if checked:
        wb.disables = [wa]
        wb.makeConsistent()

    if callback and callbackOnReturn:
        wa.enterButton, wa.placeHolder = enterButton(bi, wa.sizeHint().height())
        QObject.connect(wa, SIGNAL("valueChanged(const QString &)"), wa.onChange)
        QObject.connect(wa, SIGNAL("editingFinished()"), wa.onEnter)
        QObject.connect(wa.enterButton, SIGNAL("clicked()"), wa.onEnter)
        if hasattr(wa, "upButton"):
            QObject.connect(wa.upButton(), SIGNAL("clicked()"), lambda c=wa.editor(): c.setFocus())
            QObject.connect(wa.downButton(), SIGNAL("clicked()"), lambda c=wa.editor(): c.setFocus())

    if posttext:
        widgetLabel(bi, posttext)

    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("spin", wa, value, min, max, step, callback)]

    if checked:
        return wb, wa
    else:
        return wa


class DoubleSpinBoxWFocusOut(QDoubleSpinBox):
    def __init__(self, min, max, step, bi):
        QDoubleSpinBox.__init__(self, bi)
        self.setDecimals(math.ceil(-math.log10(step)))
        self.setRange(min, max)
        self.setSingleStep(step)
        self.inSetValue = False
        self.enterButton = None

    def onChange(self, value):
        if not self.inSetValue:
            self.placeHolder.hide()
            self.enterButton.show()

    def onEnter(self):
        if self.enterButton.isVisible():
            self.enterButton.hide()
            self.placeHolder.show()
            if self.cback:
                self.cback(float(str(self.text()).replace(",", ".")))
            if self.cfunc:
                self.cfunc()

    # doesn't work: it's probably LineEdit's focusOut that we should (and can't) catch
    def focusOutEvent(self, *e):
        QDoubleSpinBox.focusOutEvent(self, *e)
        if self.enterButton and self.enterButton.isVisible():
            self.onEnter()

    def setValue(self, value):
        self.inSetValue = True
        QDoubleSpinBox.setValue(self, value)
        self.inSetValue = False

def doubleSpin(widget, master, value, min, max, step=1,
               box=None, label=None, labelWidth=None, orientation=None, tooltip=None,
               callback=None, debuggingEnabled = 1, controlWidth = None, callbackOnReturn = False,
               checked = "", checkCallback = None, posttext = None, addToLayout=True, alignment = Qt.AlignLeft,
               keyboardTracking=True, decimals=None):
    if box or label and not checked:
        b = widgetBox(widget, box, orientation)
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False

    if not hasHBox and (checked or callback and callbackOnReturn or posttext):
        bi = widgetBox(b, "", 0)
    else:
        bi = b

    if checked:
        wb = checkBox(bi, master, checked, label, labelWidth = labelWidth, callback=checkCallback, debuggingEnabled = debuggingEnabled)
    elif label:
        widgetLabel(b, label, labelWidth)


    wa = bi.control = DoubleSpinBoxWFocusOut(min, max, step, bi)

    if decimals is not None:
        wa.setDecimals(decimals)

    wa.setAlignment(alignment)
    wa.setKeyboardTracking(keyboardTracking) # If false it wont emit valueChanged signals while editing the text
    if addToLayout and bi.layout() is not None:
        bi.layout().addWidget(wa)
    # must be defined because of the setText below
    if controlWidth:
        wa.setFixedWidth(controlWidth)
    if tooltip:
        wa.setToolTip(tooltip)
    if value:
        wa.setValue(getdeepattr(master, value))

    cfront, wa.cback, wa.cfunc = connectControl(wa, master, value, callback, not (callback and callbackOnReturn) and "valueChanged(double)", CallFrontDoubleSpin(wa))

    if checked:
        wb.disables = [wa]
        wb.makeConsistent()

    if callback and callbackOnReturn:
        wa.enterButton, wa.placeHolder = enterButton(bi, wa.sizeHint().height())
        QObject.connect(wa, SIGNAL("valueChanged(const QString &)"), wa.onChange)
        QObject.connect(wa, SIGNAL("editingFinished()"), wa.onEnter)
        QObject.connect(wa.enterButton, SIGNAL("clicked()"), wa.onEnter)
        if hasattr(wa, "upButton"):
            QObject.connect(wa.upButton(), SIGNAL("clicked()"), lambda c=wa.editor(): c.setFocus())
            QObject.connect(wa.downButton(), SIGNAL("clicked()"), lambda c=wa.editor(): c.setFocus())

    if posttext:
        widgetLabel(bi, posttext)

##    if debuggingEnabled and hasattr(master, "_guiElements"):
##        master._guiElements = getattr(master, "_guiElements", []) + [("spin", wa, value, min, max, step, callback)]

    if checked:
        return wb, wa
    else:
        if b==widget:
            wa.control = b.control # Backward compatibility
            return wa
        else:
            return b

def checkBox(widget, master, value, label, box=None, tooltip=None, callback=None, getwidget=None, id=None, disabled=0, labelWidth=None, disables = [], addToLayout = 1, debuggingEnabled = 1):
    if box:
        b = widgetBox(widget, box, orientation=None)
    else:
        b = widget
    wa = QCheckBox(label, b)
    wa.box = b
    if addToLayout and b.layout() is not None:
        b.layout().addWidget(wa)

    if labelWidth:
        wa.setFixedSize(labelWidth, wa.sizeHint().height())
    wa.setChecked(getdeepattr(master, value))
    if disabled:
        wa.setDisabled(1)
    if tooltip:
        wa.setToolTip(tooltip)

    cfront, cback, cfunc = connectControl(wa, master, value, None, "toggled(bool)", CallFrontCheckBox(wa),
                                          cfunc = callback and FunctionCallback(master, callback, widget=wa, getwidget=getwidget, id=id))
    wa.disables = disables or [] # need to create a new instance of list (in case someone would want to append...)
    wa.makeConsistent = Disabler(wa, master, value)
    QObject.connect(wa, SIGNAL("toggled(bool)"), wa.makeConsistent)
    wa.makeConsistent.__call__(value)
    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("checkBox", wa, value, callback)]
    return wa


def enterButton(parent, height, placeholder = True):
    button = QToolButton(parent)
    button.setFixedSize(height, height)
    button.setIcon(getEnterIcon())
    if parent.layout() is not None:
        parent.layout().addWidget(button)
    if not placeholder:
        return button

    button.hide()
    holder = QWidget(parent)
    holder.setFixedSize(height, height)
    if parent.layout() is not None:
        parent.layout().addWidget(holder)
    return button, holder


class LineEditWFocusOut(QLineEdit):
    def __init__(self, parent, master, callback, focusInCallback=None, placeholder=False):
        QLineEdit.__init__(self, parent)
        if parent.layout() is not None:
            parent.layout().addWidget(self)
        self.callback = callback
        self.focusInCallback = focusInCallback
        if placeholder:
            self.enterButton, self.placeHolder = enterButton(parent, self.sizeHint().height(), placeholder)
        else:
            self.enterButton = enterButton(parent, self.sizeHint().height(), placeholder)
            self.placeHolder = None
        QObject.connect(self.enterButton, SIGNAL("clicked()"), self.returnPressed)
        QObject.connect(self, SIGNAL("textChanged(const QString &)"), self.markChanged)
        QObject.connect(self, SIGNAL("returnPressed()"), self.returnPressed)

    def markChanged(self, *e):
        if self.placeHolder:
            self.placeHolder.hide()
        self.enterButton.show()

    def markUnchanged(self, *e):
        self.enterButton.hide()
        if self.placeHolder:
            self.placeHolder.show()

    def returnPressed(self):
        if self.enterButton.isVisible():
            self.markUnchanged()
            if hasattr(self, "cback") and self.cback:
                self.cback(self.text())
            if self.callback:
                self.callback()

    def setText(self, t):
        QLineEdit.setText(self, t)
        if self.enterButton:
            self.markUnchanged()

    def focusOutEvent(self, *e):
        QLineEdit.focusOutEvent(self, *e)
        self.returnPressed()

    def focusInEvent(self, *e):
        if self.focusInCallback:
            self.focusInCallback()
        return QLineEdit.focusInEvent(self, *e)


def lineEdit(widget, master, value,
             label=None, labelWidth=None, orientation='vertical', box=None, tooltip=None,
             callback=None, valueType = str, validator=None, controlWidth = None, callbackOnType = False, focusInCallback = None, enterPlaceholder=False, **args):
    if box or label:
        b = widgetBox(widget, box, orientation)
        widgetLabel(b, label, labelWidth)
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False

    if "baseClass" in args:
        wa = args["baseClass"](b)
        wa.enterButton = None
        if b and b.layout() is not None:
            b.layout().addWidget(wa)
    elif focusInCallback or callback and not callbackOnType:
        if not hasHBox:
            bi = widgetBox(b, "", 0)
        else:
            bi = b
        wa = LineEditWFocusOut(bi, master, callback, focusInCallback, enterPlaceholder)
    else:
        wa = QLineEdit(b)
        wa.enterButton = None
        if b and b.layout() is not None:
            b.layout().addWidget(wa)

    if value:
        wa.setText(str(getdeepattr(master, value)))

    if controlWidth:
        wa.setFixedWidth(controlWidth)

    if tooltip:
        wa.setToolTip(tooltip)
    if validator:
        wa.setValidator(validator)

    if value:
        wa.cback = connectControl(wa, master, value, callbackOnType and callback, "textChanged(const QString &)", CallFrontLineEdit(wa), fvcb = value and valueType)[1]

    wa.box = b
    return wa


def button(widget, master, label, callback = None, disabled=0, tooltip=None,
           debuggingEnabled = 1, width = None, height = None, toggleButton = False,
           value = "", addToLayout = 1, default=False, autoDefault=False):
    btn = QPushButton(label, widget)
    if addToLayout and widget.layout() is not None:
        widget.layout().addWidget(btn)

    if width:
        btn.setFixedWidth(width)
    if height:
        btn.setFixedHeight(height)
    btn.setDisabled(disabled)
    if tooltip:
        btn.setToolTip(tooltip)

    if toggleButton or value:
        btn.setCheckable(True)

    btn.setDefault(default)
    btn.setAutoDefault(autoDefault)

    if value:
        btn.setChecked(getdeepattr(master, value))
        cfront, cback, cfunc = connectControl(btn, master, value, None, "toggled(bool)", CallFrontButton(btn),
                                              cfunc = callback and FunctionCallback(master, callback, widget=btn))
    elif callback:
        QObject.connect(btn, SIGNAL("clicked()"), callback)

    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("button", btn, callback)]
    return btn

def toolButton(widget, master, label="", callback = None, width = None, height = None, tooltip = None, addToLayout = 1, debuggingEnabled = 1):
    if not isinstance(label, str) and hasattr(label, "__call__"):
        import warnings
        warnings.warn("Third positional argument to 'OWGUI.toolButton' must be a string.", DeprecationWarning)
        label, callback = "", label

    btn = QToolButton(widget)
    if addToLayout and widget.layout() is not None:
        widget.layout().addWidget(btn)
    if label:
        btn.setText(label)
    if width != None:
        btn.setFixedWidth(width)
    if height!= None:
        btn.setFixedHeight(height)
    if tooltip != None:
        btn.setToolTip(tooltip)
    if callback:
        QObject.connect(btn, SIGNAL("clicked()"), callback)
    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("button", btn, callback)]
    return btn


def separator(widget, width=4, height=4):
#    if isinstance(widget.layout(), QVBoxLayout):
#        return widget.layout().addSpacing(height)
#    elif isinstance(widget.layout(), QHBoxLayout):
#        return widget.layout().addSpacing(width)
#    return None
    sep = QWidget(widget)
#    sep.setStyleSheet("background: #000000;")
    if widget.layout() is not None:
        widget.layout().addWidget(sep)
    sep.setFixedSize(width, height)
    return sep

def rubber(widget):
    widget.layout().addStretch(100)

def createAttributePixmap(char, color = Qt.black):
    pixmap = QPixmap(13,13)
    painter = QPainter()
    painter.begin(pixmap)
    painter.setPen( color );
    painter.setBrush( color );
    painter.drawRect( 0, 0, 13, 13 );
    painter.setPen( QColor(Qt.white))
    painter.drawText(3, 11, char)
    painter.end()
    return QIcon(pixmap)


attributeIconDict = None

def attributeIcon(attr):
    from Orange.data import Variable
    if attributeIconDict is None:
        constructAttributeIcons()
    if isinstance(attr, Variable):
        return attributeIconDict[attr.var_type]
    else:
        return attributeIconDict[attr]

def attributeItem(attr):
    from Orange.data import Variable
    if attributeIconDict is None:
        constructAttributeIcons()
    return attributeIconDict[attr.var_type], attr.name

def constructAttributeIcons():
    from Orange.data import Variable
    VarTypes = Variable.VarTypes
    global attributeIconDict
    if not attributeIconDict:
        attributeIconDict = {
            VarTypes.Continuous: createAttributePixmap("C", QColor(202,0,32)),
            VarTypes.Discrete: createAttributePixmap("D", QColor(26,150,65)),
            VarTypes.String: createAttributePixmap("S", Qt.black),
            -1: createAttributePixmap("?", QColor(128, 128, 128))}
    return attributeIconDict


def listBox(widget, master, value = None, labels = None, box = None, tooltip = None, callback = None, selectionMode = QListWidget.SingleSelection, enableDragDrop = 0, dragDropCallback = None, dataValidityCallback = None, sizeHint = None, debuggingEnabled = 1):
    bg = box and widgetBox(widget, box, orientation = "horizontal") or widget
    lb = OrangeListBox(master, value, enableDragDrop, dragDropCallback, dataValidityCallback, sizeHint, bg)
    lb.box = bg
    lb.setSelectionMode(selectionMode)
    if bg.layout() is not None:
        bg.layout().addWidget(lb)

    if value != None:
        clist = getdeepattr(master, value)
        if isinstance(clist, ControlledList):
            clist = ControlledList(clist, lb)
            master.__setattr__(value, clist)

    lb.ogValue = value
    lb.ogLabels = labels
    lb.ogMaster = master
    if tooltip:
        lb.setToolTip(tooltip)

    connectControl(lb, master, value, callback, "itemSelectionChanged()", CallFrontListBox(lb), CallBackListBox(lb, master))
    if hasattr(master, "controlledAttributes") and labels != None:
        master.controlledAttributes[labels] = CallFrontListBoxLabels(lb)
    if labels != None:
        setattr(master, labels, getdeepattr(master, labels))
    if value != None:
        setattr(master, value, getdeepattr(master, value))
    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("listBox", lb, value, callback)]
    return lb


# btnLabels is a list of either char strings or pixmaps
def radioButtonsInBox(widget, master, value, btnLabels, box=None, tooltips=None, callback=None, debuggingEnabled = 1, addSpace = False, orientation = 'vertical', label = None):
    if box:
        bg = widgetBox(widget, box, orientation)
    else:
        bg = widget

    bg.group = QButtonGroup(bg)

    if addSpace:
        separator(widget)

    if not label is None:
        widgetLabel(bg, label)

    bg.buttons = []
    bg.ogValue = value
    for i in range(len(btnLabels)):
        appendRadioButton(bg, master, value, btnLabels[i], tooltips and tooltips[i], callback = callback)

    connectControl(bg.group, master, value, callback, "buttonClicked (int)", CallFrontRadioButtons(bg), CallBackRadioButton(bg, master))

    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("radioButtonsInBox", bg, value, callback)]
    return bg


def appendRadioButton(bg, master, value, label, tooltip = None, insertInto = None, callback = None, addToLayout=True):
    dest = insertInto or bg

    if not hasattr(bg, "buttons"):
        bg.buttons = []
    i = len(bg.buttons)

    if isinstance(label, str):
        w = QRadioButton(label)
    else:
        w = QRadioButton(str(i))
        w.setIcon(QIcon(label))
    #w.ogValue = value
    if addToLayout and dest.layout() is not None:
        dest.layout().addWidget(w)
    if not hasattr(bg, "group"):
        bg.group = QButtonGroup(bg)
    bg.group.addButton(w)

    w.setChecked(getdeepattr(master, value) == i)
    bg.buttons.append(w)
#    if callback == None and hasattr(bg, "callback"):
#        callback = bg.callback
#    if callback != None:
#        connectControl(w, master, value, callback, "clicked()", CallFrontRadioButtons(bg), CallBackRadioButton(w, master, bg))
    if tooltip:
        w.setToolTip(tooltip)
    return w

#def radioButton(widget, master, value, label, box = None, tooltip = None, callback = None, debuggingEnabled = 1):
#    if box:
#        bg = widgetBox(widget, box, orientation="horizontal")
#    else:
#        bg = widget
#
#    if type(label) in (str, unicode):
#        w = QRadioButton(label, bg)
#    else:
#        w = QRadioButton("X")
#        w.setPixmap(label)
#    if bg.layout(): bg.layout().addWidget(w)
#
#    w.setChecked(getattr_deep(master, value))
#    if tooltip:
#        w.setToolTip(tooltip)
#
#    connectControl(w, master, value, callback, "stateChanged(int)", CallFrontCheckBox(w))
#    if debuggingEnabled and hasattr(master, "_guiElements"):
#        master._guiElements = getattr(master, "_guiElements", []) + [("radioButton", w, value, callback)]
#    return w


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1, callback=None, label=None, labelFormat=" %d", ticks=0, divideFactor = 1.0, debuggingEnabled = 1, vertical = False, createLabel = 1, tooltip = None, width = None, intOnly = 1):
    sliderBox = widgetBox(widget, box, orientation = "horizontal")
    if label:
        lbl = widgetLabel(sliderBox, label)

    if vertical:
        sliderOrient = Qt.Vertical
    else:
        sliderOrient = Qt.Horizontal

    if intOnly:
        slider = QSlider(sliderOrient, sliderBox)
        slider.setRange(minValue, maxValue)
        if step != 0:
            slider.setSingleStep(step)
            slider.setPageStep(step)
            slider.setTickInterval(step)
        signal_signature = "valueChanged(int)"
    else:
        slider = FloatSlider(sliderOrient, minValue, maxValue, step)
        signal_signature = "valueChangedFloat(double)"
    slider.setValue(getdeepattr(master, value))

    if tooltip:
        slider.setToolTip(tooltip)

    if width != None:
        slider.setFixedWidth(width)

    if sliderBox.layout() is not None:
        sliderBox.layout().addWidget(slider)

    if ticks:
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    if createLabel:
        label = QLabel(sliderBox)
        if sliderBox.layout() is not None:
            sliderBox.layout().addWidget(label)
        label.setText(labelFormat % minValue)
        width1 = label.sizeHint().width()
        label.setText(labelFormat % maxValue)
        width2 = label.sizeHint().width()
        label.setFixedSize(max(width1, width2), label.sizeHint().height())
        txt = labelFormat % (getdeepattr(master, value)/divideFactor)
        label.setText(txt)
        label.setLbl = lambda x, l=label, f=labelFormat: l.setText(f % (x/divideFactor))
        QObject.connect(slider, SIGNAL(signal_signature), label.setLbl)

    connectControl(slider, master, value, callback, signal_signature, CallFrontHSlider(slider))
    # For compatibility with qwtSlider
    slider.box = sliderBox
    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("hSlider", slider, value, minValue, maxValue, step, callback)]
    return slider


def qwtHSlider(widget, master, value, box=None, label=None, labelWidth=None, minValue=1, maxValue=10, step=0.1, precision=1, callback=None, logarithmic=0, ticks=0, maxWidth=80, tooltip = None, showValueLabel = 1, debuggingEnabled = 1, addSpace=False, orientation=0):
    if not logarithmic:
        if type(precision) is str:
            format = precision
        elif precision == 0:
            format = " %d"
        else:
            format = " %s.%df" % ("%", precision)

        return hSlider(widget, master, value, box, minValue, maxValue, step,
                       callback, label=label, labelFormat=format,
                       width=maxWidth, tooltip=tooltip,
                       debuggingEnabled=debuggingEnabled, intOnly=0)

    import PyQt4.Qwt5 as qwt

    init = getdeepattr(master, value)

    if label:
        hb = widgetBox(widget, box, orientation)
        lbl = widgetLabel(hb, label)
        if labelWidth:
            lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
        if orientation and orientation!="horizontal":
            separator(hb, height=2)
            hb = widgetBox(hb, 0)
    else:
        hb = widgetBox(widget, box, 0)

    if ticks:
        slider = qwt.QwtSlider(hb, Qt.Horizontal, qwt.QwtSlider.Bottom, qwt.QwtSlider.BgSlot)
    else:
        slider = qwt.QwtSlider(hb, Qt.Horizontal, qwt.QwtSlider.NoScale, qwt.QwtSlider.BgSlot)
    hb.layout().addWidget(slider)

    slider.setScale(minValue, maxValue, logarithmic) # the third parameter for logaritmic scale
    slider.setScaleMaxMinor(10)
    slider.setThumbWidth(20)
    slider.setThumbLength(12)
    if maxWidth:
        slider.setMaximumSize(maxWidth,40)
    if logarithmic:
        slider.setRange(math.log10(minValue), math.log10(maxValue), step)
        slider.setValue(math.log10(init))
    else:
        slider.setRange(minValue, maxValue, step)
        slider.setValue(init)
    if tooltip:
        hb.setToolTip(tooltip)

##    format = "%s%d.%df" % ("%", precision+3, precision)
#    format = " %s.%df" % ("%", precision)
    if precision is str:
        format = precision
    else:
        format = " %s.%df" % ("%", precision)

    if showValueLabel:
        lbl = widgetLabel(hb, format % minValue)
        width1 = lbl.sizeHint().width()
        lbl.setText(format % maxValue)
        width2 = lbl.sizeHint().width()
        lbl.setFixedSize(max(width1, width2), lbl.sizeHint().height())
        lbl.setText(format % init)

    if logarithmic:
        cfront = CallFrontLogSlider(slider)
        cback = ValueCallback(master, value, f=lambda x: 10**x)
        if showValueLabel: QObject.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format, f=lambda x: 10**x))
    else:
        cfront = CallFrontHSlider(slider)
        cback = ValueCallback(master, value)
        if showValueLabel: QObject.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format))
    connectControl(slider, master, value, callback, "valueChanged(double)", cfront, cback)
    slider.box = hb

    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("qwtHSlider", slider, value, minValue, maxValue, step, callback)]
    return slider


# list box where we can use drag and drop
class OrangeListBox(QListWidget):
    def __init__(self, widget, value = None, enableDragDrop = 0, dragDropCallback = None, dataValidityCallback = None, sizeHint = None, *args):
        self.widget = widget
        self.value = value
        QListWidget.__init__(self, *args)
        self.enableDragDrop = enableDragDrop
        self.dragDopCallback = dragDropCallback
        self.dataValidityCallback = dataValidityCallback
        if not sizeHint:
            self.defaultSizeHint = QSize(150,100)
        else:
            self.defaultSizeHint = sizeHint
        if enableDragDrop:
            self.setDragEnabled(1)
            self.setAcceptDrops(1)
            self.setDropIndicatorShown(1)
            #self.setDragDropMode(QAbstractItemView.DragDrop)
            self.dragStartPosition = 0

    def setAttributes(self, data, attributes):
        if isinstance(shownAttributes[0], tuple):
            setattr(self.widget, self.ogLabels, attributes)
        else:
            domain = data.domain
            setattr(self.widget, self.ogLabels,
                    [(domain[a].name, domain[a].var_type) for a in attributes])

    def sizeHint(self):
        return self.defaultSizeHint


    def startDrag(self, supportedActions):
        if not self.enableDragDrop: return

        drag = QDrag(self)
        mime = QMimeData()

        if not self.ogValue:
            selectedItems = [i for i in range(self.count()) if self.item(i).isSelected()]
        else:
            selectedItems = getdeepattr(self.widget, self.ogValue, default = [])

        mime.setText(str(selectedItems))
        mime.source = self
        drag.setMimeData(mime)
        drag.start(Qt.MoveAction)

    def dragEnterEvent(self, ev):
        if not self.enableDragDrop: return
        if self.dataValidityCallback: return self.dataValidityCallback(ev)

        if ev.mimeData().hasText():
            ev.accept()
        else:
            ev.ignore()


    def dragMoveEvent(self, ev):
        if not self.enableDragDrop: return
        if self.dataValidityCallback: return self.dataValidityCallback(ev)

        if ev.mimeData().hasText():
            ev.setDropAction(Qt.MoveAction)
            ev.accept()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        if not self.enableDragDrop: return
        if ev.mimeData().hasText():
            item = self.itemAt(ev.pos())
            if item:
                index = self.indexFromItem(item).row()
            else:
                index = self.count()

            source = ev.mimeData().source
            selectedItemIndices = eval(str(ev.mimeData().text()))

            if self.ogLabels != None and self.ogValue != None:
                allSourceItems = getdeepattr(source.widget, source.ogLabels, default = [])
                selectedItems = [allSourceItems[i] for i in selectedItemIndices]
                allDestItems = getdeepattr(self.widget, self.ogLabels, default = [])

                if source != self:
                    setattr(source.widget, source.ogLabels, [item for item in allSourceItems if item not in selectedItems])   # TODO: optimize this code. use the fact that the selectedItemIndices is a sorted list
                    setattr(self.widget, self.ogLabels, allDestItems[:index] + selectedItems + allDestItems[index:])
                    setattr(source.widget, source.ogValue, [])  # clear selection in the source widget
                else:
                    items = [item for item in allSourceItems if item not in selectedItems]
                    if index < len(allDestItems):
                        while index > 0 and index in getdeepattr(self.widget, self.ogValue, default = []):      # if we are dropping items on a selected item, we have to select some previous unselected item as the drop target
                            index -= 1
                        destItem = allDestItems[index]
                        index = items.index(destItem)
                    else:
                        index = max(0, index - len(selectedItems))
                    setattr(self.widget, self.ogLabels, items[:index] + selectedItems + items[index:])
                setattr(self.widget, self.ogValue, list(range(index, index+len(selectedItems))))
            else:       # if we don't have variables ogValue and ogLabel
                if source != self:
                    self.insertItems(source.selectedItems())
                    for index in selectedItemIndices[::-1]:
                        source.takeItem(index)
                else:
                    if index < self.count():
                        while index > 0 and self.item(index).isSelected():      # if we are dropping items on a selected item, we have to select some previous unselected item as the drop target
                            index -= 1
                    items = [source.item(i) for i in selectedItemIndices]
                    for ind in selectedItemIndices[::-1]:
                        source.takeItem(ind)
                        if ind <= index: index-= 1
                    for item in items[::-1]:
                        self.insertItem(index, item)
                    self.clearSelection()
                    for i in range(index, index+len(items)):
                        self.item(i).setSelected(1)

            if self.dragDopCallback:        # call the callback
                self.dragDopCallback()
            ev.setDropAction(Qt.MoveAction)
            ev.accept()
        else:
            ev.ignore()

    def updateGeometries(self):
        """ A workaround for a bug in Qt (see: http://bugreports.qt.nokia.com/browse/QTBUG-14412)
        """
        if getattr(self, "_updatingGeometriesNow", False):
#            import sys
#            print >> sys.stderr, "Suppressing recursive update geometries"
            return
        self._updatingGeometriesNow = True
        try:
            return QListWidget.updateGeometries(self)
        finally:
            self._updatingGeometriesNow = False


class SmallWidgetButton(QPushButton):
    def __init__(self, widget, text = "", pixmap = None, box = None, orientation='vertical', tooltip = None, autoHideWidget = None):
        #self.parent = parent
        if pixmap != None:
            import os
            iconDir = os.path.join(os.path.dirname(__file__), "icons")
            if isinstance(pixmap, str):
                if os.path.exists(pixmap):
                    name = pixmap
                elif os.path.exists(os.path.join(iconDir, pixmap)):
                    name = os.path.join(iconDir, pixmap)
            elif type(pixmap) is QPixmap or type(pixmap) is QIcon:
                name = pixmap
            else:
                name = os.path.join(iconDir, "arrow_down.png")
            QPushButton.__init__(self, QIcon(name), text, widget)
        else:
            QPushButton.__init__(self, text, widget)
        if widget.layout() is not None:
            widget.layout().addWidget(self)
        if tooltip != None:
            self.setToolTip(tooltip)
        # create autohide widget and set a layout
        if autoHideWidget != None:
            self.autohideWidget = autoHideWidget(None, Qt.Popup)
        else:
            self.autohideWidget = AutoHideWidget(None, Qt.Popup)
        self.widget = self.autohideWidget

        if isinstance(orientation, QLayout):
            self.widget.setLayout(orientation)
        elif orientation == 'horizontal' or not orientation:
            self.widget.setLayout(QHBoxLayout())
        else:
            self.widget.setLayout(QVBoxLayout())
        #self.widget.layout().setMargin(groupBoxMargin)

        if box:
            self.widget = widgetBox(self.widget, box, orientation)
        #self.setStyleSheet("QPushButton:hover { background-color: #F4F2F0; }")

        self.autohideWidget.hide()

    def mousePressEvent(self, ev):
        QWidget.mousePressEvent(self, ev)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            #self.widget.move(self.parent.mapToGlobal(QPoint(0, 0)).x(), self.mapToGlobal(QPoint(0, self.height())).y())
            self.autohideWidget.move(self.mapToGlobal(QPoint(0, self.height())))
            self.autohideWidget.show()


class SmallWidgetLabel(QLabel):
    def __init__(self, widget, text = "", pixmap = None, box = None, orientation='vertical', tooltip = None):
        QLabel.__init__(self, widget)
        if text != "":
            self.setText("<font color=\"#C10004\">" + text + "</font>")
        elif pixmap != None:
            import os
            iconDir = os.path.join(os.path.dirname(__file__), "icons")
            if isinstance(pixmap, str):
                if os.path.exists(pixmap):
                    name = pixmap
                elif os.path.exists(os.path.join(iconDir, pixmap)):
                    name = os.path.join(iconDir, pixmap)
            elif type(pixmap) is QPixmap or type(pixmap) is QIcon:
                name = pixmap
            else:
                name = os.path.join(iconDir, "arrow_down.png")
            self.setPixmap(QPixmap(name))
        if widget.layout() is not None:
            widget.layout().addWidget(self)
        if tooltip != None:
            self.setToolTip(tooltip)
        self.autohideWidget = self.widget = AutoHideWidget(None, Qt.Popup)

        if isinstance(orientation, QLayout):
            self.widget.setLayout(orientation)
        elif orientation == 'horizontal' or not orientation:
            self.widget.setLayout(QHBoxLayout())
        else:
            self.widget.setLayout(QVBoxLayout())

        if box:
            self.widget = widgetBox(self.widget, box, orientation)

        self.autohideWidget.hide()

    def mousePressEvent(self, ev):
        QLabel.mousePressEvent(self, ev)
        if self.autohideWidget.isVisible():
            self.autohideWidget.hide()
        else:
            #self.widget.move(self.parent.mapToGlobal(QPoint(0, 0)).x(), self.mapToGlobal(QPoint(0, self.height())).y())
            self.autohideWidget.move(self.mapToGlobal(QPoint(0, self.height())))
            self.autohideWidget.show()


class AutoHideWidget(QWidget):
#    def __init__(self, parent = None):
#        QWidget.__init__(self, parent, Qt.Popup)

    def leaveEvent(self, ev):
        self.hide()



class SearchLineEdit(QLineEdit):
    def __init__(self, t, searcher):
        QLineEdit.__init__(self, t)
        self.searcher = searcher

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Down:
            curItem = self.searcher.lb.currentItem()
            if curItem+1 < self.searcher.lb.count():
                self.searcher.lb.setCurrentItem(curItem+1)
        elif k == Qt.Key_Up:
            curItem = self.searcher.lb.currentItem()
            if curItem:
                self.searcher.lb.setCurrentItem(curItem-1)
        elif k == Qt.Key_Escape:
            self.searcher.window.hide()
        else:
            return QLineEdit.keyPressEvent(self, e)

class Searcher:
    def __init__(self, control, master):
        self.control = control
        self.master = master

    def __call__(self):
        self.window = t = QFrame(self.master, "", QStyle.WStyle_Dialog + QStyle.WStyle_Tool + QStyle.WStyle_Customize + QStyle.WStyle_NormalBorder)
        la = QVBoxLayout(t).setAutoAdd(1)
        gs = self.master.mapToGlobal(QPoint(0, 0))
        gl = self.control.mapToGlobal(QPoint(0, 0))
        t.move(gl.x()-gs.x(), gl.y()-gs.y())
        self.allItems = [str(self.control.text(i)) for i in range(self.control.count())]
        le = SearchLineEdit(t, self)
        self.lb = QListBox(t)
        for i in self.allItems:
            self.lb.insertItem(i)
        t.setFixedSize(self.control.width(), 200)
        t.show()
        le.setFocus()

        QObject.connect(le, SIGNAL("textChanged(const QString &)"), self.textChanged)
        QObject.connect(le, SIGNAL("returnPressed()"), self.returnPressed)
        QObject.connect(self.lb, SIGNAL("clicked(QListBoxItem *)"), self.mouseClicked)

    def textChanged(self, s):
        s = str(s)
        self.lb.clear()
        for i in self.allItems:
            if s.lower() in i.lower():
                self.lb.insertItem(i)

    def returnPressed(self):
        if self.lb.count():
            self.conclude(self.lb.text(max(0, self.lb.currentItem())))
        else:
            self.window.hide()

    def mouseClicked(self, item):
        self.conclude(item.text())

    def conclude(self, valueQStr):
        value = str(valueQStr)
        index = self.allItems.index(value)
        self.control.setCurrentItem(index)
        if self.control.cback:
            if self.control.sendSelectedValue:
                self.control.cback(value)
            else:
                self.control.cback(index)
        if self.control.cfunc:
            self.control.cfunc()

        self.window.hide()



def comboBox(widget, master, value, box=None, label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=None, sendSelectedValue = 0, valueType = str, control2attributeDict = {}, emptyString = None, editable = 0, searchAttr = False, indent = 0, addToLayout = 1, addSpace = False, debuggingEnabled = 1):
    hb = widgetBox(widget, box, orientation)
    widgetLabel(hb, label, labelWidth)
    if tooltip:
        hb.setToolTip(tooltip)
    combo = QComboBox(hb)
    combo.setEditable(editable)
    combo.box = hb

    if addSpace:
        if isinstance(addSpace, bool):
            separator(widget)
        elif isinstance(addSpace, int):
            separator(widget, height=addSpace)
        else:
            separator(widget)

    if indent:
        hb = widgetBox(hb, orientation = "horizontal")
        hb.layout().addSpacing(indent)
    if hb.layout() is not None and addToLayout:
        hb.layout().addWidget(combo)

    if items:
        combo.addItems([str(i) for i in items])
        if len(items)>0 and value != None:
            if sendSelectedValue and getdeepattr(master, value) in items: combo.setCurrentIndex(items.index(getdeepattr(master, value)))
            elif not sendSelectedValue and getdeepattr(master, value) < combo.count():
                combo.setCurrentIndex(getdeepattr(master, value))
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
        else:
            combo.setDisabled(True)

    if value != None:
        if sendSelectedValue:
            control2attributeDict = dict(control2attributeDict)
            if emptyString:
                control2attributeDict[emptyString] = ""
            connectControl(combo, master, value, callback, "activated( const QString & )",
                           CallFrontComboBox(combo, valueType, control2attributeDict),
                           ValueCallbackCombo(master, value, valueType, control2attributeDict))
        else:
            connectControl(combo, master, value, callback, "activated(int)", CallFrontComboBox(combo, None, control2attributeDict))

    if debuggingEnabled and hasattr(master, "_guiElements"):
        master._guiElements = getattr(master, "_guiElements", []) + [("comboBox", combo, value, sendSelectedValue, valueType, callback)]
    return combo


def comboBoxWithCaption(widget, master, value, label, box=None, items=None, tooltip=None, callback = None, sendSelectedValue=0, valueType = int, labelWidth = None, debuggingEnabled = 1):
    hbox = widgetBox(widget, box = box, orientation="horizontal")
    lab = widgetLabel(hbox, label + "  ", labelWidth)
    combo = comboBox(hbox, master, value, items = items, tooltip = tooltip, callback = callback, sendSelectedValue = sendSelectedValue, valueType = valueType, debuggingEnabled = debuggingEnabled)
    return combo

# creates a widget box with a button in the top right edge, that allows you to hide all the widgets in the box and collapse the box to its minimum height
class collapsableWidgetBox(QGroupBox):
    def __init__(self, widget, box = "", master = None, value = "", orientation = "vertical", callback = None):
        QGroupBox.__init__(self, widget)
        self.setFlat(1)
        if orientation == 'vertical': self.setLayout(QVBoxLayout())
        else:                         self.setLayout(QHBoxLayout())

        if widget.layout() is not None:
            widget.layout().addWidget(self)
        if isinstance(box, str): # if you pass 1 for box, there will be a box, but no text
            self.setTitle(" " + box.strip() + " ")

        self.setCheckable(1)

        self.master = master
        self.value = value
        self.callback = callback
        QObject.connect(self, SIGNAL("clicked()"), self.toggled)


    def toggled(self, val = 0):
        if self.value:
            self.master.__setattr__(self.value, self.isChecked())
            self.updateControls()
#            self.setFlat(1)
        if self.callback != None:
            self.callback()

    def updateControls(self):
        val = self.master.getattr_deep(self.value)
        width = self.width()
        self.setChecked(val)
        self.setFlat(not val)
        if not val:
            self.setMinimumSize(QSize(width, 0))
        else:
            self.setMinimumSize(QSize(0, 0))

        for c in self.children():
            if isinstance(c, QLayout): continue
            if val:
                c.show()
            else:
                c.hide()

# creates an icon that allows you to show/hide the widgets in the widgets list
class widgetHider(QWidget):
    def __init__(self, widget, master, value, size = (19,19), widgets = [], tooltip = None):
        QWidget.__init__(self, widget)
        if widget.layout() is not None:
            widget.layout().addWidget(self)
        self.value = value
        self.master = master

        if tooltip:
            self.setToolTip(tooltip)

        import os
        iconDir = os.path.join(os.path.dirname(__file__), "icons")
        icon1 = os.path.join(iconDir, "arrow_down.png")
        icon2 = os.path.join(iconDir, "arrow_up.png")
        self.pixmaps = []

        self.pixmaps = [QPixmap(icon1), QPixmap(icon2)]
        self.setFixedSize(self.pixmaps[0].size())

        self.disables = widgets or [] # need to create a new instance of list (in case someone would want to append...)
        self.makeConsistent = Disabler(self, master, value, type = HIDER)
        if widgets != []:
            self.setWidgets(widgets)

    def mousePressEvent(self, ev):
        self.master.__setattr__(self.value, not getdeepattr(self.master, self.value))
        self.makeConsistent.__call__()


    def setWidgets(self, widgets):
        self.disables = widgets or []
        self.makeConsistent.__call__()

    def paintEvent(self, ev):
        QWidget.paintEvent(self, ev)

        if self.pixmaps != []:
            pix = self.pixmaps[getdeepattr(self.master, self.value)]
            painter = QPainter(self)
            painter.drawPixmap(0, 0, pix)


##############################################################################
# callback handlers

def setStopper(master, sendButton, stopCheckbox, changedFlag, callback):
    stopCheckbox.disables.append((-1, sendButton))
    sendButton.setDisabled(stopCheckbox.isChecked())
    QObject.connect(stopCheckbox, SIGNAL("toggled(bool)"),
                    lambda x, master=master, changedFlag=changedFlag, callback=callback: x and getdeepattr(master, changedFlag, default=True) and callback())


class ControlledList(list):
    def __init__(self, content, listBox = None):
        super().__init__(content)
        self.listBox = listBox

    def __reduce__(self):
        # cannot pickle self.listBox, but can't discard it (ControlledList may live on)
        import copyreg
        return copyreg._reconstructor, (list, list, ()), None, self.__iter__()

    def item2name(self, item):
        item = self.listBox.labels[item]
        if type(item) is tuple:
            return item[1]
        else:
            return item

    def __setitem__(self, index, item):
        if isinstance(index, int):
            self.listBox.item(self[index]).setSelected(0)
            item.setSelected(1)
        else:
            for i in self[index]:
                self.listBox.item(i).setSelected(0)
            for i in item:
                self.listBox.item(i).setSelected(1)
        super().__setitem__(index, item)

    def __delitem__(self, index):
        if isinstance(index, int):
            self.listBox.item(self[index]).setSelected(0)
        else:
            for i in self[index]:
                self.listBox.item(self[index]).setSelected(0)
        super().__delitem__(index)

    def append(self, item):
        super().append(item)
        item.setSelected(1)

    def extend(self, items):
        super().extend(items)
        for i in items:
            self.listBox.item(i).setSelected(1)

    def insert(self, index, item):
        item.setSelected(1)
        super().insert(index, item)

    def pop(self, index=-1):
        i = super().pop(index)
        self.listBox.item(i).setSelected(0)

    def remove(self, item):
        item.setSelected(0)
        super().remove(item)


def connectControlSignal(control, signal, f):
    if type(signal) is tuple:
        control, signal = signal
    QObject.connect(control, SIGNAL(signal), f)


def connectControl(control, master, value, f, signal, cfront, cback = None, cfunc = None, fvcb = None):
    cback = cback or value and ValueCallback(master, value, fvcb)
    if cback:
        if signal:
            connectControlSignal(control, signal, cback)
        cback.opposite = cfront
        if value and cfront and hasattr(master, "controlledAttributes"):
            master.controlledAttributes[value] = cfront

    cfunc = cfunc or f and FunctionCallback(master, f)
    if cfunc:
        if signal:
            connectControlSignal(control, signal, cfunc)
        cfront.opposite = cback, cfunc
    else:
        cfront.opposite = (cback,)

    return cfront, cback, cfunc


class ControlledCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        self.disabled = 0
        if type(widget) is dict:
            return     # we can't assign attributes to dict
        if not hasattr(widget, "callbackDeposit"):
            widget.callbackDeposit = []
        widget.callbackDeposit.append(self)


    def acyclic_setattr(self, value):
        if self.disabled:
            return

        if self.f:
            if self.f in [int, float] and (
                    not value or type(value) is str and value in "+-"):
                value = self.f(0)
            else:
                value = self.f(value)

        opposite = getattr(self, "opposite", None)
        if opposite:
            try:
                opposite.disabled += 1
                if type(self.widget) is dict:
                    self.widget[self.attribute] = value
                else:
                    setattr(self.widget, self.attribute, value)
            finally:
                opposite.disabled -= 1
        else:
            if type(self.widget) is dict:
                self.widget[self.attribute] = value
            else:
                setattr(self.widget, self.attribute, value)


class ValueCallback(ControlledCallback):
    def __call__(self, value):
        if value is not None:
            try:
                self.acyclic_setattr(value)
            except:
                print("OWGUI.ValueCallback: %s" % value)
                import traceback, sys
                traceback.print_exception(*sys.exc_info())


class ValueCallbackCombo(ValueCallback):
    def __init__(self, widget, attribute, f = None, control2attributeDict = {}):
        ValueCallback.__init__(self, widget, attribute, f)
        self.control2attributeDict = control2attributeDict

    def __call__(self, value):
        value = str(value)
        return ValueCallback.__call__(self, self.control2attributeDict.get(value, value))



class ValueCallbackLineEdit(ControlledCallback):
    def __init__(self, control, widget, attribute, f = None):
        ControlledCallback.__init__(self, widget, attribute, f)
        self.control = control

    def __call__(self, value):
        if value is not None:
            try:
                pos = self.control.cursorPosition()
                self.acyclic_setattr(value)
                self.control.setCursorPosition(pos)
            except:
                print("invalid value ", value, type(value))


class SetLabelCallback:
    def __init__(self, widget, label, format = "%5.2f", f = None):
        self.widget = widget
        self.label = label
        self.format = format
        self.f = f
        if hasattr(widget, "callbackDeposit"):
            widget.callbackDeposit.append(self)
        self.disabled = 0

    def __call__(self, value):
        if not self.disabled and value is not None:
            if self.f:
                value = self.f(value)
            self.label.setText(self.format % value)


class FunctionCallback:
    def __init__(self, master, f, widget=None, id=None, getwidget=None):
        self.master = master
        self.widget = widget
        self.f = f
        self.id = id
        self.getwidget = getwidget
        if hasattr(master, "callbackDeposit"):
            master.callbackDeposit.append(self)
        self.disabled = 0

    def __call__(self, *value):
        if not self.disabled and value!=None:
            kwds = {}
            if self.id != None:
                kwds['id'] = self.id
            if self.getwidget:
                kwds['widget'] = self.widget
            if isinstance(self.f, list):
                for f in self.f:
                    f(**kwds)
            else:
                self.f(**kwds)


class CallBackListBox:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = 0

    def __call__(self, *args): # triggered by selectionChange()
        if not self.disabled and self.control.ogValue != None:
            clist = getdeepattr(self.widget, self.control.ogValue)
             # skip the overloaded method to avoid a cycle
            list.__delitem__(clist, slice(0, len(clist)))
            control = self.control
            for i in range(control.count()):
                if control.item(i).isSelected():
                    list.append(clist, i)
            self.widget.__setattr__(self.control.ogValue, clist)


class CallBackRadioButton:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = False

    def __call__(self, *args): # triggered by toggled()
        if not self.disabled and self.control.ogValue != None:
            arr = [butt.isChecked() for butt in self.control.buttons]
            self.widget.__setattr__(self.control.ogValue, arr.index(1))


##############################################################################
# call fronts (through this a change of the attribute value changes the related control)


class ControlledCallFront:
    def __init__(self, control):
        self.control = control
        self.disabled = 0

    def __call__(self, *args):
        if not self.disabled:
            opposite = getattr(self, "opposite", None)
            if opposite:
                try:
                    for op in opposite:
                        op.disabled += 1
                    self.action(*args)
                finally:
                    for op in opposite:
                        op.disabled -= 1
            else:
                self.action(*args)


class CallFrontSpin(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setValue(value)


class CallFrontDoubleSpin(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setValue(value)


class CallFrontCheckBox(ControlledCallFront):
    def action(self, value):
        if value != None:
            values = [Qt.Unchecked, Qt.Checked, Qt.PartiallyChecked]
            self.control.setCheckState(values[value])

class CallFrontButton(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setChecked(bool(value))

class CallFrontComboBox(ControlledCallFront):
    def __init__(self, control, valType = None, control2attributeDict = {}):
        ControlledCallFront.__init__(self, control)
        self.valType = valType
        self.attribute2controlDict = dict([(y, x) for x, y in list(control2attributeDict.items())])

    def action(self, value):
        if value is not None:
            value = self.attribute2controlDict.get(value, value)
            if self.valType:
                for i in range(self.control.count()):
                    if self.valType(str(self.control.itemText(i))) == value:
                        self.control.setCurrentIndex(i)
                        return
                values = ""
                for i in range(self.control.count()):
                    values += str(self.control.itemText(i)) + (i < self.control.count()-1 and ", " or ".")
                print("unable to set %s to value '%s'. Possible values are %s" % (self.control, value, values))
                #import traceback
                #traceback.print_stack()
            else:
                if value < self.control.count():
                    self.control.setCurrentIndex(value)


class CallFrontHSlider(ControlledCallFront):
    def action(self, value):
        if value is not None:
            self.control.setValue(value)


class CallFrontLogSlider(ControlledCallFront):
    def action(self, value):
        if value is not None:
            if value < 1e-30:
                print("unable to set ", self.control, "to value ", value, " (value too small)")
            else:
                self.control.setValue(math.log10(value))


class CallFrontLineEdit(ControlledCallFront):
    def action(self, value):
        self.control.setText(str(value))


class CallFrontRadioButtons(ControlledCallFront):
    def action(self, value):
        if value < 0 or value >= len(self.control.buttons):
            value = 0
        self.control.buttons[value].setChecked(1)


class CallFrontListBox(ControlledCallFront):
    def action(self, value):
        if value is not None:
            if not isinstance(value, ControlledList):
                setattr(self.control.ogMaster, self.control.ogValue, ControlledList(value, self.control))
            for i in range(self.control.count()):
                shouldBe = i in value
                if shouldBe != self.control.item(i).isSelected():
                    self.control.item(i).setSelected(shouldBe)


class CallFrontListBoxLabels(ControlledCallFront):
    if attributeIconDict is None:
            constructAttributeIcons()
    unknownType = None

    def action(self, value):
#        icons = getAttributeIcons()
        self.control.clear()
        if value:
            for i in value:
                if type(i) is tuple:
                    if isinstance(i[1], int):
                        self.control.addItem(QListWidgetItem(attributeIconDict.get(i[1], self.unknownType), i[0]))
                    else:
                        self.control.addItem( QListWidgetItem(i[0],i[1]) )
                else:
                    self.control.addItem(i)


class CallFrontLabel:
    def __init__(self, control, label, master):
        self.control = control
        self.label = label
        self.master = master

    def __call__(self, *args):
        self.control.setText(self.label % self.master.__dict__)

##############################################################################
## Disabler is a call-back class for check box that can disable/enable other
## widgets according to state (checked/unchecked, enabled/disable) of the
## given check box
##
## Tricky: if self.propagateState is True (default), then if check box is
## disabled, the related widgets will be disabled (even if the checkbox is
## checked). If self.propagateState is False, the related widgets will be
## disabled/enabled if check box is checked/clear, disregarding whether the
## check box itself is enabled or not. (If you don't understand, see the code :-)
DISABLER = 1
HIDER = 2

class Disabler:
    def __init__(self, widget, master, valueName, propagateState = 1, type = DISABLER):
        self.widget = widget
        self.master = master
        self.valueName = valueName
        self.propagateState = propagateState
        self.type = type

    def __call__(self, *value):
        currState = self.widget.isEnabled()

        if currState or not self.propagateState:
            if len(value):
                disabled = not value[0]
            else:
                disabled = not getdeepattr(self.master, self.valueName)
        else:
            disabled = 1

        for w in self.widget.disables:
            if type(w) is tuple:
                if isinstance(w[0], int):
                    i = 1
                    if w[0] == -1:
                        disabled = not disabled
                else:
                    i = 0
                if self.type == DISABLER:
                    w[i].setDisabled(disabled)
                elif self.type == HIDER:
                    if disabled: w[i].hide()
                    else:        w[i].show()

                if hasattr(w[i], "makeConsistent"):
                    w[i].makeConsistent()
            else:
                if self.type == DISABLER:
                    w.setDisabled(disabled)
                elif self.type == HIDER:
                    if disabled: w.hide()
                    else:        w.show()

##############################################################################
# some table related widgets

class tableItem(QTableWidgetItem):
    def __init__(self, table, x, y, text, editType = None, backColor=None, icon=None, type = QTableWidgetItem.Type):
        QTableWidgetItem.__init__(self, type)
        if icon:
            self.setIcon(QIcon(icon))
        if editType != None:
            self.setFlags(editType)
        else:
            self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
        if backColor != None:
            self.setBackground(QBrush(backColor))
        self.setData(Qt.DisplayRole, QVariant(text))        # we add it this way so that text can also be int and sorting will be done properly (as integers and not as text)

        table.setItem(x, y, self)


TableValueRole = next(OrangeUserRole) # Role to retrieve orange.Value
TableClassValueRole = next(OrangeUserRole) # Role to retrieve the class value for the row's example
TableDistribution = next(OrangeUserRole) # Role to retrieve the distribution of the column's attribute
TableVariable = next(OrangeUserRole) # Role to retrieve the column's variable

BarRatioRole = next(OrangeUserRole) # Ratio for drawing distribution bars
BarBrushRole = next(OrangeUserRole) # Brush for distribution bar

SortOrderRole = next(OrangeUserRole) # Used for sorting


class TableBarItem(QItemDelegate):
    BarRole = next(OrangeUserRole)
    ColorRole = next(OrangeUserRole)
    def __init__(self, widget, table = None, color = QColor(255, 170, 127), color_schema=None):
        """
        :param widget: OWWidget instance
        :type widget: :class:`OWWidget.OWWidget
        :param table: Table
        :type table: :class:`Orange.data.Table`
        :param color: Color of the distribution bar.
        :type color: :class:`PyQt4.QtCore.QColor`
        :param color_schema: If not None it must be an instance of
            :class:`OWColorPalette.ColorPaletteGenerator` (note: this
            parameter, if set, overrides the ``color``)
        :type color_schema: :class:`OWColorPalette.ColorPaletteGenerator`

        """
        QItemDelegate.__init__(self, widget)
        self.color = color
        self.color_schema = color_schema
        self.widget = widget
        self.table = table

    def paint(self, painter, option, index):
        painter.save()
        self.drawBackground(painter, option, index)
        if self.table is None:
            table = getattr(index.model(), "examples", None)
        else:
            table = self.table
        ratio = index.data(TableBarItem.BarRole)
        if isinstance(ratio, float):
            if math.isnan(ratio):
                ratio = None
        elif table is not None and getattr(self.widget, "show_bars", False):
            value = index.data(Qt.DisplayRole)
            if isinstance(value, float):
                col = index.column()
                if col < len(table.normalizers):
                    max, span = table.normalizers[col]
                    ratio = (max - value) / span

        color = self.color
        if (self.color_schema is not None and table is not None and
            isinstance(table.domain.class_var, Orange.data.DiscreteVariable)):
            class_ = index.data(TableClassValueRole)
            if not math.isnan(class_):
                color = self.color_schema[int(class_)]
        else:
            color = self.color

        if ratio is not None:
            painter.save()
            painter.setPen(QPen(QBrush(color), 5, Qt.SolidLine, Qt.RoundCap))
            rect = option.rect.adjusted(3, 0, -3, -5)
            x, y = rect.x(), rect.y() + rect.height()
            painter.drawLine(x, y, x + rect.width() * ratio, y)
            painter.restore()
            text_rect = option.rect.adjusted(0, 0, 0, -3)
        else:
            text_rect = option.rect
        text = index.data(Qt.DisplayRole)
        self.drawDisplay(painter, option, text_rect, text)
        painter.restore()

class BarItemDelegate(QStyledItemDelegate):
    def __init__(self, parent, brush=QBrush(QColor(255, 170, 127)), scale=(0.0, 1.0)):
        QStyledItemDelegate.__init__(self, parent)
        self.brush = brush
        self.scale = scale

    def paint(self, painter, option, index):
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter)
        rect = option.rect
        val = index.data(Qt.DisplayRole)
        if isinstance(val, float):
            min, max = self.scale
            val = (val - min) / (max - min)
            painter.save()
            if option.state & QStyle.State_Selected:
                painter.setOpacity(0.75)
            painter.setBrush(self.brush)
            painter.drawRect(
                rect.adjusted(1, 1, - rect.width() * (1.0 - val) - 2, -2))
            painter.restore()

class IndicatorItemDelegate(QStyledItemDelegate):
    IndicatorRole = next(OrangeUserRole)
    def __init__(self, parent, role=IndicatorRole, indicatorSize=2):
        QStyledItemDelegate.__init__(self, parent)
        self.role = role
        self.indicatorSize = indicatorSize

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)
        rect = option.rect
        indicator, valid = index.data(self.role).toString(), True
        indicator = False if indicator == "false" else indicator
        if valid and indicator:
            painter.save()
            painter.setRenderHints(QPainter.Antialiasing)
            painter.setBrush(QBrush(Qt.black))
            painter.drawEllipse(rect.center(), self.indicatorSize, self.indicatorSize) #rect.adjusted(rect.width() / 2 - 5, rect.height() - 5, -rect.width() /2 + 5, -rect.height()/2 + 5))
            painter.restore()

class LinkStyledItemDelegate(QStyledItemDelegate):
    LinkRole = next(OrangeUserRole)
    def __init__(self, parent):
        QStyledItemDelegate.__init__(self, parent)
        self.mousePressState = QModelIndex(), QPoint()
        self.connect(parent, SIGNAL("entered(QModelIndex)"), self.onEntered)

    def sizeHint(self, option, index):
        size = QStyledItemDelegate.sizeHint(self, option, index)
        return QSize(size.width(), max(size.height(), 20))

    def linkRect(self, option, index):
        style = self.parent().style()
        text = self.displayText(index.data(Qt.DisplayRole), QLocale.system())
        self.initStyleOption(option, index)
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, option)
        if not textRect.isValid():
            textRect = option.rect
        margin = style.pixelMetric(QStyle.PM_FocusFrameHMargin, option) + 1
        textRect = textRect.adjusted(margin, 0, -margin, 0)
        font = index.data(Qt.FontRole)
        if font.isValid():
            font = QFont(font)
        else:
            font = option.font
        metrics = QFontMetrics(font)
        elideText = metrics.elidedText(text, option.textElideMode, textRect.width())
        try:
            str(elideText)  ## on Windows with PyQt 4.4 sometimes this fails
        except Exception as ex:
            elideText = text
        return metrics.boundingRect(textRect, option.displayAlignment, elideText)

    def editorEvent(self, event, model, option, index):
        if event.type()==QEvent.MouseButtonPress and self.linkRect(option, index).contains(event.pos()):
            self.mousePressState = QPersistentModelIndex(index), QPoint(event.pos())

        elif event.type()== QEvent.MouseButtonRelease:
            link = index.data(LinkRole)
            pressedIndex, pressPos = self.mousePressState
            if pressedIndex == index and (pressPos - event.pos()).manhattanLength() < 5 and link.isValid():
                import webbrowser
                webbrowser.open(link.toString())
            self.mousePressState = QModelIndex(), event.pos()

        elif event.type()==QEvent.MouseMove:
            link = index.data(LinkRole)
            if link.isValid() and self.linkRect(option, index).contains(event.pos()):
                self.parent().viewport().setCursor(Qt.PointingHandCursor)
            else:
                self.parent().viewport().setCursor(Qt.ArrowCursor)

        return QStyledItemDelegate.editorEvent(self, event, model, option, index)

    def onEntered(self, index):
        link = index.data(LinkRole)
        if not link.isValid():
            self.parent().viewport().setCursor(Qt.ArrowCursor)

    def paint(self, painter, option, index):
        if index.data(LinkRole).isValid():
            style = qApp.style()
            style.drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
            style.drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter)
            text = self.displayText(index.data(Qt.DisplayRole), QLocale.system())
            textRect = style.subElementRect(QStyle.SE_ItemViewItemText, option)
            if not textRect.isValid():
                textRect = option.rect
            margin = style.pixelMetric(QStyle.PM_FocusFrameHMargin, option) + 1
            textRect = textRect.adjusted(margin, 0, -margin, 0)
            elideText = QFontMetrics(option.font).elidedText(text, option.textElideMode, textRect.width())
            painter.save()
            font = index.data(Qt.FontRole)
            if font.isValid():
                painter.setFont(QFont(font))
            else:
                painter.setFont(option.font)
            painter.setPen(QPen(Qt.blue))
            painter.drawText(textRect, option.displayAlignment, elideText)
            painter.restore()
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

LinkRole = LinkStyledItemDelegate.LinkRole

def _toPyObject(variant):
    val = variant.toPyObject()
    if isinstance(val, type(NotImplemented)):
        # PyQt 4.4 converts python int, floats ... to C types and
        # cannot convert them back again and returns an exception instance.
        qtype = variant.type()
        if qtype == QVariant.Double:
            val, ok = variant.toDouble()
        elif qtype == QVariant.Int:
            val, ok = variant.toInt()
        elif qtype == QVariant.LongLong:
            val, ok = variant.toLongLong()
        elif qtype == QVariant.String:
            val = variant.toString()
    return val

class ColoredBarItemDelegate(QStyledItemDelegate):
    """ Item delegate that can also draws a distribution bar
    """
    def __init__(self, parent=None, decimals=3, color=Qt.red):
        QStyledItemDelegate.__init__(self, parent)
        self.decimals = decimals
        self.float_fmt = "%%.%if" % decimals
        self.color = QColor(color)

    def displayText(self, value, locale):
        obj = _toPyObject(value)
        if isinstance(obj, float):
            return self.float_fmt % obj
        elif isinstance(obj, str):
            return obj
        elif obj is None:
            return "NA"
        else:
            return obj.__str__()

    def sizeHint(self, option, index):
        font = self.get_font(option, index)
        metrics = QFontMetrics(font)
        height = metrics.lineSpacing() + 8 # 4 pixel margin
        width = metrics.width(self.displayText(index.data(Qt.DisplayRole), QLocale())) + 8
        return QSize(width, height)

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        text = self.displayText(index.data(Qt.DisplayRole), QLocale())

        ratio, have_ratio = self.get_bar_ratio(option, index)

        rect = option.rect
        if have_ratio:
            # The text is raised 3 pixels above the bar.
            text_rect = rect.adjusted(4, 1, -4, -4) # TODO: Style dependent margins?
        else:
            text_rect = rect.adjusted(4, 4, -4, -4)

        painter.save()
        font = self.get_font(option, index)
        painter.setFont(font)

        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter)

        # TODO: Check ForegroundRole.
        if option.state & QStyle.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
        painter.setPen(QPen(color))

        align = self.get_text_align(option, index)

        metrics = QFontMetrics(font)
        elide_text = metrics.elidedText(text, option.textElideMode, text_rect.width())
        painter.drawText(text_rect, align, elide_text)

        painter.setRenderHint(QPainter.Antialiasing, True)
        if have_ratio:
            brush = self.get_bar_brush(option, index)

            painter.setBrush(brush)
            painter.setPen(QPen(brush, 1))
            bar_rect = QRect(text_rect)
            bar_rect.setTop(bar_rect.bottom() - 1)
            bar_rect.setBottom(bar_rect.bottom() + 1)
            w = text_rect.width()
            bar_rect.setWidth(max(0, min(w * ratio, w)))
            painter.drawRoundedRect(bar_rect, 2, 2)
        painter.restore()

    def get_font(self, option, index):
        font = index.data(Qt.FontRole)
        if font.isValid():
            font = font.toPyObject()
        else:
            font = option.font
        return font

    def get_text_align(self, option, index):
        align = index.data(Qt.TextAlignmentRole)
        if align.isValid():
            align = align.toInt()
        else:
            align = Qt.AlignLeft | Qt.AlignVCenter
        return align

    def get_bar_ratio(self, option, index):
        bar_ratio = index.data(BarRatioRole)
        ratio, have_ratio = bar_ratio.toDouble()
        return ratio, have_ratio

    def get_bar_brush(self, option, index):
        bar_brush = index.data(BarBrushRole)
        if bar_brush.isValid():
            bar_brush = bar_brush.toPyObject()
            if not isinstance(bar_brush, (QColor, QBrush)):
                bar_brush = None
        else:
            bar_brush = None
        if bar_brush is None:
            bar_brush = self.color
        return QBrush(bar_brush)

##############################################################################
# progress bar management

class ProgressBar:
    def __init__(self, widget, iterations):
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()

    def advance(self, count=1):
        self.count += count
        self.widget.progressBarSet(int(self.count*100/self.iter))

    def finish(self):
        self.widget.progressBarFinished()



##############################################################################

def tabWidget(widget):
    w = QTabWidget(widget)
    if widget.layout() is not None:
        widget.layout().addWidget(w)
    return w

def createTabPage(tabWidget, name, widgetToAdd = None, canScroll = False):
    if widgetToAdd == None:
        widgetToAdd = widgetBox(tabWidget, addToLayout = 0, margin = 4)
    if canScroll:
        scrollArea = QScrollArea()
        tabWidget.addTab(scrollArea, name)
        scrollArea.setWidget(widgetToAdd)
        scrollArea.setWidgetResizable(1)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    else:
        tabWidget.addTab(widgetToAdd, name)
    return widgetToAdd

def table(widget, rows = 0, columns = 0, selectionMode = -1, addToLayout = 1):
    w = QTableWidget(rows, columns, widget)
    if widget and addToLayout and widget.layout() is not None:
        widget.layout().addWidget(w)
    if selectionMode != -1:
        w.setSelectionMode(selectionMode)
    w.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
    w.horizontalHeader().setMovable(True)
    return w

class VisibleHeaderSectionContextEventFilter(QObject):
    def __init__(self, parent, itemView=None):
        QObject.__init__(self, parent)
        self.itemView = itemView
    def eventFilter(self, view, event):
        if type(event) == QContextMenuEvent:
            model = view.model()
            headers = [(view.isSectionHidden(i), model.headerData(i, view.orientation(), Qt.DisplayRole)) for i in range(view.count())]
            menu = QMenu("Visible headers", view)

            for i, (checked, name) in enumerate(headers):
                action = QAction(name.toString(), menu)
                action.setCheckable(True)
                action.setChecked(not checked)
                menu.addAction(action)

                def toogleHidden(bool, section=i):
                    view.setSectionHidden(section, not bool)
                    if bool:
                        if self.itemView:
                            self.itemView.resizeColumnToContents(section)
                        else:
                            view.resizeSection(section, max(view.sectionSizeHint(section), 10))

                self.connect(action, SIGNAL("toggled(bool)"), toogleHidden)
#                self.connect(action, SIGNAL("toggled(bool)"), lambda bool, section=i: view.setSectionHidden(section, not bool))
            menu.exec_(event.globalPos())
            return True

        return False

def checkButtonOffsetHint(button, style=None):
    option = QStyleOptionButton()
    option.initFrom(button)
    if style is None:
        style = button.style()
    if isinstance(button, QCheckBox):
        pm_spacing = QStyle.PM_CheckBoxLabelSpacing
        pm_indicator_width = QStyle.PM_IndicatorWidth
    else:
        pm_spacing = QStyle.PM_RadioButtonLabelSpacing
        pm_indicator_width = QStyle.PM_ExclusiveIndicatorWidth
    space = style.pixelMetric(pm_spacing, option, button)
    width = style.pixelMetric(pm_indicator_width, option, button)
    style_correction = {"macintosh (aqua)": -2, "macintosh(aqua)": -2, "plastique": 1, "cde": 1, "motif": 1} #TODO: add other styles (Maybe load corrections from .cfg file?)
    return space + width + style_correction.get(str(qApp.style().objectName()).lower(), 0)


def toolButtonSizeHint(button=None, style=None):
    if button is None and style is None:
        style = qApp.style()
    elif style is None:
        style = button.style()

    button_size = style.pixelMetric(QStyle.PM_SmallIconSize) + \
                  style.pixelMetric(QStyle.PM_ButtonMargin)
    return button_size

class FloatSlider(QSlider):
    def __init__(self, orientation, min_value, max_value, step, parent=None):
        QSlider.__init__(self, orientation, parent)
        self.setScale(min_value, max_value, step)
        QObject.connect(self, SIGNAL("valueChanged(int)"), self.sendValue)

    def update(self):
        self.setSingleStep(1)
        if self.min_value != self.max_value:
            self.setEnabled(True)
            self.setMinimum(int(self.min_value/self.step))
            self.setMaximum(int(self.max_value/self.step))
        else:
            self.setEnabled(False)

    def sendValue(self, slider_value):
        value = min(max(slider_value * self.step, self.min_value), self.max_value)
        self.emit(SIGNAL("valueChangedFloat(double)"), value)

    def setValue(self, value):
        QSlider.setValue(self, int(value/self.step))

    def setScale(self, minValue, maxValue, step=0):
        if minValue >= maxValue:
            ## It would be more logical to disable the slider in this case (self.setEnabled(False))
            ## However, we do nothing to keep consistency with Qwt
            return
        if step <= 0 or step > (maxValue-minValue):
            if type(maxValue) is int and type(minValue) is int:
                step = 1
            else:
                step = float(minValue-maxValue)/100.0
        self.min_value = float(minValue)
        self.max_value = float(maxValue)
        self.step = step
        self.update()

    def setRange(self, minValue, maxValue, step=1.0):
        # For compatibility with qwtSlider
        self.setScale(minValue, maxValue, step)

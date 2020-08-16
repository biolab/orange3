"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
"""Default color theme
"""

class ColorTheme:
    """Color theme.
    """
    def __init__(self, textFormatClass):
        """Constructor gets TextFormat class as parameter for avoid cross-import problems
        """
        self.format = {
            'dsNormal':         textFormatClass(),
            'dsKeyword':        textFormatClass(bold=True),
            'dsFunction':       textFormatClass(color='#644a9a'),
            'dsVariable':       textFormatClass(color='#0057ad'),
            'dsControlFlow':    textFormatClass(bold=True),
            'dsOperator':       textFormatClass(),
            'dsBuiltIn':        textFormatClass(color='#644a9a', bold=True),
            'dsExtension':      textFormatClass(color='#0094fe', bold=True),
            'dsPreprocessor':   textFormatClass(color='#006e28'),
            'dsAttribute':      textFormatClass(color='#0057ad'),

            'dsChar':           textFormatClass(color='#914c9c'),
            'dsSpecialChar':    textFormatClass(color='#3dade8'),
            'dsString':         textFormatClass(color='#be0303'),
            'dsVerbatimString': textFormatClass(color='#be0303'),
            'dsSpecialString':  textFormatClass(color='#fe5500'),
            'dsImport':         textFormatClass(color='#b969c3'),

            'dsDataType':       textFormatClass(color='#0057ad'),
            'dsDecVal':         textFormatClass(color='#af8000'),
            'dsBaseN':          textFormatClass(color='#af8000'),
            'dsFloat':          textFormatClass(color='#af8000'),

            'dsConstant':       textFormatClass(bold=True),

            'dsComment':        textFormatClass(color='#888786'),
            'dsDocumentation':  textFormatClass(color='#608880'),
            'dsAnnotation':     textFormatClass(color='#0094fe'),
            'dsCommentVar':     textFormatClass(color='#c960c9'),

            'dsRegionMarker':   textFormatClass(color='#0057ad', background='#e0e9f8'),
            'dsInformation':    textFormatClass(color='#af8000'),
            'dsWarning':        textFormatClass(color='#be0303'),
            'dsAlert':          textFormatClass(color='#bf0303', background='#f7e6e6', bold=True),
            'dsOthers':         textFormatClass(color='#006e28'),
            'dsError':          textFormatClass(color='#bf0303', underline=True),
        }

    def getFormat(self, styleName):
        """Returns TextFormat for particular style
        """
        return self.format[styleName]

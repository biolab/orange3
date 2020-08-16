"""
Adapted from a code editor component created
for Enki editor as replacement for QScintilla.
Copyright (C) 2020  Andrei Kopats

Originally licensed under the terms of GNU Lesser General Public License
as published by the Free Software Foundation, version 2.1 of the license.
This is compatible with Orange3's GPL-3.0 license.
"""
#!/usr/bin/env python3

import os.path
import json

import sys

_MY_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_MY_PATH, '..', '..', '..'))


from qutepart.syntax.loader import loadSyntax
from qutepart.syntax import SyntaxManager, Syntax


def _add_php(targetFileName, srcFileName):
    os.system("./generate-php.pl > xml/{} < xml/{}".format(targetFileName, srcFileName))


def main():
    os.chdir(_MY_PATH)
    _add_php('javascript-php.xml', 'javascript.xml')
    _add_php('css-php.xml', 'css.xml')
    _add_php('html-php.xml', 'html.xml')

    xmlFilesPath = os.path.join(_MY_PATH, 'xml')
    xmlFileNames = [fileName for fileName in os.listdir(xmlFilesPath) \
                        if fileName.endswith('.xml')]

    syntaxNameToXmlFileName = {}
    mimeTypeToXmlFileName = {}
    extensionToXmlFileName = {}
    firstLineToXmlFileName = {}

    for xmlFileName in xmlFileNames:
        xmlFilePath = os.path.join(xmlFilesPath, xmlFileName)
        syntax = Syntax(None)
        loadSyntax(syntax, xmlFilePath)
        if not syntax.name in syntaxNameToXmlFileName or \
           syntaxNameToXmlFileName[syntax.name][0] < syntax.priority:
            syntaxNameToXmlFileName[syntax.name] = (syntax.priority, xmlFileName)

        if syntax.mimetype:
            for mimetype in syntax.mimetype:
                if not mimetype in mimeTypeToXmlFileName or \
                   mimeTypeToXmlFileName[mimetype][0] < syntax.priority:
                    mimeTypeToXmlFileName[mimetype] = (syntax.priority, xmlFileName)

        if syntax.extensions:
            for extension in syntax.extensions:
                if extension not in extensionToXmlFileName or \
                   extensionToXmlFileName[extension][0] < syntax.priority:
                    extensionToXmlFileName[extension] = (syntax.priority, xmlFileName)

        if syntax.firstLineGlobs:
            for glob in syntax.firstLineGlobs:
                if not glob in firstLineToXmlFileName or \
                   firstLineToXmlFileName[glob][0] < syntax.priority:
                    firstLineToXmlFileName[glob] = (syntax.priority, xmlFileName)

    # remove priority, leave only xml file names
    for dictionary in (syntaxNameToXmlFileName,
                       mimeTypeToXmlFileName,
                       extensionToXmlFileName,
                       firstLineToXmlFileName):
        newDictionary = {}
        for key, item in dictionary.items():
            newDictionary[key] = item[1]
        dictionary.clear()
        dictionary.update(newDictionary)

    # Fix up php first line pattern. It contains <?php, but it is generated from html, and html doesn't contain it
    firstLineToXmlFileName['<?php*'] = 'html-php.xml'

    result = {
        'syntaxNameToXmlFileName' : syntaxNameToXmlFileName,
        'mimeTypeToXmlFileName' : mimeTypeToXmlFileName,
        'extensionToXmlFileName' : extensionToXmlFileName,
        'firstLineToXmlFileName' : firstLineToXmlFileName,
    }

    with open('syntax_db.json', 'w', encoding='utf-8') as syntaxDbFile:
        json.dump(result, syntaxDbFile, sort_keys=True, indent=4)

    print('Done. Do not forget to commit the changes')

if __name__ == '__main__':
    main()

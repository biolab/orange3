# This module mimics unittest's default main with --verbose, but reduces
# the output by not showing successful tests.
#
# stdout/stderr are redirected into the same stream (stdout). The name of the
# test and its description (if any) are printed only if the test fails or the
# tests causes some printout. For this, the module defines a stream with
# 'preambule' that is, a string that is printed before the next print.
# The preambule is set at the beginning at each test function.

import os
import sys
from io import IOBase
from unittest import TestResult, TextTestRunner
from unittest.main import main

__unittest = True

if os.getenv("CI"):
    # Remove the current path to prevent tox from importing sources instead of
    # installed files (with compiled binaries)
    sys.path.remove(os.path.split(__file__)[0])


class PreambuleStream(IOBase):
    def __init__(self):
        super().__init__()
        self.preambule = ""
        self.line_before_msg = False

    def set_preambule(self, preambule):
        self.preambule = preambule

    def writable(self):
        return True

    def writelines(self, lines):
        return self.write("".join(lines))

    def write(self, s):
        if self.preambule:
            _stdout.write("\n" + self.preambule + "\n")
            self.preambule = ""
        self.line_before_msg = True
        return _stdout.write(s)

    def write_msg(self, s):
        if self.line_before_msg:
            _stdout.write("\n")
        _stdout.write(self.preambule + " ... " + s)
        self.line_before_msg = False
        self.preambule = ""

    def flush(self):
        _stdout.flush()


class QuietTestResult(TestResult):
    separator1 = '=' * 70
    separator2 = '-' * 70

    def startTest(self, test):
        super().startTest(test)
        sys.stdout.set_preambule(self.getDescription(test))

    def stopTest(self, test):
        super().stopTest(test)
        sys.stdout.set_preambule("")

    @staticmethod
    def getDescription(test):
        doc_first_line = test.shortDescription()
        if doc_first_line:
            return '\n'.join((str(test), doc_first_line))
        else:
            return str(test)

    def addError(self, test, err):
        super().addError(test, err)
        sys.stdout.write_msg("ERROR\n")

    def addFailure(self, test, err):
        super().addError(test, err)
        sys.stdout.write_msg("FAIL\n")

    def printErrors(self):
        sys.stdout.set_preambule("")
        print()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            print(self.separator1)
            print("%s: %s" % (flavour,self.getDescription(test)))
            print(self.separator2)
            print("%s" % err)


_stdout = sys.stdout
sys.stderr = sys.stdout = PreambuleStream()

testRunner = TextTestRunner(
    resultclass=QuietTestResult,
    stream=sys.stdout,
    verbosity=2)

main(module=None, testRunner=testRunner)

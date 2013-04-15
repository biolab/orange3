#!usr/bin/env python

import os, sys

try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

from distutils.core import Extension
from distutils import log

import subprocess
import glob
import numpy

NAME                = 'orangeqt-qt'
DESCRIPTION         = 'orangeqt ploting library'
URL                 = "http://orange.biolab.si"
LICENSE             = 'GNU General Public License (GPL)'
AUTHOR              = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL        = "orange@fri.uni-lj.si"
VERSION             = '0.0.1a'



import PyQt4.QtCore

from PyQt4 import pyqtconfig 

cfg = pyqtconfig.Configuration()
pyqt_sip_dir = cfg.pyqt_sip_dir

import sipdistutils

extra_compile_args = []
extra_link_args = []
include_dirs = []
library_dirs = []

if sys.platform == "darwin":
    sip_plaftorm_tag = "WS_MACX"
elif sys.platform == "win32":
    sip_plaftorm_tag = "WS_WIN"
elif sys.platform.startswith("linux"):
    sip_plaftorm_tag = "WS_X11"
else:
    sip_plaftorm_tag = ""

class PyQt4Extension(Extension):
    pass

class build_pyqt_ext(sipdistutils.build_ext):
    description = "Build a orangeqt PyQt4 extension."
    
    user_options = sipdistutils.build_ext.user_options + \
        [("required", None,  
          "orangeqt is required (failure to build will raise an error)")]
        
    boolean_options = sipdistutils.build_ext.boolean_options + \
        ["required"]
    
    def initialize_options(self):
        sipdistutils.build_ext.initialize_options(self)
        self.required = False
        
    def finalize_options(self):
        sipdistutils.build_ext.finalize_options(self)
        self.sip_opts = self.sip_opts + ["-k", "-j", "1", "-t", 
                        sip_plaftorm_tag, "-t",
                        "Qt_" + PyQt4.QtCore.QT_VERSION_STR.replace('.', '_')]
        if self.required is not None:
            self.required = True

    def build_extension(self, ext):
        if not isinstance(ext, PyQt4Extension):
            return
        cppsources = [source for source in ext.sources if source.endswith(".cpp")]
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        for source in cppsources:
            header = source.replace(".cpp", ".h")
            if os.path.exists(header):
                moc_file = os.path.basename(header).replace(".h", ".moc")
                call_arg = ["moc", "-o", os.path.join(self.build_temp, moc_file), header]
                log.info("Calling: " + " ".join(call_arg))
                try:
                    subprocess.call(call_arg)
                except OSError:
                    raise OSError("Could not locate 'moc' executable.")
        ext.extra_compile_args = ext.extra_compile_args + ["-I" + self.build_temp]
        sipdistutils.build_ext.build_extension(self, ext)
    
    def run(self):
        try:
            sipdistutils.build_ext.run(self)
        except Exception, ex:
            if self.required:
                raise
            else:
                log.info("Could not build orangeqt extension (%r)\nSkipping." % ex)

    # For sipdistutils to find PyQt4's .sip files
    def _sip_sipfiles_dir(self):
        return pyqt_sip_dir


def get_source_files(path, ext="cpp", exclude=[]):
    files = glob.glob(os.path.join(path, "*." + ext))
    files = [file for file in files if os.path.basename(file) not in exclude]
    return files


# Used Qt4 libs
qt_libs = ["QtCore", "QtGui", "QtOpenGL"]


if cfg.qt_framework:
    extra_compile_args = ["-F%s" % cfg.qt_lib_dir]
    extra_link_args = ["-F%s" % cfg.qt_lib_dir]
    for lib in qt_libs:
        include_dirs += [os.path.join(cfg.qt_lib_dir,
                                      lib + ".framework", "Headers")]
        extra_link_args += ["-framework", lib]
#    extra_link_args += ["-framework", "OpenGL"]
    qt_libs = []
else:
    include_dirs = [cfg.qt_inc_dir] + \
                   [os.path.join(cfg.qt_inc_dir, lib) for lib in qt_libs]
    library_dirs += [cfg.qt_lib_dir]

if sys.platform == "win32":
    # Qt libs on windows have a 4 added
    qt_libs = [lib + "4" for lib in qt_libs]

include_dirs += [numpy.get_include(), "./"]

orangeqt_ext = PyQt4Extension("orangeqt",
                              ["orangeqt.sip"] + get_source_files("", "cpp",
                               exclude=["canvas3d.cpp", "plot3d.cpp", "glextensions.cpp"]
                               ),
                              include_dirs=include_dirs,
                              extra_compile_args=extra_compile_args + \
                                                   ["-DORANGEQT_EXPORTS"],
                              extra_link_args=extra_link_args,
                              libraries = qt_libs,
                              library_dirs=library_dirs
                             )

ENTRY_POINTS = {
    'orange.addons': (
        'orangeqt = orangeqt',
    ),
}

def setup_package():
    setup(name = NAME,
          description = DESCRIPTION,
          version = VERSION,
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          url = URL,
          license = LICENSE,
          ext_modules = [orangeqt_ext],
          cmdclass={"build_ext": build_pyqt_ext},
          entry_points = ENTRY_POINTS,
          )

if __name__ == '__main__':
    setup_package()

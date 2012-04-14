#!usr/bin/env python
"""Orange: Machine learning and interactive data mining toolbox.

Orange is a scriptable environment for fast prototyping of new
algorithms and testing schemes. It is a collection of Python packages
that sit over the core library and implement some functionality for
which execution time is not crucial and which is easier done in Python
than in C++. This includes a variety of tasks such as attribute subset,
bagging and boosting, and alike.

Orange also includes a set of graphical widgets that use methods from
core library and Orange modules. Through visual programming, widgets
can be assembled together into an application by a visual programming
tool called Orange Canvas.
"""

DOCLINES = __doc__.splitlines()

import os, sys
try:
    from setuptools import setup
    from setuptools.command.install import install
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    from distutils.command.install import install
    have_setuptools = False

from distutils.core import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.command.install_lib import install_lib
from distutils.util import convert_path
from distutils.errors import DistutilsSetupError
from distutils.msvccompiler import MSVCCompiler
from distutils.unixccompiler import UnixCCompiler
import subprocess

CLASSIFIERS = """\
Development Status :: 4 - Beta
Programming Language :: Python
License :: OSI Approved :: GNU General Public License (GPL)
Operating System :: POSIX
Operating System :: Microsoft :: Windows
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Visualization
Intended Audience :: Education
Intended Audience :: Science/Research
"""

KEYWORDS = """\
data mining
machine learning
artificial intelligence
"""

NAME                = 'Orange'
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://orange.biolab.si"
DOWNLOAD_URL        = "https://bitbucket.org/biolab/orange/downloads"
LICENSE             = 'GNU General Public License (GPL)'
CLASSIFIERS         = filter(None, CLASSIFIERS.splitlines())
AUTHOR              = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL        = "orange@fri.uni-lj.si"
KEYWORDS            = filter(None, KEYWORDS.splitlines())
MAJOR               = 2
MINOR               = 5
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%da5' % (MAJOR, MINOR, MICRO)

if have_setuptools:
    setuptools_args = {"zip_safe": False,
                       "install_requires": ["numpy"],
                       "extras_require": {"GUI": ["PyQt4", "PyQwt"],
                                          "NETWORK": ["networkx"]}
                      }
else:
    setuptools_args = {}

import glob
from subprocess import check_call

import types

from distutils.dep_util import newer_group
from distutils.file_util import copy_file
from distutils import log

from distutils.sysconfig import get_python_inc, get_config_var

try:
    import numpy
    numpy_include_dir = numpy.get_include()
except ImportError:
    # When setup.py is first run to install orange, numpy can still be missing
    pass
    numpy_include_dir = None

python_include_dir = get_python_inc(plat_specific=1)

include_dirs = [python_include_dir, numpy_include_dir, "source/include"]

if sys.platform == "darwin":
    extra_compile_args = "-fPIC -fpermissive -fno-common -w -DDARWIN".split()
    extra_link_args = "-headerpad_max_install_names -undefined dynamic_lookup".split()
elif sys.platform == "win32":
    extra_compile_args = ["-EHsc"]
    extra_link_args = []
elif sys.platform.startswith("linux"):
    extra_compile_args = "-fPIC -fpermissive -w -DLINUX".split()
    extra_link_args = ["-Wl,-R$ORIGIN"]
else:
    extra_compile_args = []
    extra_link_args = []


# Get the command for building orangeqt extension from
# source/orangeqt/setup.py file.
# Fails  without PyQt4.
import imp
try:
    orangeqt_setup = imp.load_source("orangeqt_setup", "source/orangeqt/setup.py")
    build_pyqt_ext = orangeqt_setup.build_pyqt_ext
except ImportError:
    orangeqt_setup = None
    build_pyqt_ext = None

class LibStatic(Extension):
    pass

class PyXtractExtension(Extension):
    def __init__(self, *args, **kwargs):
        for name, default in [("extra_pyxtract_cmds", []), ("lib_type", "dynamic")]:
            setattr(self, name, kwargs.get(name, default))
            if name in kwargs:
                del kwargs[name]

        Extension.__init__(self, *args, **kwargs)

class PyXtractSharedExtension(PyXtractExtension):
    pass

class pyxtract_build_ext(build_ext):
    def run_pyxtract(self, ext, dir):
        original_dir = os.path.realpath(os.path.curdir)
        log.info("running pyxtract for %s" % ext.name)
        try:
            os.chdir(dir)
            ## we use the commands which are used for building under windows
            pyxtract_cmds = [cmd.split() for cmd in getattr(ext, "extra_pyxtract_cmds", [])]
            if os.path.exists("_pyxtract.bat"):
                pyxtract_cmds.extend([cmd.split()[1:] for cmd in open("_pyxtract.bat").read().strip().splitlines()])
            for cmd in pyxtract_cmds:
                log.info(" ".join([sys.executable] + cmd))
                check_call([sys.executable] + cmd)
            if pyxtract_cmds:
                ext.include_dirs.append(os.path.join(dir, "ppp"))
                ext.include_dirs.append(os.path.join(dir, "px"))

        finally:
            os.chdir(original_dir)

    def finalize_options(self):
        build_ext.finalize_options(self)
        # add the build_lib dir and build_temp (for 
        # liborange_include and liborange linking)
        if not self.inplace:
            # for linking with liborange.so (it is in Orange package)
            self.library_dirs.append(os.path.join(self.build_lib, "Orange"))
            # for linking with liborange_include.a
            self.library_dirs.append(self.build_temp)
        else:
            # for linking with liborange.so
            self.library_dirs.append("./Orange") 
            # for linking with liborange_include.a
            self.library_dirs.append(self.build_temp)

    def build_extension(self, ext):
        if isinstance(ext, LibStatic):
            # Build static library
            self.build_static(ext)
        elif isinstance(ext, PyXtractExtension):
            # Build pyextract extension
            self.build_pyxtract(ext)
        elif orangeqt_setup and isinstance(ext, orangeqt_setup.PyQt4Extension):
            # Skip the build (will be handled by build_pyqt_ext command)
            return
        else:
            build_ext.build_extension(self, ext)

        if isinstance(ext, PyXtractSharedExtension):
            # Fix extension modules so they can be linked
            # by other modules
            if isinstance(self.compiler, MSVCCompiler):
                # Copy ${TEMP}/orange/orange.lib to ${BUILD}/orange.lib
                ext_fullpath = self.get_ext_fullpath(ext.name)
                # Get the last component of the name
                ext_name = ext.name.rsplit(".", 1)[-1]
                libs = glob.glob(os.path.join(self.build_temp, 
                                              "*", "*", ext_name + ".lib"))
                if not libs:
                    log.info("Could not locate library %r in directory %r" \
                             %(ext_name, self.build_temp))
                else:
                    lib = libs[0]
                    copy_file(lib, os.path.splitext(ext_fullpath)[0] + ".lib")
            else:
                # Make lib{name}.so link to {name}.so
                ext_path = self.get_ext_fullpath(ext.name)
                ext_path, ext_filename = os.path.split(ext_path)
                realpath = os.path.realpath(os.curdir)
                try:
                    os.chdir(ext_path)
                    # Get the shared library name
                    _, name = ext.name.rsplit(".", 1)
                    lib_filename = self.compiler.library_filename(name, lib_type="shared")
                    # Create the link
                    copy_file(ext_filename, lib_filename, link="sym")
                except OSError, ex:
                    log.info("failed to create shared library for %s: %s" % (ext.name, str(ex)))
                finally:
                    os.chdir(realpath)

    def build_pyxtract(self, ext):
        ## mostly copied from build_extension
        sources = ext.sources
        if sources is None or type(sources) not in (types.ListType, types.TupleType):
            raise DistutilsSetupError, \
                  ("in 'ext_modules' option (extension '%s'), " +
                   "'sources' must be present and must be " +
                   "a list of source filenames") % ext.name
        sources = list(sources)

        ext_path = self.get_ext_fullpath(ext.name)

        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)

        # Run pyxtract in dir this adds ppp and px dirs to include_dirs
        dir = os.path.commonprefix([os.path.split(s)[0] for s in ext.sources])
        self.run_pyxtract(ext, dir)

        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(sources,
                                         output_dir=self.build_temp,
                                         macros=macros,
                                         include_dirs=ext.include_dirs,
                                         debug=self.debug,
                                         extra_postargs=extra_args,
                                         depends=ext.depends)

        # XXX -- this is a Vile HACK!
        #
        # The setup.py script for Python on Unix needs to be able to
        # get this list so it can perform all the clean up needed to
        # avoid keeping object files around when cleaning out a failed
        # build of an extension module.  Since Distutils does not
        # track dependencies, we have to get rid of intermediates to
        # ensure all the intermediates will be properly re-built.
        #
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.link_shared_object(
            objects, ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language)


    def build_static(self, ext):
        ## mostly copied from build_extension, changed
        sources = ext.sources
        if sources is None or type(sources) not in (types.ListType, types.TupleType):
            raise DistutilsSetupError, \
                  ("in 'ext_modules' option (extension '%s'), " +
                   "'sources' must be present and must be " +
                   "a list of source filenames") % ext.name
        sources = list(sources)

        # Static libs get build in the build_temp directory
        output_dir = self.build_temp
        if not os.path.exists(output_dir): #VSC fails if the dir does not exist
            os.makedirs(output_dir)

        lib_filename = self.compiler.library_filename(ext.name, lib_type='static', output_dir=output_dir)

        depends = sources + ext.depends
        if not (self.force or newer_group(depends, lib_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)

        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(sources,
                                         output_dir=self.build_temp,
                                         macros=macros,
                                         include_dirs=ext.include_dirs,
                                         debug=self.debug,
                                         extra_postargs=extra_args,
                                         depends=ext.depends)

        # XXX -- this is a Vile HACK!
        #
        # The setup.py script for Python on Unix needs to be able to
        # get this list so it can perform all the clean up needed to
        # avoid keeping object files around when cleaning out a failed
        # build of an extension module.  Since Distutils does not
        # track dependencies, we have to get rid of intermediates to
        # ensure all the intermediates will be properly re-built.
        #
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)
        
        #first remove old library (ar only appends the contents if archive already exists)
        try:
            os.remove(lib_filename)
        except OSError, ex:
            log.debug("failed to remove obsolete static library %s: %s" %(ext.name, str(ex)))

        # The static library is created in the temp dir, it is used during the compile step only
        # it should not be included in the final install
        self.compiler.create_static_lib(
            objects, ext.name, output_dir,
            debug=self.debug,
            target_lang=language)

    def get_libraries(self, ext):
        """ Change the 'orange' library name to 'orange_d' if
        building in debug mode. Using ``get_ext_filename`` to discover if
        _d postfix is required.
        
        """
        libraries = build_ext.get_libraries(self, ext)
        if "orange" in libraries and self.debug:
            filename = self.get_ext_filename("orange")
            basename = os.path.basename(filename)
            name, ext = os.path.splitext(basename)
            if name.endswith("_d"):
                index = libraries.index("orange")
                libraries[index] = "orange_d"

        return libraries

    if not hasattr(build_ext, "get_ext_fullpath"):
        #On mac OS X python 2.6.1 distutils does not have this method
        def get_ext_fullpath(self, ext_name):
            """Returns the path of the filename for a given extension.
            
            The file is located in `build_lib` or directly in the package
            (inplace option).
            """
            import string
            # makes sure the extension name is only using dots
            all_dots = string.maketrans('/' + os.sep, '..')
            ext_name = ext_name.translate(all_dots)
            fullname = self.get_ext_fullname(ext_name)
            modpath = fullname.split('.')
            filename = self.get_ext_filename(ext_name)
            filename = os.path.split(filename)[-1]
            if not self.inplace:
                # no further work needed
                # returning :
                #   build_dir/package/path/filename
                filename = os.path.join(*modpath[:-1] + [filename])
                return os.path.join(self.build_lib, filename)
            # the inplace option requires to find the package directory
            # using the build_py command for that
            package = '.'.join(modpath[0:-1])
            build_py = self.get_finalized_command('build_py')
            package_dir = os.path.abspath(build_py.get_package_dir(package))
            # returning
            #   package_dir/filename
            return os.path.join(package_dir, filename)


# Add build_pyqt_ext to build subcommands
class orange_build(build):
    def has_pyqt_extensions(self):
        # For now this is disabled unless specifically requested
        # using build_pyqt_ext command
        return False
#        return any([isinstance(ext, orangeqt_setup.PyQt4Extension) \
#                   for ext in self.distribution.ext_modules]
#                   )

    sub_commands = build.sub_commands
    if orangeqt_setup:
        sub_commands += [("build_pyqt_ext", has_pyqt_extensions)]


class orange_install_lib(install_lib):
    """ An command to install orange (preserves liborange.so -> orange.so symlink)
    """
    def run(self):
        install_lib.run(self)

    def copy_tree(self, infile, outfile, preserve_mode=1, preserve_times=1, preserve_symlinks=1, level=1):
        """ Run copy_tree with preserve_symlinks=1 as default
        """ 
        install_lib.copy_tree(self, infile, outfile, preserve_mode, preserve_times, preserve_symlinks, level)

    def install(self):
        """ Copy build_dir to install_dir
        """
        # A Hack to unlink liborange.so -> orange.so if it already exists,
        # because copy_tree fails to overwrite it
        # 
        liborange = os.path.join(self.install_dir, "Orange", "liborange.so")
        if os.path.exists(liborange) and os.path.islink(liborange):
            log.info("unlinking %s -> %s", liborange, os.path.join(self.install_dir, "orange.so"))
            os.unlink(liborange)

        return install_lib.install(self)


class orange_install(install):
    """ A command to install orange while also creating
    a .pth path to access the old orng* modules and orange, 
    orangeom etc. 
    
    """
    def run(self):
        install.run(self)
        # Create a .pth file with a path inside the Orange/orng directory
        # so the old modules are importable
        self.path_file, self.extra_dirs = ("Orange-orng-modules", "Orange/orng")
        self.extra_dirs = convert_path(self.extra_dirs)
        log.info("creating portal path for orange compatibility.")
        self.create_path_file()
        self.path_file, self.extra_dirs = None, None


def get_source_files(path, ext="cpp", exclude=[]):
    files = glob.glob(os.path.join(path, "*." + ext))
    files = [file for file in files if os.path.basename(file) not in exclude]
    return files


include_ext = LibStatic("orange_include",
                        get_source_files("source/include/"),
                        include_dirs=include_dirs)


if sys.platform == "win32": # ?? mingw/cygwin
    libraries = ["orange_include"]
else:
    libraries = ["stdc++", "orange_include"]


import ConfigParser
config = ConfigParser.RawConfigParser()

config.read(["setup-site.cfg",
             os.path.expanduser("~/.orange-site.cfg")]
            )

orange_sources = get_source_files("source/orange/")
orange_include_dirs = list(include_dirs)
orange_libraries = list(libraries)

if config.has_option("blas", "library"):
    # Link external blas library
    orange_libraries += [config.get("blas", "library")]
else:
    orange_sources += get_source_files("source/orange/blas/", "c")

if config.has_option("R", "library"):
    # Link external R library (for linpack)
    orange_libraries += [config.get("R", "library")]
else:
    orange_sources += get_source_files("source/orange/linpack/", "c")

if config.has_option("liblinear", "library"):
    # Link external LIBLINEAR library
    orange_libraries += [config.get("liblinear", "library")]
else:
    orange_sources += get_source_files("source/orange/liblinear/", "cpp")
    orange_include_dirs += ["source/orange/liblinear"]

if config.has_option("libsvm", "library"):
    # Link external LibSVM library
    orange_libraries += [config.get("libsvm", "library")]
else:
    orange_sources += get_source_files("source/orange/libsvm/", "cpp")


orange_ext = PyXtractSharedExtension("Orange.orange", orange_sources,
                                      include_dirs=orange_include_dirs,
                                      extra_compile_args = extra_compile_args + ["-DORANGE_EXPORTS"],
                                      extra_link_args = extra_link_args,
                                      libraries=orange_libraries,
                                      extra_pyxtract_cmds = ["../pyxtract/defvectors.py"],
                                      )

if sys.platform == "darwin":
    build_shared_cmd = get_config_var("BLDSHARED")
    # Dont link liborange.so with orangeom and orangene - MacOS X treats
    # loadable modules and shared libraries different
    if "-bundle" in build_shared_cmd.split():
        shared_libs = libraries
    else:
        shared_libs = libraries + ["orange"]
else:
    shared_libs = libraries + ["orange"]

orangeom_sources = get_source_files("source/orangeom/", exclude=["lib_vectors.cpp"])
orangeom_libraries = list(shared_libs)
orangeom_include_dirs = list(include_dirs)

if config.has_option("qhull", "library"):
    # Link external qhull library
    orangeom_libraries += [config.get("qhull", "library")]
else:
    orangeom_sources += get_source_files("source/orangeom/qhull/", "c")
    orangeom_include_dirs += ["source/orangeom"]


orangeom_ext = PyXtractExtension("Orange.orangeom", orangeom_sources,
                                  include_dirs=orangeom_include_dirs + ["source/orange/"],
                                  extra_compile_args = extra_compile_args + ["-DORANGEOM_EXPORTS"],
                                  extra_link_args = extra_link_args,
                                  libraries=orangeom_libraries,
                                  )

orangene_ext = PyXtractExtension("Orange.orangene",
    get_source_files("source/orangene/", exclude=["lib_vectors.cpp"]),
                                  include_dirs=include_dirs + ["source/orange/"], 
                                  extra_compile_args = extra_compile_args + ["-DORANGENE_EXPORTS"],
                                  extra_link_args = extra_link_args,
                                  libraries=shared_libs,
                                  )

corn_ext = Extension("Orange.corn", get_source_files("source/corn/"),
                     include_dirs=include_dirs + ["source/orange/"], 
                     extra_compile_args = extra_compile_args + ["-DCORN_EXPORTS"],
                     extra_link_args = extra_link_args,
                     libraries=libraries
                     )

statc_ext = Extension("Orange.statc", get_source_files("source/statc/"),
                      include_dirs=include_dirs + ["source/orange/"], 
                      extra_compile_args = extra_compile_args + ["-DSTATC_EXPORTS"],
                      extra_link_args = extra_link_args,
                      libraries=libraries
                      )


ext_modules = [include_ext, orange_ext, orangeom_ext,
               orangene_ext, corn_ext, statc_ext]

cmdclass = {"build": orange_build,
            "build_ext": pyxtract_build_ext,
            "install_lib": orange_install_lib,
            "install": orange_install}
                    
if orangeqt_setup:
    orangeqt_ext = orangeqt_setup.orangeqt_ext
    # Fix relative paths, name etc.
    orangeqt_ext.name = "Orange.orangeqt"
    orangeqt_ext.sources = ["source/orangeqt/orangeqt.sip"] + \
                           get_source_files("source/orangeqt", "cpp",
                                exclude=["canvas3d.cpp", "plot3d.cpp", 
                                         "glextensions.cpp"]
                                            )

    orangeqt_ext.include_dirs += ["source/orangeqt"]

    ext_modules += [orangeqt_ext]

    cmdclass["build_pyqt_ext"] = build_pyqt_ext


def get_packages():
    import fnmatch
    matches = []

    #Recursively find '__init__.py's
    for root, dirnames, filenames in os.walk('Orange'):
        # Add packages for Orange
        for filename in fnmatch.filter(filenames, '__init__.py'):
            matches.append(os.path.join(root, filename))
    return [os.path.dirname(pkg).replace(os.path.sep, '.') for pkg in matches]


def get_package_data():
    package_data = {
        "Orange":
            ["orangerc.cfg", "doc/style.css", "doc/widgets/*/*.*"] +\
             all_with_extension(path="doc/datasets", extensions=("tab", "csv", "basket")) +\
             all_with_extension(path="doc/networks", extensions=("net", "tab")) +\
             all_with_extension(path="testing/regression/tests_20", extensions=("net", "tab", "basket", "csv")),
        "Orange.OrangeCanvas": ["icons/*.png", "orngCanvas.pyw", "WidgetTabs.txt"],
        "Orange.OrangeWidgets": ["icons/*.png", "icons/backgrounds/*.png", "report/index.html"],
        "Orange.OrangeWidgets.Associate": ["icons/*.png"],
        "Orange.OrangeWidgets.Classify": ["icons/*.png"],
        "Orange.OrangeWidgets.Data": ["icons/*.png"],
        "Orange.OrangeWidgets.Evaluate": ["icons/*.png"],
        "Orange.OrangeWidgets.Prototypes": ["icons/*.png"],
        "Orange.OrangeWidgets.Regression": ["icons/*.png"],
        "Orange.OrangeWidgets.Unsupervised": ["icons/*.png"],
        "Orange.OrangeWidgets.Visualize": ["icons/*.png"],
        "Orange.OrangeWidgets.Visualize Qt": ["icons/*.png"],
        "Orange.OrangeWidgets.plot": ["*.gs", "*.vs"],
        "Orange.OrangeWidgets.plot.primitives": ["*.obj"],
    }

    return package_data

def all_with_extension(path, extensions):
    return [os.path.join(path, "*.%s"%extension) for extension in extensions]

def hg_revision():
    # Copied from numpy setup.py and modified to work with hg
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['hg', 'ide', '-i'])
        HG_REVISION = str(out.strip().decode('ascii'))
    except OSError:
        HG_REVISION = "Unknown"

    return HG_REVISION

def write_version_py(filename='Orange/version.py'):
    # Copied from numpy setup.py
    cnt = """
# THIS FILE IS GENERATED FROM ORANGE SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
hg_revision = '%(hg_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION = VERSION
    if os.path.exists('.hg'):
        HG_REVISION = hg_revision()
    elif os.path.exists('Orange/version.py'):
        # must be a source distribution, use existing version file
        version = imp.load_source("Orange.version", "Orange/version.py")
        HG_REVISION = version.hg_revision
    else:
        HG_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + HG_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'hg_revision' : HG_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def setup_package():
    write_version_py()
    setup(name = NAME,
          description = DESCRIPTION,
          version = VERSION,
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          url = URL,
          download_url = DOWNLOAD_URL,
          classifiers = CLASSIFIERS,
          long_description=LONG_DESCRIPTION,
          license = LICENSE,
          keywords = KEYWORDS,
          cmdclass=cmdclass,
          packages = get_packages(),
          package_data = get_package_data(),
          ext_modules = ext_modules,
          scripts = ["bin/orange-canvas"],
          **setuptools_args)

if __name__ == '__main__':
    setup_package()

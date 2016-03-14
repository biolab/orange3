#! /usr/bin/env python3
import os
import sys
import subprocess
from itertools import chain
from setuptools import find_packages

if sys.version_info < (3, 4):
    sys.exit('Orange requires Python >= 3.4')
try:
    from numpy.distutils.core import setup
except ImportError:
    sys.exit('setup requires numpy; install numpy first')


NAME = 'Orange'

VERSION = '3.3'
ISRELEASED = False

DESCRIPTION = 'Orange, a component-based data mining framework.'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Bioinformatics Laboratory, FRI UL'
AUTHOR_EMAIL = 'info@biolab.si'
URL = 'http://orange.biolab.si/'
LICENSE = 'GPLv3+'

KEYWORDS = (
    'data mining',
    'machine learning',
    'artificial intelligence',
)

CLASSIFIERS = (
    'Development Status :: 4 - Beta',
    'Environment :: X11 Applications :: Qt',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python',
    'Framework :: Orange',
    'License :: OSI Approved :: '
    'GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
)

requirements = ['requirements-core.txt', 'requirements-gui.txt']

INSTALL_REQUIRES = sorted(set(
    line.partition('#')[0].strip()
    for file in (os.path.join(os.path.dirname(__file__), file)
                 for file in requirements)
    for line in open(file)
) - {''})

SETUP_REQUIRES = ["setuptools-git"]

ENTRY_POINTS = {
    "orange.canvas.help": (
        "html-index = Orange.widgets:WIDGET_HELP_PATH",
    ),
    "console_scripts": (
        "orange = Orange.canvas.__main__:main",
    ),
    "gui_scripts": (
        "orange = Orange.canvas.__main__:main",
    ),
}


# Return the git revision as a string
def git_version():
    """Return the git revision as a string.

    Copied from numpy setup.py
    """
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
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


def write_version_py(filename='Orange/version.py'):
    # Copied from numpy setup.py
    cnt = """
# THIS FILE IS GENERATED FROM ORANGE SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
    short_version += ".dev"
"""
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('Orange/version.py'):
        # must be a source distribution, use existing version file
        import imp
        version = imp.load_source("Orange.version", "Orange/version.py")
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('Orange')

    config.get_version('Orange/version.py')  # sets config.version

    return config


def find_package_data(
    where='.',
    package='',
    exclude=('*.py', '*.pyc', '*$py.class', '*~', '.*', '*.bak'),
    exclude_directories=('.*', 'CVS', '_darcs', './build',
                         './dist', 'EGG-INFO', '*.egg-info'),
    only_in_packages=True,
    show_ignored=False):

    """
    Adapted from: http://svn.w4py.org/Paste/trunk/paste/util/finddata.py :
    (c) 2005 Ian Bicking and contributors; written for Paste (http://pythonpaste.org)
    Licensed under the MIT license: http://www.opensource.org/licenses/mit-license.php
    ------

    Return a dictionary suitable for use in ``package_data``
    in a distutils ``setup.py`` file.

    The dictionary looks like::

        {'package': [files]}

    Where ``files`` is a list of all the files in that package that
    don't match anything in ``exclude``.

    If ``only_in_packages`` is true, then top-level directories that
    are not packages won't be included (but directories under packages
    will).

    Directories matching any pattern in ``exclude_directories`` will
    be ignored.

    If ``show_ignored`` is true, then all the files that aren't
    included in package data are shown on stderr (for debugging
    purposes).

    Note patterns use wildcards, or can be exact paths (including
    leading ``./``), and all searching is case-insensitive.
    """
    from fnmatch import fnmatchcase
    from distutils.util import convert_path
    from os.path import join, isfile, isdir, sep
    out = {}
    stack = [(convert_path(where), '', package, only_in_packages)]
    while stack:
        where, prefix, package, only_in_packages = stack.pop(0)
        for name in os.listdir(where):
            fn = join(where, name)
            if isdir(fn):
                bad_name = False
                for pattern in exclude_directories:
                    if fnmatchcase(name, pattern) or fn.lower() == pattern.lower():
                        bad_name = True
                        if show_ignored:
                            print("find_package_data: Directory %s ignored"
                                  "by pattern %s" % (fn, pattern),
                                  file=sys.stderr)
                        break
                if bad_name:
                    continue
                if isfile(join(fn, '__init__.py')) and not prefix:
                    new_package = (package + '.' + name) if package else name
                    stack.append((fn, '', new_package, False))
                else:
                    stack.append((fn, prefix + name + sep, package, only_in_packages))
            elif package or not only_in_packages:
                # is a file
                bad_name = False
                for pattern in exclude:
                    if fnmatchcase(name, pattern) or fn.lower() == pattern.lower():
                        bad_name = True
                        if show_ignored:
                            print("find_package_data: File %s ignored"
                                  "by pattern %s" % (fn, pattern),
                                  file=sys.stderr)
                        break
                if bad_name:
                    continue
                out.setdefault(package, []).append(prefix + name)
    return out


def setup_package():
    write_version_py()
    setup(
        configuration=configuration,
        name=NAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        package_data=find_package_data('Orange', 'Orange'),
        install_requires=INSTALL_REQUIRES,
        setup_requires=SETUP_REQUIRES,
        entry_points=ENTRY_POINTS,
        zip_safe=False,
        include_package_data=True,
        test_suite='Orange.tests.test_suite',
    )

if __name__ == '__main__':
    setup_package()

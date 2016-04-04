#! /usr/bin/env python3

import os
import sys
import subprocess
from setuptools import find_packages, Command

if sys.version_info < (3, 4):
    sys.exit('Orange requires Python >= 3.4')
try:
    from numpy.distutils.core import setup
except ImportError:
    sys.exit('setup requires numpy; install numpy first')

NAME = 'Orange'

VERSION = '3.3.2'
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

ENTRY_POINTS = {
    "orange.canvas.help": (
        "html-index = Orange.widgets:WIDGET_HELP_PATH",
    ),
    "gui_scripts": (
        "orange-canvas = Orange.canvas.__main__:main",
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


PACKAGES = find_packages()

# Extra non .py, .{so,pyd} files that are installed within the package dir
# hierarchy
PACKAGE_DATA = {
    "Orange": ["datasets/*.{}".format(ext)
               for ext in ["tab", "csv", "basket", "info"]],
    "Orange.canvas": ["icons/*.png", "icons/*.svg"],
    "Orange.canvas.styles": ["*.qss", "orange/*.svg"],
    "Orange.canvas.application.tutorials": ["*.ows"],
    "Orange.canvas.report": ["icons/*.svg", "*.html"],
    "Orange.widgets": ["icons/*.png", "icons/*.svg"],
    "Orange.widgets.classify": ["icons/*.svg"],
    "Orange.widgets.data": ["icons/*.svg",
                            "icons/paintdata/*.png",
                            "icons/paintdata/*.svg"],
    "Orange.widgets.evaluate": ["icons/*.svg"],
    "Orange.widgets.visualize": ["icons/*.svg"],
    "Orange.widgets.regression": ["icons/*.svg"],
    "Orange.widgets.unsupervised": ["icons/*.svg"],
    "Orange.widgets.utils.plot": ["*.fs", "*.gs", "*.vs"],
    "Orange.widgets.utils.plot.primitives": ["*.obj"],
    "Orange.tests": ["xlsx_files/*.xlsx", "*.tab", "*.basket", "*.csv"]
}


class LintCommand(Command):
    """A setup.py lint subcommand developers can run locally."""
    description = "run code linter(s)"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Lint current branch compared to a reasonable master branch"""
        sys.exit(subprocess.call(r'''
        set -eu
        upstream="$(git remote -v |
                    awk '/[@\/]github.com[:\/]biolab\/orange3[\. ]/{ print $1; exit }')"
        git fetch -q $upstream master
        best_ancestor=$(git merge-base HEAD refs/remotes/$upstream/master)
        .travis/check_pylint_diff $best_ancestor
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


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
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        zip_safe=False,
        test_suite='Orange.tests.test_suite',
        cmdclass={
            'lint': LintCommand,
        },
    )

if __name__ == '__main__':
    setup_package()

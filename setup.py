#! /usr/bin/env python3

import os
import sys
import subprocess
from setuptools import find_packages, Command

if sys.version_info < (3, 4):
    sys.exit('Orange requires Python >= 3.4')

try:
    from numpy.distutils.core import setup
    have_numpy = True
except ImportError:
    from setuptools import setup
    have_numpy = False


try:
    # need sphinx and recommonmark for build_htmlhelp command
    from sphinx.setup_command import BuildDoc
    # pylint: disable=unused-import
    import recommonmark
    have_sphinx = True
except ImportError:
    have_sphinx = False

from distutils.command.build_ext import build_ext
from distutils.command import install_data, sdist, config, build


NAME = 'Orange3'

VERSION = '3.21.0'
ISRELEASED = True
# full version identifier including a git revision identifier for development
# build/releases (this is filled/updated in `write_version_py`)
FULLVERSION = VERSION

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


EXTRAS_REQUIRE = {
    ':python_version<="3.4"': ["typing"],
}

ENTRY_POINTS = {
    "orange.canvas.help": (
        "html-index = Orange.widgets:WIDGET_HELP_PATH",
    ),
    "gui_scripts": (
        "orange-canvas = Orange.canvas.__main__:main",
    ),
}


DATA_FILES = []

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
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
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
    global FULLVERSION
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
    return config


PACKAGES = find_packages()

# Extra non .py, .{so,pyd} files that are installed within the package dir
# hierarchy
PACKAGE_DATA = {
    "Orange": ["datasets/*.{}".format(ext)
               for ext in ["tab", "csv", "basket", "info", "dst", "metadata"]],
    "Orange.canvas": ["icons/*.png", "icons/*.svg"],
    "Orange.canvas.styles": ["*.qss", "orange/*.svg"],
    "Orange.canvas.application.workflows": ["*.ows"],
    "Orange.widgets": ["icons/*.png",
                       "icons/*.svg"],
    "Orange.widgets.report": ["icons/*.svg", "*.html"],
    "Orange.widgets.tests": ["datasets/*.tab",
                             "workflows/*.ows"],
    "Orange.widgets.data": ["icons/*.svg",
                            "icons/paintdata/*.png",
                            "icons/paintdata/*.svg"],
    "Orange.widgets.data.tests": ["origin1/*.tab",
                                  "origin2/*.tab"],
    "Orange.widgets.evaluate": ["icons/*.svg"],
    "Orange.widgets.model": ["icons/*.svg"],
    "Orange.widgets.visualize": ["icons/*.svg"],
    "Orange.widgets.unsupervised": ["icons/*.svg"],
    "Orange.widgets.utils": ["_webview/*.js"],
    "Orange.tests": ["xlsx_files/*.xlsx", "datasets/*.tab",
                     "datasets/*.basket", "datasets/*.csv",
                     "datasets/*.pkl", "datasets/*.pkl.gz"]
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

class CoverageCommand(Command):
    """A setup.py coverage subcommand developers can run locally."""
    description = "run code coverage"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Check coverage on current workdir"""
        sys.exit(subprocess.call(r'''
        coverage run --source=Orange -m unittest -v Orange.tests
        echo; echo
        coverage combine
        coverage report
        coverage html &&
            { echo; echo "See also: file://$(pwd)/htmlcov/index.html"; echo; }
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


class build_ext_error(build_ext):
    def initialize_options(self):
        raise SystemExit(
            "Cannot compile extensions. numpy is required to build Orange."
        )


# ${prefix} relative install path for html help files
DATAROOTDIR = "share/help/en/orange3/htmlhelp"


def findall(startdir, followlinks=False, ):
    files = (
        os.path.join(base, file)
        for base, dirs, files in os.walk(startdir, followlinks=followlinks)
        for file in files
    )
    return filter(os.path.isfile, files)


def find_htmlhelp_files(subdir):
    data_files = []
    thisdir = os.path.dirname(__file__)
    sourcedir = os.path.join(thisdir, subdir)
    files = filter(
        # filter out meta files
        lambda path: not path.endswith((".hhc", ".hhk", ".hhp", ".stp")),
        findall(sourcedir)
    )
    for file in files:
        relpath = os.path.relpath(file, start=subdir)
        relsubdir = os.path.dirname(relpath)
        # path.join("a", "") results in "a/"; distutils install_data does not
        # accept paths that end with "/" on windows.
        if relsubdir:
            targetdir = os.path.join(DATAROOTDIR, relsubdir)
        else:
            targetdir = DATAROOTDIR
        assert not targetdir.endswith("/")
        data_files.append((targetdir, [file]))
    return data_files


def add_with_option(option, help="", default=None, ):
    """
    A class decorator that adds a boolean --with(out)-option cmd line switch
    to a distutils.cmd.Command class

    Parameters
    ----------
    option : str
        Name of the option without the 'with-' part i.e. passing foo will
        create a `--with-foo` and `--without-foo` options
    help : str
        Help for `cmd --help`. This should document the positive option (i.e.
        --with-foo)
    default : Optional[bool]
        The default state.

    Returns
    -------
    command : Command

    Examples
    --------
    >>> @add_with_option("foo", "Build with foo enabled", default=False)
    >>> class foobuild(build):
    >>>    def run(self):
    >>>        if self.with_foo:
    >>>            ...

    """
    def decorator(cmdclass):
        # type: (Type[Command]) -> Type[Command]
        cmdclass.user_options = getattr(cmdclass, "user_options", []) + [
            ("with-" + option, None, help),
            ("without-" + option, None, ""),
        ]
        cmdclass.boolean_options = getattr(cmdclass, "boolean_options", []) + [
            ("with-" + option,),
        ]
        cmdclass.negative_opt = dict(
            getattr(cmdclass, "negative_opt", {}), **{
                "without-" + option: "with-" + option
            }
        )
        setattr(cmdclass, "with_" + option, default)
        return cmdclass
    return decorator


_HELP = "Build and include html help files in the distribution"


@add_with_option("htmlhelp", _HELP)
class config(config.config):
    # just record the with-htmlhelp option for sdist and build's default
    pass


@add_with_option("htmlhelp", _HELP)
class sdist(sdist.sdist):
    # build_htmlhelp to fill in distribution.data_files which are then included
    # in the source dist.
    sub_commands = sdist.sdist.sub_commands + [
        ("build_htmlhelp", lambda self: self.with_htmlhelp)
    ]

    def finalize_options(self):
        super().finalize_options()
        self.set_undefined_options(
            "config", ("with_htmlhelp", "with_htmlhelp")
        )


@add_with_option("htmlhelp", _HELP)
class build(build.build):
    sub_commands = build.build.sub_commands + [
        ("build_htmlhelp", lambda self: self.with_htmlhelp)
    ]

    def finalize_options(self):
        super().finalize_options()
        self.set_undefined_options(
            "config", ("with_htmlhelp", 'with_htmlhelp')
        )


# Does the sphinx source for widget help exist the sources are in the checkout
# but not in the source distribution (sdist). The sdist already contains
# build html files.
HAVE_SPHINX_SOURCE = os.path.isdir("doc/visual-programming/source")
# Doest the build htmlhelp documentation exist
HAVE_BUILD_HTML = os.path.exists("doc/visual-programming/build/htmlhelp/index.html")

if have_sphinx and HAVE_SPHINX_SOURCE:
    class build_htmlhelp(BuildDoc):
        def initialize_options(self):
            super().initialize_options()
            self.build_dir = "doc/visual-programming/build"
            self.source_dir = "doc/visual-programming/source"
            self.builder = "htmlhelp"
            self.version = VERSION

        def run(self):
            super().run()
            helpdir = os.path.join(self.build_dir, "htmlhelp")
            files = find_htmlhelp_files(helpdir)
            # add the build files to distribution
            self.distribution.data_files.extend(files)

else:
    # without sphinx we need the docs to be already build. i.e. from a
    # source dist build --with-htmlhelp
    class build_htmlhelp(Command):
        user_options = [('build-dir=', None, 'Build directory')]
        build_dir = None

        def initialize_options(self):
            self.build_dir = "doc/visual-programming/build"

        def finalize_options(self):
            pass

        def run(self):
            helpdir = os.path.join(self.build_dir, "htmlhelp")
            if not (os.path.isdir(helpdir)
                    and os.path.isfile(os.path.join(helpdir, "index.html"))):
                self.warn("Sphinx is needed to build help files. Skipping.")
                return
            files = find_htmlhelp_files(os.path.join(helpdir))
            # add the build files to distribution
            self.distribution.data_files.extend(files)


def setup_package():
    write_version_py()
    cmdclass = {
        'lint': LintCommand,
        'coverage': CoverageCommand,
        'config': config,
        'sdist': sdist,
        'build': build,
        'build_htmlhelp': build_htmlhelp,
        # Use install_data from distutils, not numpy.distutils.
        # numpy.distutils insist all data files are installed in site-packages
        'install_data': install_data.install_data
    }
    if have_numpy:
        extra_args = {
            "configuration": configuration
        }
    else:
        # substitute a build_ext command with one that raises an error when
        # building. In order to fully support `pip install` we need to
        # survive a `./setup egg_info` without numpy so pip can properly
        # query our install dependencies
        extra_args = {}
        cmdclass["build_ext"] = build_ext_error

    setup(
        name=NAME,
        version=FULLVERSION,
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
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        zip_safe=False,
        test_suite='Orange.tests.suite',
        cmdclass=cmdclass,
        **extra_args
    )


if __name__ == '__main__':
    setup_package()

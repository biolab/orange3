#!/bin/bash

# set exit on error
set -e

function print_usage {
    echo 'build-win-application.sh
Build an Windows applicaiton installer for Orange Canvas

Note: needs makensis and 7z on PATH, as well as a python environment with
installed pip (>=7)

Options:

    -b --build-base PATH    Build directory (default build/win-installer)
    -d --dist-dir           Distribution dir
    --download-cache DIR    Cache downloaded packages in DIR
    -r --requirements       Extra requirements file
    --standalone            Build a standalone application.
    -h --help               Print this help
'
}


while [[ ${1:0:1} = "-" ]]; do
    case $1 in
        -b|--build-base)
            BUILDBASE=$2
            shift 2
            ;;
        -d|--dist-dir)
            DISTDIR=$2
            shift 2
            ;;
        --download-cache)
            DOWNLOADDIR=$2
            shift 2
            ;;
        -r|--requirements)
            REQUIREMENT=$2
            shift 2
            ;;
        --standalone)
            STANDALONE=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Unkown argument $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

PLATTAG=win32

PYTHON_VER=3.4.4
PYTHON_MD5=e96268f7042d2a3d14f7e23b2535738b

PYTHON_VER_SHORT=${PYTHON_VER%.[0-9]*}
PYVER=$(echo $PYTHON_VER_SHORT | sed s/\\.//g)
PYTHON_MSI=python-$PYTHON_VER.msi

# The minimum pip version required (this is the version which was shipped
# with Python 3.4.3 (ensurepip)
PIP_VER=6.0.8
PIP_MD5=41e73fae2c86ba2270ff51c1d86f7e09

PYQT_VER=4.11.4
PYQT_MD5=b4164a0f97780fbb7c5c1e265dd37473

NUMPY_VER=1.9.2
NUMPY_MD5=0c06b7beabdc053ef63699ada0ee5e98

SCIPY_VER=0.16.1
SCIPY_MD5=30bf5159326d859a42ed7718a8a09704

DISTDIR=${DISTDIR:-dist}

BUILDBASE=${BUILDBASE:-build}/temp.$PLATTAG-py$PYTHON_VER_SHORT-installer
DOWNLOADDIR=${DOWNLOADDIR:-build/temp.download-cache}

# BUILDBASE/
#   core/
#     python/
#     msvredist/
#   wheelhouse/
#       [no]sse[2|3]/
#   nsisplugins/cpucaps.dll
#   startupscripts/
#   requirements.txt

# Clean any leftovers from previous runs
if [[ -d "$BUILDBASE" ]]; then
    rm -r "$BUILDBASE"
fi


mkdir -p "$BUILDBASE"/core/python
mkdir -p "$BUILDBASE"/core/msvredist
mkdir -p "$BUILDBASE"/wheelhouse
mkdir -p "$BUILDBASE"/nsisplugins
mkdir -p "$BUILDBASE"/startupscripts

mkdir -p "$DOWNLOADDIR"
mkdir -p "$DISTDIR"

touch "$BUILDBASE"/requirements.txt

# pinned requirements (numpy and scipy are handled separately)
echo "
#:wheel: scikit-learn https://pypi.python.org/packages/cp34/s/scikit-learn/scikit_learn-0.16.1-cp34-none-win32.whl#md5=ca5864cdf9f1938aa1a55d6092bf5c86
scikit-learn==0.16.1

#:wheel: Bottlecheset https://dl.dropboxusercontent.com/u/100248799/Bottlechest-0.7.1-cp34-none-win32.whl#md5=629ba2a148dfa784d0e6817497d42e97
--find-links https://dl.dropboxusercontent.com/u/100248799/Bottlechest-0.7.1-cp34-none-win32.whl
Bottlechest==0.7.1
" > "$BUILDBASE"/requirements.txt

function __download_url {
    local url=${1:?}
    local out=${2:?}
    curl --fail -L --max-redirs 4 -o "$out" "$url"
}

function md5sum_check {
    local filepath=${1:?}
    local checksum=${2:?}
    local md5=

    if which md5 &> /dev/null; then
        md5=$(md5 -q "$filepath")
    else
        md5=$(md5sum "$filepath" | cut -d " " -f 1)
    fi

    [ "$md5" == "$checksum" ]
}

#
# download_url URL TARGET_PATH MD5_CHECKSUM
#
# download the contants of URL and to TARGET_PATH and check that the
# md5 checksum matches.

function download_url {
    local url=${1:?}
    local targetpath=${2:?}
    local checksum=${3:?}

    if [ -f "$targetpath" ] && ! md5sum_check "$targetpath" "$checksum"; then
        rm "$targetpath"
    fi

    if [ ! -f "$targetpath" ]; then
        __download_url "$url" "$targetpath"
    fi

    if ! md5sum_check "$targetpath" "$checksum"; then
        echo "Checksum does not match for $OUT"
        exit 1
    fi
}

#
# Download python msi installer
#
function prepare_python {
    local url="https://www.python.org/ftp/python/$PYTHON_VER/$PYTHON_MSI"
    download_url "$url" "$DOWNLOADDIR/$PYTHON_MSI" $PYTHON_MD5
    cp "$DOWNLOADDIR/$PYTHON_MSI" "$BUILDBASE"/core/python
}

function prepare_msvredist {
    local url="https:/orange.biolab.si/files/3rd-party/$PYVER/vcredist_x86.exe"
    download_url $url \
                 "$BUILDBASE/core/msvredist/vcredist_x86.exe" \
                 b88228d5fef4b6dc019d69d4471f23ec
}

function prepare_msvcr100 {
    download_url "https://dl.dropboxusercontent.com/u/100248799/msvcr100.dll" \
                 "$DOWNLOADDIR/msvcr100.dll" \
                 bf38660a9125935658cfa3e53fdc7d65

    cp "$DOWNLOADDIR/msvcr100.dll" "$BUILDBASE/core/msvredist/msvcr100.dll"

    download_url "https://dl.dropboxusercontent.com/u/100248799/msvcp100.dll" \
                 "$DOWNLOADDIR/msvcp100.dll" \
                 e3c817f7fe44cc870ecdbcbc3ea36132

    cp "$DOWNLOADDIR/msvcp100.dll" "$BUILDBASE/core/msvredist/msvcp100.dll"
}


function prepare_pip {
    local version=${PIP_VER:?}
    local url=https://pypi.python.org/packages/py2.py3/p/pip/pip-${version}-py2.py3-none-any.whl
    local md5=${PIP_MD5:?}
    download_url "${url}" \
                 "${DOWNLOADDIR}"/pip-${version}-py2.py3-none-any.whl \
                 ${md5}
    cp "$DOWNLOADDIR"/pip-${version}-py2.py3-none-any.whl \
       "$BUILDBASE"/wheelhouse/
}


function prepare_pyqt4 {
    download_url \
        https://dl.dropboxusercontent.com/u/100248799/PyQt4-${PYQT_VER}-cp34-none-win32.whl \
        "$DOWNLOADDIR"/PyQt4-${PYQT_VER}-cp34-none-win32.whl \
        $PYQT_MD5

    cp "$DOWNLOADDIR"/PyQt4-${PYQT_VER}-cp34-none-win32.whl "$BUILDBASE"/wheelhouse
}

function prepare_scipy_stack {
	local numpy_superpack=numpy-$NUMPY_VER-win32-superpack-python$PYTHON_VER_SHORT.exe
	local scipy_superpack=scipy-$SCIPY_VER-win32-superpack-python$PYTHON_VER_SHORT.exe

    download_url http://sourceforge.net/projects/numpy/files/NumPy/$NUMPY_VER/$numpy_superpack/download \
                 "$DOWNLOADDIR"/$numpy_superpack \
                 $NUMPY_MD5

    download_url http://sourceforge.net/projects/scipy/files/scipy/$SCIPY_VER/$scipy_superpack/download \
                 "$DOWNLOADDIR"/$scipy_superpack \
                 $SCIPY_MD5

    7z -o"$DOWNLOADDIR"/numpy -y e "$DOWNLOADDIR"/$numpy_superpack
    7z -o"$DOWNLOADDIR"/scipy -y e "$DOWNLOADDIR"/$scipy_superpack

	local wheeltag=cp${PYVER}-none-win32
	local wheeldir=

    for SSE in nosse sse2 sse3; do
		wheeldir="$BUILDBASE"/wheelhouse/$SSE
        mkdir -p "$wheeldir"

        python -m wheel convert -d "$wheeldir" \
               "$DOWNLOADDIR"/numpy/numpy-$NUMPY_VER-$SSE.exe

        mv "$wheeldir"/numpy-$NUMPY_VER-*$SSE.whl \
		   "$wheeldir"/numpy-$NUMPY_VER-$wheeltag.whl

        python -m wheel convert -d "$wheeldir" \
			   "$DOWNLOADDIR"/scipy/scipy-$SCIPY_VER-$SSE.exe

        mv "$wheeldir"/scipy-$SCIPY_VER-*$SSE.whl \
		   "$wheeldir"/scipy-$SCIPY_VER-$wheeltag.whl
    done

    # copy the CpuCaps.dll nsis plugin into place
    cp "$DOWNLOADDIR"/numpy/cpucaps.dll \
       "$BUILDBASE"/nsisplugins
}

function prepare_req {
    python -m pip wheel \
        -w "$BUILDBASE/wheelhouse" \
        -f "$BUILDBASE/wheelhouse" \
        -f "$BUILDBASE/wheelhouse/nosse" \
        "$@"
}

function prepare_orange {
    # ensure that correct numpy and scipy are installed in the build env
    pip install --no-index -f "$BUILDBASE/wheelhouse" \
                --only-binary numpy,scipy \
                numpy==$NUMPY_VER, scipy==$SCIPY_VER

    python setup.py egg_info
    local version=$(grep -E "^Version: .*$" Orange.egg-info/PKG-INFO | awk '{ print $2 }')

    python setup.py egg_info \
        build --compiler=msvc \
        bdist_wheel -d "$BUILDBASE/wheelhouse"

    # Ensure all install dependencies are available in the wheelhouse
    prepare_req --only-binary numpy,scipy,scikit-learn,bottlechest .

    echo "# Orange " >> "$BUILDBASE/requirements.txt"
    echo "Orange==$version" >> "$BUILDBASE/requirements.txt"
}

function prepare_extra {
    python -m pip wheel \
        -w "$BUILDBASE/wheelhouse" \
        -f "$BUILDBASE/wheelhouse" \
        -f "$BUILDBASE/wheelhouse/nosse" \
        --no-deps \
        --no-index \
        -r "$1"

    echo "Inserting extra requirements"
    cat "$1" | grep -v -E '(--find-links)|(-f)' >> "$BUILDBASE"/requirements.txt
}


function create_startupscript {
    local template="@echo off
set __DIRNAME=%~dp0
set PATH=\"%__DIRNAME%\Python${PYVER}\";%PATH%
shift
start \"__TITLE__\" /B /D \"%__DIRNAME%\Python${PYVER}\" \"%__DIRNAME%\Python${PYVER}\python.exe\" __ARGS__ %*
"
    local title=${1:?}
    local args=${2}

    local script=$(echo "$template" | sed "s/__TITLE__/$title/g" | sed "s/__ARGS__/$args/g")
    echo "$script"
}

function create_aliasscript {
    local template="@echo off
set __DIRNAME=%~dp0
set PATH=\"%__DIRNAME%\Python${PYVER}\";%PATH%
shift
\"%__DIRNAME%\Python${PYVER}\python.exe\" __ARGS__ %*
"
    local args=${2}

    local script=$(echo "$template" | sed "s/__ARGS__/$args/g")
    echo "$script"
}

function prepare_startupscripts {
    local dir="$BUILDBASE"/startupscripts
    create_aliasscript "ipython" "-m IPython" > "$dir"/ipython.bat
    create_startupscript "ipython-qtconsole" "-m IPython qtconsole" > "$dir"/ipython-qtconsole.bat
    create_startupscript "ipython-notebook" "-m IPython notebook" > "$dir"/ipython-notebook.bat
    create_aliasscript "pip" "-m pip" > "$dir"/pip.bat
    create_startupscript "Orange Canvas" "-m Orange.canvas" > "$dir"/orange-canvas.bat
}

function prepare_all {
    prepare_python
    prepare_msvcr100
    prepare_pip
    prepare_scipy_stack
    prepare_pyqt4
    # Need to specifically restrict the numpy/scipy versions, otherwise
    # pip wheel will try to download/build them as soon as there is a newer
    # version available on pip.
    prepare_req numpy==$NUMPY_VER scipy==$SCIPY_VER -r "$BUILDBASE/requirements.txt"
    prepare_orange

    if [[ "$STANDALONE" ]]; then
        prepare_startupscripts
    fi

    if [[ "$REQUIREMENT" ]]; then
        prepare_extra "$REQUIREMENT"
    fi
}

function abs_dir_path {
    echo $(cd "$1"; pwd)
}

function create_installer {
    local basedir=${1:?}
    local output_path=${2:?}
    local nsis_script=${3:?}
    local basedir_abs=$(cd "$basedir"; pwd)

    # output path must be absolute.
    if [[ ${output_path:0:1} != "/" ]]; then
        output_path="$(pwd)/$output_path"
    fi

    makensis -DOUTFILENAME="$output_path" \
             -DPYTHON_VERSION=$PYTHON_VER \
             -DPYTHON_VERSION_SHORT=$PYTHON_VER_SHORT \
             -DPYVER=$PYVER \
			 -DBASEDIR="$basedir_abs" \
             -DNSIS_PLUGINS_PATH="$basedir_abs"/nsisplugins \
             "$nsis_script"
}

# Prepare prerequisites
prepare_all

VERSION=$(grep -E "^Orange==" "$BUILDBASE/requirements.txt" | sed s/^Orange==//g)

# Package everything in an installer
if [[ $STANDALONE ]]; then
    NSIS_SCRIPT=scripts/windows/install-standalone.nsi
    INSTALLER=Orange3-${VERSION:?}.$PLATTAG-py$PYTHON_VER_SHORT-install-standalone.exe
else
    NSIS_SCRIPT=scripts/windows/install.nsi
    INSTALLER=Orange3-${VERSION:?}.$PLATTAG-py$PYTHON_VER_SHORT-install.exe
fi

create_installer "$BUILDBASE" "$DISTDIR/$INSTALLER" "$NSIS_SCRIPT"

#!/usr/bin/env bash

# exit on any error
set -e

function usage() {
    echo 'build-installer.sh

Build a Windows application installer based on a set of pip requirements.
All of the requirements must to be available as .whl files

Note: This script needs makensis, curl, python  and pip (>=9) on $PATH
Note: Needs basic bash env on Windows (git-bash is sufficient/tested, but
      cygwin should work too).

Options:
    -b --build-base <path>  Build directory (default build)
    -d --dist-dir           Directory where the final installer is put
                            (default dist)
    --cache-dir DIR         Cache downloaded packages in DIR
                            (default is "build/download-cache")
    --python-version        Python version: Major.Minor.Micro
                            The python installer version from python.org
                            bundled into the installer.
    --platform              win32 or win_amd64
    -f --find-links  <url>  Index option passed to pip to download wheels
                            (see pip download --help)
    --extra-index-url <url> Index option passed to pip to download wheels
                            (see pip download --help)
    --no-index              Index option passed to pip to download wheels
    --pip-arg               Requirement args passed to pip
    -h --help               Print this help

Examples:
    $ ./scripts/windows/build-win-installer.sh \
        --python-version 3.4.4 --platform win32 \
        --pip-arg={-r,scripts/windows/specs/PY34-win32.txt,orange3==3.4.2}

    # Build the installer using a local wheels cache containing a set of
    # binary packages (for instance from Christoph Gohlke`s pythonlibs)
    $ ./scripts/windows/build/win-installer.sh \
        --python-version 3.6.1 --platform win_amd64 \
        --no-index --find-links=./wheels \
        --pip-arg=orange3~=3.3.12
'
}

NAME=Orange3
# version is determined at the end when all packages are available
VERSION=

BUILDBASE=
DISTDIR=
CACHEDIR=
PIP_INDEX_ARGS=()
PIP_ARGS=()

PYTHON_VERSION=
PLATTAG=

while [[ "${1:0:1}" = "-" ]]; do
    case $1 in
        -b|--build-base)
            BUILDBASE=${2:?}; shift 2;;
        --build-base=*)
            BUILDBASE=${1#*=}; shift 1;;
        -d|--dist-dir)
            DISTDIR=${2:?}; shift 2;;
        --dist-dir=*)
            DISTDIR=${1#*=}; shift 1;;
        --cache-dir)
            CACHEDIR=${2:?}; shift 2;;
        --cache-dir=*)
            CACHEDIR=${1#*=}; shift 1;;
        --python-version)
            PYTHON_VERSION=${2:?}; shift 2;;
        --python-version=*)
            PYTHON_VERSION=${1#*=}; shift 1;;
        --platform)
            PLATTAG=${2:?}; shift 2;;
        --platform=*)
            PLATTAG=${1#*=}; shift 1;;
        -f|--find-links)
            PIP_INDEX_ARGS+=(--find-links "${2:?}"); shift 2;;
        --find-links=*)
            PIP_INDEX_ARGS+=(--find-links "${1#*=}"); shift 1;;
        --extra-index-url)
            PIP_INDEX_ARGS+=(--extra-index-url "${2:?}"); shift 2;;
        --extra-index-url=*)
            PIP_INDEX_ARGS+=(--extra-index-url "${1#*=}"); shift 1;;
        --no-index)
            PIP_INDEX_ARGS+=( --no-index ); shift 1;;
        --pip-arg)
            PIP_ARGS+=( "${2:?}" ); shift 2;;
        --pip-arg=*)
            PIP_ARGS+=( "${1#*=}" ); shift 1;;
        -h|--help)
            usage; exit 0;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
     esac
done

if [[ ! ${PYTHON_VERSION} =~ ^([0-9]+\.){2,}[0-9]+$ ]]; then
    echo "Invalid python version: $PYTHON_VERSION (need major.minor.micro)" >&2
    exit 1
fi

if [[ ! ${PLATTAG:?} =~ (win32|win_amd64) ]]; then
    echo "Invalid platform tag: ${PLATTAG} (expected win32 or win_amd64)" >&2
    exit 1
fi

# fail early if unzip is not on path
which unzip > /dev/null || { echo "unzip not on path"; exit 1; }

# Major.Minor
PYTHON_VER=${PYTHON_VERSION%.*}
# MajorMinor
PYTAG=${PYTHON_VER/./}

ABITAG=cp${PYTAG}m

export PIP_DISABLE_PIP_VERSION_CHECK=

BUILDBASE=${BUILDBASE:-build}
BASEDIR="${BUILDBASE:?}"/temp.${PLATTAG}-${PYTHON_VER}.installer

CACHEDIR=${CACHEDIR:-build/download-cache}
DISTDIR=${DISTDIR:-dist}

if [[ -d "${BASEDIR:?}" ]]; then
    rm -r "${BASEDIR:?}"
fi

# BASEDIR/
#   wheelhouse/
#   requirements.txt

mkdir -p "${BASEDIR:?}"/wheelhouse

mkdir -p "${CACHEDIR:?}"/wheelhouse
mkdir -p "${CACHEDIR:?}"/python


# Extract the n-th version component
# $ version-component 1 1.2.3a
# 1
# $ version-component 3 1.2.3a
# 3a
version-component() {
    local n=${1:?}
    local verstring=${2:?}
    declare -i comindex
    declare -a vercomp

    if [[ ! ${n} =~ ^[0-9] ]]; then
        echo "Invalid version component number ${n}" >&2;
        return 1
    fi

    IFS=. read -r -a vercomp <<< "${verstring}"
    comindex=$(( ${n} - 1 ))
    echo "${vercomp[comindex]}"
}

# python-installer-filename MAJOR.MINOR.MICRO (win32|win_amd64)
#
# Output the filename of the official (python.org) installer
python-installer-filename(){
    local version=${1:?}
    local plattag=${2:?}
    local pymajor=$(version-component 1 ${version})
    local pyminor=$(version-component 2 ${version})
    local filename=
    local ext=
    if [[ ${pymajor}${pyminor} -ge 35 ]]; then
        ext=exe
    else
        ext=msi
    fi
    case ${plattag}/${ext} in
        win32/msi) filename=python-${version}.msi;;
        win_amd64/msi) filename=python-${version}.amd64.msi;;
        win32/exe) filename=python-${version}.exe;;
        win_amd64/exe) filename=python-${version}-amd64.exe;;
        *)
            echo "Invalid version and/or platform tag (${version}/${plattag})"\
                 >&2;
            return 1;;
    esac
    echo ${filename}
}

fetch-python() {
    # $ fetch-python major.minor.micro plattag [ dest ]
    #
    # Download an official python installer from python.org into dest
    # folder (./ by default)
    local version=${1:?}
    local plattag=${2:?}
    local dest=${3:-./}

    local filename=$(python-installer-filename ${version} ${plattag})
    if [[ ! ${filename} ]]; then
        return 1
    fi

    dest="${dest}/${filename}"
    if [[ ! -f "${dest}" ]]; then
        local tmpname=$(mktemp "${dest}.XXXXX")
        if curl -s -f -L -o "${tmpname}" \
               https://www.python.org/ftp/python/${version}/${filename}; then
            mv "${tmpname}" "${dest}"
        else
            return $?
        fi
    fi
}

fetch-requirements() {
    # Download binary packages for the specified platform (all packages
    # must be available as .whl files)
    local wheeldir="${CACHEDIR}/wheelhouse"
    pip download \
        "${PIP_INDEX_ARGS[@]}" \
        --dest "${wheeldir}" \
        --find-links "${wheeldir}" \
        --only-binary :all: \
        --python-version "${PYTAG}" \
        --platform  "${PLATTAG}" \
        --abi "${ABITAG}" \
        "$@"
}

# Package install requirements in "${BASEDIR}/wheelhouse".
# All the requirements MUST have the .whl cached in ${CACHEDIR}/wheelhouse

package-requirements() {
    local pyfilename=$(python-installer-filename ${PYTHON_VERSION} ${PLATTAG})
    cp "${CACHEDIR:?}/python/${pyfilename:?}" \
       "${BASEDIR:?}/"
    local wheeldir="${CACHEDIR:?}/wheelhouse"
    pip download \
        --no-index \
        --find-links "${wheeldir}" \
        --dest "${BASEDIR:?}/wheelhouse" \
        --only-binary :all: \
        --python-version "${PYTAG}" \
        --platform  "${PLATTAG}" \
        --abi "${ABITAG}" \
        "$@"

    echo "# Env spec " > "${BASEDIR:?}"/requirements.txt
    (
        cd "${BASEDIR:?}/wheelhouse"
        ls -1 *.whl
    ) >> "${BASEDIR:?}/requirements.txt"

    mkdir -p "${BASEDIR:?}/icons"
    cp scripts/windows/{orange.ico,OrangeOWS.ico} "${BASEDIR:?}/icons"
}


win-path() {
    case "$(uname -s)" in
        MINGW*|CYGWIN*)
            cygpath -w "$1";;
        *)
            echo "$1";;
    esac
}

# Display a named dist-info directory filename from a wheel (.whl) file
# Example:
#     $ wheel-dist-info numpy-1.12.0-cp34-none-win32.whl METADATA
wheel-dist-info() {
    local wheel=${1:?}
    local filename=${2:?}
    local filepath_path=$(
        unzip -Z -1 "${wheel}" | grep -E "^[^/]*.dist-info/${filename}$"
    )
    if [[ ${filepath_path} ]]; then
        unzip -p "${wheel}" "${filepath_path}"
    else
        return 1
    fi
}

# Extract contents of METADATA from a wheel (.whl) file
wheel-metadata() {
    wheel-dist-info "${1:?}" METADATA
}

# Extract the package version from a wheel (.whl) file
wheel-version() {
    wheel-metadata "${1:?}" | grep -E "^Version: " | cut -d " " -f 2
}

PYINSTALL_TYPE=Normal

make-installer() {
    local scriptdir="$(dirname "$0")"
    local nsis_script="${scriptdir:?}/orange-install.nsi"
    local outpath=${DISTDIR}
    local filename=${NAME}-${VERSION}-Python${PYTAG:?}-${PLATTAG:?}.exe
    local pyinstaller=$(python-installer-filename ${PYTHON_VERSION} ${PLATTAG})
    local basedir=$(win-path "${BASEDIR:?}")
    local versionstr=${VERSION}
    local major=$(version-component 1 "${versionstr}")
    local minor=$(version-component 2 "${versionstr}")
    local micro=$(version-component 3 "${versionstr}")
    local pymajor=$(version-component 1 "${PYTHON_VERSION}")
    local pyminor=$(version-component 2 "${PYTHON_VERSION}")
    local pymicro=$(version-component 3 "${PYTHON_VERSION}")

    cat <<EOF > "${BASEDIR}"/license.txt
Acknowledgments and License Agreement
-------------------------------------

EOF
    local licenses=( LICENSE )
    for file in "${licenses[@]}"; do
        cat "${file}" >> "${BASEDIR}"/license.txt
        echo "" >> "${BASEDIR}"/license.txt
    done
    mkdir -p "${DISTDIR}"

    makensis -DOUTFILENAME="${outpath}/${filename}" \
             -DAPPNAME=Orange \
             -DVERSION=${VERSION} \
             -DVERMAJOR=${major} -DVERMINOR=${minor} -DVERMICRO=${micro} \
             -DPYMAJOR=${pymajor} -DPYMINOR=${pyminor} -DPYMICRO=${pymicro} \
             -DPYARCH=${PLATTAG} \
             -DPYINSTALL_TYPE=${PYINSTALL_TYPE} \
             -DBASEDIR="${basedir}" \
             -DPYINSTALLER=${pyinstaller} \
             -DINSTALL_REGISTRY_KEY=OrangeCanvas \
             -DINSTALLERICON=scripts/windows/OrangeInstall.ico \
             -DLICENSE_FILE="${BASEDIR}"/license.txt \
             -NOCD \
             -V4 -WX \
             "-X!addincludedir $(win-path "${scriptdir}")" \
             "${nsis_script:?}"
}

DIRNAME=$(dirname "${0}")

fetch-python ${PYTHON_VERSION} ${PLATTAG} "${CACHEDIR:?}"/python
fetch-requirements "${PIP_ARGS[@]}"
package-requirements "${PIP_ARGS[@]}"

shopt -s failglob
WHEEL=( "${BASEDIR}"/wheelhouse/${NAME}*.whl )
shopt -u failglob

if [[ ! "${WHEEL}" ]]; then
    echo "Error: ${NAME} missing from the environment specification" >&2
    exit 1
fi

VERSION=$(wheel-version "${WHEEL:?}")

if [[ ! ${VERSION} ]]; then
    echo "ERROR: Could not determine version string" >&2
    exit 1
fi

make-installer

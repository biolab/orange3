#!/usr/bin/env bash

set -e

usage() {
    echo 'usage: build-macos-app.sh [--python-version VER] [--pip-arg ARG] APPPATH
Create (build) an macOS application bundle

Options:
    --python-version VERSION
        Python version to install in the application bundle (default: 3.7.5)

    --pip-arg  ARG
        Pip install arguments to populate the python environemnt in the
        application bundle. Can be used multiple times.
        If not supplied then by default the latest PyPi published Orange3 is
        used.

    -h|--help
        Print this help

Examples
    build-macos-app.sh ~/Applications/Orange3.app
        Build the application using the latest published version on pypi

    build-macos-app.sh --pip-arg={Orange3==3.3.12,PyQt5} ~/Applications/Orange3.app
        Build the application using the specified Orange version

    build-macos-app.sh --pip-arg=path-tolocal-checkout ~/Applications/Orange3-Dev.app
        Build the application using a local source checkout

    build-macos-app.sh --pip-arg={-e,path-tolocal-checkout}  ~/Applications/Orange3-Dev.app
        Build the application and install orange in editable mode

    buils-macos-app.sh --pip-arg={-r,requirements.txt} /Applications/Orange3.app
        Build the application using a fixed set of locked requirements.
'
}

DIR=$(dirname "$0")

# Python version in the bundled framework
PYTHON_VERSION=3.7.5

# Pip arguments used to populate the python environment in the application
# bundle
PIP_REQ_ARGS=( )

while [[ "${1:0:1}" == "-" ]]; do
    case "${1}" in
        --python-version=*)
            PYTHON_VERSION=${1#*=}
            shift 1;;
        --python-version)
            PYTHON_VERSION=${2:?"--python-version requires an argument"}
            shift 2;;
        --pip-arg=*)
            PIP_REQ_ARGS+=( "${1#*=}" )
            shift 1;;
        --pip-arg)
            PIP_REQ_ARGS+=( "${2:?"--pip-arg requires an argument"}" )
            shift 2;;
        --help|-h)
            usage; exit 0;;
        -*)
            echo "Invalid argument ${1}" >&2; usage >&2; exit 1;;
    esac
done

APPDIR=${1:?"Target APPDIR argument is missing"}

PYVER=${PYTHON_VERSION%.*}  # Major.Minor

if [[ ${#PIP_REQ_ARGS[@]} -eq 0 ]]; then
    PIP_REQ_ARGS+=( Orange3 'PyQt5~=5.12.0' 'PyQtWebEngine~=5.12.0' )
fi

mkdir -p "${APPDIR}"/Contents/MacOS
mkdir -p "${APPDIR}"/Contents/Frameworks
mkdir -p "${APPDIR}"/Contents/Resources

cp -a "${DIR}"/skeleton.app/Contents/{Resources,Info.plist.in} \
    "${APPDIR}"/Contents

# Layout a 'relocatable' python framework in the app directory
"${DIR}"/python-framework.sh \
    --version "${PYTHON_VERSION}" \
    --macos 10.9 \
    --install-certifi \
    "${APPDIR}"/Contents/Frameworks

ln -fs ../Frameworks/Python.framework/Versions/${PYVER}/Resources/Python.app/Contents/MacOS/Python \
    "${APPDIR}"/Contents/MacOS/PythonApp

ln -fs ../Frameworks/Python.framework/Versions/${PYVER}/bin/python${PYVER} \
    "${APPDIR}"/Contents/MacOS/python

"${APPDIR}"/Contents/MacOS/python -m ensurepip
"${APPDIR}"/Contents/MacOS/python -m pip install pip~=19.0 wheel

cat <<'EOF' > "${APPDIR}"/Contents/MacOS/Orange
#!/bin/bash

DIR=$(dirname "$0")

# LaunchServices passes the Carbon process identifier to the application with
# -psn parameter - we do not want it
if [[ "${1}" == -psn_* ]]; then
    shift 1
fi

# Disable user site packages
export PYTHONNOUSERSITE=1

exec "${DIR}"/PythonApp -m Orange.canvas "$@"
EOF
chmod +x "${APPDIR}"/Contents/MacOS/Orange

cat <<'EOF' > "${APPDIR}"/Contents/MacOS/pip
#!/bin/bash

DIR=$(dirname "$0")

# Disable user site packages
export PYTHONNOUSERSITE=1

exec -a "$0" "${DIR}"/python -m pip "$@"
EOF
chmod +x "${APPDIR}"/Contents/MacOS/pip

PYTHON="${APPDIR}"/Contents/MacOS/python

"${PYTHON}" -m pip install --no-warn-script-location "${PIP_REQ_ARGS[@]}"

VERSION=$("${PYTHON}" -m pip show orange3 | grep -E '^Version:' |
          cut -d " " -f 2)

m4 -D__VERSION__="${VERSION:?}" "${APPDIR}"/Contents/Info.plist.in \
    > "${APPDIR}"/Contents/Info.plist
rm "${APPDIR}"/Contents/Info.plist.in

# Sanity check
(
    # run from an empty dir to avoid importing/finding any packages on ./
    tempdir=$(mktemp -d)
    cleanup() { rm -r "${tempdir}"; }
    trap cleanup EXIT
    cd "${tempdir}"
    "${PYTHON}" -m pip install --no-cache-dir --no-index orange3 PyQt5
    "${PYTHON}" -m Orange.canvas --help > /dev/null
)

#!/usr/bin/env bash

set -e

usage() {
    echo 'usage: sign-app.sh -s IDENTITY PATH

Sign an Orange.app application at PATH and create a signed .dmg installer.

OPTIONS
    --sign -s IDENTITY
        Signing identity to use. The \`identity\` must name a signing
        certificate in a macOS keychain (see \`man codesign\` SIGNING
        IDENTITIES section for details).

    --help -h
        Print this help.
'
}

SCRIPTS=$( cd "$(dirname "$0")" ; pwd -P )
DIST=$( cd "$(dirname "$0")/../../dist" ; pwd -P )

IDENTITY="Developer ID"

while true; do
    case "${1}" in
        -s|--sign) IDENTITY="${2:?"no identity provided"}"; shift 2;;
        -h|--help) usage; exit 0;;
        -*) echo "unrecognized parameter: ${1}" >&2; usage >&2; exit 1;;
        *)  break;;
    esac
done

APPPATH=${1}
if [ ! "${APPPATH}" ]; then
    APPPATH=${DIST}/Orange3.app
    echo "No path supplied; using default ${APPPATH}"
fi

VERSION=$(
    "${APPPATH}"/Contents/MacOS/python -c '
import pkg_resources
print(pkg_resources.get_distribution("Orange3").version)
'
)

# Create disk image
"${SCRIPTS}"/create-dmg-installer.sh --app "${APPPATH}" \
    --sign "${IDENTITY}" \
    "${DIST}/Orange3-${VERSION}.dmg"


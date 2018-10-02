#!/usr/bin/env bash

set -e
shopt -s failglob

usage() {
    echo "sign-dmg.sh [-s IDENTITY] [-o OUTPUT] PATH

Sign a .dmg application installer at 'path'.

\`path\` must name a .dmg image.

OPTIONS
    --sign -s IDENTITY
        Signing identity to use. The \`identity\` must name a signing
        certificate in a macOS keychain (see \`man codesign\` SIGNING
        IDENTITIES section for details)

    --output -o PATH
        Specify the output path for the resulting .dmg installer. If not
        supplied, then the signed dmg image is placed next to the input path
        with the same name and .signed appended at the end.

    --help -h
        Show this help.

EXAMPLES
    $ ./sign-dmg.sh -s \"Developer ID\" SupperApp.dmg
"
}

IDENTITY="Developer ID"
OUTPUT=

while true; do
    case "${1}" in
        -s|--sign) IDENTITY=${2:?"no identity provided"}; shift 2;;
        -o|--output) OUTPUT=${2:?"${1} requires a path parameter"}; shift 2;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unrecognized parameter: ${1}" >&2; echo; usage >&2; exit 1;;
        *) break;;
    esac
done

DMG="${1:?"Missing positional argument: PATH"}"

if [ ! -f "${DMG}" ]; then
    echo "${DMG} does not exist or is not a file." >&2;
    exit 1
fi

# Temporary work directory
WORKDIR=
# Temporary mount point for the dmg
MOUNT=

cleanup() {
    if [ -d "${MOUNT}" ]; then hdiutil detach "${MOUNT}" -force || true; fi
    if [ -d "${WORKDIR}" ]; then rm -rf "${WORKDIR}"; fi
    return
}
trap cleanup EXIT

BASENAME=$(basename "${DMG}")
WORKDIR=$(mktemp -d -t "${BASENAME}")
MOUNT=${WORKDIR}/mnt

IMGRW=${WORKDIR}/image
IMG=${WORKDIR}/${BASENAME}

# convert the input dmg to a read write growable uncompressed image
# NOTE: hdiutil convert always appends the extension on the output even if
# already present (the ext for UDSB is .sparsebundle so the actual filename
# is ${IMGRW}.sparsebundle)
hdiutil convert -format UDSB -o "${IMGRW}" "${DMG}"
# 'resize' the sparse image allowing growth for signing data
# (1GB should be enough for everybody)
hdiutil resize -size 1g "${IMGRW}".sparsebundle
# mount it R/W for modification
mkdir "${MOUNT}"
hdiutil attach "${IMGRW}".sparsebundle -readwrite -noverify -noautoopen \
        -mountpoint "${MOUNT}"

codesign --sign "${IDENTITY}" --deep --verbose "${MOUNT}"/*.app

# detach/unmount to sync
hdiutil detach "${MOUNT}" -force
# resize the image to minimum required size
hdiutil resize -sectors min "${IMGRW}".sparsebundle
# convert back to compressed read only image
hdiutil convert -format UDZO -imagekey zlib-level=9 \
        -o "${IMG}" "${IMGRW}.sparsebundle"
codesign -s "${IDENTITY}" "${IMG}"

if [ ! "${OUTPUT}" ]; then
    OUTPUT=${DMG}.signed
fi

mv "${IMG}" "${OUTPUT}"

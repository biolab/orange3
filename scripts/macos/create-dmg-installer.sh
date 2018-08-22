#!/bin/bash

set -e

function print_usage() {
    echo 'create-dmg-installer.sh --app BUILD_APP_PATH OUTPUT_BUNDLE.dmg

Create an disk image installer (.dmg) for Orange OSX application.

Options:
    -a --app PATH
        Path to a build Orange3.app to include in the disk image
        (default dist/Orange3.app)

    -s --sign IDENTITY
        Sign the application and the .dmg image using the signing identity
        provided (see `man codesign` SIGNING IDENTITIES section for details)

    -k --keep-temp
        Keep the temporary files after creating the final image.

    -h --help
        Print this help
'
}


DIRNAME=$(dirname "$0")

# Path to dmg resources (volume icon, background, ...)
RES="${DIRNAME}"/dmg-resources

APP=dist/Orange3.app

KEEP_TEMP=0
IDENTITY=

while [[ "${1:0:1}" = "-" ]]; do
    case "${1}" in
        -a|--app)
            APP=${2:?"BUILD_APP_PATH is missing"}
            shift 2 ;;
        -k|--keep-temp)
            KEEP_TEMP=1
            shift 1 ;;
        -s|--sign)
            IDENTITY=${2:?"${1} is missing a parameter"}
            shift 2;;
        -h|--help)
            print_usage
            exit 0 ;;
        -*)
            echo "Unknown option $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

DMG=${1?"Output bundle dmg path not specified"}


if [[ ! -d "${APP}" ]]; then
    echo "$APP path does not exits or is not a directory."
    print_usage
    exit 1
fi

TMP_DIR=$(mktemp -d -t create-dmg-installer)
TMP_TEMPLATE="${TMP_DIR}"/template
TMP_DMG="${TMP_DIR}"/orange.dmg
TMP_MOUNT="${TMP_DIR}"/mnt

echo "Preparing an image template in ${TMP_TEMPLATE}"
echo "============================================="

# Copy necessary resources into the template

mkdir -p "${TMP_TEMPLATE}"/.background

cp -a "${RES}"/background.png "${TMP_TEMPLATE}"/.background
cp -a "${RES}"/VolumeIcon.icns "${TMP_TEMPLATE}"/.VolumeIcon.icns
cp -a "${RES}"/DS_Store "${TMP_TEMPLATE}"/.DS_Store

# Create a link to the Applications folder.
ln -s /Applications/ "${TMP_TEMPLATE}"/Applications

# Copy the .app directory in place
cp -a "${APP}" "${TMP_TEMPLATE}"/Orange3.app

if [[ "${IDENTITY}" ]]; then
    codesign -s "${IDENTITY}" --deep --verbose \
        "${TMP_TEMPLATE}"/Orange3.app
fi

# Create a regular .fseventsd/no_log file
# (see http://hostilefork.com/2009/12/02/trashes-fseventsd-and-spotlight-v100/ )

mkdir "${TMP_TEMPLATE}"/.fseventsd
touch "${TMP_TEMPLATE}"/.fseventsd/no_log


echo "Creating a temporary disk image"
hdiutil create -format UDRW -volname Orange -fs HFS+ \
       -fsargs "-c c=64,a=16,e=16" \
       -srcfolder "${TMP_TEMPLATE}" \
       "${TMP_DMG}"

mkdir "${TMP_MOUNT}"

# Mount in RW mode
echo "Mounting temporary disk image"
hdiutil attach -readwrite -noverify -noautoopen -mountpoint "${TMP_MOUNT}" \
       "${TMP_DMG}"

echo "Fixing permissions"
chmod -Rf go-w "${TMP_TEMPLATE}" || true

# Makes the disk image window open automatically when mounted
bless -openfolder "${TMP_MOUNT}"

# Hides background directory even more
SetFile -a V "${TMP_MOUNT}/.background/"

# Sets the custom icon volume flag so that volume has nice
# Orange icon after mount (.VolumeIcon.icns)
SetFile -a C "${TMP_MOUNT}"

echo "Unmouting the temporary image"
sync
hdiutil detach "${TMP_MOUNT}" -verbose -force

echo "Converting temporary image to a compressed image."

if [[ -e "${DMG}" ]]; then rm -f "${DMG}"; fi

mkdir -p "$(dirname "${DMG}")"
hdiutil convert "${TMP_DMG}" -format UDZO -imagekey zlib-level=9 -o "${DMG}"

if [[ "${IDENTITY}" ]]; then
    codesign -s "${IDENTITY}" "${DMG}"
fi

if [ ! ${KEEP_TEMP} ]; then
    echo "Cleaning up."
    rm -rf "${TMP_DIR}"
fi

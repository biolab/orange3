#!/bin/bash -e
#
# Create an .dmg installer for Orange

function print_usage() {
	echo 'create-dmg-installer.sh --app BUILD_APP_PATH OUTPUT_BUNDLE.dmg

Create an disk image installer (.dmg) for Orange OSX application.

Options:

    -a --app PATH      Path to a build Orange3.app to include in the disk.
                       (default dist/Orange3.app)
    -k --keep-temp     Keep the temporary files after creating the final image.
    -h --help          Print this help
'
}

DIRNAME=$(dirname "$0")

# Path to dmg resources (volume icon, background, ...)
RES=$DIRNAME/dmg-resources

APP="dist/Orange3.app"

KEEP_TEMP=0

while test ${1:0:1} = "-"; do
    case $1 in
        -a|--app)
            APP=$2
            shift 2
            ;;
        -k|--keep-temp)
            KEEP_TEMP=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

DMG=$1

if [[ ! -d $APP ]]; then
    echo "$APP path does not exits or is not a directory."
    print_usage
    exit 1
fi

TMP_DIR=$(mktemp -d -t orange-dmg)
TMP_TEMPLATE=$TMP_DIR/template
TMP_DMG=$TMP_DIR/orange.dmg

echo "Preparing an image template in $TMP_TEMPLATE"
echo "============================================="

# Copy neccessary resources into the template
mkdir -p "$TMP_TEMPLATE"/.background
cp -a "$RES"/background.png "$TMP_TEMPLATE"/.background

cp -a "$RES"/VolumeIcon.icns "$TMP_TEMPLATE"/.VolumeIcon.icns

cp -a "$RES"/DS_Store "$TMP_TEMPLATE"/.DS_Store

# Create a link to the Applications folder.
ln -s /Applications/ "$TMP_TEMPLATE"/Applications

# Copy the .app directory in place
cp -a "$APP" "$TMP_TEMPLATE"/Orange3.app

# Remove unnecesary files.
find "$TMP_TEMPLATE"/Orange3.app/Contents/ \( -name '*~' -or -name '*.bak' -or -name '*.pyc' -or -name '*.pyo' \) -delete

# Create a regular .fseventsd/no_log file
# (see http://hostilefork.com/2009/12/02/trashes-fseventsd-and-spotlight-v100/ )

mkdir "$TMP_TEMPLATE"/.fseventsd
touch "$TMP_TEMPLATE"/.fseventsd/no_log


echo "Creating a temporary disk image"
hdiutil create -format UDRW -volname Orange -fs HFS+ \
       -fsargs "-c c=64,a=16,e=16" \
       -srcfolder "$TMP_TEMPLATE" \
       "$TMP_DMG"

# Force detatch an image it it is mounted
hdiutil detach /Volumes/Orange -force || true

# Mount in RW mode
echo "Mounting temporary disk image"
MOUNT_OUTPUT=$(hdiutil attach -readwrite -noverify -noautoopen "$TMP_DMG" | egrep '^/dev/')

DEV_NAME=$(echo -n "$MOUNT_OUTPUT" | head -n 1 | awk '{print $1}')
MOUNT_POINT=$(echo -n "$MOUNT_OUTPUT" | tail -n 1 | awk '{print $3}')

echo "Fixing permissions."

chmod -Rf go-w "$TMP_TEMPLATE" || true

# Makes the disk image window open automatically when mounted
bless -openfolder "$MOUNT_POINT"

# Hides background directory even more
/Developer/Tools/SetFile -a V "$MOUNT_POINT/.background/"

# Sets the custom icon volume flag so that volume has nice
# Orange icon after mount (.VolumeIcon.icns)
/Developer/Tools/SetFile -a C "$MOUNT_POINT"

hdiutil detach "$DEV_NAME" -force

echo "Converting temporary image to a compressed image."

if [[ -e $DMG ]]; then
	rm -f "$DMG"
fi

hdiutil convert "$TMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG"

if [ ! $KEEP_TEMP ]; then
    echo "Cleaning up."
    rm -rf "$TMP_DIR"
fi

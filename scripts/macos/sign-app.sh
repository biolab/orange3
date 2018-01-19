#!/usr/bin/env bash

set -e

SCRIPTS="$( cd "$(dirname "$0")" ; pwd -P )"
DIST="$( cd "$(dirname "$0")/../../dist" ; pwd -P )"
APP=${DIST}/Orange3.app

# Build app
rm -rf ${APP}
${SCRIPTS}/build-macos-app.sh ${APP}
# Missing symlink Current messes with code signing
ln -s 3.6 ${APP}/Contents/Frameworks/Python.framework/Versions/Current
# sign bundle
codesign -s "Developer ID" ${APP}/Contents/Frameworks/Python.framework/Versions/3.6
codesign -s "Developer ID" ${APP}/Contents/MacOS/pip
codesign -s "Developer ID" ${APP}


VERSION=$($APP/Contents/MacOS/python -c 'import pkg_resources; print(pkg_resources.get_distribution("Orange3").version)')
# Create disk image
${SCRIPTS}/create-dmg-installer.sh --app ${APP} ${DIST}/Orange3-${VERSION}.dmg
# Sign disk image
codesign -s "Developer ID" ${DIST}/Orange3-${VERSION}.dmg

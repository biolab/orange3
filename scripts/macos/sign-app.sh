#!/usr/bin/env bash

# Build app
rm -rf ~/dev/orange3/dist/Orange3.app
./build-macos-app.sh ~/dev/orange3/dist/Orange3.app
# Missing symlink Current messes with code signing
ln -s 3.6 ../../dist/Orange3.app/Contents/Frameworks/Python.framework/Versions/Current
# sign bundle
codesign -s "Developer ID" /Users/anze/dev/orange3/dist/Orange3.app/Contents/Frameworks/Python.framework/Versions/3.6
codesign -s "Developer ID" /Users/anze/dev/orange3/dist/Orange3.app/Contents/MacOS/pip 
codesign -s "Developer ID" /Users/anze/dev/orange3/dist/Orange3.app

# Create disk image
./create-dmg-installer.sh --app ../../dist/Orange3.app ../../dist/Orange3-3.4.5-signed.dmg
# Sign disk image
codesign -s "Developer ID" /Users/anze/dev/orange3/dist/Orange3-3.4.5-signed.dmg 

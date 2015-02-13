Building Windows installer
--------------------------

These are bash scripts which and assume a cygwin/mysys environment.
In addition `7z` and `makensis` executables must be on PATH

Run from root source directory:

$ scripts/windows/build-win-application.sh

This will create a Orange3-$VERSION-install.exe in the dist/ directory
(see `scripts/windows/build-win-application.sh --help` for more information)

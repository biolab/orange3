Building Windows installer
--------------------------

Run from the root source directory

    $ scripts/windows/build-win-installer.sh

This is a bash script that expects a basic cygwin/mysys environment.
In addition `makensis` executables must be on PATH

This will create a Orange3-$VERSION.*-install.exe in the dist/ directory
(see `scripts/windows/build-win-installer.sh --help` for more information)

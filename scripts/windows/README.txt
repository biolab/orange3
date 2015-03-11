Building Windows installer
--------------------------

Run from the root source directory

	$ scripts/windows/build-win-application.sh

This is a bash script that expects a basic cygwin/mysys environment.
In addition `7z` and `makensis` executables must be on PATH

This will create a Orange3-$VERSION.*-install.exe in the dist/ directory
(see `scripts/windows/build-win-application.sh --help` for more information)

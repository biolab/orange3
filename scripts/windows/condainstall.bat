@echo off

setlocal EnableDelayedExpansion

rem Target install prefix
set PREFIX=%~1
rem Path to conda executable
set CONDA=%~2

if not exist "%PREFIX%\python.exe" (
    echo Creating a conda env in "%PREFIX%"
    rem # Create an empty initial skeleton to layout the conda, activate.bat
    rem # and other things needed to manage the environment. conda 4.1.*
    rem # requires at least one package name to succeed
    for %%f in ( vs*_runtime*.tar.bz2 ) do (
        "%CONDA%" create --yes  --quiet --prefix "%PREFIX%" "%CD%\%%f" ^
            || exit /b !ERRORLEVEL!
    )

    rem # Also install python (msvc runtime and python might be required
    rem # for any post-link scripts).
    for %%f in ( python-*.tar.bz2 ) do (
        "%CONDA%" install --yes --copy --quiet --prefix "%PREFIX%" "%CD%\%%f" ^
            || exit /b !ERRORLEVEL!
    )
)

for %%f in ( *.tar.bz2 ) do (
    echo Installing: %%f
    "%CONDA%" install --yes  --copy --quiet --prefix "%PREFIX%" "%CD%\%%f" ^
        || exit /b !ERRORLEVEL!
)

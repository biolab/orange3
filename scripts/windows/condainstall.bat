@echo off

setlocal EnableDelayedExpansion

rem Target install prefix
set PREFIX=%~1
rem Path to conda executable
set CONDA=%~2

if not exist "%PREFIX%\python.exe" (
    echo Creating a conda env in "%PREFIX%"
    rem Create an empty initial skeleton to layout the conda, activate.bat and
    rem other things needed to manage the environment.
    "%CONDA%" create --yes  --quiet --prefix "%PREFIX%" || exit /b %ERRORLEVEL%
    set BASEPKGS=
    for %%f in ( vs*_runtime*.tar.bz2 python-*.tar.bz2 ) do (
        set BASEPKGS=!BASEPKGS! %%f
    )
    "%CONDA%" install --yes --copy --quiet --prefix  "%PREFIX%" !BASEPKGS!
    if errorlevel 1 (
        echo "Error creating a conda environment. conda command exited with "
              "%ERRORLEVEL%"
        exit /b %ERRORLEVEL%
    )
)

for %%f in ( *.tar.bz2 ) do (
    echo Installing: %%f
    "%CONDA%" install --yes  --copy --quiet --prefix "%PREFIX%" %%f
    if errorlevel 1 (
        echo "Error installing %%f. conda command exited with %ERRORLEVEL%"
        exit /b %ERRORLEVEL%
    )
)

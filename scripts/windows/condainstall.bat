@echo off

setlocal EnableDelayedExpansion

rem Target install prefix
set PREFIX=%~1
rem Path to conda executable
set CONDA=%~2
rem Path to base conda env
for /f %%f in ( '"%CONDA%" info --root' ) do (
    set "CONDA_BASE_PREFIX=%%f"
)

if not exist "%PREFIX%\python.exe" (
    echo Creating a conda env in "%PREFIX%"
    rem # Create an empty initial skeleton to layout the conda, activate.bat
    rem # and other things needed to manage the environment.
    "%CONDA%" create --yes  --quiet --prefix "%PREFIX%"

    rem # Also install python (msvc runtime and python might be required
    rem # for any post-link scripts).
    for %%f in ( python-*.tar.bz2 ) do (
        "%CONDA%" install --yes --copy --quiet --prefix "%PREFIX%" "%CD%\%%f" ^
            || exit /b !ERRORLEVEL!
    )
)

rem # Create .condarc file that includes conda-forge channel
rem # We need it so add-ons can be installed from conda-forge
echo Appending conda-forge channel
echo channels:         > "%PREFIX%\.condarc"
echo   - conda-forge  >> "%PREFIX%\.condarc"
echo   - defaults     >> "%PREFIX%\.condarc"

rem # `conda create` (at least since 4.5) does not add the conda.bat script,
rem # so we create it manually (has different env activation pattern).
set "CONDA_BAT=%PREFIX%\Scripts\conda.bat"
if not exist "%CONDA_BAT%" (
    echo @echo off>                    "%CONDA_BAT%"
    echo call "%CONDA%" %%*>>          "%CONDA_BAT%"
)

rem # same for activate.bat
set "ACTIVATE_BAT=%PREFIX%\Scripts\activate.bat"
if not exist "%ACTIVATE_BAT%" (
    echo @echo off >  "%ACTIVATE_BAT%"
    echo call "%CONDA_BASE_PREFIX%\Scripts\activate.bat" "%PREFIX%" >> "%ACTIVATE_BAT%"
)

for %%f in ( *.tar.bz2 ) do (
    echo Installing: %%f
    "%CONDA%" install --yes  --copy --quiet --prefix "%PREFIX%" "%CD%\%%f" ^
        || exit /b !ERRORLEVEL!
)

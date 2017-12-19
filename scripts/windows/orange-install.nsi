#
# A NSIS installer script for Orange3 Windows application
#

# Required definitions need to be passed to the makensis call
#  - BASEDIR base location of all required files, ... (see below)
#  - PYARCH python architecture identifier (win32 or amd64)
#  - PY{MAJOR,MINOR,MICRO} Python version of the included installer
#  - APPNAME Application (short) name
#  - VER{MAJOR,MINOR,MICRO} Application version
#  - PYINSTALLER basename of the python installer
#  - INSTALL_REGISTRY_KEY reg subkey name to use for storing install infomation
#       (details will be stored under Software/${INSTALL_REGISTRY_KEY})


# Required data layout at compile time
# (by default BASEDIR is the directory where this .nsi script is located;
# use -NOCD parameter when invoking makensis to preserve invoker cwd)

# ${BASEDIR}/
#   wheelhouse/
#   requirements.txt


## InstallationTypes:
##    - Normal
##    - Private


# Python installation type: Normal | Private  (Env | Private)
# - Normal:
#   If a compatible Python installation is found it is used, otherwise an
#   Python installer is executed (this installer must register the
#   appropriate keys in the Windows registry).
# - Private:
#   A private copy of the Python interpreter is installed for use by the
#   application

#!include InstallConfig.nsh

!ifndef PYINSTALL_TYPE
    !error "PYINSTALL_TYPE must be defined"
!endif

!ifndef APPNAME
    !error "APPNAME must be defined"
!endif

!ifndef APPLICATIONNAME
    !define APPLICATIONNAME ${APPNAME}
!endif

!ifdef VERMAJOR & VERMINOR
    !ifdef VERMICRO
        !define APPVERSION ${VERMAJOR}.${VERMINOR}.${VERMICRO}
    !else
        !define APPVERSION ${VERMAJOR}.${VERMINOR}
    !endif
!else
    !error "VERMAJOR and VERMINOR are no defiend"
!endif

!define PYTHON_VERSION ${PYMAJOR}.${PYMINOR}.${PYMICRO}
!define PYTAG ${PYMAJOR}${PYMINOR}

!if ${PYARCH} == win32
    !define BITS 32
!else if ${PYARCH} == win_amd64
    !define BITS 64
!else
    !error "Invalid PYARCH ${PYARCH}"
!endif


!ifndef OUTFILENAME
    !define OUTFILENAME \
        ${APPNAME}-${APPVERSION}-python-${PYMAJOR}.${PYMINOR}-${PYARCH}-setup.exe
!endif


OutFile ${OUTFILENAME}
Name ${APPLICATIONNAME}-${VERSION}

!ifndef DEFAULT_INSTALL_FOLDER
    !define DEFAULT_INSTALL_FOLDER ${APPNAME}
!endif

# Wheel packages which are already compressed.
SetCompress "off"

!ifdef INSTALLERICON
    Icon "${INSTALLERICON}"
    UninstallIcon "${INSTALLERICON}"

    !define MUI_ICON ${INSTALLERICON}
    !define MUI_UNICON ${INSTALLERICON}
!endif

!ifndef BASEDIR
    !define BASEDIR .
!endif

# Application launcher shortcut name (in start menu or on the desktop)
!ifndef LAUNCHER_SHORTCUT_NAME
    !define LAUNCHER_SHORTCUT_NAME "${APPNAME}"
!endif

!ifndef INSTALL_REGISTRY_KEY
    !error 'INSTALL_REGISTRY_KEY must be defined'
!endif

# Registry key/values where the installation layout is saved
!define INSTALL_SETTINGS_KEY Software\${INSTALL_REGISTRY_KEY}
!define INSTALL_SETTINGS_INSTDIR InstallDir
!define INSTALL_SETTINGS_INSTMODE InstallMode

# Support for both current or all users installation layout
# (see MultiUser/Readme.html)
!define MULTIUSER_EXECUTIONLEVEL Highest
# By default select current user mode
!define MULTIUSER_INSTALLMODE_DEFAULT_CURRENTUSER

!if ${BITS} == 64
    # Use correct program files folder.
    # NSIS >= 3.02 (currently still unreleased)
    !define MULTIUSER_USE_PROGRAMFILES64
!endif

# Enable the Current/All Users selection page.
!define MULTIUSER_MUI

# Enable /AllUsers or /CurrentUser command line switch
# (see MultiUser/Readme.html)
!define MULTIUSER_INSTALLMODE_COMMANDLINE

# the default folder name where the application will be installed
!define MULTIUSER_INSTALLMODE_INSTDIR ${DEFAULT_INSTALL_FOLDER}

!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_KEY ${INSTALL_SETTINGS_KEY}
!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_VALUENAME ${INSTALL_SETTINGS_INSTDIR}

!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_KEY ${INSTALL_SETTINGS_KEY}
!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_VALUENAME ${INSTALL_SETTINGS_INSTMODE}

# A function which will restore the install dir passed on the command
# line (/D=DIR) if running in silent mode (MultiUser.nsh will not respect
# the default $InstDir).
!define MULTIUSER_INSTALLMODE_FUNCTION RestoreSilentInstallDir

!include Sections.nsh
!include MultiUser.nsh
!include MUI2.nsh
!include LogicLib.nsh
!include FileFunc.nsh

# Installer Pages Definitions
# ---------------------------

!insertmacro MUI_PAGE_WELCOME

!ifdef LICENSE_FILE
    !insertmacro MUI_PAGE_LICENSE "${LICENSE_FILE}"
!endif

!define MUI_STARTMENUPAGE_DEFAULTFOLDER ${APPNAME}
# Stay on InstFiles page to allow log view inspection
!define MUI_FINISHPAGE_NOAUTOCLOSE

Var StartMenuFolder

# All/Current user install selection selection page:
!insertmacro MULTIUSER_PAGE_INSTALLMODE

# Components Selection Page:
# - put the component description box at the bottom of the list (more
#   compact GUI)
!define MUI_COMPONENTSPAGE_SMALLDESC
!insertmacro MUI_PAGE_COMPONENTS

# Install Directory selection page:
# Custom function to call on leaving the Directory Page (validation)
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE "DirectoryLeave"
!insertmacro MUI_PAGE_DIRECTORY

# Start Menu Directory selection page:
# - custom start menu page pre callback function to skip the page if
#   Start Menu section is unselected
!define MUI_PAGE_CUSTOMFUNCTION_PRE "StartMenuPre"
# - no check box to enable/disable start menu creation
#   (is controled by the Components Page)
!define MUI_STARTMENUPAGE_NODISABLE
# Registry key path where the selected start folder name is stored
!define MUI_STARTMENUPAGE_REGISTRY_ROOT SHELL_CONTEXT
!define MUI_STARTMENUPAGE_REGISTRY_KEY ${INSTALL_SETTINGS_KEY}
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME StartMenuFolder
!insertmacro MUI_PAGE_STARTMENU StartMenuPageID $StartMenuFolder

# Install Files page:
!insertmacro MUI_PAGE_INSTFILES

# Finish Page:
# - run the application from the finish page
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION "LaunchApplication"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APPLICATIONNAME}"
# - add link at the bottom
!define MUI_FINISHPAGE_LINK "orange.biolab.si"
!define MUI_FINISHPAGE_LINK_LOCATION "http://orange.biolab.si"

!insertmacro MUI_PAGE_FINISH

# Uninstaller Page Definitions
# ----------------------------
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE English

# Temporary folder where temp data will be extracted
# ($PLUGINSDIR is the (supposedly) recommend place for this; deleted on exit)
!define TEMPDIR "$PLUGINSDIR\${APPNAME}-installer-data"

!include "PythonHelpers.nsh"

#
# ${ExtractTemp} Resource TargetLocation
#
#   Extract a Resource (available at compile time) to TargetLocation
#   (available at install time)
#
!macro _EXTRACT_TEMP_MACRO RESOURCE LOCATION
    SetOutPath ${LOCATION}
    File ${RESOURCE}
!macroend
!define ExtractTemp "!insertmacro _EXTRACT_TEMP_MACRO"

#
# ${ExtractTempRec} Resource TargetLocation
#
#   Extract a Resource (available at compile time) recursively to
#   TargetLocation (available at install time)
#
!macro _EXTRACT_TEMP_MACRO_REC RESOURCE LOCATION
    SetOutPath ${LOCATION}
    File /r ${RESOURCE}
!macroend
!define ExtractTempRec "!insertmacro _EXTRACT_TEMP_MACRO_REC"


# Key prefix where "Add/Remove Programs" entries are registerd
!define WINDOWS_UNINSTALL_REGKEY \
    "Software\Microsoft\Windows\CurrentVersion\Uninstall"

# Full key path for the application uninstall entry in Add/Remove Programs
!define APPLICATION_UNINSTALL_REGKEY "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}"

# Uninstaller base name
!ifndef UNINSTALL_EXEFILE
    !define UNINSTALL_EXEFILE "${APPLICATIONNAME}-Uninstall.exe"
!endif

# Preserved $InstDir as present in .onInit (i.e from /D=... command line
# switch)
Var SilentInstallDir

# The (env) base Python prefix
Var BasePythonPrefix

# The created env Python installation layout
Var PythonPrefix
Var PythonExecPrefix
Var PythonScriptsPrefix

# Python installaion mode (1 if installed for all users; 0 if installed for
# current user only
Var PythonInstallMode

Var LogFile
!macro __LOG_INIT
    !ifdef __UNINSTALL__
        FileOpen $LogFile "$TEMP\${UNINSTALL_EXEFILE}-uninstall-log.txt" "w"
    !else
        FileOpen $LogFile "$TEMP\$EXEFILE-install-log.txt" "w"
    !endif
    FileWrite $LogFile "------------------------$\r$\n"
    !ifdef __UNINSTALL__
        FileWrite $LogFile "${APPLICATIONNAME} ${VERSION} Uninstall Log$\r$\n"
    !else
        FileWrite $LogFile "${APPLICATIONNAME} ${VERSION} Install Log$\r$\n"
    !endif
    FileWrite $LogFile "------------------------$\r$\n"
!macroend

!define LogInit "!insertmacro __LOG_INIT"

# ${LogWrite}  STR
# ----------------
# Write a STR to $LogFile. The $LogFile file must already be open
# (by ${LogInit})
!macro __LOG_WRITE TEXT
    FileWrite $LogFile '${TEXT}$\r$\n'
!macroend
!define LogWrite "!insertmacro __LOG_WRITE"


# ${ExecToLog} COMMAND
# --------------------
# Log the COMMAND using ${LogWrite} and execute it using nsExec::ExecToLog
!macro __EXEC_TO_LOG COMMAND
    ${LogWrite} 'nsExec::ExecToLog: ${COMMAND}'
    SetDetailsPrint listonly
    DetailPrint 'Executing: ${COMMAND}'
    SetDetailsPrint both
    nsExec::ExecToLog '${COMMAND}'
!macroend
!define ExecToLog "!insertmacro __EXEC_TO_LOG"


# Validate the selected installation directory. Show a message and Abort
# if it is not suitable for installation.
Function DirectoryLeave
    ${LogWrite} "DirectoryLeave"
    ${LogWrite} "InstDir: $InstDir"
    Push $R0
    Push $R1
    Push $R2
    Push $R3
    ${DirState} "$InstDir" $R0
    ${LogWrite} "InstDir state: $R0"
    # Check if this looks like a location of an existing install
    ${If} $R0 == 1
        # Directory exists and is not empty
        ${If} ${FileExists} "$InstDir\${UNINSTALL_EXEFILE}"
!if ${PYINSTALL_TYPE} == Private
        ${AndIf} ${FileExists} "$InstDir\Python${PYTAG}-${BITS}\*.*"
!else if ${PYINSTALL_TYPE} == Normal
        ${AndIf} ${FileExists} "$InstDir\pyvenv.cfg"
        ${AndIfNot} ${FileExists} "$InstDir\python.exe"
            # Check that the existing env is binary compatible
            ${GetPythonVersion} "$InstDir\Scripts\python.exe" $R1 $R2
            ${GetPythonArch} "$InstDir\Scripts\python.exe" $R1 $R3
            ${If} $R1 != 0
            ${OrIf} "$R2/$R3" != "${PYMAJOR}.${PYMINOR}/${PYARCH}"
                ${LogWrite} 'Python version/architecture mismatch \
                             ($R2/$R3 != ${PYMAJOR}.${PYMINOR}/${PYARCH})'
                MessageBox MB_OK '\
                    "$InstDir" contains a pre-existing environment \
                    that is not binary compatible with this installer.$\r$\n\
                    Please choose another destination folder.'
                Abort '$R1 $R2 != "${PYMAJOR}.${PYMINOR} ${PYARCH}"'
            ${EndIf}
            ${LogWrite} 'Found existing python in $InstDir: $R2 - $R3'
!endif
            MessageBox MB_YESNO|MB_ICONQUESTION \
                '"$InstDir" looks like an existing ${APPLICATIONNAME} \
                 installation. Do you want to overwrite it?' \
                /SD IDYES IDYES continue_ IDNO abort_
            abort_:
                ${LogWrite} "Overwrite of an existing installation cancelled \
                             by user (InstDir: $InstDir)"
                Abort "Aborting"
            continue_:
                ${LogWrite} "Reusing an existing installation directory: \
                            $InstDir"
        ${Else}
            ${LogWrite} "$InstDir is not empty, aborting"
            MessageBox MB_OK '"$InstDir" exists and is not empty.$\r$\n \
                              Please choose annother destination folder.'
            Abort '"$InstDir" exists an is not empty'
        ${EndIf}
    ${EndIf}
    Pop $R2
    Pop $R1
    Pop $R0
FunctionEnd


Function RestoreSilentInstallDir
    ${If} $SilentInstallDir != ""
    ${AndIf} ${Silent}
        StrCpy $InstDir $SilentInstallDir
        ${LogWrite} "Restored InstDir to: $SilentInstallDir"
    ${EndIf}
FunctionEnd


# Section Python
# --------------
# Install Official Python distribution

!if ${PYINSTALL_TYPE} == Private
# Install/layout a Python installation inside the
# $InstDir\Python${PYTAG}-${BITS} folder.
Section "-Python ${PYTHON_VERSION} (${BITS} bit)" SectionPython
    ${If} $InstDir == ""
        Abort "Invalid installation prefix"
    ${EndIf}
    ${LogWrite} "Installing a private python installation"
    ${ExtractTemp} "${BASEDIR}\${PYINSTALLER}" "${TEMPDIR}"
    DetailPrint 'Installing a private Python ${PYTHON_VERSION} (${BITS} bit) \
                 in "$InstDir\Python${PYTAG}-${BITS}"'

    ${PyInstallNoRegister} "${TEMPDIR}\${PYINSTALLER}" \
                           "$InstDir\Python${PYTAG}-${BITS}" $0
    # msvcredist?
    ${If} $0 != 0
        Abort "Python installation failed (error value: $0)"
    ${EndIf}
    ${IfNot} ${FileExists} "$InstDir\Python${PYTAG}-${BITS}\python.exe"
        ${LogWrite} "Failed to install Python in $InstDir$\r$\n\
                     Python executable not found in: \
                     $InstDir\Python${PYTAG}-${BITS}"
        Abort "Failed to install Python in $InstDir"
    ${EndIf}
    ${GetPythonVersion} "$InstDir\Python${PYTAG}-${BITS}\python.exe" $0 $1
    ${If} $0 != 0
        Abort "Python installation failed (simple command returned an error: $0)"
    ${EndIf}
    StrCpy $BasePythonPrefix "$InstDir\Python${PYTAG}-${BITS}"
SectionEnd

Function un.Python
    # Uninstall a private copy of python
    ${If} $InstDir != ""
    ${AndIf} ${FileExists} "$InstDir\Python${PYTAG}-${BITS}\*.*"
        # Delete entire dir
        RMDir /R /REBOOTOK "$InstDir\Python${PYTAG}-${BITS}"
    ${EndIf}
FunctionEnd

!else if ${PYINSTALL_TYPE} == Normal
# Run the embeded Python installer to install the official python distribution,
# if one is not already installed. After the installation completes the
# location of the installation is queried from the windows registry.
Section "Python ${PYTHON_VERSION} (${BITS} bit)" SectionPython
    SectionIn RO
    ${If} $BasePythonPrefix != ""
        # The section should heve been unselected (disabled) in .onInit
        Abort "Installer logic error"
    ${EndIf}

    ${LogWrite} "Installing Python ${PYTHON_VERSION} (${BITS} bit)"
    ${ExtractTemp} "${BASEDIR}\${PYINSTALLER}" "${TEMPDIR}"
    DetailPrint "Executing external installer for Python \
                ${PYTHON_VERSION} (${BITS} bit)"
    # TODO: Ask for confirmation again?
    ${PyInstall} "${TEMPDIR}\${PYINSTALLER}" $MultiUser.InstallMode $0
    ${If} $0 != 0
        Abort "Python installation failed (error value: $0)"
    ${EndIf}
    ${GetPythonInstall} \
        "${PYMAJOR}.${PYMINOR}" ${BITS} $BasePythonPrefix $PythonInstallMode
    ${If} $BasePythonPrefix == ""
        Abort "Python installation failed (cannot determine Python \
               installation prefix)."
    ${EndIf}
SectionEnd

Function un.Python
    # Nothing to do
FunctionEnd
!endif

Section "-Python env setup" SectionEnvSetup
    ${If} $BasePythonPrefix == ""
        Abort "No base python configured. Cannot proceed."
    ${EndIf}
    ${LogWrite} "Setup effective python prefix"
!if ${PYINSTALL_TYPE} == Private
    ${LogWrite} "Using Python installed in $BasePythonPrefix"
    StrCpy $PythonPrefix "$BasePythonPrefix"
    StrCpy $PythonExecPrefix "$PythonPrefix"
    StrCpy $PythonScriptsPrefix "$PythonPrefix\Scripts"
!else if ${PYINSTALL_TYPE} == Normal
    ${IfNot} ${FileExists} "$InstDir\pyvenv.cfg"
        DetailPrint "Creating virtual env in $InstDir \
                     (using base $BasePythonPrefix)"
        ${LogWrite} "Creating virtaul env in $InstDir \
                     (using base $BasePythonPrefix)"
        ${ExecToLog} '"$BasePythonPrefix\python" -m venv "$InstDir"'
    ${EndIf}
    StrCpy $PythonPrefix "$InstDir"
    StrCpy $PythonExecPrefix "$PythonPrefix\Scripts"
    StrCpy $PythonScriptsPrefix "$PythonPrefix\Scripts"
!endif
    ${LogWrite} "Target environment layout:"
    ${LogWrite} "   PythonPrefix: $PythonPrefix"
    ${LogWrite} "   PythonExecPrefix: $PythonExecPrefix"
    ${LogWrite} "   PythonScriptsPrefix: $PythonScriptsPrefix"
SectionEnd

Function un.Environment
!if ${PYINSTALL_TYPE} == Normal
    ${If} $InstDir != ""
    ${AndIf} ${FileExists} "$InstDir\pyvenv.cfg"
    ${AndIf} ${FileExists} "$InstDir\${UNINSTALL_EXEFILE}"
        ${LogWrite} "Removing $InstDir"
        # TODO: Need a more specific marker
        RMDir /R /REBOOTOK "$InstDir"
    ${EndIf}
!endif
FunctionEnd


# Instal into the python environment in $PythonPrefix all the required packages
Section "Install required pacakges" InstallPackages
    SectionIn RO
    ${If} $PythonExecPrefix == ""
        Abort "No python executable configured. Cannot proceed."
    ${EndIf}
    ${ExtractTemp} "${BASEDIR}\requirements.txt" "${TEMPDIR}"
    ${ExtractTempRec} "${BASEDIR}\wheelhouse\*.*" "${TEMPDIR}\wheelhouse"

    # Install into PythonPrefix
    ${LogWrite} "Installing packages into $PythonPrefix"
    DetailPrint "Installing required packages"
    ${ExecToLog} '"$PythonExecPrefix\python" -m ensurepip'
    # First update pip to at least the bundled version (>=9)
    ${ExecToLog} '\
        "$PythonExecPrefix\python" -m pip install --upgrade \
             --isolated --no-cache-dir --no-index \
             --find-links "${TEMPDIR}\wheelhouse" \
             pip>=9 \
            '
    Pop $0
    ${If} $0 != 0
        ${LogWrite} "pip install exited with: $0)"
        Abort "Failed to install required packages (exit status $0)"
    ${EndIf}

    ${ExecToLog} '\
        "$PythonExecPrefix\python" -m pip install \
             --isolated --no-cache-dir --no-index \
             --find-links "${TEMPDIR}\wheelhouse" \
             -r "${TEMPDIR}\requirements.txt" \
        '
    Pop $0
    ${If} $0 != 0
        ${LogWrite} "pip install exited with: $0)"
        Abort "Failed to install required packages (exit status $0)"
    ${EndIf}
SectionEnd


Function un.InstallPackages
FunctionEnd


Section -Icons
    # Layout icons if necessary (are not present)
    ${IfNot} ${FileExists} $PythonPrefix\share\orange3\icons\*.ico"
        ${ExtractTempRec} "${BASEDIR}\icons\*.ico" "${TEMPDIR}\icons"
        CreateDirectory "$PythonPrefix\share\orange3\icons"
        CopyFiles /SILENT "${TEMPDIR}\icons\*.ico" \
                          "$PythonPrefix\share\orange3\icons"
    ${EndIf}
SectionEnd


# Create utility shortcut launchers in the $InstDir
Section -Launchers
    SetOutPath "$InstDir"
    DetailPrint "Creating launcher shortcuts"
    # Startup shortcut
    CreateShortCut \
        "$InstDir\${LAUNCHER_SHORTCUT_NAME}.lnk" \
        "$PythonExecPrefix\pythonw.exe" "-m Orange.canvas" \
        "$PythonPrefix\share\orange3\icons\orange.ico" 0
    # Utility shortcut to launch the application with max log level attached
    # to the console that remains visible after exit
    CreateShortCut \
        "$InstDir\${LAUNCHER_SHORTCUT_NAME} Debug.lnk" \
        "%COMSPEC%" '/K "$PythonExecPrefix\python.exe" -m Orange.canvas -l4' \
        "$PythonPrefix\share\orange3\icons\orange.ico" 0
!if ${PYINSTALL_TYPE} == Normal
    # A utility shortcut for activating the environment
    CreateShortCut \
        "$InstDir\${APPNAME} Command Prompt.lnk" \
        "%COMSPEC%" '/K "$PythonScriptsPrefix\activate.bat"'
!endif

SectionEnd


Function un.Launchers
    Delete "$InstDir\${LAUNCHER_SHORTCUT_NAME}.lnk"
    Delete "$InstDir\${LAUNCHER_SHORTCUT_NAME} Debug.lnk"
!if ${PYINSTALL_TYPE} == Normal
    Delete "$InstDir\${APPNAME} Command Prompt.lnk"
!endif
FunctionEnd


SectionGroup "Shortcuts" SectionShortcuts
Section "Start Menu Shortcuts" SectionStartMenu
    !insertmacro MUI_STARTMENU_WRITE_BEGIN StartMenuPageID
    DetailPrint "Creating Start Menu Shortcuts"
    ${If} $StartMenuFolder != ""
        ${LogWrite} "Creating shortcuts in $SMPROGRAMS\$StartMenuFolder"
        # Output path is used as the default CWD for the created shortcuts
        SetDetailsPrint none
        SetOutPath "%HOMEDRIVE%\%HOMEPATH%"
        SetDetailsPrint both
        CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
        CreateShortCut \
            "$SMPROGRAMS\$StartMenuFolder\${LAUNCHER_SHORTCUT_NAME}.lnk" \
            "$PythonExecPrefix\pythonw.exe" "-m Orange.canvas" \
            "$PythonPrefix\share\orange3\icons\orange.ico" 0
!if ${PYINSTALL_TYPE} == Normal
        # A utility shortcut for activating the environment
        CreateShortCut \
            "$SMPROGRAMS\$StartMenuFolder\${APPNAME} Command Prompt.lnk" \
            "%COMSPEC%" '/K "$PythonScriptsPrefix\activate.bat"'
!endif
    ${EndIf}
    !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

# StartMenuPage 'pre' callback. Skip Start Menu directory selection if
# the SectionStarMenu is not selected on the Components page
Function StartMenuPre
    ${IfNot} ${SectionIsSelected} ${SectionStartMenu}
        ${LogWrite} "Skiping start menu page because it is not selected"
        Abort
    ${EndIf}
FunctionEnd

Section "Desktop Shortcuts" SectionDesktop
    DetailPrint "Installing Desktop shortcurt"
    CreateShortCut \
        "$DESKTOP\${LAUNCHER_SHORTCUT_NAME}.lnk" \
        "$PythonExecPrefix\pythonw.exe" "-m Orange.canvas" \
        "$PythonPrefix\share\orange3\icons\orange.ico" 0
SectionEnd

SectionGroupEnd

Function un.Shortcuts
    !insertmacro MUI_STARTMENU_GETFOLDER StartMenuPageID $0
    ${If} $0 != ""
    ${AndIf} ${FileExists} "$SMPROGRAMS\$0"
        ${LogWrite} "Removing Start Menu Shortcuts (from $SMPROGRAMS\$0)"
        DetailPrint "Removing Start Menu shortcuts"
        Delete "$SMPROGRAMS\$0\${LAUNCHER_SHORTCUT_NAME}.lnk"
!if ${PYINSTALL_TYPE} == Normal
        Delete "$SMPROGRAMS\$0\${APPNAME} Command Prompt.lnk"
!endif
        RMDir "$SMPROGRAMS\$0"
    ${EndIf}
    ${LogWrite} "Removing Desktop shortcurt"
    DetailPrint "Removing Desktop shortcurt"
    Delete "$DESKTOP\${LAUNCHER_SHORTCUT_NAME}.lnk"
FunctionEnd


# Should this section be made selectable by the user. Would allow multiple
# installations.
Section -Register SectionRegister
    DetailPrint "Writing to registry"
    ${LogWrite} 'Register installation layout (${INSTALL_SETTINGS_KEY})'
    ${LogWrite} '    BasePythonPrefix "$BasePythonPrefix"'
    WriteRegStr SHELL_CONTEXT \
                ${INSTALL_SETTINGS_KEY} BasePythonPrefix "$BasePythonPrefix"
    ${LogWrite} '    PythonPrefix "$PythonPrefix"'
    WriteRegStr SHELL_CONTEXT \
                ${INSTALL_SETTINGS_KEY} PythonPrefix "$PythonPrefix"
    ${LogWrite} '    ${INSTALL_SETTINGS_INSTDIR} "$InstDir"'
    WriteRegStr SHELL_CONTEXT \
                ${INSTALL_SETTINGS_KEY} ${INSTALL_SETTINGS_INSTDIR} \
                "$InstDir"

    WriteRegStr SHELL_CONTEXT \
                ${INSTALL_SETTINGS_KEY} ${INSTALL_SETTINGS_INSTMODE} \
                $MultiUser.InstallMode
    ${LogWrite} '    InstallType "${PYINSTALL_TYPE}"'
    WriteRegStr SHELL_CONTEXT \
                ${INSTALL_SETTINGS_KEY} InstallType ${PYINSTALL_TYPE}

    ${LogWrite} "Register .ows filetype"
    WriteRegStr SHELL_CONTEXT \
        "Software\Classes\.ows" "" "OrangeCanvas"
    WriteRegStr SHELL_CONTEXT \
        "Software\Classes\OrangeCanvas" "" "Orange Workflow"
    WriteRegStr SHELL_CONTEXT \
        "Software\Classes\OrangeCanvas\DefaultIcon" "" \
        "$PythonPrefix\share\orange3\icons\OrangeOWS.ico"
    WriteRegStr SHELL_CONTEXT \
        "Software\Classes\OrangeCanvas\Shell\Open\Command\" "" \
        '"$PythonExecPrefix\pythonw.exe" -m Orange.canvas "%1"'

    WriteUninstaller "$InstDir\${UNINSTALL_EXEFILE}"

    # Register uninstaller in Add/Remove Programs

    ${LogWrite} "Register uninstaller (${WINDOWS_UNINSTALL_REGKEY}\${APPNAME})"

    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                DisplayName "${APPNAME} ${APPVERSION} (${BITS} bit)"
    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                DisplayVersion "${APPVERSION}"
    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                DisplayIcon "$InstDir\${UNINSTALL_EXEFILE}"
    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                "UninstallString" \
                '"$InstDir\${UNINSTALL_EXEFILE}" /$MultiUser.InstallMode'
    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                "QuietUninstallString" \
                '"$InstDir\${UNINSTALL_EXEFILE}" /$MultiUser.InstallMode /S'
    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                InstallLocation "$InstDir"
    WriteRegStr SHELL_CONTEXT \
                "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                URLInfoAbout http://orange.biolab.si

    WriteRegDWORD SHELL_CONTEXT \
                  "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                   NoModify 1
    WriteRegDWORD SHELL_CONTEXT \
                  "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" \
                  NoRepair 1
SectionEnd


Var un.InstallDir

Function un.Register
    DetailPrint "Unregister"
    ReadRegStr $un.InstallDir SHCTX ${INSTALL_SETTINGS_KEY} InstallDir
    ${If} $un.InstallDir != ""
    ${AndIf} $un.InstallDir == $InstDir
        ${LogWrite} "Deleting reg key: ${INSTALL_SETTINGS_KEY}"
        DeleteRegKey SHCTX "${INSTALL_SETTINGS_KEY}"
        ${LogWrite} "Deleting reg key: Software\Classes\OrangeCanvas"
        DeleteRegKey SHCTX Software\Classes\OrangeCanvas
    ${Else}
        ${LogWrite} "InstallDir from ${INSTALL_SETTINGS_KEY} does not match \
                    InstDir ($un.InstallDir != $InstDir). Leaving it."
    ${EndIf}

    ReadRegStr $un.InstallDir SHCTX \
               "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}" InstallLocation
    ${If} $un.InstallDir != ""
    ${AndIf} $un.InstallDir == $InstDir
        ${LogWrite} "Deleting reg key: ${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}"
        DeleteRegKey SHCTX "${WINDOWS_UNINSTALL_REGKEY}\${APPNAME}"
    ${Else}
        ${LogWrite} "InstallLocation from \
                     ${WINDOWS_UNINSTALL_REGKEY}\${APPNAME} does not match \
                     InstDir ($0 != $InstDir). Leaving it."
    ${EndIf}
FunctionEnd

Function LaunchApplication
    ExecShell "open" "$PythonExecPrefix\pythonw.exe" "-m Orange.canvas"
FunctionEnd


!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN

!insertmacro MUI_DESCRIPTION_TEXT ${SectionPython} \
    "Install Python ${PYTHON_VERSION} (${BITS} bit)"
!insertmacro MUI_DESCRIPTION_TEXT ${InstallPackages} \
    "Install required packages into the destination environment"
!insertmacro MUI_DESCRIPTION_TEXT ${SectionStartMenu} \
    "Install Start Menu shortcuts"
!insertmacro MUI_DESCRIPTION_TEXT ${SectionDesktop} \
    "Install Desktop shortcuts"

!insertmacro MUI_FUNCTION_DESCRIPTION_END


Section Uninstall
    ReadRegStr $BasePythonPrefix SHCTX ${INSTALL_SETTINGS_KEY} BasePythonPrefix
    ReadRegStr $PythonPrefix SHCTX ${INSTALL_SETTINGS_KEY} PythonPrefix
    ReadRegStr $un.InstallDir SHCTX ${INSTALL_SETTINGS_KEY} InstallDir

    ${LogWrite} "InstDir: $InstDir"
    ${LogWrite} "Layout from registry:"
    ${LogWrite} "    InstallDir: $un.InstallDir"
    ${LogWrite} "    PythonPrefix: $PythonPrefix"
    ${LogWrite} "    BasePythonPrefix: $BasePythonPrefix"

    Call un.Shortcuts
    Call un.Register
    Call un.Launchers
    Call un.InstallPackages
    Call un.Environment
    Call un.Python

    ${If} ${FileExists} "$InstDir\${UNINSTALL_EXEFILE}"
        Delete "$InstDir\${UNINSTALL_EXEFILE}"
    ${EndIf}
SectionEnd


Function .onInit
    ${LogInit}
    InitPluginsDir
    # $InstDir is not empty if specified by the /D command line switch.
    # Store it because MultiUser.nsh will override it with its own either
    # in MULTIUSER_INIT or MULTIUSER_PAGE_INSTALLMODE.
    ${If} $InstDir != ""
    ${AndIf} ${Silent}
        StrCpy $SilentInstallDir $InstDir
        ${LogWrite} "SilentInstallDir: $SilentInstallDir"
    ${EndIf}
    ${LogWrite} "Setting ${BITS} bit registry view."
    SetRegView ${BITS}

    # Initialize MultiUser.nsh
    !insertmacro MULTIUSER_INIT

!if ${PYINSTALL_TYPE} == Normal
    ${GetPythonInstall} ${PYMAJOR}.${PYMINOR} ${BITS} \
        $BasePythonPrefix $PythonInstallMode
    ${LogWrite} "Python Prefix: $BasePythonPrefix"
    ${LogWrite} "Python Install Type: $PythonInstallMode"
    ${If} $BasePythonPrefix != ""
        ${GetPythonVersion} "$BasePythonPrefix\python.exe" $0 $1
        ${LogWrite} "Python Version: $1"
        ${GetPythonArch} "$BasePythonPrefix\python.exe" $0 $1
        ${LogWrite} "Python platform: $1"

        # Found an appropriate python installation and can reuse it
        # Change the SectionPython to Unselected
        SectionGetText ${SectionPython} $0
        SectionSetText ${SectionPython} "$0 - Already installed"
        !insertmacro UnselectSection ${SectionPython}
        !insertmacro SetSectionFlag ${SectionPython} ${SF_RO}
    ${Else}
        !insertmacro SelectSection ${SectionPython}
        !insertmacro SetSectionFlag ${SectionPython} ${SF_RO}
    ${EndIf}
!endif
FunctionEnd


Function un.onInit
    ${LogInit}
    ${LogWrite} "Setting ${BITS} bit registry view."
    SetRegView ${BITS}
    !insertmacro MULTIUSER_UNINIT
FunctionEnd

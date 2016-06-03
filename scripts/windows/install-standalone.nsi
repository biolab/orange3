#
# Installer script for a stand-alone Orange3 installation
#

# Required definitions need to be passed to the makensis call
#  - BASEDIR base location of all required binaries, ... (see below)
#  - PYTHON_VERSION (major.minor.micro) python version e.g 3.4.2
#  - PYTHON_VERSION_SHORT (major.minor) python version e.g 3.4
#  - PYVER short (majorminor) python version e.g 34
#  - ARCH python architecture identifier (win32 or amd64)

# Required data layout at compile time
# (BASEDIR must be passed with compiler flags)
#
# ${BASEDIR}/
#   core/
#     python/
#     msvredist/
#   wheelhouse/
#       [sse-flags]/
#   startupscripts/
#   requirements.txt

OutFile ${OUTFILENAME}

Name Orange3
Icon OrangeInstall.ico
UninstallIcon OrangeInstall.ico

# Registry key/values where the installation layout is saved
!define INSTALL_SETTINGS_KEY Software\OrangeCanvas\Standalone\Current
!define INSTALL_SETTINGS_INSTDIR InstallDir
!define INSTALL_SETTINGS_INSTMODE InstallMode

!define MUI_ICON OrangeInstall.ico
!define MUI_UNICON OrangeInstall.ico

# Support for both current or all users installation layout
# (see MultiUser/Readme.html)
!define MULTIUSER_EXECUTIONLEVEL Highest

# Enable the Current/All Users selection page.
!define MULTIUSER_MUI

# Enable /AllUsers or /CurrentUser command line switch (see MultiUser/Readme.html)
!define MULTIUSER_INSTALLMODE_COMMANDLINE

# the default folder name where Orange will be installed
!define MULTIUSER_INSTALLMODE_INSTDIR "Orange3"

!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_KEY ${INSTALL_SETTINGS_KEY}
!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_VALUENAME ${INSTALL_SETTINGS_INSTDIR}

!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_KEY ${INSTALL_SETTINGS_KEY}
!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_VALUENAME ${INSTALL_SETTINGS_INSTMODE}

# A function which will restore the install dir passed on the command
# line (/D=DIR) if running in silent mode (MultiUser.nsh will not respect
# the default $InstDir).
!define MULTIUSER_INSTALLMODE_FUNCTION RestoreSilentInstDir

!include MultiUser.nsh
!include MUI2.nsh
!include LogicLib.nsh

!insertmacro MULTIUSER_PAGE_INSTALLMODE
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE English


# Temporary folder where temp data will be extracted
!define TEMPDIR $TEMP\orange-installer

!include "install-common.nsi"

!define SHELLFOLDERS \
  "Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"

!define UNINSTALLER "$INSTDIR\uninstall.exe"

Var SILENTINSTDIR

Function .onInit
	# INSTDIR is not empty if specified by the /D command line switch.
	# Store it because MultiUser.nsh will override it with it's own either
	# in MULTIUSER_INIT or MULTIUSER_PAGE_INSTALLMODE.
	${If} $INSTDIR != ""
	${AndIf} ${Silent}
		StrCpy $SILENTINSTDIR $INSTDIR
	${EndIf}

	# Initialize MultiUser
	!insertmacro MULTIUSER_INIT
	# Initialize SSE global variable
	${InitSSE}
FunctionEnd


Function un.onInit
	!insertmacro MULTIUSER_UNINIT
FunctionEnd

Function RestoreSilentInstDir
	${If} $SILENTINSTDIR != ""
	${AndIf} ${Silent}
		StrCpy $INSTDIR $SILENTINSTDIR
	${EndIf}
FunctionEnd


Section "Main"

	DetailPrint "Extracting installers"
	${ExtractTemp} "${BASEDIR}\core\python\python-${PYTHON_VERSION}.msi" "${TEMPDIR}\core\python"

	# Install python inside the application directory
	StrCpy $PythonDir "$INSTDIR\Python${PYVER}"

	DetailPrint "Installing Python"
	${InstallPythonStandalone} \
		"${TEMPDIR}\core\python\python-${PYTHON_VERSION}.msi" \
		"$PythonDir"

	# Install/copy Microsoft Visual Studio redistributable dll in python's
	# libs dir.
	${ExtractTempRec} "${BASEDIR}\core\msvredist\*.*" "${TEMPDIR}\core\msvredist"
	CopyFiles "${TEMPDIR}\core\msvredist\*.dll" "$PythonDir"

	# Ensure pip is installed in case of pre-existing python install
	${PythonExec} "-m ensurepip"

	${ExtractTempRec} "${BASEDIR}\wheelhouse\*.*" ${TEMPDIR}\wheelhouse\

	${ExtractTemp} "${BASEDIR}\requirements.txt" ${TEMPDIR}\

	# Update pip to minimum required version (pip>=6)
	DetailPrint "Updating pip"
	${PythonExec} '-m pip install --no-index -f "${TEMPDIR}\wheelhouse" -U pip'

	DetailPrint "Installing scipy stack ($SSE)"
	${Pip} 'install --no-deps --no-index \
			-f "${TEMPDIR}\wheelhouse" \
			-f "${TEMPDIR}\wheelhouse\$SSE" numpy scipy'
	Pop $0
	${If} $0 != 0
		Abort "Could not install scipy stack"
	${EndIf}

	# Install other packages.
	# Note we also add the numpy/scipy --find-links path.
	DetailPrint "Installing required packages"
	${Pip} 'install --no-index \
			-f "${TEMPDIR}\wheelhouse" \
			-f "${TEMPDIR}\wheelhouse\$SSE" \
			-r "${TEMPDIR}\requirements.txt'
	Pop $0
	${If} $0 != 0
		Abort "Could not install all requirements"
	${EndIf}

	DetailPrint "Installing PyQt4"
	${Pip} 'install --no-deps --no-index \
			-f ${TEMPDIR}\wheelhouse \
			PyQt4'

	DetailPrint "Installing Orange"
	${Pip} 'install --no-deps --no-index \
			-f "${TEMPDIR}\wheelhouse" Orange3'

	Pop $0
	${If} $0 != 0
		Abort "Could not install Orange"
	${EndIf}

	CreateDirectory "$PythonDir\share\Orange\canvas\icons"
	SetOutPath "$PythonDir\share\Orange\canvas\icons"

	File orange.ico
	File OrangeOWS.ico

	DetailPrint "Creating shortcuts"

	SetOutPath "$INSTDIR"
	File "${BASEDIR}\startupscripts\*.bat"

	# $OUTDIR is used to set the working directory for the shortcuts
	# created using CreateShortCut (it needs to be PythonDir)
	SetOutPath "$PythonDir"

	# Create shortcut at the root install directory
	CreateShortCut "$INSTDIR\Orange Canvas.lnk" \
					"$PythonDir\pythonw.exe" "-m Orange.canvas" \
					"$PythonDir\share\Orange\canvas\icons\orange.ico" 0

	# Start Menu
	CreateDirectory "$SMPROGRAMS\Orange3"
	CreateShortCut "$SMPROGRAMS\Orange3\Orange Canvas.lnk" \
					"$PythonDir\pythonw.exe" "-m Orange.canvas" \
					"$PythonDir\share\Orange\canvas\icons\orange.ico" 0

	CreateShortCut "$SMPROGRAMS\Orange3\Uninstall Orange.lnk" ${UNINSTALLER}

	# Desktop shortcut
	CreateShortCut "$DESKTOP\Orange Canvas.lnk" \
					"$PythonDir\pythonw.exe" "-m Orange.canvas" \
					"$PythonDir\share\Orange\canvas\icons\orange.ico" 0

	WriteRegStr SHELL_CONTEXT \
				${INSTALL_SETTINGS_KEY} ${INSTALL_SETTINGS_INSTDIR} \
				"$INSTDIR"

	WriteRegStr SHELL_CONTEXT \
				${INSTALL_SETTINGS_KEY} "PythonDir" \
				"$PythonDir"

	WriteRegStr SHELL_CONTEXT \
				${INSTALL_SETTINGS_KEY} ${INSTALL_SETTINGS_INSTMODE} \
				$MultiUser.Privileges

	WriteRegStr SHELL_CONTEXT "Software\Classes\.ows" "" "OrangeCanvas"
	WriteRegStr SHELL_CONTEXT "Software\Classes\OrangeCanvas\DefaultIcon" "" "$PythonDir\share\Orange\canvas\icons\OrangeOWS.ico"
	WriteRegStr SHELL_CONTEXT "Software\Classes\OrangeCanvas\Shell\Open\Command\" "" '$PythonDir\python.exe -m Orange.canvas "%1"'

	WriteUninstaller ${UNINSTALLER}

	WriteRegStr SHELL_CONTEXT \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3" \
				"DisplayName" "Orange3 (remove only)"

	WriteRegStr SHELL_CONTEXT \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3" \
				"UninstallString" ${UNINSTALLER}

	DetailPrint "Cleanup"
	RmDir /R ${TEMPDIR}

SectionEnd


Section Uninstall

	ReadRegStr $PythonDir SHELL_CONTEXT ${INSTALL_SETTINGS_KEY} PythonDir

	${If} ${FileExists} "$PythonDir\python.exe"
		RmDir /R $PythonDir
	${EndIf}

	ReadRegStr $0 SHELL_CONTEXT ${INSTALL_SETTINGS_KEY} ${INSTALL_SETTINGS_INSTDIR}

	${If} ${FileExists} "$0"
		Delete "$0\*.bat"
		Delete "$0\Orange Canvas.lnk"
		Delete "$0\uninstall.exe"
		RmDir "$0"
	${EndIf}

	RmDir /R "$SMPROGRAMS\Orange3"

	# Remove application settings folder (?)
	ReadRegStr $0 HKCU "${SHELLFOLDERS}" AppData
	${If} $0 != ""
		ReadRegStr $0 HKLM "${SHELLFOLDERS}" "Common AppData"
	${Endif}

	${If} "$0" != ""
	${AndIf} ${FileExists} "$0\Orange3"
		RmDir /R "$0\Orange3"
	${EndIf}

	DeleteRegKey SHELL_CONTEXT "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3"

	Delete "$DESKTOP\Orange Canvas.lnk"

	DeleteRegKey SHELL_CONTEXT "Software\Classes\.ows"
	DeleteRegKey SHELL_CONTEXT "Software\Classes\OrangeCanvas"

	DeleteRegKey SHELL_CONTEXT ${INSTALL_SETTINGS_KEY}

SectionEnd

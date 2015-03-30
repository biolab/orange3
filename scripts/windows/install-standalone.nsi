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

# Default installation directory
InstallDir $PROGRAMFILES\Orange3\

# Ask the user for a target install dir.
Page directory
DirText "Choose a folder in which to install Orange3"
Page instfiles


AutoCloseWindow false

# Temporary folder where temp data is extracted
!define TEMPDIR $TEMP\orange-installer

!include "LogicLib.nsh"

!include "install-common.nsi"

!define SHELLFOLDERS \
  "Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"


Function .onInit
	# Initialize AdminInstall and PythonDir global variables.
	${InitAdminInstall}
	${InitSSE}
FunctionEnd

Function .onInstSuccess
	MessageBox MB_OK "Orange3 has been successfully installed." /SD IDOK
FunctionEnd


Section ""

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

	DetailPrint "Installing scipy stack ($SSE)"
	${Pip} 'install --no-deps --no-index \
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
			-f "${TEMPDIR}\wheelhouse" Orange'

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

	CreateShortCut "$SMPROGRAMS\Orange3\Uninstall Orange.lnk" \
					"$PythonDir\share\Orange\canvas\uninst.exe"

	# Desktop shortcut
	CreateShortCut "$DESKTOP\Orange Canvas.lnk" \
					"$PythonDir\pythonw.exe" "-m Orange.canvas" \
					"$PythonDir\share\Orange\canvas\icons\orange.ico" 0


	WriteRegStr SHELL_CONTEXT \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3" \
				"DisplayName" "Orange3 (remove only)"

	WriteRegStr SHELL_CONTEXT \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3" \
				"UninstallString" '$PythonDir\share\Orange\canvas\uninst.exe'

	WriteRegStr SHELL_CONTEXT \
				"Software\OrangeCanvas\Standalone\Current" "InstallDir" \
				"$INSTDIR"

	WriteRegStr SHELL_CONTEXT \
				"Software\OrangeCanvas\Standalone\Current" "PythonDir" \
				"$PythonDir"

	WriteRegStr HKEY_CLASSES_ROOT ".ows" "" "OrangeCanvas"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\DefaultIcon" "" "$PythonDir\share\Orange\canvas\icons\OrangeOWS.ico"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\Shell\Open\Command\" "" '$PythonDir\python.exe -m Orange.canvas "%1"'

	WriteUninstaller "$PythonDir\share\Orange\canvas\uninst.exe"

	DetailPrint "Cleanup"
	RmDir /R ${TEMPDIR}
SectionEnd


Section Uninstall

	${If} $AdminInstall = 0
	    SetShellVarContext all
	${Else}
	    SetShellVarContext current
	${EndIf}

	MessageBox MB_YESNO "Are you sure you want to remove Orange?" /SD IDYES IDNO abort

	ReadRegStr $PythonDir SHELL_CONTEXT Software\OrangeCanvas\Standalone\Current PythonDir

	${If} ${FileExists} "$PythonDir\python.exe"
		RmDir /R $PythonDir
	${EndIf}

	ReadRegStr $0 SHELL_CONTEXT Software\OrangeCanvas\Standalone\Current InstallDir

	${If} ${FileExists} "$0"
		Delete "$0\*.bat"
		Delete "$0\Orange Canvas.lnk"
		RmDir "$0"
	${EndIf}

	RmDir /R "$SMPROGRAMS\Orange3"

	# Remove application settings folder
	ReadRegStr $0 HKCU "${SHELLFOLDERS}" AppData
	${If} $0 != ""
		ReadRegStr $0 HKLM "${SHELLFOLDERS}" "Common AppData"
	${Endif}

	${If} "$0" != ""
	${AndIf} ${FileExists} "$0\Orange3"
		RmDir /R "$0\Orange3"
	${EndIf}

	${If} $AdminInstall == 1
		DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3"
	${Else}
		DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3"
	${Endif}

	Delete "$DESKTOP\Orange Canvas.lnk"

	DeleteRegKey HKEY_CLASSES_ROOT ".ows"
	DeleteRegKey HKEY_CLASSES_ROOT "OrangeCanvas"

	DeleteRegKey SHELL_CONTEXT Software\OrangeCanvas\Standalone\Current

	MessageBox MB_OK "Orange has been succesfully removed from your system." /SD IDOK

  abort:


SectionEnd

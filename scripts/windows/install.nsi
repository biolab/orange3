#
# Standard Orange installer
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
#   requirements.txt

Name "Orange3"
Icon OrangeInstall.ico
UninstallIcon OrangeInstall.ico

# ShowInstDetails nevershow

AutoCloseWindow false

OutFile ${OUTFILENAME}

#
# Temporary folder where temp data is extracted
#
!define TEMPDIR $TEMP\orange-installer

!include "LogicLib.nsh"
!include "install-common.nsi"

!define SHELLFOLDERS \
  "Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"


Function .onInit
	# Initialize AdminInstall and PythonDir global variables.
	${InitAdminInstall}
	${InitPythonDir}
	${InitSSE}
FunctionEnd


Function .onInstSuccess
	MessageBox MB_OK "Orange3 has been successfully installed." /SD IDOK
FunctionEnd


Section ""
	# First install python if not already installed
	${If} $PythonDir == ""
		askpython:

		MessageBox MB_OKCANCEL \
			"Orange installer will first launch installation of Python ${PYTHON_VERSION}." \
			/SD IDOK \
			IDOK installpython \
			IDCANCEL askpythonretry

		askpythonretry:

		MessageBox MB_YESNO "Orange cannot run without Python.$\r$\nAbort the installation?" \
			IDNO askpython
		Quit

		installpython:

		DetailPrint "Extracting installers"
		${ExtractTemp} "${BASEDIR}\core\python\python-${PYTHON_VERSION}.msi" "${TEMPDIR}\core\python"

		DetailPrint "Installing Python"
		${InstallPython} "${TEMPDIR}\core\python\python-${PYTHON_VERSION}.msi"

		# Get the location of the interpreter from registry
		${InitPythonDir}

		${If} $PythonDir == ""
			MessageBox MB_OK "Python installation failed.$\r$\nOrange installation cannot continue."
			Quit
		${EndIf}

	${EndIf}

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

	${IfNot} ${FileExists} $PythonDir\Lib\site-packages\PyQt4\QtCore.pyd
		DetailPrint "Installing PyQt4"
		${Pip} 'install --no-deps --no-index \
				-f ${TEMPDIR}\wheelhouse \
				PyQt4'
	${EndIf}

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

	# $OUTDIR is set as working directory for the shortcuts
	SetOutPath $PythonDir

	CreateDirectory "$SMPROGRAMS\Orange3"

	CreateShortCut "$SMPROGRAMS\Orange3\Orange Canvas.lnk" \
					"$PythonDir\pythonw.exe" "-m Orange.canvas" \
					"$PythonDir\share\Orange\canvas\icons\orange.ico" 0

	CreateShortCut "$DESKTOP\Orange Canvas.lnk" \
					"$PythonDir\pythonw.exe" "-m Orange.canvas" \
					"$PythonDir\share\Orange\canvas\icons\orange.ico" 0

	CreateShortCut "$SMPROGRAMS\Orange3\Uninstall Orange.lnk" \
					"$PythonDir\share\Orange\canvas\uninst.exe"

	WriteRegStr SHELL_CONTEXT \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3" \
				"DisplayName" "Orange3 (remove only)"

	WriteRegStr SHELL_CONTEXT \
				"Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange3" \
				"UninstallString" '$PythonDir\share\Orange\canvas\uninst.exe'

	WriteRegStr SHELL_CONTEXT \
				"Software\OrangeCanvas\Current" "PythonDir" \
				"$PythonDir"

	WriteRegStr HKEY_CLASSES_ROOT ".ows" "" "OrangeCanvas"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\DefaultIcon" "" "$PythonDir\share\Orange\canvas\icons\OrangeOWS.ico"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\Shell\Open\Command\" "" '$PythonDir\python.exe -m Orange.canvas "%1"'

	WriteUninstaller "$PythonDir\share\Orange\canvas\uninst.exe"

	DetailPrint "Cleanup"
	RmDir /R ${TEMPDIR}

SectionEnd


Section Uninstall
	MessageBox MB_YESNO "Are you sure you want to remove Orange?" /SD IDYES IDNO abort

	${If} $AdminInstall = 0
	    SetShellVarContext all
	${Else}
	    SetShellVarContext current
	${EndIf}

	ReadRegStr $PythonDir SHELL_CONTEXT Software\OrangeCanvas\Current "PythonDir"

	${PythonExec} "-m pip uninstall -y Orange3"

	RmDir /R $PythonDir\share\Orange

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
		DeleteRegKey HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Orange3"
	${Else}
		DeleteRegKey HKCU "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Orange3"
	${Endif}

	Delete "$DESKTOP\Orange Canvas.lnk"

	DeleteRegKey HKEY_CLASSES_ROOT ".ows"
	DeleteRegKey HKEY_CLASSES_ROOT "OrangeCanvas"

	MessageBox MB_OK "Orange has been succesfully removed from your system." /SD IDOK

  abort:

SectionEnd
#
# Common install macros
#

# 1 if this is an admin install 0 otherwise
Var AdminInstall

# Directory of an existing Python installation if/when available
Var PythonDir

# Cpu's SSE capabilities (nosse|sse2|sse3)
Var SSE


#
# ${InitAdminInstall}
#
#     Initialize the $AdminInstall variable
#
!macro GET_ACCOUNT_TYPE
	StrCpy $AdminInstall 1
	UserInfo::GetAccountType
	Pop $1
	SetShellVarContext all
	${If} $1 != "Admin"
		SetShellVarContext current
		StrCpy $AdminInstall 0
	${Else}
		SetShellVarContext all
		StrCpy $AdminInstall 1
	${EndIf}
!macroend
!define InitAdminInstall "!insertmacro GET_ACCOUNT_TYPE"

#
# ${InitPythonDir}
#
#     Initialize Python installation directory ($PythonDir variable)
#
!macro GET_PYTHON_DIR

    ${If} $AdminInstall == 0
	    ReadRegStr $PythonDir HKCU Software\Python\PythonCore\${PYTHON_VERSION_SHORT}\InstallPath ""
		StrCmp $PythonDir "" 0 trim_backslash
		ReadRegStr $PythonDir HKLM Software\Python\PythonCore\${PYTHON_VERSION_SHORT}\InstallPath ""
		StrCmp $PythonDir "" return
		MessageBox MB_OK "Please ask the administrator to install Orange$\r$\n(this is because Python was installed by him, too)."
		Quit
	${Else}
	    ReadRegStr $PythonDir HKLM Software\Python\PythonCore\${PYTHON_VERSION_SHORT}\InstallPath ""
		StrCmp $PythonDir "" 0 trim_backslash
		ReadRegStr $PythonDir HKCU Software\Python\PythonCore\${PYTHON_VERSION_SHORT}\InstallPath ""
		StrCmp $PythonDir "" return
		StrCpy $AdminInstall 0
	${EndIf}

	trim_backslash:
	StrCpy $0 $PythonDir "" -1
    ${If} $0 == "\"
        StrLen $0 $PythonDir
        IntOp $0 $0 - 1
        StrCpy $PythonDir $PythonDir $0 0
    ${EndIf}

	return:
!macroend
!define InitPythonDir "!insertmacro GET_PYTHON_DIR"

#
# ${InitSSE}
#
# Initialize the SSE global variable with nosse|sse2|sse3 string
# depending on the executing Cpu sse support
# (requires CpuCaps.dll nsis plugin)
#
!define InitSSE "!insertmacro __INIT_SSE"
!macro __INIT_SSE
	!ifdef NSIS_PLUGINS_PATH
		!addplugindir ${NSIS_PLUGINS_PATH}
	!endif
	Push $0

	CpuCaps::hasSSE3
	Pop $0
	${If} $0 == "Y"
		StrCpy $SSE "sse3"
	${Else}
		CpuCaps::hasSSE2
		Pop $0
		${If} $0 == "Y"
			StrCpy $SSE "sse2"
		${Else}
			StrCpy $SSE "nosse"
		${EndIf}
	${EndIf}

	Pop $0
!macroend

#
# ${PythonExec} COMMAND_STR
#
# Execute a python interpreter with a command string.
# (example: ${PythonExec} '-c "import this"')
#
!macro PYTHON_EXEC_MACRO COMMAND_LINE_STR
	#ExecWait '$PythonDir\python ${COMMAND_LINE_STR}' $0
	nsExec::ExecToLog '"$PythonDir\python" ${COMMAND_LINE_STR}'
!macroend
!define PythonExec "!insertmacro PYTHON_EXEC_MACRO"

#
# Check if a python package dist_name is present in the python's
# site-packages directory (the result is stored in $0)
# (example  ${IsInstalled} Orange )
#
!define IsDistInstalled '!insertmacro IS_DIST_INSTALLED_MACRO'
!macro IS_DIST_INSTALLED_MACRO DIST_NAME
	${If} ${FileExists} ${DIST_NAME}.egg-info
	${OrIf} $FileExists ${DIST_NAME}*.egg
	${OrIf} ${FileExists} ${DIST_NAME}.dist-info
		StrCpy $0 1
	${Else}
		StrCpy $0 0
	${EndId}
!macroend


# ${InstallPython} python.msi
#
# 	Install Python from a msi installer
#
!macro INSTALL_PYTHON INSTALLER
	Push $1
	${If} ${Silent}
		StrCpy $1 "-qn"
	${Else}
		StrCpy $1 ""
	${EndIf}

	${If} $AdminInstall == 1
		ExecWait 'msiexec.exe $1 -i "${INSTALLER}" ALLUSERS=1' $0
	${Else}
		ExecWait 'msiexec.exe $1 -i "${INSTALLER}"' $0
	${EndIf}

	${If} $0 != 0
		Abort "Error. Could not install required package Python."
	${EndIF}
	Pop $1
!macroend
!define InstallPython "!insertmacro INSTALL_PYTHON"


# ${InstallPythonStandalone} INSTALLER TARGETDIR
#
#	Install python from a msi installer into TARGETDIR but do not
#	register it with Windows.
#
!macro INSTALL_PYTHON_STANDALONE INSTALLER TARGETDIR
	Push $0

	ExecWait 'msiexec.exe -qn -a "${INSTALLER}" TARGETDIR="${TARGETDIR}"' $0

	${If} $0 != 0
		Abort "Error. Could not install required package Python."
	${EndIF}
	Pop $0
!macroend
!define InstallPythonStandalone "!insertmacro INSTALL_PYTHON_STANDALONE"


# Install PyWin32 from a bdist_wininst .exe installer
# (INSTALLER must point to an existing file at install time)

!macro INSTALL_PYWIN32 INSTALLER
#	${If} ${FileExists} "$SysDir\${NAME_MFC}"
#		SetOutPath $SysDir
#		File ${PARTY}\${NAME_MFC}
#	${EndIf}

#	SetOutPath $DESKTOP
#	File ${PARTY}\${INSTALLER}

	${If} ${Silent}
		${PythonExec} '-m easy_install "${INSTALLER}"'
		${PythonExec} '$PythonDir\Scripts\pywin32_postinstall.py'
#		ExecWait "$EASY_INSTALL $DESKTOP\${INSTALLER}"
#		ExecWait "$PYTHON $PYTHON_BIN\pywin32_postinstall.py"
	${Else}
		ExecWait "${INSTALLER}
#		ExecWait "$DESKTOP\${INSTALLER}"
	${EndIf}
	Delete "$DESKTOP\${INSTALLER}"
!macroend


!macro INSTALL_BDIST_WININST INSTALLER
	${If} ${Silent}
		${PythonExec} '-m easy_install "${INSTALLER}"'
	${Else}
		ExecWait "${INSTALLER}"
	${EndIf}
!macroend


#
# ${ExtractTemp} Resource TargetLocation
#
#   Extract a Resource (available at compile time) to
#   Target Location (available at install time)
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
#   Target Location (available at install time)
#
!macro _EXTRACT_TEMP_MACRO_REC RESOURCE LOCATION
	SetOutPath ${LOCATION}
	File /r ${RESOURCE}
!macroend
!define ExtractTempRec "!insertmacro _EXTRACT_TEMP_MACRO_REC"


#
# Ensure pip is installed
#
!macro PIP_BOOTSTRAP
	${PythonExec} '-m ensurepip'
!macroend
!define PipBootstrap "!insertmacro PIP_BOOTSTRAP"


#
# ${PipExec} CMD
#
# Run pip.exe CMD
#
!macro _PIP_EXEC_MACRO COMMAND_LINE_STR
	nsExec::ExecToLog '"$PythonDir\Scripts\pip.exe" ${COMMAND_LINE_STR}'
!macroend
!define PipExec "!insertmacro _PIP_EXEC_MACRO"


#
#  ${Pip} COMMAND_STRING
#
#  Run python -m pip COMMAND_STRING
#
!macro _PIP_MACRO COMMAND_STRING
	${PythonExec} '-m pip ${COMMAND_STRING}'
!macroend
!define Pip "!insertmacro _PIP_MACRO"

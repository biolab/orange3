
!include TextFunc.nsh

# ${GetPythonInstallPEP514} COMPANY TAG $(user_var: INSTALL_PREFIX) $(user_var: INSTALL_MODE)
#
# Retrive the registered install prefix for Python (as documented by PEP 514)
# - input: COMPANY distibutor (use PythonCore for default pyhton.org
#          distributed python)
# - input: TAG environemnt tag (e.g 3.6 or 3.6-32; sys.winver for PythonCore)
# - output: INSTALL_PREFIX installation path (e.g C:\Python35) or empty if
#           not found
# - output: INSTALL_MODE
#           1 if Python was installed for all users, 0 if
#           current user only or -1 when not found.
#
# Example
# -------
# ${GetPythonInstallPEP14} PythonCore 3.5 $PythonDir $InstallMode
# ${GetPythonInstallPEP14} ContinuumAnalytics Anaconda36-64 $1 $2

!macro __GET_PYTHON_INSTALL_PEP514 COMPANY TAG INSTALL_PREFIX INSTALL_MODE
    ReadRegStr ${INSTALL_PREFIX} \
        HKCU Software\Python\${COMPANY}\${TAG}\InstallPath ""
    ${If} ${INSTALL_PREFIX} != ""
        StrCpy ${INSTALL_MODE} 0
    ${Else}
        ReadRegStr ${INSTALL_PREFIX} \
            HKLM Software\Python\${COMPANY}\${TAG}\InstallPath ""
        ${If} ${INSTALL_PREFIX} != ""
            StrCpy ${INSTALL_MODE} 1
        ${Else}
            StrCpy ${INSTALL_MODE} -1
        ${EndIf}
    ${EndIf}

    ${If} ${INSTALL_PREFIX} != ""
        # Strip (single) trailing '\' if present
        Push $0
        StrCpy $0 ${INSTALL_PREFIX} "" -1
        ${If} $0 == "\"
            StrLen $0 ${INSTALL_PREFIX}
            IntOp $0 $0 - 1
            StrCpy ${INSTALL_PREFIX} ${INSTALL_PREFIX} $0 0
        ${EndIf}
        Pop $0
     ${EndIf}
!macroend
!define GetPythonInstallPEP514 "!insertmacro __GET_PYTHON_INSTALL_PEP514"


# ${GetPythonInstall} VERSION BITS $(user_var: INSTALL_PREFIX) $(user_var: INSTALL_MODE)
#
# Retrive the registered install prefix for Python
# - input: VERSION is a Major.Minor version number
# - input: BITS is 32 or 64 constant speciying which python arch to lookup
# - output: INSTALL_PREFIX installation path (e.g C:\Python35) or empty if
#           not found
# - output: INSTALL_MODE
#           1 if Python was installed for all users, 0 if
#           current user only or -1 when not found.
#
# Example
# -------
# ${GetPythonInstall} 3.5 64 $PythonDir $InstallMode

!macro __GET_PYTHON_INSTALL VERSION BITS INSTALL_PREFIX INSTALL_MODE
    !if ${VERSION} < 3.5
        !define __TAG ${VERSION}
    !else if ${BITS} == 64
        !define __TAG ${VERSION}
    !else
        !define __TAG ${VERSION}-${BITS}
    !endif
    ${GetPythonInstallPEP514} PythonCore ${__TAG} ${INSTALL_PREFIX} ${INSTALL_MODE}
    !undef __TAG
!macroend
!define GetPythonInstall "!insertmacro __GET_PYTHON_INSTALL"

# ${GetAnacondaInstall} VERSIONTAG BITS $(user_var: INSTALL_PREFIX) $(user_var: INSTALL_MODE)
#
# Retrive the registered install prefix for Python
# - input: VERSIONTAG is a MajorMinor version number (no dots)
# - input: BITS 34 or 64 (constant)
# - output: INSTALL_PREFIX installation path (e.g C:\Python35)
# - output: INSTALL_MODE
#           1 if Python was installed for all users or 0 if
#           current user only (-1 when not installed).
#
# Example
# -------
# ${GetAnacondaInstall} 35 64 $PythonDir $InstallMode

!macro __GET_CONDA_INSTALL VERSION_TAG BITS INSTALL_PREFIX INSTALL_MODE
    !define __TAG Anaconda${VERSION_TAG}-${BITS}
    ${GetPythonInstallPEP514} ContinuumAnalytics ${__TAG} ${INSTALL_PREFIX} ${INSTALL_MODE}
    !undef __TAG
!macroend
!define GetAnacondaInstall "!insertmacro __GET_CONDA_INSTALL"

# ${FindAnacondaInstall} ROOT_KEY $(user_var: PREFIX)
#    ROOT_KEY: HKCU or HKLM (see ReadRegStr for details)
#    PREFIX: User variable where the install prefix is stored. Empty if
#            no anaconda installation was found.
# Find a registerd anaconda python installation
# In case there are multiple registered installs the last
# one under 'Software\Python\ContinuumAnalytics' registry key is returned
!macro FindAnacondaInstallCall ROOT_KEY PREFIX
    !define __CONDA_REG_PREFIX Software\Python\ContinuumAnalytics
    Push $0
    Push $1
    Push $2
    Push ""  # <stack> $0, $1, $2, ""
    StrCpy $0 0
    StrCpy $1 ""
    StrCpy $2 ""
    ${Do}
        EnumRegKey $1 ${ROOT_KEY} ${__CONDA_REG_PREFIX} $0
        ${If} $1 != ""
            ReadRegStr $2 ${ROOT_KEY} \
                       "${__CONDA_REG_PREFIX}\$1\InstallPath" ""
            ${If} $2 != ""
            ${AndIf} ${FileExists} "$2\python.exe"
            ${AndIf} ${FileExists} "$2\Scripts\conda.exe"
                ${LogWrite} "${ROOT_KEY} ${__CONDA_REG_PREFIX}\$1\InstallPath: $2"
                Exch $2  # <stack> $0, $1, $2, "prefix"
            ${EndIf}
        ${EndIf}
        IntOp $0 $0 + 1
    ${LoopUntil} $1 == ""
    Exch      # <stack> $0, $1, prefix, $2
    Pop $2    # <stack> $0, $1, prefix
    Exch      # <stack> $0, prefix, $1
    Pop $1    # <stack> $0, prefix
    Exch      # <stack> prefix, $0
    Pop $0    # <stack> prefix
    Pop ${PREFIX}
    !undef __CONDA_REG_PREFIX
!macroend
!define FindAnacondaInstall "!insertmacro FindAnacondaInstallCall"

!macro GetAnyAnacondaInstalCall INSTALL_PREFIX INSTALL_MODE
    ${FindAnacondaInstall} HKCU ${INSTALL_PREFIX}
    ${LogWrite} "Anaconda in HKCU: ${INSTALL_PREFIX}"
    ${If} ${INSTALL_PREFIX} == ""
        ${FindAnacondaInstall} HKLM ${INSTALL_PREFIX}
        ${LogWrite} "Anaconda in HKLM: ${INSTALL_PREFIX}"
        ${If} ${INSTALL_PREFIX} != ""
            StrCpy ${INSTALL_MODE} 1
        ${Else}
            StrCpy ${INSTALL_MODE} -1
        ${Endif}
    ${Else}
        StrCpy ${INSTALL_MODE} 0
    ${EndIf}
!macroend
!define GetAnyAnacondaInstall "!insertmacro GetAnyAnacondaInstalCall"

# ${GetPythonVersion} PYTHONEXE $(user_var: RVAL) $(user_var: OUTPUT)
#
# Get the Major.Minor python version string
# - input: PYTHONEXE: Full path to the python.exe interpreter executable
# - output: $(user_var: RVAL) return value (exit status)
# - output: $(user_var: OUTPUT) The Major.Minor version string if command
#           completed successfully
# Example
# -------
# ${GetPythonVersion} 'C:\Python34\python.exe' $0 $1

!macro GetPythonVersionCall PYTHONEXE RVAL OUT
    nsExec::ExecToStack '"${PYTHONEXE}" -c \
        "import sys; print($\'{}.{}$\'.format(*sys.version_info[:2]))"'
    Pop ${RVAL}  # return value (int) or 'error'
    Pop ${OUT}   # output
    ${TrimNewLines} "${OUT}" ${OUT}
!macroend
!define GetPythonVersion "!insertmacro GetPythonVersionCall"

# ${GetPythonArch} PYTHONEXE $(user_var: RVAL) $(user_var: OUTPUT)
#
# Get the python architecute tag (win32 or win_amd64)
# - input: PYTHONEXE: Full path to the python.exe interpreter executable
# - output: $(user_var: RVAL) return value (exit status)
# - output: $(user_var: OUTPUT) 'win32' or 'win_amd64' string if command
#           completed successfully
# Example
# -------
# ${GetPythonArch} 'C:\Python34\python.exe' $0 $1
!macro GetPythonArchCall PYTHONEXE RVAL OUT
    nsExec::ExecToStack '"${PYTHONEXE}" -c \
         "import sys; print($\'win32$\' if sys.maxsize == 2 ** 31 - 1 else \
                            $\'win_amd64$\')"'
    Pop ${RVAL}  # return value  (int) or 'error'
    Pop ${OUT}   # output
    ${TrimNewLines} "${OUT}" ${OUT}
!macroend
!define GetPythonArch "!insertmacro GetPythonArchCall"

# ${CommandOutput} EXECUTABLE ARGS $(user_val:RVAL) $(user_var:OUT)
#
# Capture the output of executing '"${EXECUTABLE}" ${ARGS}' (trailing new
# lines are striped)

!macro CommandOutputCall EXECUTABLE ARGS RVAL OUT
    nsExec::ExecToStack '"${EXECUTABLE}" ${ARGS}'
    Pop ${RVAL}  # return value  (int) or 'error'
    Pop ${OUT}   # command output
    ${TrimNewLines} "${OUT}" ${OUT}
!macroend
!define CommandOutput "!insertmacro CommandOutputCall"


!macro GetCondaVersionCall PYTHONEXE RVAL OUT
    ${CommandOutput} "${PYTHONEXE}" \
        '-c "import sys, conda; print(conda.__version__)"' \
        ${RVAL} ${OUT}
!macroend
!define GetCondaVersion "!insertmacro GetCondaVersionCall"


!macro PyInstallNoRegisterCall INSTALLER TARGETDIR RETVAL
    Push $0
    Push $1
    ${GetFileExt} "${INSTALLER}" $0
    ${GetFileName} "${INSTALLER}" $1
    DetailPrint "Installing Python: $1"
    SetDetailsPrint listonly
    ${If} $0 == "msi"
        DetailPrint 'Executing: \
            msiexec.exe -qn -a "${INSTALLER}" TARGETDIR="${TARGETDIR}"'
        nsExec::ExecToLog \
            'msiexec.exe -qn -a "${INSTALLER}" TARGETDIR="${TARGETDIR}"'
    ${ElseIf} $0 == "exe"
        # !! This is wrong !! the new .exe installers always add at least
        # some reg keys in uninstall section (http://bugs.python.org/issue29231)
        nsExec::ExecToLog \
            '"${INSTALLER}" -quiet \
                InstallAllUsers=0 AssociateFiles=0 PrependPath=0 \
                Shortcuts=0 Include_launcher=0 InstallLauncherAllUsers=0 \
                TargetDir="${TargetDir}"'
    ${Else}
        Abort "PyInstallNoRegister Error: ${INSTALLER} - \
               invalid filename (extension)"
    ${EndIf}
    Exch
    Pop $1
    Exch
    Pop $0
    Pop ${RETVAL}
!macroend
!define PyInstallNoRegister "!insertmacro PyInstallNoRegisterCall"


!macro PyInstallCall INSTALLER INSTALLMODE RETVAL
    Push $0
    Push $1
    ${GetFileExt} "${INSTALLER}" $0
    ${GetFileName} "${INSTALLER}" $1
    DetailPrint "Installing Python: $1"
    SetDetailsPrint listonly
    ${If} $0 == "msi"
        # Python (<3.5) .msi installer
        ${If} "${INSTALLMODE}" == AllUsers
            StrCpy $1 " ALLUSERS=1"
        ${Else}
            StrCpy $1 ""
        ${EndIf}

        ${If} ${Silent}
            DetailPrint 'Executing: msiexec.exe -qn -i "${INSTALLER}"$1'
            nsExec::ExecToLog 'msiexec.exe -qn -i "${INSTALLER}"$1'
        ${Else}
            DetailPrint 'Executing: msiexec.exe -i "${INSTALLER}"$1'
            nsExec::ExecToLog 'msiexec.exe -i "${INSTALLER}"$1'
        ${EndIf}
    ${ElseIf} $0 == "exe"
        # Python (<=3.5).exe installer
        ${If} "${INSTALLMODE}" == AllUsers
            StrCpy $1 "1"
        ${Else}
            StrCpy $1 "0"
        ${EndIf}
        ${If} ${Silent}
            DetailPrint 'Executing: "${INSTALLER}" -quiet InstallAllUsers=$1'
            nsExec::ExecToLog '"${INSTALLER}" -quiet InstallAllUsers=$1'
        ${Else}
            DetailPrint 'Executing: "${INSTALLER}" InstallAllUsers=$1'
            nsExec::ExecToLog '"${INSTALLER}" InstallAllUsers=$1'
        ${EndIf}
    ${Else}
        Abort "PyInstall Error: ${INSTALLER} - invalid filename (extension)"
    ${EndIf}
    SetDetailsPrint lastused
    Exch
    Pop $1
    Exch
    Pop $0
    Pop ${RETVAL}
!macroend
!define PyInstall "!insertmacro PyInstallCall"
